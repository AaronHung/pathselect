#!/usr/bin/env python3
"""S2 — SeqFT：序列訓練下的遺忘（沒有任何 CL 機制）。

一個 shared selector 依序學 4 個 task，每學完一個就在**所有已學過的 task** 上評估。
joint 訓練只能證明 multi-task interference，不能代替這個實驗。

架構 L3b：joint 架構（單一 shared selector，非 per-task）但**不給 q_tau** ——
  use_query=False、use_state=False、hierarchy=False。

三個軸（缺一不可；舊實驗已證明 selection 行為可以崩掉而 accuracy 幾乎不動）
  A1  accuracy forgetting        acc(T_i | 學完 T_i) − acc(T_i | 學完 T_4)
  A2  selection-behaviour forgetting
        同一批 T_i slide，「學完 T_i 時選的 patch 集合」vs「學完 T_4 後選的集合」
        的 Jaccard；另報 group 配額分佈的 KL divergence
  A3  utility retention
        U(S) = 沿選取順序累加的 counterfactual gain 總和。frozen head 不隨訓練改變，
        所以 U 只取決於「選了哪些 patch」。retention = ΣU(學完 T_4 的選擇)
        / ΣU(學完 T_i 的選擇) —— 選的東西變了，新選的東西一樣有用嗎。

每個 (order, seed, task) 跑完就落檔（模型 checkpoint + 逐 slide 記錄），中斷可續跑。
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from selector.classifier import conch_classify, softmax_weights          # noqa: E402
from selector.evaluate import read_slide, slide_dataset                  # noqa: E402
from selector.grouping import (NUM_GROUPS, TISSUE_GROUP_NAMES,           # noqa: E402
                               assign_groups, tissue_text_features)
from selector.model import GroupSelector, PatchSelector                  # noqa: E402
from selector.priors import MAINLINE_PRIOR, semantic_prior               # noqa: E402
from selector.rounds import run_rounds                                   # noqa: E402
from selector.task_query import encode_task_query                        # noqa: E402
from selector.text_encoder import build_f_txt, load_config               # noqa: E402
from selector.train import evidence_loss, frozen_head                    # noqa: E402
from selector.utility import counterfactual_gain                         # noqa: E402

OUT_DIR = REPO_ROOT / "outputs" / "exp2" / "seqft"
#: L3b：shared selector、無 q_tau、無 state、無 hierarchy
L3B = dict(use_query=False, use_state=False, hierarchy=False)
ORDERS = {
    "reverse": ["tcga_esca", "tcga_rcc", "tcga_brca", "tcga_lung"],
    "main": ["tcga_lung", "tcga_brca", "tcga_rcc", "tcga_esca"],
}
SLIDE_CACHE = 256
EPS = 1e-9


class Ctx:
    def __init__(self, cfg, device="cpu"):
        from collections import OrderedDict
        self.cfg = cfg
        self.device = torch.device(device)
        self.label_space = list(cfg["tasks"])        # 8-way label space 的疊放順序
        self.f_txt = torch.cat([build_f_txt(t, cfg, device=device).f_txt
                                for t in self.label_space], 0)
        self.logit_scale = build_f_txt(self.label_space[0], cfg, device=device).logit_scale
        self.tissue = tissue_text_features(cfg, device=device)
        self._ds, self._lru = {}, OrderedDict()
        # q_tau 在 L3b 不使用，但 run_rounds 仍需一個佔位張量（use_query=False 會填零）
        self.q0 = torch.zeros(512, device=self.device)

    def dataset(self, task, split):
        key = (task, split)
        if key not in self._ds:
            self._ds[key] = slide_dataset(self.cfg, task,
                                          self.label_space.index(task), split)
        return self._ds[key]

    def n_slides(self, task, split):
        return len(self.dataset(task, split)[0])

    def get(self, task, split, i):
        key = (task, split, i)
        if key in self._lru:
            self._lru.move_to_end(key)
            return self._lru[key]
        ds, shift = self.dataset(task, split)
        rec = read_slide(ds, shift, i)
        val = (rec, assign_groups(rec.Z, self.tissue))
        self._lru[key] = val
        if len(self._lru) > SLIDE_CACHE:
            self._lru.popitem(last=False)
        return val


# ── utility ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def utility_total(ctx, Z, selected_idx, label) -> float:
    """沿選取順序累加 counterfactual gain。frozen head 固定，故只取決於選了誰。"""
    S = torch.zeros(Z.shape[1], dtype=Z.dtype, device=Z.device)
    total, n = 0.0, 0
    for i in selected_idx.tolist():
        x = Z[i].reshape(1, -1)
        u = counterfactual_gain(S, n, x, ctx.f_txt, ctx.logit_scale, label)
        total += float(u[0])
        S = S + Z[i]
        n += 1
    return total


# ── train / eval ────────────────────────────────────────────────────────────

def train_task(ctx, models, task, seed, args):
    f_g, f_p = models
    f_g.train(); f_p.train()
    opt = torch.optim.Adam(list(f_g.parameters()) + list(f_p.parameters()),
                           lr=args.lr, weight_decay=1e-4)
    n = ctx.n_slides(task, "train")
    idxs = list(range(n if args.max_train <= 0 else min(args.max_train, n)))
    for epoch in range(args.epochs):
        g = torch.Generator().manual_seed(seed * 1000 + epoch)
        order = torch.randperm(len(idxs), generator=g).tolist()
        total = 0.0
        for k in order:
            rec, grp = ctx.get(task, "train", idxs[k])
            res = run_rounds(rec.Z, grp, ctx.q0, f_g, f_p,
                             budget=args.budget, chunk=args.chunk, **L3B)
            ste = sum(r.ste_mask for r in res.records)
            s_last = res.records[-1].s
            logits = frozen_head(rec.Z, s_last, ste, ctx.f_txt, ctx.logit_scale)
            prior = semantic_prior(rec.Z, ctx.f_txt, kind=args.prior,
                                   n_candidate_classes=ctx.f_txt.shape[0],
                                   logit_scale=ctx.logit_scale)
            loss, _ = evidence_loss(logits, rec.label, s_last, prior, beta_s=args.beta_s)
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.detach())
        print(f"      epoch {epoch + 1}/{args.epochs} n={len(idxs)} "
              f"mean_loss={total / len(idxs):.4f}", flush=True)
    f_g.eval(); f_p.eval()
    return models


@torch.no_grad()
def eval_task(ctx, models, task, order_name, seed, stage, args):
    f_g, f_p = models
    out = []
    for i in range(ctx.n_slides(task, "test")):
        rec, grp = ctx.get(task, "test", i)
        res = run_rounds(rec.Z, grp, ctx.q0, f_g, f_p,
                         budget=args.budget, chunk=args.chunk, **L3B)
        idx = res.selected
        s = res.records[-1].s
        w = softmax_weights(s, idx)
        quota = [0] * NUM_GROUPS
        for j in grp.assignment.index_select(0, idx).tolist():
            quota[j] += 1
        out.append({
            "order": order_name, "seed": seed, "stage": stage, "task": task,
            "slide_id": rec.sid, "true": rec.label,
            "pred_softmax": int(conch_classify(rec.Z.index_select(0, idx), w,
                                               ctx.f_txt, ctx.logit_scale)
                                .reshape(-1).argmax()),
            "pred_uniform": int(conch_classify(rec.Z.index_select(0, idx), None,
                                               ctx.f_txt, ctx.logit_scale)
                                .reshape(-1).argmax()),
            "selected_idx": idx.tolist(),
            "weights_softmax": [round(float(x), 6) for x in w],
            "weights_uniform": [round(1.0 / max(idx.numel(), 1), 6)] * idx.numel(),
            "group_quota": quota,
            "utility_total": utility_total(ctx, rec.Z, idx, rec.label),
            "B": args.budget,
        })
    return out


# ── metrics ─────────────────────────────────────────────────────────────────

def acc(records, weighting="softmax") -> float:
    return sum(r[f"pred_{weighting}"] == r["true"] for r in records) / len(records)


def jaccard(a: list[int], b: list[int]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 1.0


def kl_quota(p_counts, q_counts) -> float:
    """KL(P || Q)，group 配額分佈，加 Laplace 平滑避免 log 0。"""
    p = [(c + 1) for c in p_counts]
    q = [(c + 1) for c in q_counts]
    sp, sq = sum(p), sum(q)
    return sum((pi / sp) * math.log((pi / sp) / (qi / sq)) for pi, qi in zip(p, q))


def axes_for(records, order_tasks, seed):
    """回傳每個 task 的 A1 / A2 / A3。"""
    last = len(order_tasks) - 1
    out = {}
    for i, task in enumerate(order_tasks):
        at_i = [r for r in records if r["seed"] == seed and r["stage"] == i
                and r["task"] == task]
        at_end = [r for r in records if r["seed"] == seed and r["stage"] == last
                  and r["task"] == task]
        if not at_i or not at_end:
            continue
        by_sid = {r["slide_id"]: r for r in at_end}
        pairs = [(r, by_sid[r["slide_id"]]) for r in at_i if r["slide_id"] in by_sid]
        q_i = [sum(r["group_quota"][j] for r in at_i) for j in range(NUM_GROUPS)]
        q_e = [sum(r["group_quota"][j] for r in at_end) for j in range(NUM_GROUPS)]
        u_i = sum(r["utility_total"] for r in at_i)
        u_e = sum(r["utility_total"] for r in at_end)
        out[task] = {
            "acc_at_learn": acc(at_i), "acc_at_end": acc(at_end),
            "A1_acc_forgetting": (acc(at_i) - acc(at_end)) * 100,
            "A2_jaccard": statistics.mean(
                [jaccard(a["selected_idx"], b["selected_idx"]) for a, b in pairs]),
            "A2_quota_kl": kl_quota(q_i, q_e),
            "A3_utility_at_learn": u_i, "A3_utility_at_end": u_e,
            "A3_retention": (u_e / u_i) if abs(u_i) > EPS else float("nan"),
        }
    return out


# ── main ────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orders", default="reverse,main")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--beta-s", type=float, default=0.1)
    ap.add_argument("--prior", default=MAINLINE_PRIOR)
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=1)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--report-only", action="store_true",
                    help="從既有的 results.json 重繪報告，不訓練也不評估")
    args = ap.parse_args()

    cfg = load_config()
    orders = [o.strip() for o in args.orders.split(",") if o.strip()]
    seeds = [int(x) for x in args.seeds.split(",")]
    (OUT_DIR / "per_slide").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "ckpt").mkdir(parents=True, exist_ok=True)

    ctx = Ctx(cfg)
    if args.report_only:
        recs = json.loads((OUT_DIR / "results.json").read_text())
        write_report(ctx, recs, orders, seeds, args)
        print(f"→ {OUT_DIR / 'SEQFT.md'}（report-only）")
        return 0
    print(f"SeqFT  orders={orders}  seeds={seeds}  B={args.budget} c={args.chunk} "
          f"epochs={args.epochs}  arch=L3b(no q_tau, no state, flat)  無任何 CL 機制",
          flush=True)

    all_recs = []
    for order_name in orders:
        tasks = ORDERS[order_name]
        for seed in seeds:
            torch.manual_seed(seed)
            f_g, f_p = GroupSelector().to(ctx.device), PatchSelector().to(ctx.device)
            print(f"  ▶ order={order_name} seed={seed}  序列 {' → '.join(tasks)}",
                  flush=True)
            for stage, task in enumerate(tasks):
                tag = f"{order_name}_seed{seed}_stage{stage}"
                rec_path = OUT_DIR / "per_slide" / f"{tag}.json"
                ck_path = OUT_DIR / "ckpt" / f"{tag}.pt"
                if rec_path.exists() and ck_path.exists() and not args.no_resume:
                    blob = torch.load(ck_path, map_location=ctx.device)
                    f_g.load_state_dict(blob["f_g"]); f_p.load_state_dict(blob["f_p"])
                    recs = json.loads(rec_path.read_text())
                    all_recs += recs
                    print(f"    ▷ stage{stage} {task} 已有存檔，跳過", flush=True)
                    continue
                print(f"    ── stage {stage}: 訓練 {task}", flush=True)
                train_task(ctx, (f_g, f_p), task, seed, args)
                recs = []
                for seen in tasks[:stage + 1]:          # 所有已學過的 task
                    r = eval_task(ctx, (f_g, f_p), seen, order_name, seed, stage, args)
                    recs += r
                    print(f"       eval {seen:10s} acc={acc(r):.4f}", flush=True)
                rec_path.write_text(json.dumps(recs, indent=1))
                torch.save({"f_g": f_g.state_dict(), "f_p": f_p.state_dict()}, ck_path)
                all_recs += recs

    (OUT_DIR / "results.json").write_text(json.dumps(all_recs, indent=1))
    write_report(ctx, all_recs, orders, seeds, args)
    print(f"\n→ {OUT_DIR / 'SEQFT.md'}")
    return 0


def ms(vals, fmt="{:.4f}"):
    vals = [v for v in vals if v == v]
    if not vals:
        return "—"
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f"{fmt.format(statistics.mean(vals))} ± {fmt.format(sd)}"


def write_report(ctx, recs, orders, seeds, args) -> None:
    n_test = {t: ctx.n_slides(t, "test") for t in ctx.label_space}
    L = [
        "# S2 — SeqFT：序列訓練下的遺忘",
        "",
        "一個 shared selector 依序學 4 個 task，**沒有任何 CL 機制**"
        "（無 replay、無 distillation、無 LoRA merge）。每學完一個 task 就在所有"
        "已學過的 task 上評估。joint 訓練只能證明 multi-task interference，"
        "不能代替這個實驗。",
        "",
        f"架構 L3b（shared selector，不給 q_tau、無 state、flat）、"
        f"B={args.budget}、c={args.chunk}、seeds {seeds}、epochs {args.epochs}、"
        f"lr {args.lr}、beta_s {args.beta_s}、prior {args.prior}。"
        "訓練用 train split、評估用 test split、8-way label space。",
        "",
        "## 事前預測（跑之前寫下，跑完不得修改）",
        "",
        "> PI 判讀（承 S1）：task identity 在 F_g 的實際輸入（group prototype）上",
        "> 98.57% 線性可讀，因此 q_tau 在此 benchmark 中結構性冗餘，L4 ≈ L3b 為預期",
        "> 結果而非模型缺陷。",
        ">",
        "> 延伸預測：shared selector 可隱式按器官路由，故 **accuracy 層 forgetting",
        "> 可能偏輕**，但 **selection 行為層仍可能嚴重漂移**。此預測於 S2 跑完後",
        "> 對照，不得事後修改。",
        "",
        "⚠️ 不得因為 accuracy forgetting 小就下結論說沒有遺忘 —— 三個軸一起看。",
        "",
    ]
    for order_name in orders:
        tasks = ORDERS[order_name]
        short = [t.replace("tcga_", "") for t in tasks]
        rs = [r for r in recs if r["order"] == order_name]
        L += [f"## order = {order_name}　（{' → '.join(short)}）", "",
              "### 表 1：accuracy matrix（3 seeds mean ± std，softmax 權重）", "",
              "| 學完 | " + " | ".join(f"eval {s}" for s in short) + " |",
              "|---" * (len(short) + 1) + "|"]
        for stage in range(len(tasks)):
            cells = []
            for j, t in enumerate(tasks):
                if j > stage:
                    cells.append("—")
                    continue
                cells.append(ms([acc([r for r in rs if r["seed"] == sd
                                      and r["stage"] == stage and r["task"] == t])
                                 for sd in seeds]))
            L.append(f"| T{stage + 1} {short[stage]} | " + " | ".join(cells) + " |")
        L += ["", f"n（test）：" + "、".join(f"{s} {n_test[t]}"
                                            for s, t in zip(short, tasks)) + "。",
              "", "### 表 2：三個軸（3 seeds mean ± std）", "",
              "| task | n | A1 accuracy forgetting (pp) | A2 Jaccard | "
              "A2 quota KL | A3 ΣU 學完 T_i | A3 ΣU 學完 T_4 | A3 retention |",
              "|---|---|---|---|---|---|---|---|"]
        per_seed = {sd: axes_for(rs, tasks, sd) for sd in seeds}
        for t in tasks:
            g = lambda k: [per_seed[sd][t][k] for sd in seeds if t in per_seed[sd]]
            L.append(f"| {t} | {n_test[t]} | {ms(g('A1_acc_forgetting'), '{:+.2f}')} | "
                     f"{ms(g('A2_jaccard'))} | {ms(g('A2_quota_kl'))} | "
                     f"{ms(g('A3_utility_at_learn'), '{:.1f}')} | "
                     f"{ms(g('A3_utility_at_end'), '{:.1f}')} | "
                     f"{ms(g('A3_retention'))} |")
        L += ["",
              "- **A1** = acc(T_i | 學完 T_i) − acc(T_i | 學完 T_4)，正值代表退步。",
              "- **A2 Jaccard** = 同一批 slide 在兩個時點選到的 patch 集合重疊度；"
              "1.0 = 完全沒變，0.0 = 完全換掉。",
              "- **A2 quota KL** = group 配額分佈 KL(學完 T_i ‖ 學完 T_4)，"
              "Laplace 平滑；0 = 分佈沒變。",
              "- **A3** ΣU 是該 task 全部 test slide 的 utility 加總。"
              "U 沿選取順序累加 counterfactual gain；frozen head 不隨訓練改變，"
              "所以 U 只取決於選了哪些 patch。retention = ΣU(學完 T_4) / ΣU(學完 T_i)："
              "1.0 = 效用完全保留，0 = 新選的東西完全沒用，**負值 = 新選的東西是反效果**"
              "（把證據推向錯誤類別，比什麼都不看還糟）。",
              "", "### T4 學完後，各 task 的 group 配額分佈", "",
              "| task | 時點 | " + " | ".join(TISSUE_GROUP_NAMES) + " |",
              "|---" * (len(TISSUE_GROUP_NAMES) + 2) + "|"]
        for i, t in enumerate(tasks):
            stages = [("學完 T%d" % (i + 1), i)]
            if i != len(tasks) - 1:      # 最後一個 task 的兩個時點是同一個，不重複列
                stages.append(("學完 T%d" % len(tasks), len(tasks) - 1))
            for label, stage in stages:
                sub = [r for r in rs if r["stage"] == stage and r["task"] == t]
                if not sub:
                    continue
                tot = [sum(r["group_quota"][j] for r in sub) for j in range(NUM_GROUPS)]
                s = sum(tot) or 1
                L.append(f"| {t.replace('tcga_', '')} | {label} | "
                         + " | ".join(f"{v / s:.3f}" for v in tot) + " |")
        L.append("")

    L += ["## 事前預測對照（跑完後填入；上面的預測段落未修改）", "", "| 預測 | 觀察 |",
          "|---|---|"]
    for order_name in orders:
        tasks = ORDERS[order_name]
        rs = [r for r in recs if r["order"] == order_name]
        per_seed = {sd: axes_for(rs, tasks, sd) for sd in seeds}
        early = [t for t in tasks[:-1]]
        a1 = [per_seed[sd][t]["A1_acc_forgetting"] for sd in seeds for t in early
              if t in per_seed[sd]]
        jac = [per_seed[sd][t]["A2_jaccard"] for sd in seeds for t in early
               if t in per_seed[sd]]
        L.append(f"| ({order_name}) accuracy 層 forgetting **偏輕** | "
                 f"前 3 個 task 的 A1 平均 **{statistics.mean(a1):+.2f} pp**"
                 f"（範圍 {min(a1):+.2f} ~ {max(a1):+.2f}） |")
        L.append(f"| ({order_name}) selection 行為層**嚴重漂移** | "
                 f"前 3 個 task 的 Jaccard 平均 **{statistics.mean(jac):.4f}**"
                 f"（範圍 {min(jac):.4f} ~ {max(jac):.4f}） |")
    L += ["", "判讀由 PI 進行；此處只陳述數字。", "",
          "## 產出檔案", "",
          "- 逐 slide 預測與選中 index：`outputs/exp2/seqft/per_slide/"
          "{order}_seed{seed}_stage{k}.json`",
          "- 曲線資料（accuracy 隨訓練進度）：`outputs/exp2/seqft/curves.json`",
          "- 模型 checkpoint：`outputs/exp2/seqft/ckpt/`（每個 (order, seed, task) 一份，"
          "中斷可續跑）", ""]
    (OUT_DIR / "SEQFT.md").write_text("\n".join(L) + "\n")

    curves = []
    for order_name in orders:
        tasks = ORDERS[order_name]
        for t in tasks:
            for stage in range(len(tasks)):
                for sd in seeds:
                    sub = [r for r in recs if r["order"] == order_name
                           and r["seed"] == sd and r["stage"] == stage and r["task"] == t]
                    if sub:
                        curves.append({"order": order_name, "task": t, "stage": stage,
                                       "seed": sd, "acc_softmax": acc(sub),
                                       "acc_uniform": acc(sub, "uniform"),
                                       "n": len(sub)})
    (OUT_DIR / "curves.json").write_text(json.dumps(curves, indent=1))


if __name__ == "__main__":
    raise SystemExit(main())
