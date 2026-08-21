#!/usr/bin/env python3
"""Exp 1 — 單任務 / joint 消融階梯。

共同設定（不得逐級變動）：fold 1、reverse order、8-way label space、B=8、c=1、
3 seeds (0,1,2)、frozen head（score-weighted pooling → L2 normalize → CONCH
class-text logits）。每一級都同時報 softmax 與 uniform 兩種權重（PI 裁定 B）。

階梯（開關差異；架構與參數量在各級之間相同，只有輸入資訊量不同）
  L3  flat learned selector      per-task   query✗ state✗ hierarchy✗
  L4  + task conditioning        joint      query✓ state✗ hierarchy✗
  L5  + Group → Patch hierarchy  joint      query✓ state✗ hierarchy✓
  L6  + E_t / B_t stateful       joint      query✓ state✓ hierarchy✓

⚠️ L3 用新 pipeline 重訓，不沿用 reference/v9 的 skill bank（那是舊 f_txt 下訓練的）。
⚠️ L4–L6 一律 joint 訓練：per_task 模式下 q_tau 是常數會被第一層 bias 吸收，
   L4 相對 L3 的差異在數學上保證為零。主表分開標示 per-task / joint。

指標（pre-register）
  主要 = 四 task 在 B=8 的平均 accuracy（3 seeds，mean ± std）
  次要 = budget 曲線 B ∈ {1,2,4,8,16}
兩者都報。

裁定 C：每一次評估都落一份逐 slide JSON
（slide_id, task, true, pred, selected_idx, weights）。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from selector.classifier import conch_classify, softmax_weights          # noqa: E402
from selector.evaluate import read_slide, slide_dataset                  # noqa: E402
from selector.grouping import TISSUE_GROUP_NAMES, assign_groups, tissue_text_features  # noqa: E402
from selector.model import GroupSelector, PatchSelector                  # noqa: E402
from selector.rounds import run_rounds                                   # noqa: E402
from selector.task_query import encode_task_query                        # noqa: E402
from selector.text_encoder import build_f_txt, load_config               # noqa: E402
from selector.train import evidence_loss, frozen_head                    # noqa: E402
from selector.priors import MAINLINE_PRIOR, semantic_prior               # noqa: E402

OUT_ROOT = REPO_ROOT / "outputs" / "exp1"
PRIMARY_B = 8
BUDGET_CURVE = (1, 2, 4, 8, 16)
WEIGHTINGS = ("softmax", "uniform")
DEFAULT_SEEDS = (0, 1, 2)
#: esca 只有 15 張 test slide，一張 = 6.67 pp
INDISTINGUISHABLE_PP = 100.0 / 15

LEVELS = {
    "L3": dict(name="Flat learned selector", mode="per_task",
               use_query=False, use_state=False, hierarchy=False),
    "L4": dict(name="+ task conditioning q_tau", mode="joint",
               use_query=True, use_state=False, hierarchy=False),
    "L5": dict(name="+ Group → Patch hierarchy", mode="joint",
               use_query=True, use_state=False, hierarchy=True),
    "L6": dict(name="+ E_t / B_t stateful", mode="joint",
               use_query=True, use_state=True, hierarchy=True),
}


# ── shared context ──────────────────────────────────────────────────────────

#: slide 的 LRU 快取上限。單張 ~7 MB（3400 patch × 512 × 4 B），機器只有 16 GB，
#: 把 2273 張訓練 slide 全快取會直接 swap，所以必須有界。
SLIDE_CACHE = 256


class Ctx:
    """共用的凍結素材 + 有界的 slide 快取（LRU）。"""

    def __init__(self, cfg, device="cpu"):
        from collections import OrderedDict
        self.cfg = cfg
        self.device = torch.device(device)
        self.tasks = list(cfg["tasks"])
        self.f_txt = torch.cat([build_f_txt(t, cfg, device=device).f_txt
                                for t in self.tasks], 0)              # [8, 512]
        self.logit_scale = build_f_txt(self.tasks[0], cfg, device=device).logit_scale
        self.tissue = tissue_text_features(cfg, device=device)
        self.q = {t: encode_task_query(t, cfg, device=device) for t in self.tasks}
        self._ds: dict[tuple[str, str], tuple] = {}
        self._lru = OrderedDict()

    def dataset(self, task, split):
        key = (task, split)
        if key not in self._ds:
            self._ds[key] = slide_dataset(self.cfg, task, self.tasks.index(task), split)
        return self._ds[key]

    def n_slides(self, task, split) -> int:
        return len(self.dataset(task, split)[0])

    def get(self, task, split, i):
        """回傳 (SlideRecord, Grouping)；有界 LRU，避免把整個 split 吃進記憶體。"""
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

    def indices(self, task, split, limit=0):
        n = self.n_slides(task, split)
        return list(range(n if limit <= 0 else min(limit, n)))


# ── training ────────────────────────────────────────────────────────────────

def train_one(ctx, level: str, seed: int, tasks: list[str], epochs: int,
              lr: float, beta_s: float, prior_kind: str, budget: int, chunk: int,
              q_override=None, max_train: int = 0):
    """回傳訓練好的 (f_group, f_patch)。tasks 決定訓練資料來源。"""
    spec = LEVELS[level]
    torch.manual_seed(seed)
    f_g, f_p = GroupSelector().to(ctx.device), PatchSelector().to(ctx.device)
    opt = torch.optim.Adam(list(f_g.parameters()) + list(f_p.parameters()),
                           lr=lr, weight_decay=1e-4)
    stream = [(t, i) for t in tasks for i in ctx.indices(t, "train", max_train)]
    for epoch in range(epochs):
        g = torch.Generator().manual_seed(seed * 1000 + epoch)
        order = torch.randperm(len(stream), generator=g).tolist()
        total = 0.0
        for k in order:
            task, si = stream[k]
            rec, grp = ctx.get(task, "train", si)
            q = q_override(task) if q_override else ctx.q[task]
            res = run_rounds(rec.Z, grp, q, f_g, f_p, budget=budget, chunk=chunk,
                             use_query=spec["use_query"], use_state=spec["use_state"],
                             hierarchy=spec["hierarchy"])
            ste = sum(r.ste_mask for r in res.records)
            s_last = res.records[-1].s
            logits = frozen_head(rec.Z, s_last, ste, ctx.f_txt, ctx.logit_scale)
            prior = semantic_prior(rec.Z, ctx.f_txt, kind=prior_kind,
                                   n_candidate_classes=ctx.f_txt.shape[0],
                                   logit_scale=ctx.logit_scale)
            loss, _ = evidence_loss(logits, rec.label, s_last, prior, beta_s=beta_s)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
        print(f"    [{level} seed{seed}] epoch {epoch + 1}/{epochs} "
              f"n={len(stream)} mean_loss={total / len(stream):.4f}", flush=True)
    return f_g.eval(), f_p.eval()


# ── evaluation ──────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(ctx, level, seed, models, task, budgets, chunk, q_override=None):
    """回傳逐 slide 記錄（每個 B 一筆，內含兩種 weighting 的預測）。"""
    spec = LEVELS[level]
    f_g, f_p = models
    q = q_override(task) if q_override else ctx.q[task]
    out = []
    for si in ctx.indices(task, "test"):
        rec, grp = ctx.get(task, "test", si)
        for B in budgets:
            res = run_rounds(rec.Z, grp, q, f_g, f_p, budget=B, chunk=chunk,
                             use_query=spec["use_query"], use_state=spec["use_state"],
                             hierarchy=spec["hierarchy"])
            idx = res.selected
            s = res.records[-1].s
            w_soft = softmax_weights(s, idx)
            preds = {
                "softmax": int(conch_classify(rec.Z.index_select(0, idx), w_soft,
                                              ctx.f_txt, ctx.logit_scale)
                               .reshape(-1).argmax()),
                "uniform": int(conch_classify(rec.Z.index_select(0, idx), None,
                                              ctx.f_txt, ctx.logit_scale)
                               .reshape(-1).argmax()),
            }
            out.append({
                "level": level, "seed": seed, "task": task, "B": B,
                "slide_id": rec.sid, "true": rec.label,
                "pred_softmax": preds["softmax"], "pred_uniform": preds["uniform"],
                "selected_idx": idx.tolist(),
                "weights_softmax": [round(float(x), 6) for x in w_soft],
                "weights_uniform": [round(1.0 / max(idx.numel(), 1), 6)] * idx.numel(),
                "eff_K": float(1.0 / w_soft.pow(2).sum()),
                "group_seq": [int(r.b.argmax()) for r in res.records],
                "group_quota": res.records[-1].b.tolist() if spec["hierarchy"] else
                               [sum(int(r.b[j]) for r in res.records)
                                for j in range(len(TISSUE_GROUP_NAMES))],
            })
    return out


def acc(records, task, B, weighting) -> float:
    rs = [r for r in records if r["task"] == task and r["B"] == B]
    return sum(r[f"pred_{weighting}"] == r["true"] for r in rs) / len(rs)


# ── reporting ───────────────────────────────────────────────────────────────

def ms(vals) -> str:
    if not vals:
        return "—"
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f"{statistics.mean(vals):.4f} ± {sd:.4f}"


def write_results(ctx, all_recs, levels, seeds, out_dir, args) -> dict:
    """RESULTS.md：列 = task × level，欄 = softmax / uniform。"""
    n_test = {t: len({r["slide_id"] for r in all_recs if r["task"] == t})
              for t in ctx.tasks}
    per = {}   # (level, task, B, weighting) -> [acc per seed]
    for lv in levels:
        for t in ctx.tasks:
            for B in BUDGET_CURVE:
                for w in WEIGHTINGS:
                    per[(lv, t, B, w)] = [
                        acc([r for r in all_recs if r["level"] == lv and r["seed"] == s],
                            t, B, w) for s in seeds]

    def mean_over_tasks(lv, B, w):
        return [statistics.mean([per[(lv, t, B, w)][i] for t in ctx.tasks])
                for i in range(len(seeds))]

    L = [
        f"# Exp 1 Stage {args.stage} — 消融階梯結果",
        "",
        f"fold 1、reverse order、8-way label space、B={PRIMARY_B}、c={args.chunk}、"
        f"seeds {list(seeds)}、epochs {args.epochs}、lr {args.lr}、"
        f"beta_s {args.beta_s}、prior {args.prior}。",
        "訓練用 train split，評估用 test split。frozen head = score-weighted "
        "pooling → L2 normalize → CONCH class-text logits，無 trained diagnosis head。",
        "",
        "**權重（PI 裁定 B）**：同一個訓練好的 selector，評估時分別用兩種聚合權重。",
        "拆解方式：",
        "",
        "```",
        "learned(uniform) vs random(uniform)   = 純選取能力",
        "learned(softmax) vs learned(uniform)  = 加權貢獻",
        "```",
        "",
        "## 主要指標 —— B=8 accuracy（3 seeds mean ± std）",
        "",
        "| task | n | 訓練模式 | level | softmax | uniform |",
        "|---|---|---|---|---|---|",
    ]
    for t in ctx.tasks:
        for lv in levels:
            L.append(f"| {t} | {n_test[t]} | {LEVELS[lv]['mode']} | "
                     f"{lv} {LEVELS[lv]['name']} | "
                     f"{ms(per[(lv, t, PRIMARY_B, 'softmax')])} | "
                     f"{ms(per[(lv, t, PRIMARY_B, 'uniform')])} |")
    L += ["", "### 四 task 平均（B=8）", "",
          "| level | 訓練模式 | softmax | uniform |", "|---|---|---|---|"]
    summary = {}
    for lv in levels:
        soft, uni = mean_over_tasks(lv, PRIMARY_B, "softmax"), mean_over_tasks(lv, PRIMARY_B, "uniform")
        summary[lv] = {"softmax": soft, "uniform": uni}
        L.append(f"| {lv} {LEVELS[lv]['name']} | {LEVELS[lv]['mode']} | "
                 f"{ms(soft)} | {ms(uni)} |")

    L += ["", "⚠️ **per-task 與 joint 不在同一欄比較**：L3 是 per-task 訓練"
          "（每個 task 一個模型），L4+ 是 joint 訓練（一個模型跑全部 task）。",
          "",
          "## 次要指標 —— budget 曲線（四 task 平均，3 seeds mean ± std）", "",
          "| level | weighting | " + " | ".join(f"B={b}" for b in BUDGET_CURVE) + " |",
          "|---" * (len(BUDGET_CURVE) + 2) + "|"]
    for lv in levels:
        for w in WEIGHTINGS:
            L.append(f"| {lv} | {w} | " + " | ".join(
                ms(mean_over_tasks(lv, b, w)) for b in BUDGET_CURVE) + " |")

    # 誠實性：esca 的可區分門檻
    L += ["", "## 誠實性註記", "",
          f"- esca 只有 n={n_test.get('tcga_esca', 0)} 張 test slide，一張 = "
          f"{INDISTINGUISHABLE_PP:.2f} pp。**esca 上任何小於 "
          f"{INDISTINGUISHABLE_PP:.2f} pp 的差異一律視為不可區分。**"]
    if len(levels) >= 2:
        a, b = levels[0], levels[1]
        for t in ctx.tasks:
            d = (statistics.mean(per[(b, t, PRIMARY_B, "softmax")])
                 - statistics.mean(per[(a, t, PRIMARY_B, "softmax")])) * 100
            tag = ("　（esca：不可區分）" if t == "tcga_esca"
                   and abs(d) < INDISTINGUISHABLE_PP else "")
            L.append(f"- {t}：{b} − {a} = {d:+.2f} pp{tag}")

    (out_dir / "RESULTS.md").write_text("\n".join(L) + "\n")
    return summary


def write_diagnostics(ctx, all_recs, levels, seeds, out_dir) -> None:
    L = ["# Exp 1 診斷紀錄", "",
         "1. 每個 (task, level) 的 eff_K（沿用 D2 的算法 1 / sum(w_i^2)）",
         "2. group 配額分佈：各 task 把 B 個名額分給哪些 tissue group",
         "3. L6 每一輪選中的 group（看 e_t 更新後有沒有換組）", "",
         "## 1. eff_K @ B=8（逐 slide 後平均 ± std，跨 seed 合併）", "",
         "| task | " + " | ".join(levels) + " |", "|---" * (len(levels) + 1) + "|"]
    for t in ctx.tasks:
        cells = []
        for lv in levels:
            v = [r["eff_K"] for r in all_recs
                 if r["level"] == lv and r["task"] == t and r["B"] == PRIMARY_B]
            cells.append(f"{statistics.mean(v):.2f} ± {statistics.stdev(v):.2f}"
                         if len(v) > 1 else "—")
        L.append(f"| {t} | " + " | ".join(cells) + " |")

    L += ["", "## 2. group 配額分佈 @ B=8（各 task 選中的 patch 落在哪些 tissue group，"
          "跨 slide 與 seed 加總後的比例）", ""]
    for lv in levels:
        L += [f"### {lv} {LEVELS[lv]['name']}", "",
              "| task | " + " | ".join(TISSUE_GROUP_NAMES) + " |",
              "|---" * (len(TISSUE_GROUP_NAMES) + 1) + "|"]
        for t in ctx.tasks:
            rs = [r for r in all_recs if r["level"] == lv and r["task"] == t
                  and r["B"] == PRIMARY_B]
            tot = [0] * len(TISSUE_GROUP_NAMES)
            for r in rs:
                for j, v in enumerate(r["group_quota"]):
                    tot[j] += v
            s = sum(tot) or 1
            L.append(f"| {t} | " + " | ".join(f"{v / s:.3f}" for v in tot) + " |")
        L.append("")

    if "L6" in levels:
        L += ["## 3. L6 的輪間換組率 @ B=8", "",
              "每張 slide 的 8 輪中，相鄰兩輪選到**不同** group 的比例。"
              "接近 0 代表 e_t 更新後仍停在同一組。", "",
              "| task | 換組率 | 平均用到幾個不同 group |", "|---|---|---|"]
        for t in ctx.tasks:
            rs = [r for r in all_recs if r["level"] == "L6" and r["task"] == t
                  and r["B"] == PRIMARY_B]
            switches, pairs, distinct = 0, 0, []
            for r in rs:
                seq = r["group_seq"]
                switches += sum(1 for a, b in zip(seq, seq[1:]) if a != b)
                pairs += max(len(seq) - 1, 0)
                distinct.append(len(set(seq)))
            L.append(f"| {t} | {switches / max(pairs, 1):.3f} | "
                     f"{statistics.mean(distinct):.2f} |" if distinct else f"| {t} | — | — |")
    (out_dir / "DIAGNOSTICS.md").write_text("\n".join(L) + "\n")


# ── main ────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--levels", default="")
    ap.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--beta-s", type=float, default=0.1)
    ap.add_argument("--prior", default=MAINLINE_PRIOR)
    ap.add_argument("--chunk", type=int, default=1)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-resume", action="store_true",
                    help="忽略既有的 per_slide 存檔，全部重跑")
    args = ap.parse_args()

    cfg = load_config()
    levels = ([x.strip() for x in args.levels.split(",") if x.strip()]
              or (["L3", "L4"] if args.stage == 1 else ["L5", "L6"]))
    seeds = [int(x) for x in args.seeds.split(",")]
    out_dir = OUT_ROOT / (f"stage{args.stage}" + (f"_{args.tag}" if args.tag else ""))
    (out_dir / "per_slide").mkdir(parents=True, exist_ok=True)

    ctx = Ctx(cfg)
    print(f"Exp1 stage{args.stage}  levels={levels}  seeds={seeds}  "
          f"epochs={args.epochs}  B={PRIMARY_B}  c={args.chunk}  prior={args.prior}")

    all_recs = []
    for lv in levels:
        spec = LEVELS[lv]
        for seed in seeds:
            # 續跑：(level, seed) 是最小的存檔單位，跑完就落檔，中斷後可從這裡接。
            done = out_dir / "per_slide" / f"{lv}_seed{seed}.json"
            if done.exists() and not args.no_resume:
                recs = json.loads(done.read_text())
                all_recs += recs
                print(f"  ▷ {lv} seed={seed} 已有存檔，跳過（{len(recs)} 筆）", flush=True)
                continue
            print(f"  ▶ {lv} ({spec['mode']}) seed={seed}", flush=True)
            if spec["mode"] == "per_task":
                recs = []
                for t in ctx.tasks:
                    models = train_one(ctx, lv, seed, [t], args.epochs, args.lr,
                                       args.beta_s, args.prior, PRIMARY_B, args.chunk,
                                       max_train=args.max_train)
                    recs += evaluate(ctx, lv, seed, models, t, BUDGET_CURVE, args.chunk)
            else:
                models = train_one(ctx, lv, seed, ctx.tasks, args.epochs, args.lr,
                                   args.beta_s, args.prior, PRIMARY_B, args.chunk,
                                   max_train=args.max_train)
                recs = [r for t in ctx.tasks
                        for r in evaluate(ctx, lv, seed, models, t, BUDGET_CURVE,
                                          args.chunk)]
            all_recs += recs
            (out_dir / "per_slide" / f"{lv}_seed{seed}.json").write_text(
                json.dumps(recs, indent=1))
            for t in ctx.tasks:
                print(f"      {t:10s} B=8  softmax={acc(recs, t, PRIMARY_B, 'softmax'):.4f}"
                      f"  uniform={acc(recs, t, PRIMARY_B, 'uniform'):.4f}", flush=True)

    (out_dir / "results.json").write_text(json.dumps(all_recs, indent=1))
    summary = write_results(ctx, all_recs, levels, seeds, out_dir, args)
    write_diagnostics(ctx, all_recs, levels, seeds, out_dir)
    print(f"\n→ {out_dir / 'RESULTS.md'}\n→ {out_dir / 'DIAGNOSTICS.md'}"
          f"\n→ {out_dir / 'per_slide'}/")

    if args.stage == 1 and {"L3", "L4"} <= set(levels):
        d = (statistics.mean(summary["L4"]["softmax"])
             - statistics.mean(summary["L3"]["softmax"])) * 100
        print(f"\nGATE 1：L4 − L3（四 task 平均，softmax）= {d:+.2f} pp"
              f"  →  {'通過，可進 Stage 2' if d > 0 else '未通過，停下來回報'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
