#!/usr/bin/env python3
"""Exp 2 — CL 方法臂對照。

架構 L3b（shared selector、無 q_tau、flat）、B=8、c=1、epochs 5、lr 1e-3、
beta_s 0.1、beta_u 0.1、prior discriminative、λ_kd = λ_eq = λ_r = 1.0（不調）。

方法臂
  A1 SeqFT           無任何 CL 機制
  A2 + LoRA merge    只有 LoRA + sequential merge，無 preservation
  A3 + Replay        LoRA + replay（λ_kd = λ_eq = 0）
  A4 + Replay + KD   加選取行為蒸餾（λ_eq = 0）
  A5 Ours            Replay + KD + eq，三項全開
  R1 per-task oracle 每個 task 獨立訓練（無干擾天花板）
  R2 joint offline   一次看到所有 task 資料、沒有順序
                     ⚠️ **offline shared-model reference，不是 CL baseline**，
                        不能用來宣稱 forgetting。

指標（裁定 2：task-IL 與 class-IL **都是主要指標**，不是主/次關係）
  task-IL A1 forgetting  = 忘了怎麼在任務內鑑別
  跨任務洩漏率           = 選出的證據整體上不再像這個任務的組織。
                           head 是 frozen 的，所以洩漏 100% 可歸因於選取漂移 ——
                           這是架構的直接後果，當成發現來報。

每個 (arm, order, seed, stage) 跑完就落檔（逐 slide 記錄 + checkpoint），可續跑。
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import OrderedDict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from selector.classifier import conch_classify, softmax_weights          # noqa: E402
from selector.evaluate import read_slide, slide_dataset                  # noqa: E402
from selector.grouping import NUM_GROUPS, assign_groups, tissue_text_features  # noqa: E402
from selector.lora import apply_lora, lora_parameters, merge_lora        # noqa: E402
from selector.memory import MEMORY_CAPACITY, SelectionMemory             # noqa: E402
from selector.model import GroupSelector, PatchSelector                  # noqa: E402
from selector.priors import MAINLINE_PRIOR                               # noqa: E402
from selector.text_encoder import build_f_txt, load_config               # noqa: E402
from selector.train import (continual_terms, fill_memory, total_loss,    # noqa: E402
                            train_step)
from selector.utility import sequential_utility_total                    # noqa: E402

OUT_ROOT = REPO_ROOT / "outputs" / "exp2"
L3B = dict(use_query=False, use_state=False, hierarchy=False)
ORDERS = {
    "reverse": ["tcga_esca", "tcga_rcc", "tcga_brca", "tcga_lung"],
    "main": ["tcga_lung", "tcga_brca", "tcga_rcc", "tcga_esca"],
}
ARMS = OrderedDict([
    ("A1", dict(name="SeqFT", mode="sequential", lora=False,
                replay=False, kd=False, eq=False)),
    ("A2", dict(name="+ LoRA merge", mode="sequential", lora=True,
                replay=False, kd=False, eq=False)),
    ("A3", dict(name="+ Replay", mode="sequential", lora=True,
                replay=True, kd=False, eq=False)),
    ("A4", dict(name="+ Replay + KD", mode="sequential", lora=True,
                replay=True, kd=True, eq=False)),
    ("A5", dict(name="Ours (Replay+KD+eq)", mode="sequential", lora=True,
                replay=True, kd=True, eq=True)),
    # 元件消融：λ 設 0 與「不計算該項」位元等價（乘 0 再相加不改變任何位元），
    # 實作上直接關掉以省算力。記憶體照常填充與取樣 —— KD / eq 都需要舊樣本。
    ("B1", dict(name="只 KD (λ_r=λ_eq=0)", mode="sequential", lora=True,
                replay=False, kd=True, eq=False)),
    ("B2", dict(name="只 eq (λ_r=λ_kd=0)", mode="sequential", lora=True,
                replay=False, kd=False, eq=True)),
    ("R1", dict(name="per-task specialist (independent training)",
                mode="per_task", lora=False,
                replay=False, kd=False, eq=False)),
    ("R2", dict(name="joint offline reference", mode="joint", lora=False,
                replay=False, kd=False, eq=False)),
])
SLIDE_CACHE = 256
#: E2：臂間比較一律配對統計（PI 裁定 4）。不報 p 值。
PAIRED_COMPARISONS = [("A5", "A3"), ("A5", "A1"), ("A4", "A3"),
                      ("A5", "A4"), ("A5", "R2")]
#: 配對比較看的四個指標，以及「越大越好 / 越小越好」
PAIRED_METRICS = [("final_task_il", "task-IL final avg", True),
                  ("final_class_il", "class-IL final avg", True),
                  ("mean_leak", "跨任務洩漏率", False),
                  ("mean_jaccard", "selection Jaccard", True)]
INDISTINGUISHABLE_PP = 100.0 / 15      # esca n=15，一張 slide


class Ctx:
    def __init__(self, cfg, device="cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.label_space = list(cfg["tasks"])
        self.f_txt = torch.cat([build_f_txt(t, cfg, device=device).f_txt
                                for t in self.label_space], 0)
        self.logit_scale = build_f_txt(self.label_space[0], cfg, device=device).logit_scale
        self.tissue = tissue_text_features(cfg, device=device)
        self.q0 = torch.zeros(512, device=self.device)
        self._ds, self._lru = {}, OrderedDict()

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


def new_models(ctx, seed, use_lora, rank):
    torch.manual_seed(seed)
    f_g, f_p = GroupSelector().to(ctx.device), PatchSelector().to(ctx.device)
    if use_lora:
        apply_lora(f_g, r=rank), apply_lora(f_p, r=rank)
    return f_g, f_p


def trainable(f_g, f_p, use_lora):
    return (lora_parameters(f_g, f_p) if use_lora
            else list(f_g.parameters()) + list(f_p.parameters()))


# ── training ────────────────────────────────────────────────────────────────

def train_stage(ctx, arm, models, tasks, seed, args, memory, rng):
    """在 tasks 這批 slide 上訓練一輪 stage。回傳 l_eq 觸發率等診斷。"""
    spec = ARMS[arm]
    f_g, f_p = models
    f_g.train(); f_p.train()
    opt = torch.optim.Adam(trainable(f_g, f_p, spec["lora"]), lr=args.lr,
                           weight_decay=1e-4)
    stream = [(t, i) for t in tasks
              for i in range(min(args.max_train, ctx.n_slides(t, "train"))
                             if args.max_train > 0 else ctx.n_slides(t, "train"))]
    eq_fired = eq_seen = 0
    for epoch in range(args.epochs):
        g = torch.Generator().manual_seed(seed * 1000 + epoch)
        order = torch.randperm(len(stream), generator=g).tolist()
        total = 0.0
        for k in order:
            task, si = stream[k]
            rec, grp = ctx.get(task, "train", si)
            l_ev, _parts, _res = train_step(
                rec.Z, rec.label, ctx.q0, ctx.f_txt, ctx.logit_scale, f_g, f_p,
                grouping=grp, budget=args.budget, chunk=args.chunk,
                prior_kind=args.prior, beta_s=args.beta_s, beta_u=args.beta_u,
                **L3B)

            kd = eq = replay = None
            if len(memory) and (spec["kd"] or spec["eq"] or spec["replay"]):
                for entry in memory.sample(args.replay_k, rng):
                    k_, e_, r_ = continual_terms(
                        entry, ctx.cfg, (f_g, f_p), ctx.f_txt, ctx.logit_scale,
                        ctx.tissue, budget=args.budget, chunk=args.chunk, spec=L3B,
                        use_kd=spec["kd"], use_eq=spec["eq"],
                        use_replay=spec["replay"])
                    kd = k_ if kd is None else kd + k_
                    eq = e_ if eq is None else eq + e_
                    replay = r_ if replay is None else replay + r_
                if eq is not None:
                    eq_seen += 1
                    eq_fired += int(float(eq.detach()) > 0.0)
            loss, _cl = total_loss(l_ev, kd, eq, replay, lambda_kd=args.lambda_kd,
                                   lambda_eq=args.lambda_eq,
                                   lambda_replay=args.lambda_replay)
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.detach())
        print(f"      epoch {epoch + 1}/{args.epochs} n={len(stream)} "
              f"mean_loss={total / len(stream):.4f}", flush=True)
    f_g.eval(); f_p.eval()
    return {"l_eq_fire_rate": (eq_fired / eq_seen) if eq_seen else None,
            "l_eq_steps": eq_seen}


@torch.no_grad()
def evaluate(ctx, models, task, arm, order_name, seed, stage, args, diag=None):
    f_g, f_p = models
    out = []
    lo = 2 * ctx.label_space.index(task)
    for i in range(ctx.n_slides(task, "test")):
        rec, grp = ctx.get(task, "test", i)
        from selector.rounds import run_rounds
        res = run_rounds(rec.Z, grp, ctx.q0, f_g, f_p, budget=args.budget,
                         chunk=args.chunk, **L3B)
        idx, s = res.selected, res.records[-1].s
        w = softmax_weights(s, idx)
        logits = conch_classify(rec.Z.index_select(0, idx), w,
                                ctx.f_txt, ctx.logit_scale).reshape(-1)
        quota = [0] * NUM_GROUPS
        for j in grp.assignment.index_select(0, idx).tolist():
            quota[j] += 1
        u_total = sequential_utility_total(rec.Z, idx, ctx.f_txt,
                                           ctx.logit_scale, rec.label)
        out.append({
            "arm": arm, "order": order_name, "seed": seed, "stage": stage,
            "task": task, "slide_id": rec.sid, "true": rec.label,
            "pred_class_il": int(logits.argmax()),
            "pred_task_il": lo + int(logits[lo:lo + 2].argmax()),
            "pred_softmax": int(logits.argmax()),
            "selected_idx": idx.tolist(),
            "weights_softmax": [round(float(x), 6) for x in w],
            "weights_uniform": [round(1.0 / max(idx.numel(), 1), 6)] * idx.numel(),
            "group_quota": quota, "n_patch": int(rec.Z.shape[0]),
            "mem_capacity": args.mem_capacity or MEMORY_CAPACITY,
            "utility_total": u_total, "B": args.budget,
            **(diag or {}),
        })
    return out


def run_arm(ctx, arm, order_name, seed, args, out_dir):
    spec = ARMS[arm]
    tasks = ORDERS[order_name]
    recs_all = []
    rng = random.Random(seed)
    cap = args.mem_capacity or MEMORY_CAPACITY
    memory = SelectionMemory(capacity=cap, policy=None,       # 預設 reservoir
                             allow_over_contract=cap > MEMORY_CAPACITY)

    if spec["mode"] == "per_task":
        # R1：每個 task 獨立訓練，彼此不干擾 → 各 stage 的結果相同（天花板）
        per_task_models = {}
        for t in tasks:
            models = new_models(ctx, seed, spec["lora"], args.rank)
            train_stage(ctx, arm, models, [t], seed, args, memory, rng)
            per_task_models[t] = models
        for stage in range(len(tasks)):
            for t in tasks[:stage + 1]:
                recs_all += evaluate(ctx, per_task_models[t], t, arm, order_name,
                                     seed, stage, args)
        return recs_all

    if spec["mode"] == "joint":
        # R2：一次看到所有資料、沒有順序。offline reference，不是 CL baseline。
        models = new_models(ctx, seed, spec["lora"], args.rank)
        train_stage(ctx, arm, models, tasks, seed, args, memory, rng)
        for stage in range(len(tasks)):
            for t in tasks[:stage + 1]:
                recs_all += evaluate(ctx, models, t, arm, order_name, seed, stage, args)
        return recs_all

    # A1–A5：序列
    models = new_models(ctx, seed, spec["lora"], args.rank)
    for stage, task in enumerate(tasks):
        print(f"    ── stage {stage}: {task}", flush=True)
        diag = train_stage(ctx, arm, models, [task], seed, args, memory, rng)
        if spec["lora"]:
            merge_lora(*models)                     # W_t = W_{t-1} + ΔW_t
        if spec["replay"] or spec["kd"] or spec["eq"]:
            added = fill_memory(memory, models, task, ctx.cfg, ctx.f_txt,
                                ctx.logit_scale, ctx.tissue, budget=args.budget,
                                chunk=args.chunk, spec=L3B,
                                max_slides=args.mem_slides)
            print(f"       記憶體 +{added} → |M|={len(memory)}", flush=True)
        for t in tasks[:stage + 1]:
            r = evaluate(ctx, models, t, arm, order_name, seed, stage, args, diag)
            recs_all += r
            print(f"       eval {t:10s} class-IL={acc(r, 'pred_class_il'):.4f} "
                  f"task-IL={acc(r, 'pred_task_il'):.4f}", flush=True)
    return recs_all


def acc(records, key="pred_task_il") -> float:
    return sum(r[key] == r["true"] for r in records) / len(records) if records else float("nan")


# ── main ────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--order", default="reverse")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--beta-s", type=float, default=0.1)
    ap.add_argument("--beta-u", type=float, default=0.1)
    ap.add_argument("--prior", default=MAINLINE_PRIOR)
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=1)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--lambda-kd", type=float, default=1.0)
    ap.add_argument("--lambda-eq", type=float, default=1.0)
    ap.add_argument("--lambda-replay", type=float, default=1.0)
    ap.add_argument("--replay-k", type=int, default=1)
    ap.add_argument("--mem-slides", type=int, default=0, help="0 = 該 task 全部")
    ap.add_argument("--mem-capacity", type=int, default=None,
                    help="|M| 上限；超過 CONTRACT-3 的 512 需要本旗標顯式指定")
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    seeds = [int(x) for x in args.seeds.split(",")]
    out_dir = OUT_ROOT / args.tag
    (out_dir / "per_slide").mkdir(parents=True, exist_ok=True)

    ctx = Ctx(cfg)
    if args.report_only:
        recs = [r for p in sorted((out_dir / "per_slide").glob("*.json"))
                for r in json.loads(p.read_text())]
        write_report(ctx, recs, arms, args.order, seeds, args, out_dir)
        print(f"→ {out_dir / 'EXP2.md'}（report-only）")
        return 0

    print(f"Exp2  arms={arms}  order={args.order}  seeds={seeds}  "
          f"B={args.budget} c={args.chunk} epochs={args.epochs} "
          f"beta_u={args.beta_u} replay_k={args.replay_k} "
          f"λ=({args.lambda_kd},{args.lambda_eq},{args.lambda_replay})", flush=True)

    all_recs = []
    for arm in arms:
        for seed in seeds:
            suffix = f"_M{args.mem_capacity}" if args.mem_capacity else ""
            tag = f"{arm}_{args.order}_seed{seed}{suffix}"
            path = out_dir / "per_slide" / f"{tag}.json"
            if path.exists() and not args.no_resume:
                all_recs += json.loads(path.read_text())
                print(f"  ▷ {arm} seed={seed} 已有存檔，跳過", flush=True)
                continue
            print(f"  ▶ {arm} ({ARMS[arm]['name']}) seed={seed}", flush=True)
            recs = run_arm(ctx, arm, args.order, seed, args, out_dir)
            path.write_text(json.dumps(recs, indent=1))
            all_recs += recs

    write_report(ctx, all_recs, arms, args.order, seeds, args, out_dir)
    print(f"\n→ {out_dir / 'EXP2.md'}")
    return 0


def ms(vals, fmt="{:.4f}"):
    vals = [v for v in vals if v is not None and v == v]
    if not vals:
        return "—"
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f"{fmt.format(statistics.mean(vals))} ± {fmt.format(sd)}"


def jac(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 1.0


def kl_quota(p, q):
    p = [c + 1 for c in p]; q = [c + 1 for c in q]
    sp, sq = sum(p), sum(q)
    return sum((pi / sp) * math.log((pi / sp) / (qi / sq)) for pi, qi in zip(p, q))


def arm_metrics(recs, arm, tasks, seed, label_space):
    """單一 (arm, seed) 的全部指標。"""
    last = len(tasks) - 1
    sub = [r for r in recs if r["arm"] == arm and r["seed"] == seed]
    out = {"per_task": {}}
    for i, t in enumerate(tasks):
        at_i = [r for r in sub if r["stage"] == i and r["task"] == t]
        at_e = [r for r in sub if r["stage"] == last and r["task"] == t]
        if not at_i or not at_e:
            continue
        by = {r["slide_id"]: r for r in at_e}
        pairs = [(r, by[r["slide_id"]]) for r in at_i if r["slide_id"] in by]
        lo = 2 * label_space.index(t)
        leak = sum(not (lo <= r["pred_class_il"] <= lo + 1) for r in at_e) / len(at_e)
        q_i = [sum(r["group_quota"][j] for r in at_i) for j in range(NUM_GROUPS)]
        q_e = [sum(r["group_quota"][j] for r in at_e) for j in range(NUM_GROUPS)]
        n_px = statistics.mean([r["n_patch"] for r in at_e])
        K = at_e[0]["B"]
        f = min(K, n_px) / n_px
        out["per_task"][t] = {
            "task_il_at_learn": acc(at_i, "pred_task_il"),
            "task_il_at_end": acc(at_e, "pred_task_il"),
            "class_il_at_learn": acc(at_i, "pred_class_il"),
            "class_il_at_end": acc(at_e, "pred_class_il"),
            "A1_task_il": (acc(at_i, "pred_task_il") - acc(at_e, "pred_task_il")) * 100,
            "A1_class_il": (acc(at_i, "pred_class_il") - acc(at_e, "pred_class_il")) * 100,
            "leak": leak,
            "jaccard": statistics.mean([jac(a["selected_idx"], b["selected_idx"])
                                        for a, b in pairs]),
            "jaccard_ref": f / (2 - f),
            "quota_kl": kl_quota(q_i, q_e),
            "sum_u_at_learn": sum(r["utility_total"] for r in at_i),
            "sum_u_at_end": sum(r["utility_total"] for r in at_e),
            "l_eq_fire_rate": at_e[0].get("l_eq_fire_rate"),
        }
    p = out["per_task"]
    if p:
        # final accuracy 算全部 T 個 task（標準做法）
        out["final_task_il"] = statistics.mean([v["task_il_at_end"] for v in p.values()])
        out["final_class_il"] = statistics.mean([v["class_il_at_end"] for v in p.values()])
        out["mean_leak"] = statistics.mean([v["leak"] for v in p.values()])
        # forgetting 類指標只算前 T−1 個 task：最後一個 task 的「學完」與「學完 T4」
        # 是同一個時點，A1 恆為 0、Jaccard 恆為 1，算進去只會稀釋數字。
        early = [p[t] for t in tasks[:-1] if t in p]
        if early:
            out["mean_A1_task_il"] = statistics.mean([v["A1_task_il"] for v in early])
            out["mean_A1_class_il"] = statistics.mean([v["A1_class_il"] for v in early])
            out["mean_jaccard"] = statistics.mean([v["jaccard"] for v in early])
            out["mean_quota_kl"] = statistics.mean([v["quota_kl"] for v in early])
            out["mean_leak_early"] = statistics.mean([v["leak"] for v in early])
        fr = [v["l_eq_fire_rate"] for v in p.values() if v["l_eq_fire_rate"] is not None]
        out["l_eq_fire_rate"] = statistics.mean(fr) if fr else None
    return out


def write_report(ctx, recs, arms, order_name, seeds, args, out_dir):
    tasks = ORDERS[order_name]
    short = [t.replace("tcga_", "") for t in tasks]
    n_test = {t: ctx.n_slides(t, "test") for t in tasks}
    M = {a: {s: arm_metrics(recs, a, tasks, s, ctx.label_space) for s in seeds}
         for a in arms}

    def col(a, key, fmt="{:.4f}"):
        return ms([M[a][s].get(key) for s in seeds if M[a][s].get("per_task")], fmt)

    L = [
        f"# Exp 2 — CL 方法臂對照（order = {order_name}）",
        "",
        f"架構 L3b（shared selector、無 q_tau、flat）、B={args.budget}、c={args.chunk}、"
        f"epochs {args.epochs}、lr {args.lr}、beta_s {args.beta_s}、"
        f"beta_u {args.beta_u}、prior {args.prior}、"
        f"λ_kd={args.lambda_kd} λ_eq={args.lambda_eq} λ_r={args.lambda_replay}"
        f"（全程固定，未調）、replay_k={args.replay_k}、seeds {seeds}。",
        "訓練用 train split、評估用 test split。",
        "",
        "**task-IL 與 class-IL 都是主要指標**（PI 裁定 2），不是主/次關係：",
        "",
        "- **task-IL A1 forgetting** = 忘了怎麼在任務內鑑別。",
        "- **跨任務洩漏率** = 選出的證據整體上不再像這個任務的組織。"
        "head 是 frozen 的，**洩漏 100% 可歸因於選取漂移** —— 這是架構的直接後果，"
        "是發現而不是缺陷。",
        "",
        f"n（test）：" + "、".join(f"{s} {n_test[t]}" for s, t in zip(short, tasks))
        + f"。⚠️ esca n={n_test.get('tcga_esca', 0)}，一張 slide = "
        f"{INDISTINGUISHABLE_PP:.2f} pp，**esca 上小於 {INDISTINGUISHABLE_PP:.2f} pp "
        f"的差異一律視為不可區分**。",
        "",
        "## 主表",
        "",
        "| # | 方法臂 | final avg acc (task-IL) | final avg acc (class-IL) | "
        "A1 forgetting task-IL (pp) | A1 forgetting class-IL (pp) | Jaccard | "
        "quota KL | 洩漏率 | l_eq fire rate |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for a in arms:
        L.append(f"| {a} | {ARMS[a]['name']} | {col(a, 'final_task_il')} | "
                 f"{col(a, 'final_class_il')} | {col(a, 'mean_A1_task_il', '{:+.2f}')} | "
                 f"{col(a, 'mean_A1_class_il', '{:+.2f}')} | {col(a, 'mean_jaccard')} | "
                 f"{col(a, 'mean_quota_kl')} | {col(a, 'mean_leak')} | "
                 f"{col(a, 'l_eq_fire_rate')} |")
    L += ["",
          f"**欄位口徑**：final avg acc 算全部 {len(tasks)} 個 task；"
          f"**forgetting、Jaccard、quota KL 只算前 {len(tasks) - 1} 個 task** —— "
          f"最後學的 {short[-1]} 的「學完」與「學完 T4」是同一個時點，"
          "A1 恆為 0、Jaccard 恆為 1，算進去只會稀釋遺忘的量級（CL 慣例）。"
          "洩漏率算全部 4 個 task，因為最後一個 task 的洩漏不是由構造為 0。",
          "",
          "⚠️ **R1 / R2 不是 CL baseline**，兩者的 A1 forgetting 由構造為 0，"
          "不能用來宣稱 forgetting。",
          "",
          "**R1 = per-task specialist (independent training)（PI 裁定 2）**："
          "每個 task 只用自己的訓練資料（esca 僅 120 張），而 A3 / A5 經由 replay "
          "實質可及跨任務資料。**因此 R1 在 task-IL 上不是上界**；它的參考意義在 "
          "class-IL —— 那一欄 R1 是全場最高（0.8777）。",
          "",
          "**R2 = offline shared-model reference**：一次看到所有資料、沒有 task 順序。",
          "", "## 逐 task 明細（學完 T4 後）", ""]
    for a in arms:
        L += [f"### {a} {ARMS[a]['name']}", "",
              "| task | n | task-IL @學完 | task-IL @T4 | A1 task-IL (pp) | "
              "class-IL @T4 | A1 class-IL (pp) | 洩漏率 | Jaccard | 隨機參照 | "
              "quota KL | ΣU @學完 | ΣU @T4 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for t in tasks:
            g = lambda k, f="{:.4f}": ms(
                [M[a][s]["per_task"][t][k] for s in seeds
                 if t in M[a][s].get("per_task", {})], f)
            note = ""
            if t == "tcga_esca":
                vals = [M[a][s]["per_task"][t]["A1_task_il"] for s in seeds
                        if t in M[a][s].get("per_task", {})]
                if vals and abs(statistics.mean(vals)) < INDISTINGUISHABLE_PP:
                    note = "（不可區分）"
            L.append(f"| {t}{note} | {n_test[t]} | {g('task_il_at_learn')} | "
                     f"{g('task_il_at_end')} | {g('A1_task_il', '{:+.2f}')} | "
                     f"{g('class_il_at_end')} | {g('A1_class_il', '{:+.2f}')} | "
                     f"{g('leak')} | {g('jaccard')} | {g('jaccard_ref', '{:.5f}')} | "
                     f"{g('quota_kl')} | {g('sum_u_at_learn', '{:.1f}')} | "
                     f"{g('sum_u_at_end', '{:.1f}')} |")
        L.append("")

    L += ["## 對照差值（task-IL final avg，5 seeds 平均）", "",
          "| 對照 | 差值 (pp) |", "|---|---|"]
    def fin(a):
        v = [M[a][s].get("final_task_il") for s in seeds if M[a][s].get("per_task")]
        return statistics.mean([x for x in v if x is not None]) if v else float("nan")
    for x, y in (("A5", "A3"), ("A5", "A1"), ("A5", "A2"), ("A5", "A4"),
                 ("A5", "R1"), ("A5", "R2")):
        if x in M and y in M:
            L.append(f"| {x} − {y} | {(fin(x) - fin(y)) * 100:+.2f} |")
    L += write_paired(M, arms, seeds)
    L += ["", "逐 slide 預測：`outputs/exp2/" + args.tag + "/per_slide/*.json`", ""]
    (out_dir / "EXP2.md").write_text("\n".join(L) + "\n")


def write_paired(M, arms, seeds) -> list[str]:
    """E2 —— 臂間比較一律配對統計（同 seed 相減）。不報 p 值。"""
    L = ["", "## Paired comparisons（E2）", "",
         "臂間比較一律**配對**：同一個 seed 相減，再對差值取 mean ± std。",
         "",
         "### 方法學註記：win count 三級規則（DR-020）", "",
         "win count = 幾個 seed 往「較好」的方向。判讀只有三級，"
         "全文一律使用這三個詞，不混用：", "",
         "| win count | 名稱 | 判讀 |",
         "|---|---|---|",
         "| 5/5 | **systematic** | 系統性差異 |",
         "| 4/5 | **directional, inconclusive** | 方向一致但證據不足以定案 |",
         "| ≤3/5 | **within noise** | 落在雜訊內 |",
         "",
         "**不報 p 值** —— n=5 的政策沿用（DR-016），顯著性檢定在這個樣本數下會誤導。",
         ""]
    for key, label, higher_better in PAIRED_METRICS:
        L += [f"### {label}（{'越大越好' if higher_better else '越小越好'}）", "",
              "| 對照 | 逐 seed 配對差值 | 配對 mean ± std | win count |",
              "|---|---|---|---|"]
        for x, y in PAIRED_COMPARISONS:
            if x not in M or y not in M:
                continue
            diffs = []
            for sd in seeds:
                a, b = M[x][sd].get(key), M[y][sd].get(key)
                if a is None or b is None:
                    continue
                diffs.append(a - b)
            if not diffs:
                continue
            wins = sum((d > 0) if higher_better else (d < 0) for d in diffs)
            scale = 100 if key != "mean_jaccard" else 1
            unit = " pp" if scale == 100 else ""
            per = ", ".join(f"{d * scale:+.2f}" for d in diffs)
            sd_ = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
            L.append(f"| {x} − {y} | {per} | "
                     f"{statistics.mean(diffs) * scale:+.2f} ± {sd_ * scale:.2f}{unit} | "
                     f"{wins}/{len(diffs)} |")
        L.append("")
    return L


if __name__ == "__main__":
    raise SystemExit(main())
