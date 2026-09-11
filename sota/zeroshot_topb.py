#!/usr/bin/env python3
"""DR-051 E1：zero-shot **top-B** 參照線 —— 不訓練 selector，用類別文字本身選 patch。

每張 slide：`score_i = max_c cos(x_i, t_c)`（c 跑遍**固定 8 類**的 class-text 特徵，
與凍結頭同一組 `f_txt`，C-30），取分數最高的 B（預設 8）個 patch，**等權**池化，
送進同一顆凍結的 CONCH 頭（CONTRACT-4）。

與 `sota/zeroshot.py` 的兩條參照線同一家族：沒有可訓練參數，所以跨 stage 完全不變，
Forgetting／BWT 必然為 0（「沒有東西可遺忘」，不是「不遺忘」）。它回答的是：
「在同樣只看 8 片的條件下，**用類別語意直接挑 8 片**能到哪」—— 這是 learned selector
必須勝過的最強零樣本選片基準；`ZS-rand8` 只回答「隨機 8 片能到哪」。

per_slide 欄位與 `scripts/run_exp2.py::evaluate` 對齊（`pooling = "zs_top"`），
`sota/report_sota.py` 直接吃。檔名 `ZS-top8_{order}_seed{k}{_f{k}}.json`。
完全決定性（無隨機），seed 只是協定欄位（seed = fold）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_exp2 import ARCH, DEFAULT_ARCH, NUM_GROUPS, ORDERS, Ctx, acc   # noqa: E402
from selector.classifier import conch_classify                          # noqa: E402
from selector.memory import MEMORY_CAPACITY                             # noqa: E402
from selector.text_encoder import load_config                           # noqa: E402
from selector.utility import sequential_utility_total                   # noqa: E402

OUT_ROOT = ROOT / "outputs" / "exp2"
ARM = "ZS-top8"
POOLING = "zs_top"


def select_topb(Z: torch.Tensor, f_txt: torch.Tensor, budget: int) -> torch.Tensor:
    """[n, D] × [C, D] → 依 max_c cos 取 top-B 的 index（升冪排序，僅為可讀）。"""
    cos = F.normalize(Z, dim=-1) @ F.normalize(f_txt, dim=-1).t()       # [n, C]
    score = cos.max(dim=1).values                                       # [n]
    k = min(budget, Z.shape[0])
    return score.topk(k).indices.sort().values


def evaluate_task(ctx, task: str, order_name: str, seed: int, stage: int, args):
    lo = 2 * ctx.label_space.index(task)
    out = []
    for i in range(ctx.n_slides(task, "test")):
        rec, grp = ctx.get(task, "test", i)
        idx = select_topb(rec.Z, ctx.f_txt, args.budget)
        logits = conch_classify(rec.Z.index_select(0, idx), None,
                                ctx.f_txt, ctx.logit_scale).reshape(-1)
        quota = [0] * NUM_GROUPS
        for j in grp.assignment.index_select(0, idx).tolist():
            quota[j] += 1
        k = idx.numel()
        out.append({
            "arm": ARM, "order": order_name, "seed": seed, "stage": stage,
            "task": task, "slide_id": rec.sid, "true": rec.label,
            "pred_class_il": int(logits.argmax()),
            "pred_task_il": lo + int(logits[lo:lo + 2].argmax()),
            "pred_softmax": int(logits.argmax()),
            "selected_idx": idx.tolist(),
            "weights_softmax": [round(1.0 / k, 6)] * k,
            "weights_uniform": [round(1.0 / k, 6)] * k,
            "group_quota": quota, "n_patch": int(rec.Z.shape[0]), "pooling": POOLING,
            "mem_capacity": args.mem_capacity or MEMORY_CAPACITY,
            "arch": args.arch, "prior": args.prior,
            "allocation": args.allocation, "fold": args.fold,
            "utility_total": sequential_utility_total(rec.Z, idx, ctx.f_txt,
                                                      ctx.logit_scale, rec.label),
            "B": k,
        })
    return out


def run(ctx, order_name: str, seed: int, args) -> list[dict]:
    tasks = ORDERS[order_name]
    recs = []
    for stage in range(len(tasks)):
        for t in tasks[:stage + 1]:
            r = evaluate_task(ctx, t, order_name, seed, stage, args)
            recs += r
            print(f"    stage {stage}  {t:10s} class-IL={acc(r, 'pred_class_il'):.4f} "
                  f"task-IL={acc(r, 'pred_task_il'):.4f}", flush=True)
    return recs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--order", default="reverse", choices=list(ORDERS))
    ap.add_argument("--arch", default=DEFAULT_ARCH, choices=list(ARCH))
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--seed", type=int, default=None,
                    help="預設 = --fold（SOTA 協定：seed 就是折號）")
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--allocation", default="per_budget")
    ap.add_argument("--prior", default="tissue")
    ap.add_argument("--mem-capacity", type=int, default=None)
    ap.add_argument("--tag", default="sota")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args(argv)

    seed = args.fold if args.seed is None else args.seed
    cfg = load_config()
    cfg["fold"] = args.fold
    ctx = Ctx(cfg)
    out_dir = OUT_ROOT / args.tag / "per_slide"
    out_dir.mkdir(parents=True, exist_ok=True)
    arch_sfx = "" if args.arch == DEFAULT_ARCH else "_" + args.arch
    fold_sfx = "" if args.fold == 1 else f"_f{args.fold}"
    path = out_dir / f"{ARM}_{args.order}_seed{seed}{arch_sfx}{fold_sfx}.json"
    if path.exists() and not args.no_resume:
        print(f"▷ 跳過（已存在）{path.name}", flush=True)
        return 0
    print(f"═══ zero-shot top-{args.budget} fold={args.fold} seed={seed} "
          f"order={args.order}", flush=True)
    path.write_text(json.dumps(run(ctx, args.order, seed, args), indent=1))
    print(f"    → {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
