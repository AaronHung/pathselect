#!/usr/bin/env python3
"""DR-048 B6：zero-shot 參照線 —— 完全不訓練 selector。

兩個變體，共用同一顆凍結的 CONCH 分類頭（CONTRACT-4）：

* **`meanpool`**：整張 slide 的**所有** patch 等權平均 → L2 normalize → 文字 logits。
  沒有任何隨機性，也沒有預算限制。它回答的是「不做 patch 選擇能到哪」。
* **`rand8`**：均勻隨機抽 `B`（預設 8）個 patch，等權平均。它與主線臂**同預算**，
  回答的是「在同樣只看 8 片的條件下，選擇本身值多少」。

兩者都沒有可訓練參數，所以**跨 stage 完全不變** —— 準確率矩陣的每一欄都是常數，
Forgetting 與 BWT 必然為 0。這不是「不遺忘的好方法」，而是「沒有東西可遺忘」；
`docs/SOTA_TABLE.md` 必須把它標成參照線而不是 CL 方法。

輸出的 per_slide 欄位與 `scripts/run_exp2.py` 的 `evaluate` 對齊，
好讓 `sota/metrics.py` 直接吃。每筆都帶 `pooling` 欄位（`"all"` / `"random"`）
明講這一筆是怎麼來的。

⚠️ **`meanpool` 不帶 `B` / `selected_idx` / `weights_*`**，理由有兩層：

1. **語意**：它沒有預算，也沒有做選取 —— 整張 slide 一起彙總。
   `tests/test_per_slide_records.py` 正是以「有沒有 `B`」判定一份存檔算不算
   選取類評估，並註明沒有 patch 選取這回事的只需滿足 REQUIRED。
2. **量體**：曾經照抄選取類的欄位寫過一版，`selected_idx` 就是 `range(n)`、
   權重是 n 份相同的數，**單折 57 MB**（n 最大到 8466），十折 570 MB 全是冗餘；
   而且 `round(1/n, 6)` 在 n 幾千時累積誤差達 1.7e-3，權重根本歸不了一
   （被上面那條測試抓到）。`n_patch` + `pooling` 已經完整說明了發生什麼事。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_exp2 import ARCH, DEFAULT_ARCH, NUM_GROUPS, ORDERS, Ctx, acc   # noqa: E402
from selector.classifier import conch_classify                          # noqa: E402
from selector.memory import MEMORY_CAPACITY                             # noqa: E402
from selector.text_encoder import load_config                           # noqa: E402
from selector.utility import sequential_utility_total                   # noqa: E402

#: 與 `scripts/run_exp2.py` 同一個根，好讓 SOTA 主表只讀一個目錄
OUT_ROOT = ROOT / "outputs" / "exp2"
VARIANTS = {"meanpool": "ZS-mean", "rand8": "ZS-rand8"}
#: 記錄裡的 `pooling` 欄位 —— 明講這一筆是「全部彙總」還是「抽樣選取」
POOLING = {"meanpool": "all", "rand8": "random"}


def select(variant: str, n_patch: int, budget: int,
           gen: torch.Generator) -> torch.Tensor:
    """回傳要送進分類頭的 patch index。`meanpool` 是全部，`rand8` 是隨機 B 個。"""
    if variant == "meanpool":
        return torch.arange(n_patch)
    k = min(budget, n_patch)
    return torch.randperm(n_patch, generator=gen)[:k].sort().values


def evaluate_task(ctx, task: str, variant: str, order_name: str, seed: int,
                  stage: int, args) -> list[dict]:
    arm = VARIANTS[variant]
    lo = 2 * ctx.label_space.index(task)
    out = []
    for i in range(ctx.n_slides(task, "test")):
        rec, grp = ctx.get(task, "test", i)
        n = int(rec.Z.shape[0])
        # 每張 slide 用固定的推導種子 → 同 seed 重跑逐位元相同，且與 stage 無關
        gen = torch.Generator().manual_seed(seed * 1_000_003 + i)
        idx = select(variant, n, args.budget, gen)
        logits = conch_classify(rec.Z.index_select(0, idx), None,
                                ctx.f_txt, ctx.logit_scale).reshape(-1)
        quota = [0] * NUM_GROUPS
        for j in grp.assignment.index_select(0, idx).tolist():
            quota[j] += 1
        r = {
            "arm": arm, "order": order_name, "seed": seed, "stage": stage,
            "task": task, "slide_id": rec.sid, "true": rec.label,
            "pred_class_il": int(logits.argmax()),
            "pred_task_il": lo + int(logits[lo:lo + 2].argmax()),
            "pred_softmax": int(logits.argmax()),
            "group_quota": quota, "n_patch": n, "pooling": POOLING[variant],
            "mem_capacity": args.mem_capacity or MEMORY_CAPACITY,
            "arch": args.arch, "prior": args.prior,
            "allocation": args.allocation, "fold": args.fold,
            "utility_total": sequential_utility_total(rec.Z, idx, ctx.f_txt,
                                                      ctx.logit_scale, rec.label),
        }
        if variant != "meanpool":
            # 有預算才算「選取類評估」：`tests/test_per_slide_records.py` 以
            # `B` 欄位的有無判定，沒有 B 的只需滿足 REQUIRED。
            r |= {"B": idx.numel(), "selected_idx": idx.tolist(),
                  "weights_softmax": [round(1.0 / idx.numel(), 6)] * idx.numel(),
                  "weights_uniform": [round(1.0 / idx.numel(), 6)] * idx.numel()}
        out.append(r)
    return out


def run(ctx, variant: str, order_name: str, seed: int, args) -> list[dict]:
    tasks = ORDERS[order_name]
    recs = []
    for stage in range(len(tasks)):
        for t in tasks[:stage + 1]:
            r = evaluate_task(ctx, t, variant, order_name, seed, stage, args)
            recs += r
            print(f"    stage {stage}  {t:10s} class-IL={acc(r, 'pred_class_il'):.4f} "
                  f"task-IL={acc(r, 'pred_task_il'):.4f}", flush=True)
    return recs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--variants", default=",".join(VARIANTS))
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

    for variant in [v.strip() for v in args.variants.split(",") if v.strip()]:
        if variant not in VARIANTS:
            raise SystemExit(f"❌ 未知變體 {variant}；可用：{list(VARIANTS)}")
        arch_sfx = "" if args.arch == DEFAULT_ARCH else "_" + args.arch
        fold_sfx = "" if args.fold == 1 else f"_f{args.fold}"
        path = out_dir / f"{VARIANTS[variant]}_{args.order}_seed{seed}{arch_sfx}{fold_sfx}.json"
        if path.exists() and not args.no_resume:
            print(f"▷ 跳過（已存在）{path.name}", flush=True)
            continue
        print(f"═══ zero-shot {variant} fold={args.fold} seed={seed} "
              f"order={args.order}", flush=True)
        path.write_text(json.dumps(run(ctx, variant, args.order, seed, args), indent=1))
        print(f"    → {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
