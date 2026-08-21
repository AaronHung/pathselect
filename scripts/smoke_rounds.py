#!/usr/bin/env python3
"""Smoke — 單一 task、少量 slide 跑完整個 chunked sequential loop。

驗的是「跑得動、契約沒被違反」，不是任何實驗結論：
  - 每一輪 sum_j b_j = c
  - 累積已選數為 c, 2c, ..., B
  - 沒有 patch 被選第二次

    python scripts/smoke_rounds.py --task tcga_lung --slides 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from selector.evaluate import iter_test_slides                          # noqa: E402
from selector.grouping import (TISSUE_GROUP_NAMES, assign_groups,       # noqa: E402
                               tissue_text_features)
from selector.model import GroupSelector, PatchSelector                 # noqa: E402
from selector.rounds import DEFAULT_BUDGET, DEFAULT_CHUNK, run_rounds   # noqa: E402
from selector.task_query import encode_task_query                       # noqa: E402
from selector.text_encoder import load_config                           # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="tcga_lung")
    ap.add_argument("--slides", type=int, default=10)
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    ap.add_argument("--chunk", type=int, default=DEFAULT_CHUNK)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    cfg = load_config()
    device = torch.device("cpu")
    task_pos = cfg["tasks"].index(args.task)

    tissue = tissue_text_features(cfg, device=device)
    q_tau = encode_task_query(args.task, cfg, device=device)
    f_group, f_patch = GroupSelector().eval(), PatchSelector().eval()

    print(f"task={args.task} (pos {task_pos})  B={args.budget}  c={args.chunk}  "
          f"rounds={-(-args.budget // args.chunk)}  slides={args.slides}")
    print(f"groups: {', '.join(TISSUE_GROUP_NAMES)}\n")

    header = ("slide  n_patch  round  " + "  ".join(f"{g[:4]:>4}" for g in TISSUE_GROUP_NAMES)
              + "   sum   cum")
    print(header)
    print("-" * len(header))

    ok = True
    for i, rec in enumerate(iter_test_slides(cfg, args.task, task_pos, limit=args.slides)):
        grouping = assign_groups(rec.Z, tissue)
        with torch.no_grad():
            res = run_rounds(rec.Z, grouping, q_tau, f_group, f_patch,
                             budget=args.budget, chunk=args.chunk)
        for r in res.records:
            b = r.b.tolist()
            print(f"{i:>5}  {rec.Z.shape[0]:>7}  {r.t:>5}  "
                  + "  ".join(f"{x:>4}" for x in b)
                  + f"   {sum(b):>3}   {r.n_selected_after:>3}")
            if sum(b) != min(args.chunk, args.budget):
                ok = False
        sel = res.selected.tolist()
        if len(sel) != len(set(sel)) or len(sel) != args.budget:
            ok = False
        print(f"{'':>5}  {'':>7}  total  已選 {len(sel)}  相異 {len(set(sel))}  "
              f"覆蓋率 {len(sel) / rec.Z.shape[0]:.3%}")
        print()

    print("契約檢查：每輪 sum(b_j)=c 且無重複選取 →", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
