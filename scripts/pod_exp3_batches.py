#!/usr/bin/env python3
"""DR-052 第 3 步的三批 run 清單（給 scripts/pod_run_exp3.sh 吃）。

    python scripts/pod_exp3_batches.py 1        # 反向十折 hier + flat（20 run）
    python scripts/pod_exp3_batches.py 2        # 正向十折 hier + flat（20 run）
    python scripts/pod_exp3_batches.py 3        # fold-1 五 seed 組件消融（30 run）

每行 = `run 名稱|run_exp2 參數`。所有 run 都帶 `--head accumulating --out-root outputs/exp3`，
per_slide 落在 outputs/exp3/<tag>/per_slide/，檔名規約同 exp2 另加 `_acc`。
組件消融沿用 DR-046 的 flat 協定（fold 1、seeds 0–4；E2/E3 亦為 flat）。
"""
from __future__ import annotations

import sys

COMMON = "--head accumulating --out-root outputs/exp3"
BATCHES = {
    1: [(f"A5_{arch}_rev_f{k}",
         f"--arms A5 --order reverse --arch {arch} --fold {k} --seeds {k} --tag sota_acc {COMMON}")
        for arch in ("hier", "flat") for k in range(1, 11)],
    2: [(f"A5_{arch}_fwd_f{k}",
         f"--arms A5 --order main --arch {arch} --fold {k} --seeds {k} --tag sota_acc {COMMON}")
        for arch in ("hier", "flat") for k in range(1, 11)],
    3: [(f"{arm}_flat_rev_f1_s{s}",
         f"--arms {arm} --order reverse --arch flat --fold 1 --seeds {s} --tag ablation_acc {COMMON}")
        for arm in ("A2", "A3", "A4", "A5", "B1", "B2") for s in range(5)],
}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1 or argv[0] not in {"1", "2", "3"}:
        print(__doc__); return 2
    for name, args in BATCHES[int(argv[0])]:
        print(f"{name}|{args}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
