#!/usr/bin/env python3
"""DR-048：把外部基準**自身輸出**的指標檔折成四個指標。

外部實作每折會寫出（`<run>/metrics/`）：

    test_acc.txt        [[a00], [a10, a11], [a20, a21, a22], [a30, ...]]
                        —— 就是**準確率矩陣** A[i][j]（下三角，未遮罩）
    test_mask_acc.txt   [m0, m1, m2, m3]  —— 各任務最終的**遮罩**準確率

因此不必重新推導：直接把它們餵進 `sota/metrics.py` 既有的
`bwt` / `forgetting`，**用與我們自己那張表完全相同的定義**去算。
這正是「以其自身輸出的指標彙整」的意思 —— 數字是它算的，定義是我們統一的。

⚠️ 路徑一律由呼叫端傳入，本檔不寫死任何外部方法的目錄名。
"""
from __future__ import annotations

import argparse
import ast
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sota.metrics import bwt, forgetting                              # noqa: E402

ACC_FILE = "test_acc.txt"
MASK_FILE = "test_mask_acc.txt"

#: 本 benchmark 的任務數。**必須對照** —— 中途掛掉的折會留下一個較小但
#: 形狀合法的下三角矩陣（例如跑到 task 2 就死，留下 2×2），
#: 若不檢查就會被當成「完整的 2 任務實驗」平均進十折，悄悄拉偏結果。
EXPECTED_TASKS = 4


def read_matrix(metrics_dir: Path) -> list[list[float]]:
    """讀準確率矩陣，補成方陣（上三角填 None）。"""
    rows = ast.literal_eval((metrics_dir / ACC_FILE).read_text().strip())
    T = len(rows)
    if any(len(r) != i + 1 for i, r in enumerate(rows)):
        raise ValueError(f"{metrics_dir/ACC_FILE} 不是下三角：各列長度 "
                         f"{[len(r) for r in rows]}")
    return [list(r) + [None] * (T - len(r)) for r in rows]


def read_masked(metrics_dir: Path) -> list[float]:
    return list(ast.literal_eval((metrics_dir / MASK_FILE).read_text().strip()))


def fold_metrics(metrics_dir: Path, expected_tasks: int = EXPECTED_TASKS) -> dict:
    A = read_matrix(metrics_dir)
    m = read_masked(metrics_dir)
    if len(A) != expected_tasks:
        raise ValueError(f"{metrics_dir}：矩陣只有 {len(A)} 個階段，"
                         f"預期 {expected_tasks} —— 這一折沒跑完，不計入")
    if len(m) != expected_tasks:
        raise ValueError(f"{metrics_dir}：masked 只有 {len(m)} 個任務，"
                         f"預期 {expected_tasks}")
    final = [x for x in A[-1] if x is not None]
    if len(final) != len(A):
        raise ValueError(f"{metrics_dir}：最終階段只有 {len(final)}/{len(A)} 個任務")
    return {"acc": statistics.mean(final),
            "masked_acc": statistics.mean(m),
            "forgetting": forgetting(A),
            "bwt": bwt(A),
            "matrix": A}


def find_folds(root: Path) -> dict[int, Path]:
    """找出 `<...>/train-data_split_seed_{k}/metrics/`。折號取自目錄名。"""
    out = {}
    for d in sorted(root.rglob("train-data_split_seed_*")):
        md = d / "metrics"
        if not (md / ACC_FILE).is_file():
            continue
        k = int(d.name.rsplit("_", 1)[1])
        out[k] = md              # 同折有多份時取路徑排序最後的（最新一次）
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root", help="結果根目錄（會遞迴找 train-data_split_seed_*）")
    ap.add_argument("--json", default=None, help="另存一份 JSON")
    args = ap.parse_args(argv)

    folds = find_folds(Path(args.root))
    if not folds:
        print(f"⚠️ {args.root} 底下找不到任何完成的折")
        return 1
    per, bad = {}, []
    for k, md in sorted(folds.items()):
        try:
            per[k] = fold_metrics(md)
        except (ValueError, SyntaxError) as exc:
            bad.append(f"fold {k}: {exc}")
    print(f"{'fold':>5} {'ACC':>8} {'Masked':>8} {'Forget':>8} {'BWT':>8}")
    for k, m in sorted(per.items()):
        print(f"{k:>5} {m['acc']:8.4f} {m['masked_acc']:8.4f} "
              f"{m['forgetting']:8.4f} {m['bwt']:8.4f}")
    agg = {}
    for key in ("acc", "masked_acc", "forgetting", "bwt"):
        v = [m[key] for m in per.values() if m[key] is not None]
        agg[key] = (statistics.mean(v), statistics.stdev(v) if len(v) > 1 else 0.0)
        print(f"  {key:11s} {agg[key][0]:.4f} ± {agg[key][1]:.3f}")
    print(f"  n folds = {len(per)}")
    for b in bad:
        print(f"  ⚠️ {b}")
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"per_fold": {str(k): {kk: vv for kk, vv in m.items() if kk != "matrix"}
                          for k, m in per.items()},
             "aggregate": agg, "n": len(per), "errors": bad}, indent=1))
        print(f"→ {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
