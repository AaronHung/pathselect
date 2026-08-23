#!/usr/bin/env python3
"""憲法 §3.5 —— 批次產物的存在性檢查。

批次腳本在印「完成」之前必須呼叫本檔。**流程走到最後一行不等於成功。**

    python scripts/check_batch_products.py --tag hier2 --arms A5,A3 --seeds 0,1,2,3,4

缺項時以非零狀態結束並列出缺了哪些檔案。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--order", default="reverse")
    ap.add_argument("--suffix", default="", help="檔名後綴，例如 _M64 或 _hier")
    ap.add_argument("--also", default="", help="另外必須存在的檔案，逗號分隔（相對 repo）")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
    d = REPO_ROOT / "outputs" / "exp2" / args.tag / "per_slide"

    expected = [f"{a}_{args.order}_seed{s}{args.suffix}.json" for a in arms for s in seeds]
    missing = [f for f in expected if not (d / f).is_file()]
    empty = [f for f in expected if (d / f).is_file() and (d / f).stat().st_size == 0]
    for extra in (x.strip() for x in args.also.split(",") if x.strip()):
        if not (REPO_ROOT / extra).is_file():
            missing.append(extra)

    print(f"[check] tag={args.tag} 期望 {len(expected)} 份、"
          f"實得 {len(expected) - len(missing)} 份")
    if missing:
        print(f"[check] ❌ 缺 {len(missing)} 份：")
        for f in missing:
            print(f"          {f}")
    if empty:
        print(f"[check] ❌ 空檔 {len(empty)} 份：{empty}")
    if missing or empty:
        print("[check] 批次未完成 —— 不得宣告完成。")
        return 1
    print("[check] ✅ 產物齊全")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
