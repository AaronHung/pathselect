#!/usr/bin/env python3
"""G1' 的結構性指標 —— 只看「每張 slide 用到幾個 group」，可在批次跑完前先看。

PI 指定的停止條件：若仍有 >50% 的 slide 落在單一 group，代表配額口徑不是真因。
本檔獨立於批次腳本，讀多少算多少，不影響跑中的 job。
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DIRS = {"hier (per_chunk, G1)": REPO_ROOT / "outputs" / "exp2" / "hier" / "per_slide",
        "hier2 (per_budget, G1')": REPO_ROOT / "outputs" / "exp2" / "hier2" / "per_slide",
        "flat (參照)": REPO_ROOT / "outputs" / "exp2" / "main" / "per_slide"}
THRESHOLD = 0.5


def stats(d: Path, arch: str | None):
    rs = []
    for f in sorted(d.glob("*.json")):
        for r in json.loads(f.read_text()):
            if r["stage"] != 3 or r.get("mem_capacity", 512) != 512:
                continue
            if arch and r.get("arch", "flat") != arch:
                continue
            rs.append(r)
    if not rs:
        return None
    ng = [sum(1 for v in r["group_quota"] if v > 0) for r in rs]
    share = [max(r["group_quota"]) / sum(r["group_quota"]) for r in rs]
    return (len(rs), statistics.mean(ng), statistics.mean(share),
            dict(sorted(Counter(ng).items())),
            Counter(ng)[1] / len(ng))


def main() -> int:
    rc = 0
    for name, d in DIRS.items():
        arch = "hier" if "hier" in name else "flat"
        if not d.is_dir():
            print(f"{name}: 尚無資料"); continue
        st = stats(d, arch)
        if st is None:
            print(f"{name}: 尚無資料"); continue
        n, mean_ng, mean_share, hist, single = st
        flag = ""
        if "G1'" in name:
            flag = ("  ⚠️ >50% 單組 → 配額口徑不是真因，停下來回報"
                    if single > THRESHOLD else "  ✅ 低於 50%，階層有作用空間")
            rc = 1 if single > THRESHOLD else 0
        print(f"{name}: n={n} slide-評估")
        print(f"    平均用到 {mean_ng:.2f} 組、最大組佔比 {mean_share:.3f}、"
              f"單組比例 {single:.1%}{flag}")
        print(f"    分佈 {hist}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
