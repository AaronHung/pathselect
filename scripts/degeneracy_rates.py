#!/usr/bin/env python3
"""G1（per_chunk）退化率的兩個口徑（DR-045）。

repo 裡「單組比例」有兩個數字並存：**88.6%** 與 **84.5%**。兩者都對，差在範圍。
本檔把兩者都算出來寫成產物，避免再有人以為其中一個沒有依據。
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs" / "exp2" / "hier" / "per_slide"
OUT = ROOT / "outputs" / "exp2" / "hier" / "DEGENERACY_RATES.md"
LAST_STAGE = 3


def rate(recs):
    ng = [sum(1 for v in r["group_quota"] if v > 0) for r in recs]
    c = Counter(ng)
    return c[1], len(ng), statistics.mean(ng), dict(sorted(c.items()))


def main() -> int:
    recs = [r for f in sorted(SRC.glob("*.json")) for r in json.loads(f.read_text())]
    if not recs:
        print("尚無 hier（G1）資料"); return 1
    arms = sorted({r["arm"] for r in recs})

    rows = [("全部 arm、學完 T4 後", [r for r in recs if r["stage"] == LAST_STAGE]),
            ("全部 arm、全部 stage", recs)]
    rows += [(f"只算 {a}、學完 T4 後",
              [r for r in recs if r["arm"] == a and r["stage"] == LAST_STAGE])
             for a in arms]

    L = ["# G1（per_chunk）退化率的兩個口徑", "",
         "「單組比例」= 學完後只用到一個組織 group 的 slide 佔比。",
         "`outputs/exp2/hier/per_slide/*.json` 的 `group_quota` 欄重算，5 seeds 合併計數。", "",
         "| 口徑 | 單組 / 全部 | 比例 | 平均用到幾組 | 分佈 |",
         "|---|---|---|---|---|"]
    for label, sub in rows:
        if not sub:
            continue
        one, n, mean_ng, hist = rate(sub)
        L.append(f"| {label} | {one} / {n} | **{one / n:.1%}** | {mean_ng:.2f} | {hist} |")
    L += ["",
          "## 兩個數字的來歷", "",
          "- **88.6%** = 全部 arm、學完 T4 後。這是 CLAIMS C-02、憲法 §3.6b、"
          "`report_prior.py`、`tests/test_report_scripts.py` 引用的數字。",
          "- **84.5%** = 只算 A5、學完 T4 後。這是 `HIER.md` 結構性診斷表那一列"
          "（`{1: 1179, ...}` 共 1395 張）的來源，也是 RESULTS_DOSSIER §4.6 用的數字。",
          "",
          "**兩者都對，差在範圍**（DR-045）。引用時必須寫明是哪一個口徑。",
          "不論用哪一個，結論都一樣：per_chunk 在 c=1 下退化，G1 測到的不是階層。",
          ""]
    OUT.write_text("\n".join(L) + "\n")
    print(f"→ {OUT}")
    for label, sub in rows:
        if sub:
            one, n, mean_ng, _ = rate(sub)
            print(f"  {label:22s} {one}/{n} = {one / n:.1%}  平均 {mean_ng:.2f} 組")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
