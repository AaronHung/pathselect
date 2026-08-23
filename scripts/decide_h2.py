#!/usr/bin/env python3
"""H1 判準：只有在 B2 ≥ A5（class-IL、5 seeds、非 within noise）時才跑 H2。

exit 0 = 跑 H2；exit 1 = 不跑，直接進 F3。
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_exp2 import ORDERS, arm_metrics                         # noqa: E402
from selector.text_encoder import load_config                            # noqa: E402


def main() -> int:
    cfg = load_config()
    recs = [r for d in ("ablation", "main")
            for f in sorted((REPO_ROOT / "outputs" / "exp2" / d / "per_slide").glob("*.json"))
            for r in json.loads(f.read_text())
            if r["order"] == "reverse" and r.get("mem_capacity", 512) == 512
            and r.get("arch", "flat") == "flat"]
    seen, uniq = set(), []
    for r in recs:
        k = (r["arm"], r["seed"], r["stage"], r["task"], r["slide_id"])
        if k not in seen:
            seen.add(k); uniq.append(r)
    M = {}
    for arm in ("A5", "B2"):
        sub = [r for r in uniq if r["arm"] == arm]
        seeds = sorted({r["seed"] for r in sub})
        M[arm] = ({s: arm_metrics(sub, arm, ORDERS["reverse"], s, cfg["tasks"])
                   for s in seeds}, seeds)
    common = sorted(set(M["A5"][1]) & set(M["B2"][1]))
    d = [M["B2"][0][s]["final_class_il"] - M["A5"][0][s]["final_class_il"] for s in common]
    wins = sum(x > 0 for x in d)
    mean = statistics.mean(d) * 100
    print(f"H1：B2 − A5 class-IL 共同 seeds {common}")
    print(f"    逐 seed {[f'{x*100:+.2f}' for x in d]}")
    print(f"    配對 {mean:+.2f} pp，win {wins}/{len(d)}")
    go = mean > 0 and wins >= 4 and len(d) >= 5
    print(f"    → {'跑 H2（B2 記憶體曲線）' if go else '不跑 H2，直接進 F3'}")
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main())
