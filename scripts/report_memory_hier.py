#!/usr/bin/env python3
"""E1 階層版 —— 記憶體效率曲線（per_budget 配額、hier 架構）。

與 flat 版 `report_memory.py` 相同格式，另外兩件事：
  1. **每個 |M| 都報結構性指標**（單組比例）。若某個容量下階層退化，
     那格的判讀無效，明確標出。
  2. 產出 flat vs hier 的曲線對照 —— `hier-A3 − flat-A3 = −3.11 pp` 已證明
     replay 在階層下的行為與 flat 不同，該曲線不可假設可移植。
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_exp2 import ARMS, ORDERS, arm_metrics                   # noqa: E402
from selector.text_encoder import load_config                            # noqa: E402

HIER_DIR = REPO_ROOT / "outputs" / "exp2" / "memory_hier" / "per_slide"
FLAT_DIR = REPO_ROOT / "outputs" / "exp2" / "memory" / "per_slide"
OUT = REPO_ROOT / "outputs" / "exp2" / "memory_hier" / "MEMORY_HIER.md"
METRICS = [("final_task_il", "task-IL final avg", True),
           ("final_class_il", "class-IL final avg", True),
           ("mean_leak", "跨任務洩漏率", False),
           ("mean_jaccard", "selection Jaccard", True)]
ARMS_E1 = ["A3", "A5"]
CONTRACT_CAP = 512
DEGENERATE = 0.5      # 單組比例超過即視為階層退化，該格判讀無效


def load(d: Path, arch: str) -> list[dict]:
    if not d.is_dir():
        return []
    return [r for f in sorted(d.glob("*.json")) for r in json.loads(f.read_text())
            if r.get("arch", "flat") == arch and r["order"] == "reverse"]


def ms(vals, scale=1.0, fmt="{:.2f}"):
    vals = [v * scale for v in vals if v is not None and v == v]
    if not vals:
        return "—"
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f"{fmt.format(statistics.mean(vals))} ± {fmt.format(sd)}"


def verdict(w, n):
    return ("**systematic**" if w in (0, n) else "within noise" if w <= 3
            else "directional, inconclusive")


def structural(recs, tasks) -> tuple[float, float, dict, int]:
    rs = [r for r in recs if r["stage"] == len(tasks) - 1]
    ng = [sum(1 for v in r["group_quota"] if v > 0) for r in rs]
    share = [max(r["group_quota"]) / sum(r["group_quota"]) for r in rs]
    return (Counter(ng)[1] / len(ng), statistics.mean(ng),
            dict(sorted(Counter(ng).items())), len(rs))


def build(recs, tasks, label_space, seeds):
    M = {}
    for arm in ARMS_E1:
        for cap in sorted({r.get("mem_capacity", CONTRACT_CAP) for r in recs}):
            sub = [r for r in recs if r["arm"] == arm
                   and r.get("mem_capacity", CONTRACT_CAP) == cap]
            if sub:
                M[(arm, cap)] = {s: arm_metrics(sub, arm, tasks, s, label_space)
                                 for s in seeds}
    return M


def mean_of(M, arm, cap, seeds, key):
    if (arm, cap) not in M:
        return None
    v = [M[(arm, cap)][s].get(key) for s in seeds if M[(arm, cap)][s].get("per_task")]
    v = [x for x in v if x is not None]
    return statistics.mean(v) if v else None


def main() -> int:
    cfg = load_config()
    tasks = ORDERS["reverse"]
    hier, flat = load(HIER_DIR, "hier"), load(FLAT_DIR, "flat")
    if not hier:
        print("尚無 memory_hier 資料"); return 1
    caps = sorted({r.get("mem_capacity", CONTRACT_CAP) for r in hier})
    seeds = sorted({r["seed"] for r in hier})
    print(f"hier {len(hier)} 筆、flat {len(flat)} 筆；|M| = {caps}；seeds = {seeds}")

    Mh = build(hier, tasks, cfg["tasks"], seeds)
    Mf = build(flat, tasks, cfg["tasks"], seeds) if flat else {}

    # 結構性把關（逐 |M|）
    struct = {}
    for cap in caps:
        sub = [r for r in hier if r.get("mem_capacity", CONTRACT_CAP) == cap]
        if sub:
            struct[cap] = structural(sub, tasks)

    L = ["# E1 階層版 — 記憶體效率曲線（per_budget 配額）", "",
         f"arms = {ARMS_E1}、|M| ∈ {caps}、reverse order、seeds {seeds}、"
         "arch = **hier**、allocation = per_budget。其餘設定與主表相同"
         "（B=8、c=1、epochs 5、lr 1e-3、beta_s 0.1、beta_u 0.1、λ 全 1.0、replay_k=1）。",
         "",
         f"⚠️ |M| = {CONTRACT_CAP} 沿用 G1' 的存檔，不重跑。"
         f"|M| = 1024 超出 CONTRACT-3，需顯式 opt-in。",
         "",
         "## 結構性把關（逐 |M|）", "",
         "階層在某個容量下若退化成單組選取，**該格的判讀無效**。", "",
         "| \\|M\\| | 單組比例 | 平均用到幾組 | 分佈 | 判讀 |",
         "|---|---|---|---|---|"]
    invalid = []
    for cap in caps:
        if cap not in struct:
            continue
        single, mean_ng, hist, n = struct[cap]
        ok = single <= DEGENERATE
        if not ok:
            invalid.append(cap)
        L.append(f"| {cap} | {single:.1%} | {mean_ng:.2f} | {hist} | "
                 f"{'✅ 有效' if ok else '❌ **退化，該格判讀無效**'} |")
    L.append("")
    if invalid:
        L += [f"⚠️ **|M| ∈ {invalid} 的階層退化，下方所有涉及這些容量的數字一律不得採用。**", ""]
    else:
        L += ["✅ 所有容量下階層都有作用空間。", ""]

    for key, label, higher in METRICS:
        L += [f"## {label}", "",
              "| \\|M\\| | " + " | ".join(f"{a} {ARMS[a]['name']}" for a in ARMS_E1)
              + " | A5 − A3（配對） |", "|---" * (len(ARMS_E1) + 2) + "|"]
        for cap in caps:
            cells = []
            for arm in ARMS_E1:
                v = [Mh[(arm, cap)][s].get(key) for s in seeds
                     if (arm, cap) in Mh and Mh[(arm, cap)][s].get("per_task")]
                cells.append(ms([x for x in v if x is not None],
                                1 if key == "mean_jaccard" else 100,
                                "{:.4f}" if key == "mean_jaccard" else "{:.2f}"))
            paired = "—"
            if ("A3", cap) in Mh and ("A5", cap) in Mh:
                d = [Mh[("A5", cap)][s][key] - Mh[("A3", cap)][s][key] for s in seeds
                     if Mh[("A5", cap)][s].get("per_task")
                     and Mh[("A3", cap)][s].get("per_task")]
                if d:
                    sc = 1 if key == "mean_jaccard" else 100
                    w = sum((x > 0) if higher else (x < 0) for x in d)
                    sd = statistics.stdev(d) if len(d) > 1 else 0.0
                    paired = (f"{statistics.mean(d) * sc:+.2f} ± {sd * sc:.2f}"
                              f"（{w}/{len(d)}，{verdict(w, len(d))}）")
            mark = "  ❌退化" if cap in invalid else ""
            L.append(f"| {cap}{mark} | " + " | ".join(cells) + f" | {paired} |")
        L.append("")

    # flat vs hier 對照
    if Mf:
        L += ["## flat vs hier 的曲線對照", "",
              "`hier-A3 − flat-A3 = −3.11 pp`（G1'）已證明 **replay 在階層下的行為與 "
              "flat 不同**，因此 flat 的記憶體曲線不可假設可移植。以下為兩者並列。", "",
              "| \\|M\\| | arm | flat class-IL | hier class-IL | hier − flat |",
              "|---|---|---|---|---|"]
        for cap in caps:
            for arm in ARMS_E1:
                f_ = mean_of(Mf, arm, cap, seeds, "final_class_il")
                h_ = mean_of(Mh, arm, cap, seeds, "final_class_il")
                if f_ is None or h_ is None:
                    continue
                L.append(f"| {cap} | {arm} | {f_ * 100:.2f} | {h_ * 100:.2f} | "
                         f"{(h_ - f_) * 100:+.2f} pp |")
        L.append("")

    # 記憶體效率主張（階層版）
    valid_caps = [c for c in caps if c not in invalid]
    best = [(c, mean_of(Mh, "A3", c, seeds, "final_class_il")) for c in valid_caps]
    best = [(c, v) for c, v in best if v is not None]
    L += ["## 記憶體效率主張（階層版）", ""]
    if best:
        bc, bv = max(best, key=lambda x: x[1])
        hit = next((c for c in valid_caps
                    if (v := mean_of(Mh, "A5", c, seeds, "final_class_il")) is not None
                    and v >= bv), None)
        L += [f"A3 的 class-IL 全域最佳 = **{bv:.4f}**（|M|={bc}）。",
              "",
              (f"→ **A5 在 |M|={hit} 達到 A3 的全域最佳 → "
               f"{bc // hit if hit and bc >= hit else 1}× 記憶體效率。**"
               if hit is not None else
               "→ A5 在所有有效容量下都沒有達到 A3 的全域最佳。"),
              "",
              "各 |M| 的 class-IL：",
              "- A3：" + "、".join(f"{c}→{v:.4f}" for c, v in best),
              "- A5：" + "、".join(
                  f"{c}→{v:.4f}" for c in valid_caps
                  if (v := mean_of(Mh, "A5", c, seeds, "final_class_il")) is not None),
              ""]
    L += ["逐 slide 預測：`outputs/exp2/memory_hier/per_slide/*.json`", ""]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n")
    print(f"→ {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
