#!/usr/bin/env python3
"""E1 — 記憶體效率曲線報告。

目的：反駁「replay 做了全部的事，把 |M| 開大就好」。
若 A3 在夠大的 |M| 下追上 A5 在 |M|=512 的表現，那是重要的負面資訊，照實報。

從 outputs/exp2/memory/per_slide/*.json 彙總，不重跑任何訓練。
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_exp2 import ARMS, ORDERS, arm_metrics                   # noqa: E402
from selector.text_encoder import load_config                            # noqa: E402

MEM_DIR = REPO_ROOT / "outputs" / "exp2" / "memory"
OUT = MEM_DIR / "MEMORY.md"
METRICS = [("final_task_il", "task-IL final avg", True),
           ("final_class_il", "class-IL final avg", True),
           ("mean_leak", "跨任務洩漏率", False),
           ("mean_jaccard", "selection Jaccard", True)]
ARMS_E1 = ["A3", "A5"]
CONTRACT_CAP = 512


def ms(vals, scale=1.0, fmt="{:.4f}"):
    vals = [v * scale for v in vals if v is not None and v == v]
    if not vals:
        return "—"
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f"{fmt.format(statistics.mean(vals))} ± {fmt.format(sd)}"


def mean_of(M, arm, cap, seeds, key):
    vals = [M[(arm, cap)][s].get(key) for s in seeds
            if (arm, cap) in M and s in M[(arm, cap)]
            and M[(arm, cap)][s].get("per_task")]
    vals = [v for v in vals if v is not None]
    return statistics.mean(vals) if vals else None


def main() -> int:
    cfg = load_config()
    label_space = list(cfg["tasks"])
    tasks = ORDERS["reverse"]
    recs = [r for p in sorted((MEM_DIR / "per_slide").glob("*.json"))
            for r in json.loads(p.read_text())]
    if not recs:
        print("尚無資料")
        return 1
    caps = sorted({r["mem_capacity"] for r in recs})
    seeds = sorted({r["seed"] for r in recs})
    print(f"載入 {len(recs)} 筆；|M| = {caps}；seeds = {seeds}")

    M = {}
    for arm in ARMS_E1:
        for cap in caps:
            sub = [r for r in recs if r["arm"] == arm and r["mem_capacity"] == cap]
            if not sub:
                continue
            M[(arm, cap)] = {s: arm_metrics(sub, arm, tasks, s, label_space)
                             for s in seeds}

    L = [
        "# E1 — 記憶體效率曲線",
        "",
        "目的：反駁「replay 做了全部的事，把 |M| 開大就好」。",
        "",
        f"arms = {ARMS_E1}、|M| ∈ {caps}、reverse order、seeds {seeds}。"
        "其餘設定與主表完全相同（B=8、c=1、epochs 5、lr 1e-3、beta_s 0.1、"
        "beta_u 0.1、λ 全 1.0 不調、replay_k=1）。",
        "",
        f"⚠️ **|M| = 1024 超出 CONTRACT-3 的 |M| ≤ {CONTRACT_CAP}**，"
        "是刻意探測契約之外的診斷點，需在程式中顯式 opt-in "
        "（`SelectionMemory(..., allow_over_contract=True)`）。",
        f"⚠️ |M| = {CONTRACT_CAP} 的 A3 / A5 直接沿用主表存檔（同設定的決定性重跑）。",
        "",
    ]
    for key, label, higher in METRICS:
        L += [f"## {label}（{'越大越好' if higher else '越小越好'}）", "",
              "| \\|M\\| | " + " | ".join(f"{a} {ARMS[a]['name']}" for a in ARMS_E1)
              + " | A5 − A3（配對） |",
              "|---" * (len(ARMS_E1) + 2) + "|"]
        for cap in caps:
            cells = []
            for arm in ARMS_E1:
                if (arm, cap) not in M:
                    cells.append("—")
                    continue
                vals = [M[(arm, cap)][s].get(key) for s in seeds
                        if M[(arm, cap)][s].get("per_task")]
                cells.append(ms([v for v in vals if v is not None],
                                scale=1 if key == "mean_jaccard" else 100,
                                fmt="{:.4f}" if key == "mean_jaccard" else "{:.2f}"))
            paired = "—"
            if ("A3", cap) in M and ("A5", cap) in M:
                d = [M[("A5", cap)][s][key] - M[("A3", cap)][s][key] for s in seeds
                     if M[("A5", cap)][s].get("per_task")
                     and M[("A3", cap)][s].get("per_task")]
                if d:
                    sc = 1 if key == "mean_jaccard" else 100
                    wins = sum((x > 0) if higher else (x < 0) for x in d)
                    sd_ = statistics.stdev(d) if len(d) > 1 else 0.0
                    paired = (f"{statistics.mean(d) * sc:+.2f} ± {sd_ * sc:.2f}"
                              f"（{wins}/{len(d)}）")
            mark = "  ⚠️ 契約外" if cap > CONTRACT_CAP else ""
            L.append(f"| {cap}{mark} | " + " | ".join(cells) + f" | {paired} |")
        L.append("")

    # 追平點
    ref = mean_of(M, "A3", CONTRACT_CAP, seeds, "final_class_il")
    L += ["## 追平點", ""]
    if ref is None:
        L.append("（缺 A3 @ |M|=512 的資料，無法計算）")
    else:
        hit = next((c for c in caps
                    if (v := mean_of(M, "A5", c, seeds, "final_class_il")) is not None
                    and v >= ref), None)
        L.append(f"A3 在 |M|={CONTRACT_CAP} 的 class-IL final avg = **{ref:.4f}**。")
        L.append("")
        L.append(f"→ **A5 在 |M|={hit} 時追平 A3 在 |M|={CONTRACT_CAP} 的 class-IL 表現。**"
                 if hit is not None else
                 f"→ **A5 在所有測試的 |M| 都沒有追平 A3 在 |M|={CONTRACT_CAP} 的 "
                 f"class-IL 表現。**")
        # 反向：A3 開大能不能追上 A5@512
        ref5 = mean_of(M, "A5", CONTRACT_CAP, seeds, "final_class_il")
        if ref5 is not None:
            over = [c for c in caps
                    if (v := mean_of(M, "A3", c, seeds, "final_class_il")) is not None
                    and v >= ref5]
            L += ["",
                  f"反向檢查（PI 指定的負面資訊）：A5 在 |M|={CONTRACT_CAP} 的 "
                  f"class-IL = **{ref5:.4f}**；"
                  + (f"**A3 在 |M| ∈ {over} 追上或超過它**。"
                     if over else "A3 在所有測試的 |M| 都沒有追上它。")]
    # 穩健性：曲線非單調時，只對 A3@512 比較會讓結論依賴單一低點
    L += ["", "## 穩健性檢查：對 A3 的**最佳** |M| 比較", "",
          "⚠️ 這條曲線**不是單調的**（見上表）。只拿 A5 去比 A3@512 有可能踩到 A3 "
          "的低點，所以再對 A3 在所有 |M| 上的最佳值比一次。", ""]
    best = [(c, mean_of(M, "A3", c, seeds, "final_class_il")) for c in caps]
    best = [(c, v) for c, v in best if v is not None]
    if best:
        bc, bv = max(best, key=lambda x: x[1])
        hit2 = next((c for c in caps
                     if (v := mean_of(M, "A5", c, seeds, "final_class_il")) is not None
                     and v >= bv), None)
        L += [f"- A3 的 class-IL 最佳值 = **{bv:.4f}**，出現在 |M|={bc}。",
              (f"- **A5 在 |M|={hit2} 就達到或超過 A3 的最佳值**"
               f"（A5@{hit2} = {mean_of(M, 'A5', hit2, seeds, 'final_class_il'):.4f}）。"
               if hit2 is not None else
               "- A5 在所有測試的 |M| 都沒有達到 A3 的最佳值。"),
              ""]
        L += ["- 各 |M| 的 class-IL（看非單調性）：",
              "  - A3：" + "、".join(
                  f"{c}→{v:.4f}" for c, v in best),
              "  - A5：" + "、".join(
                  f"{c}→{mean_of(M, 'A5', c, seeds, 'final_class_il'):.4f}"
                  for c in caps if mean_of(M, "A5", c, seeds, "final_class_il")),
              ""]
    L += ["", "逐 slide 預測：`outputs/exp2/memory/per_slide/*.json`", ""]
    OUT.write_text("\n".join(L) + "\n")
    print(f"→ {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
