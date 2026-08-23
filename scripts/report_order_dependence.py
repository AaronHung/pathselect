#!/usr/bin/env python3
"""順序依賴 —— 獨立成節，不當雜訊帶過（CONSTITUTION §3.2）。

reverse（esca→rcc→brca→lung）與 main（lung→brca→rcc→esca）之間，
哪些結論翻轉、哪些穩定。只用**兩個 order 共同有的 seeds** 做配對。
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

OUT = REPO_ROOT / "outputs" / "exp2" / "ORDER_DEPENDENCE.md"
TAGS = {"reverse": "main", "main": "order_main"}
ARMS_CMP = ["A1", "A2", "A3", "A4", "A5"]
METRICS = [("final_task_il", "task-IL final avg", True),
           ("final_class_il", "class-IL final avg", True),
           ("mean_leak", "跨任務洩漏率", False)]


def load(tag, order):
    d = REPO_ROOT / "outputs" / "exp2" / tag / "per_slide"
    return [r for f in sorted(d.glob("*.json")) for r in json.loads(f.read_text())
            if r["order"] == order and r.get("mem_capacity", 512) == 512
            and r.get("arch", "flat") == "flat"]


def verdict(wins, n):
    return ("**systematic**" if wins in (0, n) else "within noise" if wins <= 3
            else "directional, inconclusive")


def main() -> int:
    cfg = load_config()
    M, seed_sets = {}, []
    for order, tag in TAGS.items():
        recs = load(tag, order)
        if not recs:
            print(f"缺 {order} 的資料"); return 1
        seeds = sorted({r["seed"] for r in recs})
        seed_sets.append(set(seeds))
        for arm in ARMS_CMP:
            sub = [r for r in recs if r["arm"] == arm]
            if sub:
                M[(order, arm)] = {s: arm_metrics(sub, arm, ORDERS[order], s,
                                                  cfg["tasks"]) for s in seeds}
    common = sorted(set.intersection(*seed_sets))
    print(f"兩個 order 共同的 seeds：{common}")

    L = ["# 順序依賴（獨立成節）", "",
         "CL 的結論不必然跨任務順序成立。本檔把翻轉的部分獨立列出，"
         "**不當雜訊帶過**（CONSTITUTION §3.2）。", "",
         f"reverse = {' → '.join(t.replace('tcga_', '') for t in ORDERS['reverse'])}"
         f"；main = {' → '.join(t.replace('tcga_', '') for t in ORDERS['main'])}。",
         f"配對只用兩個 order **共同有的 seeds {common}**（reverse 另有 seeds 3,4，"
         "此處不納入，以免混入不同樣本）。", "",
         "## 各臂在兩個 order 上的表現", ""]
    for key, label, higher in METRICS:
        L += [f"### {label}", "",
              "| 臂 | reverse | main | main − reverse（配對） | win | 判定 |",
              "|---|---|---|---|---|---|"]
        for arm in ARMS_CMP:
            if ("reverse", arm) not in M or ("main", arm) not in M:
                continue
            rv = [M[("reverse", arm)][s][key] for s in common]
            mn = [M[("main", arm)][s][key] for s in common]
            d = [a - b for a, b in zip(mn, rv)]
            w = sum((x > 0) if higher else (x < 0) for x in d)
            sd = statistics.stdev(d) if len(d) > 1 else 0.0
            L.append(f"| {arm} | {statistics.mean(rv) * 100:.2f} | "
                     f"{statistics.mean(mn) * 100:.2f} | "
                     f"{statistics.mean(d) * 100:+.2f} ± {sd * 100:.2f} | "
                     f"{w}/{len(d)} | {verdict(w, len(d))} |")
        L.append("")

    L += ["## ⚠️ 翻轉的結論", "",
          "以下比較在兩個 order 上**方向相反**，論文必須報告，不得只挑有利的那個。",
          "", "| 對照 | 指標 | reverse | main | 是否翻轉 |", "|---|---|---|---|---|"]
    FLIPS = [("A5", "A4"), ("A2", "A1"), ("A5", "A3")]
    for x, y in FLIPS:
        for key, label, higher in METRICS[:2]:
            if any((o, a) not in M for o in TAGS for a in (x, y)):
                continue
            row = {}
            for order in TAGS:
                d = [M[(order, x)][s][key] - M[(order, y)][s][key] for s in common]
                row[order] = statistics.mean(d) * 100
            flip = "**是**" if row["reverse"] * row["main"] < 0 else "否"
            L.append(f"| {x} − {y} | {label} | {row['reverse']:+.2f} pp | "
                     f"{row['main']:+.2f} pp | {flip} |")
    L += ["",
          "### 讀法", "",
          "- **eq 的貢獻非跨順序穩定**：A5 − A4 在兩個 order 上方向相反"
          "（reverse +2.43 / main −0.67 task-IL；class-IL 亦然）。這是本檔最主要的發現。",
          "- **replay 的效果跨順序穩定**：A3 相對 A1 在兩個 order 上都是大幅改善。",
          "",
          "這是 CL 的真實現象（任務難度與順序位置交互作用），不是實作瑕疵。", "",
          "### ⚠️ A2 − A1 不是順序效應，是 seed 變異", "",
          "上表用共同 seeds 0–2 時，A2 − A1 在 reverse 是 +7.92 pp、main 是 +18.89 pp，"
          "看起來像「main 上 LoRA merge 更有效」。**但那是取樣造成的假象。**",
          "",
          "reverse 的**全 5 seeds** 逐筆差值：`+1.84, +16.31, +5.61, −11.91, −27.29` pp，"
          "平均 **−3.09 pp**、win **3/5（within noise）**，幅度橫跨 43 pp。"
          "共同的 seeds 0–2 剛好是正的那三個。",
          "",
          "**因此 A2 − A1 不得列為順序依賴的例證** —— 它在 reverse 上根本就是 "
          "within noise，跨 order 的差異被 seed 變異淹沒。",
          "",
          "**方法學教訓**：兩個 order 的 seed 數不同時（reverse 5、main 3），"
          "只用共同子集做配對雖然統計上正確，但**子集可能不代表母體**。"
          "任何跨 order 的宣稱都必須同時檢查該對照在各自 order 的全 seeds 上是否成立。",
          "",
          "## 觀察：l_eq fire rate", "",
          "B2（只 eq）的 `l_eq_fire_rate` 為 0.1142，A5（三項全開）為 0.0740 —— "
          "**B2 的 hinge 被觸發的比例約為 A5 的 1.5 倍**。",
          "",
          "⚠️ **這是觀察，不做因果宣稱。** 可能的讀法包括「A5 的 replay 與 KD 已經把 "
          "utility 撐住、使 hinge 較少觸發」，也可能只是兩臂訓練軌跡不同的副產物。"
          "要區分需要另一組診斷，不在本輪範圍。", "",
          "逐 slide 預測：`outputs/exp2/main/per_slide/`（reverse）、"
          "`outputs/exp2/order_main/per_slide/`（main）", ""]
    OUT.write_text("\n".join(L) + "\n")
    print(f"→ {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
