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


def fire_rate_lines() -> list[str]:
    """從 ablation 的逐 slide 存檔重算 l_eq fire rate。

    ⚠️ 這個數字原本寫死在腳本裡（0.1142），是 3-seed 時代的舊值，與 5-seed 的
    `ablation/EXP2.md`（0.1114）對不上。改為重算，並標明 n（DR-020 / 憲法 §1.2）。
    """
    src = REPO_ROOT / "outputs" / "exp2" / "ablation" / "per_slide"
    if not src.is_dir():
        return ["⚠️ 缺 `outputs/exp2/ablation/per_slide`，無法重算。"]
    recs = [r for f in sorted(src.glob("*.json")) for r in json.loads(f.read_text())]
    out = {}
    for arm in ("B2", "A5"):
        vals = {}
        for r in recs:
            if r["arm"] == arm and r.get("l_eq_fire_rate") is not None:
                vals[r["seed"]] = r["l_eq_fire_rate"]
        if vals:
            out[arm] = (statistics.mean(vals.values()), len(vals))
    if len(out) < 2:
        return ["⚠️ 資料不足，無法重算。"]
    b2, a5 = out["B2"], out["A5"]
    return [f"B2（只 eq）的 `l_eq_fire_rate` 為 **{b2[0]:.4f}**（n={b2[1]} seeds），"
            f"A5（三項全開）為 **{a5[0]:.4f}**（n={a5[1]}）—— "
            f"**B2 的 hinge 被觸發的比例約為 A5 的 {b2[0] / a5[0]:.1f} 倍**。",
            "",
            "（本節數字由 `outputs/exp2/ablation/per_slide/*.json` 重算，不寫死；"
            "先前寫死的 0.1142 是 3-seed 舊值，與 5-seed 產物對不上 —— DR-044 同批修正。）"]


def main() -> int:
    cfg = load_config()
    M, per_arm_seeds = {}, {}
    for order, tag in TAGS.items():
        recs = load(tag, order)
        if not recs:
            print(f"缺 {order} 的資料"); return 1
        for arm in ARMS_CMP:
            sub = [r for r in recs if r["arm"] == arm]
            if not sub:
                continue
            seeds = sorted({r["seed"] for r in sub})
            per_arm_seeds[(order, arm)] = set(seeds)
            M[(order, arm)] = {s: arm_metrics(sub, arm, ORDERS[order], s,
                                              cfg["tasks"]) for s in seeds}
    # ⚠️ 共同 seeds 必須**逐 arm** 計算：main order 只有 A3/A5 補到 5 seeds，
    # 其餘仍是 3 seeds。在 order 層取聯集會讓 A1 之類的臂被要求提供不存在的 seed。
    common_of = {arm: sorted(per_arm_seeds.get(("reverse", arm), set())
                             & per_arm_seeds.get(("main", arm), set()))
                 for arm in ARMS_CMP}
    print("各 arm 的共同 seeds：" + "、".join(
        f"{a}={common_of[a]}" for a in ARMS_CMP if common_of[a]))

    L = ["# 順序依賴（獨立成節）", "",
         "CL 的結論不必然跨任務順序成立。本檔把翻轉的部分獨立列出，"
         "**不當雜訊帶過**（CONSTITUTION §3.2）。", "",
         f"reverse = {' → '.join(t.replace('tcga_', '') for t in ORDERS['reverse'])}"
         f"；main = {' → '.join(t.replace('tcga_', '') for t in ORDERS['main'])}。",
         "配對**逐 arm** 使用該臂在兩個 order 都有的 seeds（表中標 n）—— "
         "main order 只有 A3/A5 補到 5 seeds，其餘仍是 3 seeds，"
         "在 order 層取聯集會混入不存在的樣本（憲法 §1.3）。", "",
         "## 各臂在兩個 order 上的表現", ""]
    for key, label, higher in METRICS:
        L += [f"### {label}", "",
              "| 臂 | reverse | main | main − reverse（配對） | win | 判定 |",
              "|---|---|---|---|---|---|"]
        for arm in ARMS_CMP:
            cm = common_of.get(arm) or []
            if ("reverse", arm) not in M or ("main", arm) not in M or not cm:
                continue
            rv = [M[("reverse", arm)][s][key] for s in cm]
            mn = [M[("main", arm)][s][key] for s in cm]
            d = [a - b for a, b in zip(mn, rv)]
            w = sum((x > 0) if higher else (x < 0) for x in d)
            sd = statistics.stdev(d) if len(d) > 1 else 0.0
            L.append(f"| {arm} (n={len(cm)}) | {statistics.mean(rv) * 100:.2f} | "
                     f"{statistics.mean(mn) * 100:.2f} | "
                     f"{statistics.mean(d) * 100:+.2f} ± {sd * 100:.2f} | "
                     f"{w}/{len(d)} | {verdict(w, len(d))} |")
        L.append("")

    # 量級門檻：一側幾乎為零時不算翻轉，而是「一邊有效、一邊無效」
    NEGLIGIBLE_PP = 0.5
    L += ["## 跨順序的穩定性", "",
          "方向相反才算**翻轉**；若一側的效果量小於 "
          f"{NEGLIGIBLE_PP} pp（幾乎為零），標為「一邊有效、一邊無效」而非翻轉 —— "
          "兩者的論文含義不同。", "",
          "| 對照 | 指標 | reverse | main | 判定 |", "|---|---|---|---|---|"]
    CMP = [("A5", "A4"), ("A5", "A3"), ("A3", "A1")]
    for x, y in CMP:
        for key, label, higher in METRICS[:2]:
            if any((o, a) not in M for o in TAGS for a in (x, y)):
                continue
            cm = sorted(set(common_of.get(x) or []) & set(common_of.get(y) or []))
            if not cm:
                continue
            row = {}
            for order in TAGS:
                d = [M[(order, x)][s][key] - M[(order, y)][s][key] for s in cm]
                row[order] = statistics.mean(d) * 100
            rv, mn = row["reverse"], row["main"]
            if min(abs(rv), abs(mn)) < NEGLIGIBLE_PP:
                weak = "main" if abs(mn) < abs(rv) else "reverse"
                strong = "reverse" if weak == "main" else "main"
                call = f"{strong} 有效、{weak} 無效"
            elif rv * mn < 0:
                call = "**翻轉**"
            else:
                call = "跨順序穩定"
            L.append(f"| {x} − {y} | {label} | {rv:+.2f} pp | {mn:+.2f} pp | {call} |")
    L += ["",
          "### 讀法", "",
          "- **唯一乾淨的翻轉是 A5 − A4**（task-IL 與 class-IL 皆翻轉）："
          "reverse 上 eq 有正貢獻、main 上是負的。**eq 的貢獻非跨順序穩定。**"
          "這是本檔最主要的發現。",
          "- **A5 − A3 是「reverse 有效、main 無效」，不是翻轉**：main 側的 "
          "class-IL 差值幾乎為零（量級小於 0.5 pp），把它寫成翻轉會誇大。",
          "- **A3 − A1 跨順序穩定**：replay 相對 SeqFT 在兩個 order 上都是大幅改善，"
          "方向與量級都一致。這是本專案最穩固的結果。",
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
          "## 觀察：l_eq fire rate", "", *fire_rate_lines(),
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
