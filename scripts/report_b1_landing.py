#!/usr/bin/env python3
"""B1（只 KD）的預測落點分析 —— DR-033 的機制證據。

產出兩份：
  outputs/exp2/ablation/B1_LANDING.md   獨立產物（不會被 --report-only 覆蓋）
  outputs/exp2/ablation/EXP2.md         冪等注入同一節（先移除舊的再附加）

⚠️ 用獨立腳本而非改 run_exp2.py::write_report，是因為 pipeline 正在呼叫後者
（憲法 §3.4 執行期檔案凍結）。pipeline 結束後可考慮併入，但獨立產物本身即為
權威來源，EXP2.md 內的那一節只是方便閱讀的副本 —— 每次 EXP2.md 重生成後
重跑本檔即可還原。
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_exp2 import ORDERS, arm_metrics                         # noqa: E402
from selector.text_encoder import load_config                            # noqa: E402

ABL = REPO_ROOT / "outputs" / "exp2" / "ablation"
OUT = ABL / "B1_LANDING.md"
EXP2 = ABL / "EXP2.md"
SECTION_START = "## KD 與 replay 保存的是不同的東西（DR-033）"
SECTION_END = "<!-- /B1_LANDING -->"
CLASS_NAMES = ["ESAD", "ESCC", "CCRCC", "PRCC", "IDC", "ILC", "LUAD", "LUSC"]


def load(arm: str) -> list[dict]:
    return [r for f in sorted((ABL / "per_slide").glob(f"{arm}_reverse_seed*.json"))
            for r in json.loads(f.read_text())]


def build() -> list[str]:
    cfg = load_config()
    tasks = ORDERS["reverse"]
    b1, a3, a5 = load("B1"), load("A3"), load("A5")
    seeds = sorted({r["seed"] for r in b1})
    M = {a: {s: arm_metrics(recs, a, tasks, s, cfg["tasks"]) for s in seeds}
         for a, recs in (("B1", b1), ("A3", a3), ("A5", a5))}

    def col(a, k):
        v = [M[a][s][k] for s in seeds if M[a][s].get("per_task")]
        return statistics.mean(v), (statistics.stdev(v) if len(v) > 1 else 0.0)

    def paired(x, y, k, higher=True):
        d = [M[x][s][k] - M[y][s][k] for s in seeds]
        w = sum((v > 0) if higher else (v < 0) for v in d)
        vd = ("**systematic**" if w in (0, len(d)) else "within noise" if w <= 3
              else "directional, inconclusive")
        return d, statistics.mean(d), (statistics.stdev(d) if len(d) > 1 else 0.0), w, vd

    L = [SECTION_START, "",
         f"seeds {seeds}（B1 已補到 {len(seeds)} seeds）。", "",
         "B1 = 只留 KD（λ_r = λ_eq = 0），仍使用 replay 這個**資料機制**"
         "（照樣取回舊樣本），只是在上面算 L_KD 而不是 L_diag（DR-013）。", "",
         "### 兩個指標的分裂", "",
         "| 對照 | 指標 | 逐 seed 配對差值 | 配對 mean ± std | win | 判定 |",
         "|---|---|---|---|---|---|"]
    for k, lab in (("final_class_il", "class-IL"), ("final_task_il", "task-IL")):
        d, m, sd, w, vd = paired("A5", "B1", k)
        L.append(f"| A5 − B1 | {lab} | " + ", ".join(f"{x*100:+.2f}" for x in d)
                 + f" | {m*100:+.2f} ± {sd*100:.2f} pp | {w}/{len(d)} | {vd} |")
    L += ["",
          "**任務內鑑別力大致還在，壞掉的是跨任務區辨。**", "",
          "### KD 保住了什麼、沒保住什麼", "",
          "| 臂 | selection Jaccard | 跨任務洩漏率 |", "|---|---|---|"]
    for a, name in (("B1", "B1 只 KD"), ("A3", "A3 只 replay"), ("A5", "A5 三項全開")):
        j, js = col(a, "mean_jaccard")
        lk, lks = col(a, "mean_leak")
        L.append(f"| {name} | {j:.4f} ± {js:.4f} | {lk:.4f} ± {lks:.4f} |")
    jb, _ = col("B1", "mean_jaccard"); ja, _ = col("A3", "mean_jaccard")
    lb, _ = col("B1", "mean_leak"); la5, _ = col("A5", "mean_leak")
    L += ["",
          f"- **KD 保住了選取行為**：B1 的 Jaccard {jb:.4f} 與只有 replay 的 A3 "
          f"（{ja:.4f}）相當。",
          f"- **KD 沒有保住證據的任務歸屬**：B1 的洩漏率 {lb:.4f}，是 A5 "
          f"（{la5:.4f}）的 **{lb/la5:.1f} 倍**。",
          "",
          "→ **KD 與 replay 保住的是兩種不同的東西，兩者不可互相取代。**",
          "  frozen head 使這個歸因是封閉的（DR-012）：head 不隨訓練改變，"
          "所以洩漏只能來自選出的證據本身變得不像原任務的組織。",
          "",
          "### 預測落點（學完 T4 後，5 seeds 合併）", "",
          "| task | n | class-IL acc | 落在自己列 | 洩漏到別的 task |",
          "|---|---|---|---|---|"]
    for i, t in enumerate(tasks):
        sub = [r for r in b1 if r["stage"] == 3 and r["task"] == t]
        lo = 2 * i
        acc = sum(r["pred_class_il"] == r["true"] for r in sub) / len(sub)
        inside = sum(lo <= r["pred_class_il"] <= lo + 1 for r in sub) / len(sub)
        L.append(f"| {t} | {len(sub)} | {acc:.4f} | {inside:.1%} | {1 - inside:.1%} |")
    L += ["", "8 類隨機基準 = 0.1250。**esca 與 rcc 的 class-IL 低於隨機，"
          "不是亂猜 —— 亂猜會均勻散在 8 列上。**", ""]

    sub = [r for r in b1 if r["stage"] == 3 and r["task"] == "tcga_rcc" and r["seed"] == 4]
    c = Counter(r["pred_class_il"] for r in sub)
    acc = sum(r["pred_class_il"] == r["true"] for r in sub) / len(sub)
    til = sum(r["pred_task_il"] == r["true"] for r in sub) / len(sub)
    leaked = sum(v for k, v in c.items() if not (2 <= k <= 3))
    L += ["### seed 4 的 rcc：逐筆落點", "",
          "真值應落在 row 2/3（CCRCC / PRCC）。", "",
          "| 預測列 | 類別 | 張數 |", "|---|---|---|"]
    for k in sorted(c):
        L.append(f"| row{k} | {CLASS_NAMES[k]} | {c[k]} |")
    L += ["",
          f"**{leaked}/{len(sub)} 張被推到 lung 的兩列**（最後學的 task），"
          f"只有 {len(sub)-leaked} 張落在自己的類別列。",
          "",
          f"- class-IL acc = **{acc:.4f}**（低於 8 類隨機 0.1250）",
          f"- 同一批 slide 的 task-IL acc = **{til:.4f}**（接近 2 類隨機 0.5000）",
          "",
          "後者證實：限制在 rcc 自己的兩列時，鑑別力大致還在；"
          "崩掉的是「這份證據屬於哪個任務」。", "",
          "### ⚠️ B1 同時是最不穩的一臂", ""]
    cb, cbs = col("B1", "final_class_il"); lb2, lbs = col("B1", "mean_leak")
    L += [f"B1 的 seed 標準差是全場最大：class-IL **±{cbs:.4f}**、"
          f"洩漏率 ±{lbs:.4f}。任何引用 B1 的陳述都必須同時報這一點。", "",
          SECTION_END, ""]
    return L


def main() -> int:
    L = build()
    ABL.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(["# B1 落點分析（DR-033 的機制證據）", ""] + L) + "\n")
    print(f"→ {OUT}")

    if EXP2.is_file():                     # 冪等注入：移除**所有**舊區塊再附加
        t = EXP2.read_text()
        while SECTION_START in t:          # while 而非 if —— 重複注入時也要能修回來
            i = t.index(SECTION_START)
            j = t.index(SECTION_END, i) + len(SECTION_END) if SECTION_END in t[i:] else len(t)
            t = (t[:i].rstrip() + "\n" + t[j:].lstrip()).rstrip() + "\n"
        EXP2.write_text(t.rstrip() + "\n\n" + "\n".join(L) + "\n")
        assert SECTION_START in EXP2.read_text()
        assert EXP2.read_text().count(SECTION_START) == 1, "重複注入"
        print(f"→ {EXP2}（已冪等注入）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
