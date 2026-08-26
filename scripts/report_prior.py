#!/usr/bin/env python3
"""G2 — L_sem 的 semantic prior 三臂消融（none / max_sim / discriminative）。

beta_s = 0.1 在每一次跑裡都開著，從未被消融過。若 L_sem 無作用，
HistoSelect 這條連結就變薄。discriminative 是 pre-registered 主線（DR-007）；
若 max_sim 或 none 勝出，照實報 —— 那是有價值的發現。
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

OUT_DIR = REPO_ROOT / "outputs" / "exp2" / "prior"
OUT = OUT_DIR / "PRIOR.md"
PRIORS = ["none", "max_sim", "discriminative"]
MAINLINE = "discriminative"
METRICS = [("final_task_il", "task-IL final avg", True),
           ("final_class_il", "class-IL final avg", True),
           ("mean_leak", "跨任務洩漏率", False),
           ("mean_jaccard", "selection Jaccard", True)]


def ms(vals, scale=1.0, fmt="{:.4f}"):
    vals = [v * scale for v in vals if v is not None and v == v]
    if not vals:
        return "—"
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f"{fmt.format(statistics.mean(vals))} ± {fmt.format(sd)}"


def verdict(wins, n):
    return ("**systematic**" if wins in (0, n) else "within noise" if wins <= 3
            else "directional, inconclusive")


def collect(arch: str, allocation: str = "per_budget") -> dict:
    """從所有相關 tag 蒐集 A5 的記錄，依 prior 分組。

    ⚠️ **必須同時過濾 allocation。** `hier` tag 裡是 G1 的 per_chunk 紀錄
    （階層退化成單組選取，88.6%；全部 arm 口徑，只算 A5 為 84.5%，DR-045），`hier2` 才是 per_budget 的主線。
    早期紀錄沒有 allocation 欄位，缺欄位一律視為 per_chunk。
    2026-08-24 曾因缺這道過濾，把 G1 的退化紀錄當成 discriminative 主線臂，
    產出「主線遠差於 none/max_sim」的假結論。
    """
    out = {p: [] for p in PRIORS}
    roots = [REPO_ROOT / "outputs" / "exp2" / t / "per_slide"
             for t in ("prior", "hier2", "main")]
    seen = set()
    for d in roots:
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.json")):
            for r in json.loads(f.read_text()):
                if (r["arm"] != "A5" or r["order"] != "reverse"
                        or r.get("mem_capacity", 512) != 512
                        or r.get("arch", "flat") != arch):
                    continue
                if arch == "hier" and r.get("allocation", "per_chunk") != allocation:
                    continue
                pr = r.get("prior", MAINLINE)
                key = (pr, r["seed"], r["stage"], r["task"], r["slide_id"])
                if key in seen:
                    continue
                seen.add(key)
                out[pr].append(r)
    return out


def main() -> int:
    arch = sys.argv[1] if len(sys.argv) > 1 else "hier"
    cfg = load_config()
    tasks = ORDERS["reverse"]
    groups = collect(arch)
    have = [p for p in PRIORS if groups[p]]
    if not have:
        print(f"尚無 arch={arch} 的資料"); return 1
    seeds = sorted(set.intersection(*[{r["seed"] for r in groups[p]} for p in have]))
    print(f"arch={arch}；有資料的 prior：{have}；共同 seeds {seeds}")

    M = {p: {s: arm_metrics(groups[p], "A5", tasks, s, cfg["tasks"]) for s in seeds}
         for p in have}

    L = ["# G2 — semantic prior 三臂消融（L_sem）", "",
         f"arm = A5、arch = **{arch}**（allocation = per_budget）、reverse order、"
         f"|M| = 512、seeds {seeds}、"
         "beta_s = 0.1、其餘設定與主表相同（λ 全 1.0，不調）。", "",
         f"**{MAINLINE} 是 pre-registered 主線（DR-007）。**"
         "若 max_sim 或 none 勝出，照實報 —— 那是有價值的發現"
         "（relevance 勝過 discriminability，或 semantic prior 非必要）。", "",
         "## 主表", "",
         "| prior | " + " | ".join(l for _k, l, _h in METRICS) + " |",
         "|---" * (len(METRICS) + 1) + "|"]
    for p in have:
        cells = []
        for key, _l, _h in METRICS:
            v = [M[p][s].get(key) for s in seeds if M[p][s].get("per_task")]
            cells.append(ms([x for x in v if x is not None],
                            1 if key == "mean_jaccard" else 100,
                            "{:.4f}" if key == "mean_jaccard" else "{:.2f}"))
        star = " ⭐主線" if p == MAINLINE else ""
        L.append(f"| {p}{star} | " + " | ".join(cells) + " |")

    L += ["", "## 配對比較（同 seed 相減；三級規則見 DR-020）", ""]
    pairs = [(MAINLINE, p) for p in have if p != MAINLINE]
    if "max_sim" in have and "none" in have:
        pairs.append(("max_sim", "none"))
    for key, label, higher in METRICS:
        rows = []
        for a, b in pairs:
            if a not in M or b not in M:
                continue
            d = [M[a][s][key] - M[b][s][key] for s in seeds
                 if M[a][s].get("per_task") and M[b][s].get("per_task")]
            if not d:
                continue
            sc = 1 if key == "mean_jaccard" else 100
            wins = sum((x > 0) if higher else (x < 0) for x in d)
            sd = statistics.stdev(d) if len(d) > 1 else 0.0
            rows.append(f"| {a} − {b} | " + ", ".join(f"{x * sc:+.2f}" for x in d)
                        + f" | {statistics.mean(d) * sc:+.2f} ± {sd * sc:.2f} | "
                        f"{wins}/{len(d)} | {verdict(wins, len(d))} |")
        if rows:
            L += [f"### {label}", "",
                  "| 對照 | 逐 seed 配對差值 | 配對 mean ± std | win | 判定 |",
                  "|---|---|---|---|---|"] + rows + [""]

    # 判讀骨架（只陳述數字，不下結論）
    if MAINLINE in M and "none" in M:
        d = [M[MAINLINE][s]["final_class_il"] - M["none"][s]["final_class_il"]
             for s in seeds if M[MAINLINE][s].get("per_task")
             and M["none"][s].get("per_task")]
        wins = sum(x > 0 for x in d)
        L += ["## L_sem 是否有作用（class-IL）", "",
              f"{MAINLINE} − none = **{statistics.mean(d) * 100:+.2f} pp**，"
              f"win **{wins}/{len(d)}**（{verdict(wins, len(d))}）。",
              "", "判讀由 PI 進行；此處只陳述數字。", ""]
    L += ["## 結論（DR-036）", "",
          "> **L_sem 移除不損害準確率。** class-IL 上三臂全部 within noise；"
          "task-IL 上 discriminative 相對 max_sim 為 +0.76 pp（5/5 systematic）"
          "但量級極小。**不得宣稱 L_sem 改善準確率。**",
          "",
          "可以宣稱的替代說法：semantic prior 作為**弱正則**，"
          "**在階層架構下**其移除不損害準確率 —— 這與 β_s 刻意設為 0.1 的設計一致。",
          "",
          "⚠️ DR-038 已刪去 DR-036 原本的「HistoSelect 的貢獻在於分組結構而非語意先驗」"
          "一句：那是**循環論證** —— 我們正是在「分組結構壓過 patch 分數」的架構裡"
          "測 patch 層先驗。",
          "",
          "**選 discriminative 為主線的理由不受影響**（DR-007）：max_sim 實質上就是"
          "simple similarity，正是指導教授指名批評之處。本次結果反而給了新支持 ——"
          "兩者效果相當，而我們選了不是 similarity 的那個。",
          "",
          "⚠️ 須同時報：**max_sim 的洩漏率最低（9.74，對照 discriminative 12.20）**。",
          "",
          "### ⚠️ 範圍限定：本結論只適用階層架構", "",
          "L_sem 只錨定 **patch 分數 s**（`semantic_prior(Z, ...)`，程式中沒有 group 層"
          "的 prior 項）。因此它的槓桿依架構而異：", "",
          "| 架構 | patch 分數對最終選取的作用 | L_sem 的槓桿 |",
          "|---|---|---|",
          "| flat | s 單獨決定選哪 B 個 patch | **完整** |",
          "| hier | r 先決定各組名額，s 只在組內排序 | **被稀釋** |",
          "",
          "**所以 prior 與選取架構並非正交，階層下的 null 不可外推到 flat。**",
          "",
          "#### 更根本的一點：group 層語意先驗從未實作", "",
          "L_sem 的原始規格是**兩項**："
          "KL(B(r_j) ‖ B(p_j^sem)) + KL(B(s_i) ‖ B(p_i^sem))。"
          "實作中 `l_sem()` **只有 patch 項** —— 沒有 r 參數、沒有第二個 KL，"
          "訓練中也從未計算 group prior。",
          "",
          "已用 mutation 實測確認：把 group prototype 擾動 5 倍，"
          "**L_sem 的數值位元不變**（0.0226687789 → 0.0226687789）；"
          "反向對照擾動 patch 特徵則會變（0.022669 → 0.020234），證明擾動本身有效。",
          "",
          "因此本節測到的「L_sem 無效」，測的是**半邊的 L_sem**。",
          "目前沒有 flat 的 prior 消融資料（flat 全部是 discriminative）。"
          "要外推需補跑 none / max_sim × flat × 5 seeds = 10 輪（約 4.7 h）。", "",
          "逐 slide 預測：`outputs/exp2/prior/per_slide/*.json`", ""]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n")
    print(f"→ {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
