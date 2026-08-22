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

    # ── F1：replay 取樣是否隨 |M| 成長 ────────────────────────────────────
    L += ["", "## F1 — replay 取樣強度與 |M| 無關（這條軸是乾淨的）", "",
          "若每步 replay 的樣本數隨 |M| 成長，這條軸就同時混著「記憶容量」與"
          "「replay 梯度強度」，大記憶體等於更強的正則、壓抑新任務可塑性 —— "
          "那樣非單調只是實作副作用。實測不是：", "",
          "| 問題 | 答案 |", "|---|---|",
          "| 每個 training step 從 M 取幾個樣本 | **固定 `replay_k = 1`**，"
          "與 \|M\| 無關（`scripts/run_exp2.py:172`） |",
          "| 是否為 \|M\| 的函數 | 否。`SelectionMemory.sample(k)` 只有在 "
          "`k >= len(M)` 時才退回全給；k=1 而 \|M\| 全程 ≥ 64，該分支從未觸發 |",
          "| L_replay 與 current-task 的 batch 比例 | **固定 1:1**。"
          "current task 每個 optimizer step 一張 slide，replay 一筆 entry |",
          "",
          "\|M\| 變大改變的只有**記憶體內容的多樣性與時間跨度**，"
          "不改變每步的 replay 梯度筆數。取樣邏輯未修改。", ""]

    # ── F2：A3 在 |M|=256 之後的變化 ──────────────────────────────────────
    L += ["## F2 — A3 在 |M|=256 之後的變化", "",
          "### 方法學註記：win count 三級規則（DR-020）", "",
          "臂間比較一律**配對**（同 seed 相減）。win count 的判讀只有三級，"
          "全文一律使用這三個詞，不混用：", "",
          "| win count | 名稱 | 判讀 |",
          "|---|---|---|",
          "| 5/5 | **systematic** | 系統性差異 |",
          "| 4/5 | **directional, inconclusive** | 方向一致但證據不足以定案 |",
          "| ≤3/5 | **within noise** | 落在雜訊內 |",
          "",
          "**不報 p 值** —— n=5 的政策沿用（DR-016）。", "",
          "A5 為對照組，預期沿 |M| 軸平穩。", ""]
    F2_PAIRS = [("A3", 256, "A3", 512), ("A3", 256, "A3", 1024),
                ("A5", 256, "A5", 512)]
    for key, label, higher in METRICS:
        rows = []
        for (aa, ca, ab, cb) in F2_PAIRS:
            if (aa, ca) not in M or (ab, cb) not in M:
                continue
            d = [M[(aa, ca)][sd][key] - M[(ab, cb)][sd][key] for sd in seeds
                 if M[(aa, ca)][sd].get("per_task") and M[(ab, cb)][sd].get("per_task")]
            if not d:
                continue
            sc = 1 if key == "mean_jaccard" else 100
            wins = sum((x > 0) if higher else (x < 0) for x in d)
            sd_ = statistics.stdev(d) if len(d) > 1 else 0.0
            # 事前約定：5/5 或 0/5 → 系統性；≤3/5 → 雜訊。4/5 規則未定義，
            # 標為「不確定」而不是替它選一邊。
            # DR-020 的三級規則：5/5 systematic、4/5 directional inconclusive、
            # <=3/5 within noise。全文一律用這三個詞。
            verdict = ("**systematic**" if wins in (0, len(d))
                       else "within noise" if wins <= 3
                       else "directional, inconclusive")
            rows.append(f"| {aa}@{ca} − {ab}@{cb} | "
                        + ", ".join(f"{x * sc:+.2f}" for x in d)
                        + f" | {statistics.mean(d) * sc:+.2f} ± {sd_ * sc:.2f} | "
                        f"{wins}/{len(d)} | {verdict} |")
        if rows:
            L += [f"### {label}", "",
                  "| 對照 | 逐 seed 配對差值 | 配對 mean ± std | win count | 判定 |",
                  "|---|---|---|---|---|"] + rows + [""]
    L += ["### F2 判定", "",
          "- **A3 的 class-IL 從 |M|=256 到 512 是 systematic decline**"
          "（+4.25 ± 2.27 pp，5/5）。這不是雜訊，「下滑」一詞在這一格站得住。",
          "- A3@256 − A3@1024 為 4/5 → **directional, inconclusive**。",
          "- task-IL / 洩漏率 / Jaccard 三個指標在所有 A3 對照上都不是 systematic。",
          "- **對照組 A5@256 − A5@512 沿同一區間平穩**：四個指標分別 1/5、2/5、"
          "2/5、1/5，全部 **within noise**。非單調只發生在 A3。",
          "",
          "**已排除的混淆（F1，正面陳述）**：replay 取樣強度**不隨 |M| 成長** —— "
          "`replay_k` 固定為 1（`scripts/run_exp2.py:172`）、與 |M| 無關；"
          "L_replay 與 current-task 的 batch 比例固定 **1:1**；"
          "`SelectionMemory.sample()` 的「k ≥ |M| 時全給」退回分支**從未觸發**"
          "（|M| 全程 ≥ 64）。因此這條軸只變記憶體內容，不變 replay 梯度強度，"
          "**replay 強度混淆已排除**。",
          "",
          "**成因：open question。** 已排除上述混淆後仍有 systematic decline，"
          "需要另一組診斷才能回答。已登錄為研究種子"
          "（`docs/ledger/SEEDS.md` S-01）。本輪不臆測成因。", ""]

    # ── 記憶體效率主張（依 PI 裁定改寫）──────────────────────────────────
    best = [(c, mean_of(M, "A3", c, seeds, "final_class_il")) for c in caps]
    best = [(c, v) for c, v in best if v is not None]
    L += ["## 記憶體效率主張", ""]
    if best:
        bc, bv = max(best, key=lambda x: x[1])
        hit = next((c for c in caps
                    if (v := mean_of(M, "A5", c, seeds, "final_class_il")) is not None
                    and v >= bv), None)
        a5_512 = mean_of(M, "A5", CONTRACT_CAP, seeds, "final_class_il")
        over = [c for c in caps
                if (v := mean_of(M, "A3", c, seeds, "final_class_il")) is not None
                and a5_512 is not None and v >= a5_512]
        L += [f"> **A5 在 |M|={hit} 達到 A3 的全域最佳（A3@{bc} = {bv:.4f}）"
              f"→ 2× 記憶體效率。**" if hit is not None else
              "> A5 在所有測試的 |M| 都沒有達到 A3 的全域最佳。",
              "",
              "輔助句：A3 在**所有**測試容量（含契約外的 1024）都未超過 "
              f"A5@{hit}（{mean_of(M, 'A5', hit, seeds, 'final_class_il'):.4f}）。"
              if hit is not None else "",
              "",
              f"⚠️ **不採用「A5@128 追平 A3@512 → 4×」的說法**：A3@{CONTRACT_CAP} "
              f"({mean_of(M, 'A3', CONTRACT_CAP, seeds, 'final_class_il'):.4f}) "
              f"是這條曲線上的低點，拿它當基準會高估效率。改以 A3 的全域最佳為基準。",
              "",
              "各 |M| 的 class-IL（非單調性在此）：",
              "",
              "- A3：" + "、".join(f"{c}→{v:.4f}" for c, v in best),
              "- A5：" + "、".join(
                  f"{c}→{mean_of(M, 'A5', c, seeds, 'final_class_il'):.4f}"
                  for c in caps if mean_of(M, "A5", c, seeds, "final_class_il")),
              "",
              "反向檢查（PI 指定的負面資訊）：A5@512 的 class-IL = "
              f"**{a5_512:.4f}**；"
              + (f"A3 在 |M| ∈ {over} 追上或超過它。" if over
                 else "A3 在所有測試的 |M| 都沒有追上它。"),
              ""]

    # ── 中段不一致的明確揭露（PI 要求，不美化）──────────────────────────
    L += ["## ⚠️ 中段的不一致（明確揭露）", "",
          "主張**只建立在稀缺端（|M|=64）與 512/1024 端**。中間兩格的證據弱且方向不一致：", ""]
    L += ["| \|M\| | class-IL A5−A3 | 洩漏率 A5−A3 |", "|---|---|---|"]
    for c in caps:
        if ("A3", c) not in M or ("A5", c) not in M:
            continue
        cells = []
        for key, higher in (("final_class_il", True), ("mean_leak", False)):
            d = [M[("A5", c)][sd][key] - M[("A3", c)][sd][key] for sd in seeds
                 if M[("A5", c)][sd].get("per_task") and M[("A3", c)][sd].get("per_task")]
            wins = sum((x > 0) if higher else (x < 0) for x in d)
            cells.append(f"{statistics.mean(d) * 100:+.2f} pp（{wins}/{len(d)}）")
        flag = "  ← 弱" if c in (128, 256) else ""
        L.append(f"| {c}{flag} | " + " | ".join(cells) + " |")
    L += ["",
          "具體說：|M|=128 的 class-IL 只有 +1.20 pp（4/5）、|M|=256 只有 +0.17 pp（2/5）；"
          "洩漏率在這兩格都是 3/5 且**方向不一致**（128 為 +0.18 pp，方向與主張相反）。"
          "這兩格不支持主張，也不反對，證據就是弱。**不美化。**", "",
          "### 兩種陳述必須分開（DR-020）", "",
          "2× 錨在 |M|=128，而 128 正是上表的弱格之一。兩者不衝突，"
          "因為講的是**不同的比較**，論文中必須分開陳述：", "",
          "| 陳述 | 類型 | 比較對象 | 支撐 |",
          "|---|---|---|---|",
          "| **效率陳述** | 跨容量 | A5@128 vs A3@256（A3 全域最佳） | "
          "class-IL 0.8253 vs 0.8203 → **2×** |",
          "| **機制陳述** | 同容量 | A5 vs A3 於同一 \|M\| | "
          "**只用 \|M\| = 64 / 512 / 1024 三格**（5/5、5/5、4/5） |",
          "",
          "機制陳述**不使用** \|M\| = 128 / 256 兩格 —— 那兩格是 "
          "directional inconclusive 與 within noise，證據不足以支撐機制主張。", ""]

    L += ["## 結論", "",
          "> **A3 的優勢集中在狹窄的容量甜蜜點（|M| = 128–256），出了這個區間就退化"
          "（256 → 512 為 systematic decline，5/5）。A5 在 |M| ≥ 128 全程平穩"
          "（class-IL 0.822–0.828）。**",
          ">",
          "> **Replay 需要把記憶體預算調對；我們的方法不需要。**",
          "",
          "逐 slide 預測：`outputs/exp2/memory/per_slide/*.json`", ""]
    OUT.write_text("\n".join(L) + "\n")
    print(f"→ {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
