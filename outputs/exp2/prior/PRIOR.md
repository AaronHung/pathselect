# G2 — semantic prior 三臂消融（L_sem）

arm = A5、arch = **hier**（allocation = per_budget）、reverse order、|M| = 512、seeds [0, 1, 2, 3, 4]、beta_s = 0.1、其餘設定與主表相同（λ 全 1.0，不調）。

**discriminative 是 pre-registered 主線（DR-007）。**若 max_sim 或 none 勝出，照實報 —— 那是有價值的發現（relevance 勝過 discriminability，或 semantic prior 非必要）。

## 主表

| prior | task-IL final avg | class-IL final avg | 跨任務洩漏率 | selection Jaccard |
|---|---|---|---|---|
| none | 88.92 ± 2.35 | 81.20 ± 2.38 | 10.57 ± 2.95 | 0.1387 ± 0.0140 |
| max_sim | 90.13 ± 0.52 | 82.27 ± 2.10 | 9.74 ± 1.62 | 0.1657 ± 0.0243 |
| discriminative ⭐主線 | 90.89 ± 0.77 | 81.19 ± 1.62 | 12.20 ± 2.37 | 0.1419 ± 0.0482 |

## 配對比較（同 seed 相減；三級規則見 DR-020）

### task-IL final avg

| 對照 | 逐 seed 配對差值 | 配對 mean ± std | win | 判定 |
|---|---|---|---|---|
| discriminative − none | +2.27, +2.58, -2.26, +2.32, +4.98 | +1.98 ± 2.63 | 4/5 | directional, inconclusive |
| discriminative − max_sim | +1.33, +0.81, +0.02, +0.01, +1.66 | +0.76 ± 0.75 | 5/5 | **systematic** |
| max_sim − none | +0.94, +1.77, -2.28, +2.31, +3.32 | +1.21 ± 2.14 | 4/5 | directional, inconclusive |

### class-IL final avg

| 對照 | 逐 seed 配對差值 | 配對 mean ± std | win | 判定 |
|---|---|---|---|---|
| discriminative − none | +1.34, +0.06, -0.87, +0.13, -0.75 | -0.02 ± 0.89 | 3/5 | within noise |
| discriminative − max_sim | -5.87, -3.66, +2.10, +1.22, +0.80 | -1.08 ± 3.48 | 3/5 | within noise |
| max_sim − none | +7.21, +3.72, -2.97, -1.09, -1.55 | +1.07 ± 4.26 | 2/5 | within noise |

### 跨任務洩漏率

| 對照 | 逐 seed 配對差值 | 配對 mean ± std | win | 判定 |
|---|---|---|---|---|
| discriminative − none | -1.07, +3.86, -1.66, +3.00, +4.00 | +1.62 ± 2.76 | 2/5 | within noise |
| discriminative − max_sim | +6.67, +6.14, -1.82, +0.19, +1.13 | +2.46 ± 3.76 | 1/5 | within noise |
| max_sim − none | -7.74, -2.28, +0.16, +2.81, +2.87 | -0.84 ± 4.41 | 2/5 | within noise |

### selection Jaccard

| 對照 | 逐 seed 配對差值 | 配對 mean ± std | win | 判定 |
|---|---|---|---|---|
| discriminative − none | -0.04, +0.08, +0.04, -0.07, +0.01 | +0.00 ± 0.06 | 3/5 | within noise |
| discriminative − max_sim | -0.06, +0.06, +0.02, -0.12, -0.03 | -0.02 ± 0.07 | 2/5 | within noise |
| max_sim − none | +0.02, +0.01, +0.01, +0.05, +0.03 | +0.03 ± 0.01 | 5/5 | **systematic** |

## L_sem 是否有作用（class-IL）

discriminative − none = **-0.02 pp**，win **3/5**（within noise）。

判讀由 PI 進行；此處只陳述數字。

## 結論（DR-036）

> **L_sem 移除不損害準確率。** class-IL 上三臂全部 within noise；task-IL 上 discriminative 相對 max_sim 為 +0.76 pp（5/5 systematic）但量級極小。**不得宣稱 L_sem 改善準確率。**

可以宣稱的替代說法：semantic prior 作為**弱正則**，**在階層架構下**其移除不損害準確率 —— 這與 β_s 刻意設為 0.1 的設計一致。

⚠️ DR-038 已刪去 DR-036 原本的「HistoSelect 的貢獻在於分組結構而非語意先驗」一句：那是**循環論證** —— 我們正是在「分組結構壓過 patch 分數」的架構裡測 patch 層先驗。

**選 discriminative 為主線的理由不受影響**（DR-007）：max_sim 實質上就是simple similarity，正是指導教授指名批評之處。本次結果反而給了新支持 ——兩者效果相當，而我們選了不是 similarity 的那個。

⚠️ 須同時報：**max_sim 的洩漏率最低（9.74，對照 discriminative 12.20）**。

### ⚠️ 範圍限定：本結論只適用階層架構

L_sem 只錨定 **patch 分數 s**（`semantic_prior(Z, ...)`，程式中沒有 group 層的 prior 項）。因此它的槓桿依架構而異：

| 架構 | patch 分數對最終選取的作用 | L_sem 的槓桿 |
|---|---|---|
| flat | s 單獨決定選哪 B 個 patch | **完整** |
| hier | r 先決定各組名額，s 只在組內排序 | **被稀釋** |

**所以 prior 與選取架構並非正交，階層下的 null 不可外推到 flat。**

#### 更根本的一點：group 層語意先驗從未實作

L_sem 的原始規格是**兩項**：KL(B(r_j) ‖ B(p_j^sem)) + KL(B(s_i) ‖ B(p_i^sem))。實作中 `l_sem()` **只有 patch 項** —— 沒有 r 參數、沒有第二個 KL，訓練中也從未計算 group prior。

已用 mutation 實測確認：把 group prototype 擾動 5 倍，**L_sem 的數值位元不變**（0.0226687789 → 0.0226687789）；反向對照擾動 patch 特徵則會變（0.022669 → 0.020234），證明擾動本身有效。

因此本節測到的「L_sem 無效」，測的是**半邊的 L_sem**。
目前沒有 flat 的 prior 消融資料（flat 全部是 discriminative）。要外推需補跑 none / max_sim × flat × 5 seeds = 10 輪（約 4.7 h）。

逐 slide 預測：`outputs/exp2/prior/per_slide/*.json`

