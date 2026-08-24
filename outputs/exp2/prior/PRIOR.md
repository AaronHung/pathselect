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

逐 slide 預測：`outputs/exp2/prior/per_slide/*.json`

