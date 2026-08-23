# G1 — flat vs hierarchical 選取器

**唯一的差異是 hierarchy**：q_tau 與 state 在兩側都關閉（Gate 1 的教訓 —— 同時打開多件事就無法歸因）。同 seed、|M|=512、λ 全 1.0、epochs 5、B=8、c=1、reverse order。flat 側沿用主表存檔。

seeds [0, 1, 2, 3, 4]。

## 主表

### task-IL final avg

| 架構 | A3 + Replay | A5 Ours (Replay+KD+eq) | A5nG Ours − group-KD（只留 patch 蒸餾） |
|---|---|---|---|
| flat | 90.73 ± 1.82 | 91.47 ± 1.41 | — |
| hier | 74.02 ± 9.64 | 73.21 ± 9.76 | 78.95 ± 6.67 |

### class-IL final avg

| 架構 | A3 + Replay | A5 Ours (Replay+KD+eq) | A5nG Ours − group-KD（只留 patch 蒸餾） |
|---|---|---|---|
| flat | 77.78 ± 1.48 | 82.39 ± 2.84 | — |
| hier | 60.29 ± 7.72 | 63.70 ± 8.44 | 67.91 ± 6.97 |

### 跨任務洩漏率

| 架構 | A3 + Replay | A5 Ours (Replay+KD+eq) | A5nG Ours − group-KD（只留 patch 蒸餾） |
|---|---|---|---|
| flat | 14.21 ± 1.80 | 10.05 ± 2.55 | — |
| hier | 22.45 ± 5.36 | 15.22 ± 3.72 | 16.56 ± 6.27 |

### selection Jaccard

| 架構 | A3 + Replay | A5 Ours (Replay+KD+eq) | A5nG Ours − group-KD（只留 patch 蒸餾） |
|---|---|---|---|
| flat | 0.0669 ± 0.0147 | 0.1294 ± 0.0617 | — |
| hier | 0.1224 ± 0.1257 | 0.2909 ± 0.0866 | 0.1316 ± 0.0917 |

## 配對比較（同 seed 相減）

### 方法學註記：win count 三級規則（DR-020）

| win count | 名稱 |
|---|---|
| 5/5 | **systematic** |
| 4/5 | **directional, inconclusive** |
| ≤3/5 | **within noise** |

**不報 p 值**（DR-016）。

### task-IL final avg

| 對照 | 逐 seed 配對差值 | 配對 mean ± std | win | 判定 |
|---|---|---|---|---|
| hier-A5 − flat-A5 | -9.69, -21.74, -7.54, -25.17, -27.13 | -18.26 ± 9.04 | 0/5 | **systematic** |
| hier-A3 − flat-A3 | -5.76, -24.07, -26.85, -6.45, -20.37 | -16.70 ± 9.94 | 0/5 | **systematic** |
| hier-A5 − hier-A3 | -0.07, +3.15, +18.78, -18.03, -7.89 | -0.81 ± 13.66 | 2/5 | within noise |
| hier-A5 − hier-A5nG | +2.01, -15.07, +8.12, -2.32, -21.42 | -5.74 ± 12.21 | 2/5 | within noise |

### class-IL final avg

| 對照 | 逐 seed 配對差值 | 配對 mean ± std | win | 判定 |
|---|---|---|---|---|
| hier-A5 − flat-A5 | -4.76, -17.80, -14.59, -23.48, -32.80 | -18.69 ± 10.41 | 0/5 | **systematic** |
| hier-A3 − flat-A3 | -9.82, -22.77, -30.09, -8.83, -15.91 | -17.48 ± 8.98 | 0/5 | **systematic** |
| hier-A5 − hier-A3 | +8.59, +6.70, +19.65, -8.76, -9.15 | +3.41 ± 12.32 | 3/5 | within noise |
| hier-A5 − hier-A5nG | +4.07, -8.86, +11.91, -6.62, -21.54 | -4.21 ± 12.81 | 2/5 | within noise |

### 跨任務洩漏率

| 對照 | 逐 seed 配對差值 | 配對 mean ± std | win | 判定 |
|---|---|---|---|---|
| hier-A5 − flat-A5 | -0.42, +0.31, +12.75, +3.49, +9.69 | +5.17 ± 5.82 | 1/5 | within noise |
| hier-A3 − flat-A3 | +7.98, +7.13, +17.86, +6.51, +1.72 | +8.24 ± 5.90 | 0/5 | **systematic** |
| hier-A5 − hier-A3 | -10.26, -7.41, -9.52, -9.23, +0.24 | -7.24 ± 4.31 | 4/5 | directional, inconclusive |
| hier-A5 − hier-A5nG | -3.67, -7.02, -3.28, +6.69, +0.53 | -1.35 ± 5.23 | 3/5 | within noise |

### selection Jaccard

| 對照 | 逐 seed 配對差值 | 配對 mean ± std | win | 判定 |
|---|---|---|---|---|
| hier-A5 − flat-A5 | +0.34, +0.29, +0.06, +0.03, +0.09 | +0.16 ± 0.14 | 5/5 | **systematic** |
| hier-A3 − flat-A3 | -0.05, +0.06, +0.08, -0.04, +0.23 | +0.06 ± 0.11 | 3/5 | within noise |
| hier-A5 − hier-A3 | +0.39, +0.23, +0.07, +0.22, -0.06 | +0.17 ± 0.17 | 4/5 | directional, inconclusive |
| hier-A5 − hier-A5nG | +0.31, +0.17, +0.20, -0.02, +0.14 | +0.16 ± 0.12 | 4/5 | directional, inconclusive |

## ⚠️ 跨模式比較的限制（DR-022）

**A3 在 flat 下 F_g 完全無梯度、停在初始值**（`ste_allocation` 的注入只發生在 hierarchy 迴圈內）；在 hier 下 A3 的 F_g 會實際訓練。**因此 hier-A3 是比 flat-A3 更強的 baseline，不得直接跨模式比較 A3。**上表的 hier-A3 − flat-A3 一列只作記錄，不得單獨用來宣稱階層的效果。

## group 配額分佈（學完 T4 後，A5）

flat 的配額是**量測層**（選完之後統計落在哪一組）；hier 的配額是**決策層**（Group Selector 分配的名額）。

| 架構 | task | tumor | stroma | lymphocyte | necrosis | normal_epithelium | vessel | adipose | background |
|---|---|---|---|---|---|---|---|---|---|
| flat | esca | 0.335 | 0.198 | 0.208 | 0.012 | 0.143 | 0.040 | 0.000 | 0.063 |
| flat | rcc | 0.675 | 0.057 | 0.117 | 0.013 | 0.001 | 0.075 | 0.001 | 0.061 |
| flat | brca | 0.080 | 0.348 | 0.229 | 0.008 | 0.008 | 0.001 | 0.011 | 0.315 |
| flat | lung | 0.585 | 0.099 | 0.087 | 0.031 | 0.007 | 0.004 | 0.000 | 0.186 |
| hier | esca | 0.000 | 0.223 | 0.013 | 0.013 | 0.188 | 0.468 | 0.027 | 0.067 |
| hier | rcc | 0.095 | 0.130 | 0.121 | 0.036 | 0.031 | 0.496 | 0.055 | 0.036 |
| hier | brca | 0.120 | 0.216 | 0.024 | 0.039 | 0.075 | 0.412 | 0.011 | 0.103 |
| hier | lung | 0.067 | 0.139 | 0.075 | 0.025 | 0.121 | 0.473 | 0.055 | 0.045 |
## group-level distillation 是否有用（DR-022，首次驗證）

`hier-A5` 與 `hier-A5nG` 的**唯一差異**是 L_KD 的 group 項係數（1.0 vs 0.0，後者完全不計算、r_new 不進計算圖）。架構圖 Panel I 畫了這一項，但在 flat 模式下 F_g 對選取零影響，所以它從未被實際測試過。

- **task-IL final avg**：-5.74（2/5，within noise）
- **class-IL final avg**：-4.21（2/5，within noise）
- **跨任務洩漏率**：-1.35（3/5，within noise）
- **selection Jaccard**：+0.16（4/5，directional, inconclusive）

⚠️ 若 group 項無作用（≤3/5 且差值小），照實報 —— 論文會把 L_distill 誠實寫成 patch-level，並在 limitation 說明 group-level 在此設定下未顯示效果。**不調 λ 搶救。**


## Pre-registered 判準（DR-021，看到結果後不得修改）

hier-A5 − flat-A5（class-IL）= **-18.69 pp**，win **0/5**（**systematic**）。

→ ⚠️ **hier 顯著劣於 flat（win ≤ 1/5 且差值大）→ 停下來回報**，不自行調參數搶救。由 PI 裁定改標題或改用其他方式對齊架構圖。

逐 slide 預測：`outputs/exp2/hier/per_slide/*.json`

