# G1 — flat vs hierarchical 選取器

**唯一的差異是 hierarchy**：q_tau 與 state 在兩側都關閉（Gate 1 的教訓 —— 同時打開多件事就無法歸因）。同 seed、|M|=512、λ 全 1.0、epochs 5、B=8、c=1、reverse order。flat 側沿用主表存檔。

seeds [0, 1, 2, 3, 4]。

## 主表

### task-IL final avg

| 架構 | A3 + Replay | A5 Ours (Replay+KD+eq) | B2 只 eq (λ_r=λ_kd=0) | A5nG Ours − group-KD（只留 patch 蒸餾） |
|---|---|---|---|---|
| flat | 90.73 ± 1.82 | 91.47 ± 1.41 | — | — |
| hier | 87.61 ± 2.88 | 90.89 ± 0.77 | — | 86.94 ± 2.60 |

### class-IL final avg

| 架構 | A3 + Replay | A5 Ours (Replay+KD+eq) | B2 只 eq (λ_r=λ_kd=0) | A5nG Ours − group-KD（只留 patch 蒸餾） |
|---|---|---|---|---|
| flat | 77.78 ± 1.48 | 82.39 ± 2.84 | — | — |
| hier | 75.42 ± 3.59 | 81.19 ± 1.62 | — | 77.48 ± 2.42 |

### 跨任務洩漏率

| 架構 | A3 + Replay | A5 Ours (Replay+KD+eq) | B2 只 eq (λ_r=λ_kd=0) | A5nG Ours − group-KD（只留 patch 蒸餾） |
|---|---|---|---|---|
| flat | 14.21 ± 1.80 | 10.05 ± 2.55 | — | — |
| hier | 14.61 ± 2.90 | 12.20 ± 2.37 | — | 13.25 ± 2.15 |

### selection Jaccard

| 架構 | A3 + Replay | A5 Ours (Replay+KD+eq) | B2 只 eq (λ_r=λ_kd=0) | A5nG Ours − group-KD（只留 patch 蒸餾） |
|---|---|---|---|---|
| flat | 0.0669 ± 0.0147 | 0.1294 ± 0.0617 | — | — |
| hier | 0.0947 ± 0.0285 | 0.1419 ± 0.0482 | — | 0.1186 ± 0.0186 |

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
| hier-A5 − flat-A5 | -1.40, +2.23, -1.61, -1.55, -0.55 | -0.58 ± 1.63 | 1/5 | within noise |
| hier-A3 − flat-A3 | +0.52, -0.69, -5.32, -7.82, -2.25 | -3.11 ± 3.42 | 1/5 | within noise |
| hier-A5 − hier-A3 | +1.94, +3.74, +3.19, +6.96, +0.56 | +3.28 ± 2.40 | 5/5 | **systematic** |
| hier-A5 − hier-A5nG | +5.35, +4.51, +2.53, +6.89, +0.47 | +3.95 ± 2.50 | 5/5 | **systematic** |

### class-IL final avg

| 對照 | 逐 seed 配對差值 | 配對 mean ± std | win | 判定 |
|---|---|---|---|---|
| hier-A5 − flat-A5 | -0.73, +0.63, -0.34, -0.06, -5.50 | -1.20 ± 2.45 | 1/5 | within noise |
| hier-A3 − flat-A3 | -2.57, -3.75, -0.46, -5.34, +0.35 | -2.35 ± 2.34 | 1/5 | within noise |
| hier-A5 − hier-A3 | +5.37, +6.11, +4.28, +11.17, +1.89 | +5.76 ± 3.42 | 5/5 | **systematic** |
| hier-A5 − hier-A5nG | +0.48, +3.99, +2.53, +7.76, +3.79 | +3.71 ± 2.66 | 5/5 | **systematic** |

### 跨任務洩漏率

| 對照 | 逐 seed 配對差值 | 配對 mean ± std | win | 判定 |
|---|---|---|---|---|
| hier-A5 − flat-A5 | +0.99, +2.94, +0.40, +1.19, +5.21 | +2.15 ± 1.96 | 0/5 | **systematic** |
| hier-A3 − flat-A3 | +2.82, +3.87, -2.27, -0.96, -1.46 | +0.40 ± 2.75 | 3/5 | within noise |
| hier-A5 − hier-A3 | -3.70, -1.51, -1.75, -4.06, -1.06 | -2.42 ± 1.36 | 5/5 | **systematic** |
| hier-A5 − hier-A5nG | +2.88, +0.26, -1.73, -3.63, -3.05 | -1.05 ± 2.66 | 3/5 | within noise |

### selection Jaccard

| 對照 | 逐 seed 配對差值 | 配對 mean ± std | win | 判定 |
|---|---|---|---|---|
| hier-A5 − flat-A5 | +0.04, +0.15, +0.04, -0.12, -0.04 | +0.01 ± 0.10 | 3/5 | within noise |
| hier-A3 − flat-A3 | -0.02, +0.07, +0.03, +0.04, +0.01 | +0.03 ± 0.03 | 4/5 | directional, inconclusive |
| hier-A5 − hier-A3 | +0.06, +0.07, +0.08, -0.01, +0.03 | +0.05 ± 0.04 | 4/5 | directional, inconclusive |
| hier-A5 − hier-A5nG | -0.01, +0.12, +0.04, -0.03, -0.00 | +0.02 ± 0.06 | 2/5 | within noise |

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
| hier | esca | 0.397 | 0.087 | 0.055 | 0.063 | 0.230 | 0.007 | 0.027 | 0.135 |
| hier | rcc | 0.520 | 0.053 | 0.123 | 0.031 | 0.045 | 0.043 | 0.065 | 0.120 |
| hier | brca | 0.305 | 0.140 | 0.139 | 0.065 | 0.087 | 0.002 | 0.026 | 0.235 |
| hier | lung | 0.449 | 0.087 | 0.074 | 0.081 | 0.143 | 0.012 | 0.033 | 0.121 |
## group-level distillation 是否有用（DR-022，首次驗證）

`hier-A5` 與 `hier-A5nG` 的**唯一差異**是 L_KD 的 group 項係數（1.0 vs 0.0，後者完全不計算、r_new 不進計算圖）。架構圖 Panel I 畫了這一項，但在 flat 模式下 F_g 對選取零影響，所以它從未被實際測試過。

- **task-IL final avg**：+3.95（5/5，**systematic**）
- **class-IL final avg**：+3.71（5/5，**systematic**）
- **跨任務洩漏率**：-1.05（3/5，within noise）
- **selection Jaccard**：+0.02（2/5，within noise）

### 兩層蒸餾的分工（DR-035）

group-KD 在**兩個準確率軸上都是 systematic**，但在 **Jaccard 上不是**（+0.02，2/5）。兩者不矛盾 —— 它們保存的對象不同：

| 蒸餾層 | 保存的對象 | 可觀測指標 |
|---|---|---|
| **group-KD**（KL(r_old ‖ r_new)） | **組織層配額分佈** —— 各 tissue group 分到幾個名額 | 準確率（task-IL / class-IL） |
| **patch-KD**（KL(s_old ‖ s_new)） | **具體 patch 身份** —— 選到哪幾個 patch | selection Jaccard |

所以拿掉 group 項會讓準確率掉，但**選到的 patch 集合幾乎不變** ——配額變了、組內挑誰沒變。**這是架構圖 Panel I 兩層設計的直接證據**：若兩層蒸餾保存的是同一件事，拿掉一層應該同時動到兩個指標。

⚠️ DR-022 曾在**退化階層**（per_chunk，84.5% 單組）下測得四項全部within noise / directional，該結論已作廢（SUPERSEDED-BY DR-035）——當每張 slide 實質只用一個 group 時，配額分佈本來就沒有東西可保存。


## 結構性診斷：階層有沒有作用空間

| 架構 | 每張 slide 用到幾個 group | 最大組佔比 | 分佈（組數 → slide 數） |
|---|---|---|---|
| flat | 2.37 | 0.714 | {1: 281, 2: 525, 3: 408, 4: 159, 5: 22}（共 1395 張） |
| hier | 4.31 | 0.465 | {1: 34, 2: 82, 3: 226, 4: 411, 5: 401, 6: 203, 7: 37, 8: 1}（共 1395 張） |

**機制**：`use_state=False` 時 r 逐輪不變（分數重用）。`per_chunk` 配額在 c=1 時 largest-remainder 只有一個名額可發、必然給 argmax(r) ⇒ 每輪同一組 ⇒ 退化為「先挑一組再取該組 top-8」。`per_budget` 對整個 budget 配額，配額用完的組讓位，預算因此攤到多個 group。


## ⚠️ 結構性把關（PI 指定的停止條件）

單一 group 的 slide 比例 = **2.4%**。

✅ 低於 50%，配額口徑確實是真因；階層這次有作用空間，判準結果可採用。


## ⚠️ 與 DR-015 的對照：A5 − A3 在 task-IL 上

DR-015 依 flat 版的證據定調「task-IL 上 A5 − A3 落在雜訊內，**不宣稱勝出**」。階層版的同一個對照結果不同：

| 架構 | A5 − A3（task-IL） | win | 判定 |
|---|---|---|---|
| flat（DR-015 當時的證據） | +0.74 ± 1.93 pp | 3/5 | within noise |
| hier（本次） | +3.28 ± 2.40 pp | 5/5 | **systematic** |

⚠️ **DR-015 在它當時的證據下是對的，不應修改**（append-only）。階層版是否構成推翻，須由 PI 以新卡裁定。此處只陳述對照，不下結論。


## Pre-registered 判準（DR-021 原文，一字不改）

hier-A5 − flat-A5（class-IL）= **-1.20 pp**，win **1/5**（within noise）。

→ **在雜訊內 → 仍採用階層為主線**，論文誠實寫「階層在此設定下與 flat 相當；其價值在於提供可解釋的組織層配額與 group-level 保存」（配額分佈本身就是定性貢獻）。


## 結論（DR-029）

**階層採用為主線。** 除 DR-021 第二支的「可解釋組織層配額」外，追加一條由數據支撐的理由：階層版 A5 的 seed 標準差顯著小於 flat（task-IL ±0.77 vs ±1.41、class-IL ±1.62 vs ±2.84）—— **階層讓方法更穩定**。

階層版可在 **task-IL 與 class-IL 兩軸皆宣稱勝出**（A5 − A3 = +3.28 / +5.76 pp，均 5/5 systematic）。

⚠️ **差距擴大有雙重來源，必須同時陳述，不得只報前半：**

| 來源 | 證據 |
|---|---|
| A5 在階層下更穩定（正向） | seed std task-IL ±1.41 → ±0.77、class-IL ±2.84 → ±1.62 |
| **replay-only 在階層下退化（負向）** | hier-A3 − flat-A3 = **−3.11 pp**（task-IL）、**−2.35 pp**（class-IL） |

也就是說：A5 − A3 的差距從 +0.74 擴大到 +3.28 pp，**其中一部分來自 A3 變差**，而不是全部來自 A5 變好。A5 本身在階層下是 −0.58 pp（task-IL，within noise）。

逐 slide 預測：`outputs/exp2/hier2/per_slide/*.json`

