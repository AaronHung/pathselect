# Exp 2 — CL 方法臂對照（order = reverse）

架構 L3b（shared selector、無 q_tau、flat）、B=8、c=1、epochs 5、lr 0.001、beta_s 0.1、beta_u 0.1、prior max_sim、λ_kd=1.0 λ_eq=1.0 λ_r=1.0（全程固定，未調）、replay_k=1、seeds [0, 1, 2, 3, 4]。
訓練用 train split、評估用 test split。

**task-IL 與 class-IL 都是主要指標**（PI 裁定 2），不是主/次關係：

- **task-IL A1 forgetting** = 忘了怎麼在任務內鑑別。
- **跨任務洩漏率** = 選出的證據整體上不再像這個任務的組織。head 是 frozen 的，**洩漏 100% 可歸因於選取漂移** —— 這是架構的直接後果，是發現而不是缺陷。

n（test）：esca 15、rcc 76、brca 93、lung 95。⚠️ esca n=15，一張 slide = 6.67 pp，**esca 上小於 6.67 pp 的差異一律視為不可區分**。

## 主表

| # | 方法臂 | final avg acc (task-IL) | final avg acc (class-IL) | A1 forgetting task-IL (pp) | A1 forgetting class-IL (pp) | Jaccard | quota KL | 洩漏率 | l_eq fire rate |
|---|---|---|---|---|---|---|---|---|---|
| A5 | Ours (Replay+KD+eq) | 0.9013 ± 0.0052 | 0.8227 ± 0.0210 | -0.57 ± +1.06 | +5.90 ± +1.82 | 0.1657 ± 0.0243 | 0.0295 ± 0.0192 | 0.0974 ± 0.0162 | 0.0842 ± 0.0079 |

**欄位口徑**：final avg acc 算全部 4 個 task；**forgetting、Jaccard、quota KL 只算前 3 個 task** —— 最後學的 lung 的「學完」與「學完 T4」是同一個時點，A1 恆為 0、Jaccard 恆為 1，算進去只會稀釋遺忘的量級（CL 慣例）。洩漏率算全部 4 個 task，因為最後一個 task 的洩漏不是由構造為 0。

⚠️ **R1 / R2 不是 CL baseline**，兩者的 A1 forgetting 由構造為 0，不能用來宣稱 forgetting。

**R1 = per-task specialist (independent training)（PI 裁定 2）**：每個 task 只用自己的訓練資料（esca 僅 120 張），而 A3 / A5 經由 replay 實質可及跨任務資料。**因此 R1 在 task-IL 上不是上界**；它的參考意義在 class-IL —— 那一欄 R1 是全場最高（0.8777）。

**R2 = offline shared-model reference**：一次看到所有資料、沒有 task 順序。

## 逐 task 明細（學完 T4 後）

### A5 Ours (Replay+KD+eq)

| task | n | task-IL @學完 | task-IL @T4 | A1 task-IL (pp) | class-IL @T4 | A1 class-IL (pp) | 洩漏率 | Jaccard | 隨機參照 | quota KL | ΣU @學完 | ΣU @T4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tcga_esca（不可區分） | 15 | 0.8800 ± 0.0298 | 0.8667 ± 0.0471 | +1.33 ± +2.98 | 0.6933 ± 0.1011 | +17.33 ± +5.96 | 0.2400 ± 0.0760 | 0.0674 ± 0.0071 | 0.00106 ± 0.00000 | 0.0601 ± 0.0413 | 22.0 ± 2.3 | 1.6 ± 13.0 |
| tcga_rcc | 76 | 0.9526 ± 0.0118 | 0.9658 ± 0.0118 | -1.32 ± +1.32 | 0.9421 ± 0.0256 | +0.79 ± +3.17 | 0.0237 ± 0.0195 | 0.2027 ± 0.0255 | 0.00123 ± 0.00000 | 0.0125 ± 0.0109 | 143.4 ± 2.0 | 135.6 ± 12.9 |
| tcga_brca | 93 | 0.8839 ± 0.0352 | 0.9011 ± 0.0192 | -1.72 ± +2.59 | 0.8280 ± 0.0240 | -0.43 ± +2.10 | 0.0731 ± 0.0140 | 0.2270 ± 0.0584 | 0.00129 ± 0.00000 | 0.0160 ± 0.0090 | 128.0 ± 11.3 | 116.7 ± 11.3 |
| tcga_lung | 95 | 0.8716 ± 0.0360 | 0.8716 ± 0.0360 | +0.00 ± +0.00 | 0.8274 ± 0.0330 | +0.00 ± +0.00 | 0.0526 ± 0.0074 | 1.0000 ± 0.0000 | 0.00133 ± 0.00000 | 0.0000 ± 0.0000 | 139.4 ± 10.3 | 139.4 ± 10.3 |

## 對照差值（task-IL final avg，5 seeds 平均）

| 對照 | 差值 (pp) |
|---|---|

## Paired comparisons（E2）

臂間比較一律**配對**：同一個 seed 相減，再對差值取 mean ± std。

### 方法學註記：win count 三級規則（DR-020）

win count = 幾個 seed 往「較好」的方向。判讀只有三級，全文一律使用這三個詞，不混用：

| win count | 名稱 | 判讀 |
|---|---|---|
| 5/5 | **systematic** | 系統性差異 |
| 4/5 | **directional, inconclusive** | 方向一致但證據不足以定案 |
| ≤3/5 | **within noise** | 落在雜訊內 |

**不報 p 值** —— n=5 的政策沿用（DR-016），顯著性檢定在這個樣本數下會誤導。

### task-IL final avg（越大越好）

| 對照 | 逐 seed 配對差值 | 配對 mean ± std | win count |
|---|---|---|---|

### class-IL final avg（越大越好）

| 對照 | 逐 seed 配對差值 | 配對 mean ± std | win count |
|---|---|---|---|

### 跨任務洩漏率（越小越好）

| 對照 | 逐 seed 配對差值 | 配對 mean ± std | win count |
|---|---|---|---|

### selection Jaccard（越大越好）

| 對照 | 逐 seed 配對差值 | 配對 mean ± std | win count |
|---|---|---|---|


逐 slide 預測：`outputs/exp2/prior/per_slide/*.json`

