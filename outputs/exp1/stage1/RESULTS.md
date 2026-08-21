# Exp 1 Stage 1 — 消融階梯結果

fold 1、reverse order、8-way label space、B=8、c=1、seeds [0, 1, 2]、epochs 5、lr 0.001、beta_s 0.1、prior discriminative。
訓練用 train split，評估用 test split。frozen head = score-weighted pooling → L2 normalize → CONCH class-text logits，無 trained diagnosis head。

**權重（PI 裁定 B）**：同一個訓練好的 selector，評估時分別用兩種聚合權重。
拆解方式：

```
learned(uniform) vs random(uniform)   = 純選取能力
learned(softmax) vs learned(uniform)  = 加權貢獻
```

## 主要指標 —— B=8 accuracy（3 seeds mean ± std）

| task | n | 訓練模式 | level | softmax | uniform |
|---|---|---|---|---|---|
| tcga_esca | 15 | per_task | L3 Flat learned selector | 0.8222 ± 0.0385 | 0.8444 ± 0.0385 |
| tcga_esca | 15 | joint | L4 + task conditioning q_tau | 0.7333 ± 0.0667 | 0.7111 ± 0.0770 |
| tcga_rcc | 76 | per_task | L3 Flat learned selector | 0.9342 ± 0.0348 | 0.9430 ± 0.0201 |
| tcga_rcc | 76 | joint | L4 + task conditioning q_tau | 0.9035 ± 0.0402 | 0.9035 ± 0.0402 |
| tcga_brca | 93 | per_task | L3 Flat learned selector | 0.8853 ± 0.0062 | 0.8853 ± 0.0062 |
| tcga_brca | 93 | joint | L4 + task conditioning q_tau | 0.8029 ± 0.1022 | 0.7957 ± 0.0919 |
| tcga_lung | 95 | per_task | L3 Flat learned selector | 0.8351 ± 0.0219 | 0.8386 ± 0.0122 |
| tcga_lung | 95 | joint | L4 + task conditioning q_tau | 0.7965 ± 0.0717 | 0.8246 ± 0.0399 |

### 四 task 平均（B=8）

| level | 訓練模式 | softmax | uniform |
|---|---|---|---|
| L3 Flat learned selector | per_task | 0.8692 ± 0.0219 | 0.8778 ± 0.0090 |
| L4 + task conditioning q_tau | joint | 0.8091 ± 0.0499 | 0.8087 ± 0.0504 |

⚠️ **per-task 與 joint 不在同一欄比較**：L3 是 per-task 訓練（每個 task 一個模型），L4+ 是 joint 訓練（一個模型跑全部 task）。

## 次要指標 —— budget 曲線（四 task 平均，3 seeds mean ± std）

| level | weighting | B=1 | B=2 | B=4 | B=8 | B=16 |
|---|---|---|---|---|---|---|
| L3 | softmax | 0.8137 ± 0.0131 | 0.8518 ± 0.0298 | 0.8595 ± 0.0256 | 0.8692 ± 0.0219 | 0.8765 ± 0.0024 |
| L3 | uniform | 0.8137 ± 0.0131 | 0.8498 ± 0.0311 | 0.8612 ± 0.0206 | 0.8778 ± 0.0090 | 0.8776 ± 0.0017 |
| L4 | softmax | 0.6958 ± 0.0384 | 0.7518 ± 0.0584 | 0.8024 ± 0.0321 | 0.8091 ± 0.0499 | 0.8194 ± 0.0423 |
| L4 | uniform | 0.6958 ± 0.0384 | 0.7429 ± 0.0719 | 0.8026 ± 0.0294 | 0.8087 ± 0.0504 | 0.8127 ± 0.0326 |

## 誠實性註記

- esca 只有 n=15 張 test slide，一張 = 6.67 pp。**esca 上任何小於 6.67 pp 的差異一律視為不可區分。**
- tcga_esca：L4 − L3 = -8.89 pp
- tcga_rcc：L4 − L3 = -3.07 pp
- tcga_brca：L4 − L3 = -8.24 pp
- tcga_lung：L4 − L3 = -3.86 pp
