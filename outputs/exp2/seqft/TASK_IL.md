# S3 — Task-incremental 評估（從已存的逐 slide 記錄重算）

SEQFT.md 的評估是 **8-way argmax = class-incremental**。在 class-IL 下「遺忘」總是看起來像災難，但其中很大一部分是**跨任務混淆**，而不是「該任務內的鑑別力退化」。這兩者是不同的科學主張，本檔把它們分開。

**task-IL**：argmax 只在該 task 自己的兩列上取（esca rows 0-1、rcc 2-3、brca 4-5、lung 6-7），即 2-way，隨機基準 0.5000。

## 重算的正當性

per_slide JSON 沒有存 8-way logits，但存了 `selected_idx` 與 `weights_softmax`。CONTRACT-4 的 head 是 frozen 且決定性的：

```
logits = logit_scale * normalize(Σ_i w_i z_i) @ f_txt.T
```

z 來自特徵檔、f_txt 與 logit_scale 不隨訓練改變，所以 logits 可以精確重建。
**驗證**：4185 筆記錄全部重建，8-way argmax 與已存的 `pred_softmax` 不符 **0** 筆 → 重建無損。**沒有重跑任何訓練或 selector 前向。**

## order = reverse　（esca → rcc → brca → lung）

### 表 1-IL：task-IL accuracy matrix（2-way，3 seeds mean ± std）

| 學完 | eval esca | eval rcc | eval brca | eval lung |
|---|---|---|---|---|
| T1 esca | 0.9333 ± 0.0000 | — | — | — |
| T2 rcc | 0.5333 ± 0.0667 | 0.9211 ± 0.0132 | — | — |
| T3 brca | 0.5556 ± 0.1540 | 0.6623 ± 0.1542 | 0.8710 ± 0.0284 | — |
| T4 lung | 0.7556 ± 0.2037 | 0.6974 ± 0.1721 | 0.8423 ± 0.0062 | 0.8421 ± 0.0279 |

### 表 2-IL：class-IL vs task-IL 的 A1 forgetting 對照

| task | n | class-IL A1 (pp) | task-IL A1 (pp) | class-IL acc @T4 | task-IL acc @T4 |
|---|---|---|---|---|---|
| tcga_esca | 15 | +71.11 ± +3.85 | +17.78 ± +20.37 | 0.1111 ± 0.0385 | 0.7556 ± 0.2037 |
| tcga_rcc | 76 | +59.65 ± +35.07 | +22.37 ± +17.41 | 0.3158 ± 0.3383 | 0.6974 ± 0.1721 |
| tcga_brca | 93 | +33.69 ± +15.07 | +2.87 ± +2.48 | 0.5269 ± 0.1455 | 0.8423 ± 0.0062 |
| tcga_lung | 95 | +0.00 ± +0.00 | +0.00 ± +0.00 | 0.8421 ± 0.0279 | 0.8421 ± 0.0279 |

class-IL 隨機基準 0.1250（8 類）、task-IL 隨機基準 0.5000（2 類）。

### 表 3-IL：跨任務洩漏率（學完 T4 後）

預測落在**別的 task 的類別列**上的比例。這個數字直接量化 class-IL 崩潰裡有多少是「跑錯 task」而非「task 內判錯」。

| task | 洩漏率 | 落在自己列且判對 | 落在自己列但判錯 |
|---|---|---|---|
| tcga_esca | 0.8889 ± 0.0385 | 0.1111 ± 0.0385 | 0.0000 ± 0.0000 |
| tcga_rcc | 0.6360 ± 0.3462 | 0.3158 ± 0.3383 | 0.0482 ± 0.0532 |
| tcga_brca | 0.3513 ± 0.1722 | 0.5269 ± 0.1455 | 0.1219 ± 0.0271 |
| tcga_lung | 0.0000 ± 0.0000 | 0.8421 ± 0.0279 | 0.1579 ± 0.0279 |

### Jaccard 的隨機重疊參照值

從 n 個 patch 隨機抽兩次 K 個，期望 Jaccard = (K/n) / (2 − K/n)。K = 8，n 取該 task 每張 test slide 的實際 patch 數後平均。

| task | 平均 n（patch/slide） | 隨機參照 Jaccard | SEQFT 實測 Jaccard | 實測 vs 參照 |
|---|---|---|---|---|
| tcga_esca | 3773 | 0.00225 | 0.00000 | **低於**參照 |
| tcga_rcc | 3245 | 0.00220 | 0.00029 | **低於**參照 |
| tcga_brca | 3101 | 0.00186 | 0.02587 | 高於參照 |

## order = main　（lung → brca → rcc → esca）

### 表 1-IL：task-IL accuracy matrix（2-way，3 seeds mean ± std）

| 學完 | eval lung | eval brca | eval rcc | eval esca |
|---|---|---|---|---|
| T1 lung | 0.8351 ± 0.0219 | — | — | — |
| T2 brca | 0.6982 ± 0.1080 | 0.8674 ± 0.0530 | — | — |
| T3 rcc | 0.6702 ± 0.0580 | 0.7491 ± 0.0691 | 0.9123 ± 0.0201 | — |
| T4 esca | 0.6526 ± 0.0657 | 0.8065 ± 0.0599 | 0.6886 ± 0.0548 | 0.8889 ± 0.0385 |

### 表 2-IL：class-IL vs task-IL 的 A1 forgetting 對照

| task | n | class-IL A1 (pp) | task-IL A1 (pp) | class-IL acc @T4 | task-IL acc @T4 |
|---|---|---|---|---|---|
| tcga_lung | 95 | +57.89 ± +6.90 | +18.25 ± +5.80 | 0.2561 ± 0.0797 | 0.6526 ± 0.0657 |
| tcga_brca | 93 | +29.75 ± +15.96 | +6.09 ± +11.19 | 0.5663 ± 0.1184 | 0.8065 ± 0.0599 |
| tcga_rcc | 76 | +60.53 ± +23.13 | +22.37 ± +6.58 | 0.3070 ± 0.2191 | 0.6886 ± 0.0548 |
| tcga_esca | 15 | +0.00 ± +0.00 | +0.00 ± +0.00 | 0.8000 ± 0.0667 | 0.8889 ± 0.0385 |

class-IL 隨機基準 0.1250（8 類）、task-IL 隨機基準 0.5000（2 類）。

### 表 3-IL：跨任務洩漏率（學完 T4 後）

預測落在**別的 task 的類別列**上的比例。這個數字直接量化 class-IL 崩潰裡有多少是「跑錯 task」而非「task 內判錯」。

| task | 洩漏率 | 落在自己列且判對 | 落在自己列但判錯 |
|---|---|---|---|
| tcga_lung | 0.5579 ± 0.0759 | 0.2561 ± 0.0797 | 0.1860 ± 0.0122 |
| tcga_brca | 0.2473 ± 0.0880 | 0.5663 ± 0.1184 | 0.1864 ± 0.0508 |
| tcga_rcc | 0.6009 ± 0.2829 | 0.3070 ± 0.2191 | 0.0921 ± 0.0658 |
| tcga_esca | 0.1111 ± 0.0385 | 0.8000 ± 0.0667 | 0.0889 ± 0.0385 |

### Jaccard 的隨機重疊參照值

從 n 個 patch 隨機抽兩次 K 個，期望 Jaccard = (K/n) / (2 − K/n)。K = 8，n 取該 task 每張 test slide 的實際 patch 數後平均。

| task | 平均 n（patch/slide） | 隨機參照 Jaccard | SEQFT 實測 Jaccard | 實測 vs 參照 |
|---|---|---|---|---|
| tcga_lung | 3016 | 0.00230 | 0.00070 | **低於**參照 |
| tcga_brca | 3101 | 0.00186 | 0.00413 | 高於參照 |
| tcga_rcc | 3245 | 0.00220 | 0.00510 | 高於參照 |

## 產出

本檔由 `python scripts/recompute_task_il.py` 從 `outputs/exp2/seqft/per_slide/*.json` 重算，與 `SEQFT.md` 並列，不覆蓋任何既有結果。

