# S2 — SeqFT：序列訓練下的遺忘

一個 shared selector 依序學 4 個 task，**沒有任何 CL 機制**（無 replay、無 distillation、無 LoRA merge）。每學完一個 task 就在所有已學過的 task 上評估。joint 訓練只能證明 multi-task interference，不能代替這個實驗。

架構 L3b（shared selector，不給 q_tau、無 state、flat）、B=8、c=1、seeds [0, 1, 2]、epochs 5、lr 0.001、beta_s 0.1、prior discriminative。訓練用 train split、評估用 test split、8-way label space。

## 事前預測（跑之前寫下，跑完不得修改）

> PI 判讀（承 S1）：task identity 在 F_g 的實際輸入（group prototype）上
> 98.57% 線性可讀，因此 q_tau 在此 benchmark 中結構性冗餘，L4 ≈ L3b 為預期
> 結果而非模型缺陷。
>
> 延伸預測：shared selector 可隱式按器官路由，故 **accuracy 層 forgetting
> 可能偏輕**，但 **selection 行為層仍可能嚴重漂移**。此預測於 S2 跑完後
> 對照，不得事後修改。

⚠️ 不得因為 accuracy forgetting 小就下結論說沒有遺忘 —— 三個軸一起看。

## order = reverse　（esca → rcc → brca → lung）

### 表 1：accuracy matrix（3 seeds mean ± std，softmax 權重）

| 學完 | eval esca | eval rcc | eval brca | eval lung |
|---|---|---|---|---|
| T1 esca | 0.8222 ± 0.0385 | — | — | — |
| T2 rcc | 0.2222 ± 0.1388 | 0.9123 ± 0.0201 | — | — |
| T3 brca | 0.1111 ± 0.0770 | 0.0570 ± 0.0423 | 0.8638 ± 0.0310 | — |
| T4 lung | 0.1111 ± 0.0385 | 0.3158 ± 0.3383 | 0.5269 ± 0.1455 | 0.8421 ± 0.0279 |

n（test）：esca 15、rcc 76、brca 93、lung 95。

### 表 2：三個軸（3 seeds mean ± std）

| task | n | A1 accuracy forgetting (pp) | A2 Jaccard | A2 quota KL | A3 ΣU 學完 T_i | A3 ΣU 學完 T_4 | A3 retention |
|---|---|---|---|---|---|---|---|
| tcga_esca | 15 | +71.11 ± +3.85 | 0.0000 ± 0.0000 | 0.4666 ± 0.2877 | 23.4 ± 3.5 | -52.0 ± 13.7 | -2.2490 ± 0.5651 |
| tcga_rcc | 76 | +59.65 ± +35.07 | 0.0003 ± 0.0005 | 0.8407 ± 0.1834 | 141.4 ± 8.2 | -306.7 ± 297.4 | -2.1088 ± 2.0429 |
| tcga_brca | 93 | +33.69 ± +15.07 | 0.0259 ± 0.0354 | 0.5475 ± 0.3761 | 163.0 ± 5.0 | -48.9 ± 130.8 | -0.3120 ± 0.8080 |
| tcga_lung | 95 | +0.00 ± +0.00 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 154.4 ± 7.2 | 154.4 ± 7.2 | 1.0000 ± 0.0000 |

- **A1** = acc(T_i | 學完 T_i) − acc(T_i | 學完 T_4)，正值代表退步。
- **A2 Jaccard** = 同一批 slide 在兩個時點選到的 patch 集合重疊度；1.0 = 完全沒變，0.0 = 完全換掉。
- **A2 quota KL** = group 配額分佈 KL(學完 T_i ‖ 學完 T_4)，Laplace 平滑；0 = 分佈沒變。
- **A3** ΣU 是該 task 全部 test slide 的 utility 加總。U 沿選取順序累加 counterfactual gain；frozen head 不隨訓練改變，所以 U 只取決於選了哪些 patch。retention = ΣU(學完 T_4) / ΣU(學完 T_i)：1.0 = 效用完全保留，0 = 新選的東西完全沒用，**負值 = 新選的東西是反效果**（把證據推向錯誤類別，比什麼都不看還糟）。

### T4 學完後，各 task 的 group 配額分佈

| task | 時點 | tumor | stroma | lymphocyte | necrosis | normal_epithelium | vessel | adipose | background |
|---|---|---|---|---|---|---|---|---|---|
| esca | 學完 T1 | 0.575 | 0.064 | 0.133 | 0.072 | 0.094 | 0.008 | 0.000 | 0.053 |
| esca | 學完 T4 | 0.586 | 0.075 | 0.044 | 0.108 | 0.014 | 0.025 | 0.003 | 0.144 |
| rcc | 學完 T2 | 0.640 | 0.109 | 0.135 | 0.016 | 0.001 | 0.031 | 0.023 | 0.046 |
| rcc | 學完 T4 | 0.345 | 0.110 | 0.043 | 0.158 | 0.001 | 0.162 | 0.020 | 0.160 |
| brca | 學完 T3 | 0.010 | 0.388 | 0.116 | 0.024 | 0.000 | 0.003 | 0.021 | 0.438 |
| brca | 學完 T4 | 0.275 | 0.253 | 0.069 | 0.099 | 0.001 | 0.024 | 0.018 | 0.260 |
| lung | 學完 T4 | 0.704 | 0.042 | 0.016 | 0.095 | 0.001 | 0.008 | 0.000 | 0.134 |

## order = main　（lung → brca → rcc → esca）

### 表 1：accuracy matrix（3 seeds mean ± std，softmax 權重）

| 學完 | eval lung | eval brca | eval rcc | eval esca |
|---|---|---|---|---|
| T1 lung | 0.8351 ± 0.0219 | — | — | — |
| T2 brca | 0.1333 ± 0.0161 | 0.8638 ± 0.0508 | — | — |
| T3 rcc | 0.3719 ± 0.0747 | 0.3333 ± 0.1037 | 0.9123 ± 0.0201 | — |
| T4 esca | 0.2561 ± 0.0797 | 0.5663 ± 0.1184 | 0.3070 ± 0.2191 | 0.8000 ± 0.0667 |

n（test）：lung 95、brca 93、rcc 76、esca 15。

### 表 2：三個軸（3 seeds mean ± std）

| task | n | A1 accuracy forgetting (pp) | A2 Jaccard | A2 quota KL | A3 ΣU 學完 T_i | A3 ΣU 學完 T_4 | A3 retention |
|---|---|---|---|---|---|---|---|
| tcga_lung | 95 | +57.89 ± +6.90 | 0.0007 ± 0.0007 | 0.5643 ± 0.2219 | 145.7 ± 20.9 | -244.4 ± 78.4 | -1.7512 ± 0.8126 |
| tcga_brca | 93 | +29.75 ± +15.96 | 0.0041 ± 0.0034 | 0.7486 ± 0.5411 | 161.6 ± 11.4 | -21.4 ± 89.0 | -0.1206 ± 0.5303 |
| tcga_rcc | 76 | +60.53 ± +23.13 | 0.0051 ± 0.0029 | 0.8703 ± 0.4031 | 141.1 ± 7.3 | -303.0 ± 225.7 | -2.1156 ± 1.5805 |
| tcga_esca | 15 | +0.00 ± +0.00 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 22.9 ± 3.1 | 22.9 ± 3.1 | 1.0000 ± 0.0000 |

- **A1** = acc(T_i | 學完 T_i) − acc(T_i | 學完 T_4)，正值代表退步。
- **A2 Jaccard** = 同一批 slide 在兩個時點選到的 patch 集合重疊度；1.0 = 完全沒變，0.0 = 完全換掉。
- **A2 quota KL** = group 配額分佈 KL(學完 T_i ‖ 學完 T_4)，Laplace 平滑；0 = 分佈沒變。
- **A3** ΣU 是該 task 全部 test slide 的 utility 加總。U 沿選取順序累加 counterfactual gain；frozen head 不隨訓練改變，所以 U 只取決於選了哪些 patch。retention = ΣU(學完 T_4) / ΣU(學完 T_i)：1.0 = 效用完全保留，0 = 新選的東西完全沒用，**負值 = 新選的東西是反效果**（把證據推向錯誤類別，比什麼都不看還糟）。

### T4 學完後，各 task 的 group 配額分佈

| task | 時點 | tumor | stroma | lymphocyte | necrosis | normal_epithelium | vessel | adipose | background |
|---|---|---|---|---|---|---|---|---|---|
| lung | 學完 T1 | 0.827 | 0.022 | 0.006 | 0.095 | 0.002 | 0.000 | 0.000 | 0.048 |
| lung | 學完 T4 | 0.424 | 0.143 | 0.165 | 0.061 | 0.026 | 0.068 | 0.020 | 0.093 |
| brca | 學完 T2 | 0.028 | 0.385 | 0.203 | 0.003 | 0.000 | 0.001 | 0.133 | 0.247 |
| brca | 學完 T4 | 0.119 | 0.289 | 0.299 | 0.049 | 0.013 | 0.056 | 0.029 | 0.146 |
| rcc | 學完 T3 | 0.683 | 0.045 | 0.100 | 0.036 | 0.003 | 0.082 | 0.022 | 0.029 |
| rcc | 學完 T4 | 0.139 | 0.285 | 0.163 | 0.068 | 0.007 | 0.197 | 0.035 | 0.105 |
| esca | 學完 T4 | 0.494 | 0.078 | 0.231 | 0.058 | 0.056 | 0.014 | 0.000 | 0.069 |

## 事前預測對照（跑完後填入；上面的預測段落未修改）

| 預測 | 觀察 |
|---|---|
| (reverse) accuracy 層 forgetting **偏輕** | 前 3 個 task 的 A1 平均 **+54.82 pp**（範圍 +19.74 ~ +85.53） |
| (reverse) selection 行為層**嚴重漂移** | 前 3 個 task 的 Jaccard 平均 **0.0087**（範圍 0.0000 ~ 0.0668） |
| (main) accuracy 層 forgetting **偏輕** | 前 3 個 task 的 A1 平均 **+49.39 pp**（範圍 +16.13 ~ +77.63） |
| (main) selection 行為層**嚴重漂移** | 前 3 個 task 的 Jaccard 平均 **0.0033**（範圍 0.0000 ~ 0.0083） |

判讀由 PI 進行；此處只陳述數字。

## 產出檔案

- 逐 slide 預測與選中 index：`outputs/exp2/seqft/per_slide/{order}_seed{seed}_stage{k}.json`
- 曲線資料（accuracy 隨訓練進度）：`outputs/exp2/seqft/curves.json`
- 模型 checkpoint：`outputs/exp2/seqft/ckpt/`（每個 (order, seed, task) 一份，中斷可續跑）

