# S1 — 任務可分離性 probe

⚠️ **這是診斷，不是方法的一部分。** 本 probe 不被 selector 匯入，不影響任何 Exp 0 / Exp 1 的數字。

問題：四個 task 是四個不同器官，task identity 可能從影像特徵本身就免費可得。

做法：每張 slide 取一個固定長度的表徵，訓一個 multinomial logistic regression 預測 4-way task id（train split 訓練、test split 評估）。特徵先做 StandardScaler，max_iter=5000，random_state=0。

- train：2273 張（esca 120、rcc 616、brca 763、lung 774）
- test：279 張（esca 15、rcc 76、brca 93、lung 95）
- 多數類基準（test）：0.3405；隨機猜：0.2500

## 結果

| 輸入表徵 | 維度 | train accuracy | test accuracy |
|---|---|---|---|
| slide 平均 CONCH patch feature | 512 | 0.9996 | 0.9821 |
| 8 個 tissue group prototype 串接 | 4096 | 1.0000 | 0.9857 |

group prototype 的順序為 tumor、stroma、lymphocyte、necrosis、normal_epithelium、vessel、adipose、background；空 group 補零向量。

## Confusion matrix（test split）

### A. slide 平均 patch feature（512-d）

| true \ pred | esca | rcc | brca | lung | n |
|---|---|---|---|---|---|
| **esca** | 14 | 0 | 0 | 1 | 15 |
| **rcc** | 0 | 74 | 1 | 1 | 76 |
| **brca** | 0 | 0 | 93 | 0 | 93 |
| **lung** | 2 | 0 | 0 | 93 | 95 |

### B. group prototype 串接（4096-d）

| true \ pred | esca | rcc | brca | lung | n |
|---|---|---|---|---|---|
| **esca** | 14 | 0 | 0 | 1 | 15 |
| **rcc** | 0 | 76 | 0 | 0 | 76 |
| **brca** | 1 | 0 | 91 | 1 | 93 |
| **lung** | 1 | 0 | 0 | 94 | 95 |

逐 slide 預測：`outputs/exp1/diag/per_slide_task_probe.json`

