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
| **單一 patch feature（附錄 A）** | 512 | 0.9088 | **0.8930** |

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

## 附錄 A — 單一 patch 的 task 可分離性

S1 的主 probe 用全片平均，但 F_p 的輸入是**單一 512-d patch**。這一節把輸入換成單一 patch feature，label 是該 slide 的 task id，split 沿用同一組。

- **train 端子取樣**：全集是 2273 張 × 平均約 3400 個 patch ≈ 7.7M 個，float32 約 15.8 GB，這台機器放不下。改為每張 train slide 固定隨機取 64 個（`np.random.RandomState(0)`），共 145,429 個 patch。
- **test 端不取樣**：279 張 test slide 的**全部** 878,117 個 patch 都評估（逐 slide 串流預測，不一次載入）。
- max_iter=1000（patch 數量大，比主 probe 的 5000 低）。

| 指標 | 值 |
|---|---|
| patch 層 train accuracy | 0.9088 |
| **patch 層 test accuracy** | **0.8930** |
| 同一個 patch probe 做 slide 多數決 | 0.9713 |
| 多數類基準（patch 層） | 0.3284 |

### Confusion matrix（test split，patch 層，單位：patch 數）

| true \ pred | esca | rcc | brca | lung | n |
|---|---|---|---|---|---|
| **esca** | 34997 | 2902 | 13192 | 5505 | 56596 |
| **rcc** | 451 | 220449 | 10487 | 15265 | 246652 |
| **brca** | 2524 | 4545 | 267169 | 14112 | 288350 |
| **lung** | 4905 | 13255 | 6830 | 261529 | 286519 |

逐 slide 預測：`outputs/exp1/diag/per_slide_task_probe.json`（slide 層）、`outputs/exp1/diag/per_slide_patch_probe.json`（patch 層，含每張 slide 的 patch 正確率與多數決）

