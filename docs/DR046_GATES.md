# DR-046 gate 總表

五個 gate 的**逐 seed 數字與 win count**。三級規則（DR-020）：5/5 = systematic、4/5 = directional inconclusive、≤3/5 = within noise。**不報 p 值**（DR-016）。

⚠️ **本檔只放數字，不寫解讀** —— 判讀的文字在 [`docs/ledger/DR-046.md`](ledger/DR-046.md) 與論文裡。

⚠️ 「觸發分支」一欄是 **PI 逐條裁定後照錄**，不由腳本推導；用條款名稱標註而不引節號 —— pre-registration 文件（PI 的 DR-046 v2）不在本 repo 內，無法逐字核對節次編號。

⚠️ **win count 帶方向**：`0/5` 是 `5/5` 的鏡像（對照式右邊那個臂系統性勝出），不是 within noise。沿用 `report_memory_hier.verdict` 的既有慣例。

資料源：`outputs/exp2/main/per_slide/*.json`，沿用 `run_exp2.arm_metrics` / `filter_arch` 重算；flat 與 hier 分開取（同名臂在兩種架構下是兩個實驗）。

---

## G-W1　warm-start（task1 全參數 FT）

對照：**W1 − A5**（arch = `flat`，order = `reverse`，seeds = [0, 1, 2, 3, 4]）

| 指標 | 逐 seed 差值 | mean ± sd | win count | 三級判讀 |
|---|---|---|---|---|
| task-IL final avg | -4.79, +4.71, -2.53, -0.47, -1.74 | -0.96 ± 3.54 pp | 1/5 | directional, inconclusive（A5 向） |
| class-IL final avg | +0.54, +2.77, -1.81, -2.31, -8.22 | -1.80 ± 4.12 pp | 2/5 | within noise |
| 跨任務洩漏率 | -2.00, +0.21, +2.35, +3.11, +4.87 | +1.71 ± 2.67 pp | 1/5 | directional, inconclusive（A5 向） |
| selection Jaccard | +0.109, +0.102, +0.026, +0.042, -0.002 | +0.055 ± 0.05 | 4/5 | directional, inconclusive（W1 向） |

觸發分支：**分裂** → 依預註冊 **split 規則**維持現行 LoRA-from-task-1。

---

## G-L2　single continual adapter

對照：**L2 − A5**（arch = `flat`，order = `reverse`，seeds = [0, 1, 2, 3, 4]）

| 指標 | 逐 seed 差值 | mean ± sd | win count | 三級判讀 |
|---|---|---|---|---|
| task-IL final avg | -3.28, +3.05, -3.79, -2.94, -3.33 | -2.06 ± 2.87 pp | 1/5 | directional, inconclusive（A5 向） |
| class-IL final avg | +0.93, -3.08, -4.68, -2.40, -8.46 | -3.54 ± 3.43 pp | 1/5 | directional, inconclusive（A5 向） |
| 跨任務洩漏率 | -2.28, +4.40, +2.82, +0.99, +6.53 | +2.49 ± 3.36 pp | 1/5 | directional, inconclusive（A5 向） |
| selection Jaccard | -0.037, +0.008, -0.096, -0.153, -0.108 | -0.077 ± 0.06 | 1/5 | directional, inconclusive（A5 向） |

觸發分支：A5 勝 4/5（class-IL）→ 觸發 **fresh-per-task 容量正當化條款**。

---

## G-C1　post-hoc composition（獨立 delta 相加）

對照：**A2 − C1**（arch = `flat`，order = `reverse`，seeds = [0, 1, 2, 3, 4]）

| 指標 | 逐 seed 差值 | mean ± sd | win count | 三級判讀 |
|---|---|---|---|---|
| task-IL final avg | -8.19, -4.51, -6.94, -3.62, -9.71 | -6.59 ± 2.53 pp | 0/5 | **systematic**（C1 勝） |
| class-IL final avg | -22.10, -9.56, -31.47, -24.16, -35.63 | -24.59 ± 10.02 pp | 0/5 | **systematic**（C1 勝） |
| 跨任務洩漏率 | +21.49, +9.25, +32.34, +23.44, +37.92 | +24.89 ± 11.00 pp | 0/5 | **systematic**（C1 勝） |
| selection Jaccard | -0.347, -0.224, -0.311, -0.048, -0.416 | -0.269 ± 0.14 | 0/5 | **systematic**（C1 勝） |

觸發分支：**翻盤** → 觸發**翻盤條款**。

PI 核可：依**翻盤條款**誠實報告。

---

## G-α　damped merging（α = 0.5）

對照：**A5 − A5H**（arch = `flat`，order = `reverse`，seeds = [0, 1, 2, 3, 4]）

| 指標 | 逐 seed 差值 | mean ± sd | win count | 三級判讀 |
|---|---|---|---|---|
| task-IL final avg | +2.47, -2.18, +2.47, +0.48, +1.66 | +0.98 ± 1.95 pp | 4/5 | directional, inconclusive（A5 向） |
| class-IL final avg | -3.55, -2.42, +3.53, -0.29, +7.35 | +0.92 ± 4.49 pp | 2/5 | within noise |
| 跨任務洩漏率 | +3.83, -0.75, -2.98, -1.03, -4.81 | -1.15 ± 3.23 pp | 4/5 | directional, inconclusive（A5 向） |
| selection Jaccard | -0.109, -0.183, +0.012, +0.058, -0.161 | -0.077 ± 0.11 | 2/5 | within noise |

觸發分支：**分裂** → 維持 **α = 1**。

PI 核可：維持 **α = 1**；A5H 留存為消融證據。

---

## L2@hier　single continual adapter，階層底盤（配對版）

對照：**L2 − A5**（arch = `hier`，order = `reverse`，seeds = [0, 1, 2, 3, 4]）

| 指標 | 逐 seed 差值 | mean ± sd | win count | 三級判讀 |
|---|---|---|---|---|
| task-IL final avg | -0.54, -1.05, -1.54, -2.27, -0.85 | -1.25 ± 0.67 pp | 0/5 | **systematic**（A5 勝） |
| class-IL final avg | +3.57, +2.21, +0.11, -3.49, +1.56 | +0.79 ± 2.70 pp | 4/5 | directional, inconclusive（L2 向） |
| 跨任務洩漏率 | -5.52, -6.33, -1.58, -0.12, -2.68 | -3.25 ± 2.63 pp | 5/5 | **systematic**（L2 勝） |
| selection Jaccard | -0.033, -0.139, -0.072, +0.019, -0.075 | -0.060 ± 0.06 | 1/5 | directional, inconclusive（A5 向） |

觸發分支：**未預註冊之觀察**（非 gate）。

PI 核可：**未預註冊之觀察**。方法維持 fresh-per-task（現任者＋主指標 task-IL 5/5）；底盤交互作用於論文如實報告。

---

產生：`python scripts/report_dr046_gates.py`。數值由 `scripts/verify_doc_numbers.py` 溯源把關。

