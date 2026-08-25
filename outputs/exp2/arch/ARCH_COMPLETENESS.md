# G3 / G4 / G5 —— 架構完整性

圖上有三個元件在所有主結果中都是關閉或未實作的：E_t/B_t 狀態迴圈、q_tau 任務條件化、group 層 L_sem。本檔把它們逐一測出來 —— **有效就保留，無效就從圖上移除並寫成有數據的 limitation**。

通用設定：架構 hier（per_budget）、|M|=512、B=8、c=1、order=reverse、5 seeds、epochs 5、lr 1e-3、prior=discriminative、beta_s=0.1、beta_u=0.1、λ 全 1.0。**每個實驗只動一個變因**，對照組沿用 G1' 已有的 hier-A5 存檔（不重跑）。

**判準讀法（PI 裁定 1）**：G5 = 任一準確率軸滿足即通過（描述性用字）；G4 / G3 = task-IL 與 class-IL 兩軸皆須滿足（效能宣稱）。G5 決定的是 "stateful" 這個**描述性用字**，機制存在性只需單軸一致增益即可支持；G4 / G3 決定的是**元件對方法有貢獻的效能宣稱**，門檻較高。前例：A5−A3 在 flat 下 class-IL 5/5 而 task-IL 3/5，採單軸即會誤宣稱勝出，DR-015 正確擋下。**單軸通過而另一軸不動 → 照實報為「僅在 X 軸有效」，不計為通過。**

⚠️ 此讀法為**事前修訂**（PROMPT G345-CRITERIA-20260824），時間早於任何 G345 結果產出，非事後調整。

## G5 前置：no-op 檢查

判準：state 開啟後，c=1 八輪與 c=8 一輪的選取集合必須不同。

設定：synthetic slide + 未訓練的隨機初始化模型，20 組、B=8、arch=hier。比較 c=1 跑八輪 vs c=8 跑一輪。

| use_state | 選取集合相同 | 選取順序相同 | 判讀 |
|---|---|---|---|
| False（現行主線） | 20/20 | 0/20 | **no-op** —— 與 CLAIMS C-01 一致 |
| True（G5） | 16/20 | 0/20 | **非 no-op**，state 確實進入計算 |

**判定：PASS** —— G5 可以進行。

### ⚠️ 這個數字怎麼讀（PI 裁定 2）

「4/20 的選取集合改變」是**下界，不是效果量**。

- 這批用的是**未訓練的隨機初始化權重**與 synthetic slide。state 對選取的影響取決於 F_g / F_p 學到多重視 e_t 與 B_tilde 這兩段輸入；隨機權重下該敏感度沒有理由代表訓練後的敏感度。
- **不可讀成「state 只影響 20% 的選取」。** 本檢查回答的是二元問題「state 有沒有進入計算」，不是「影響多大」。
- **以訓練後的模型在真實 slide 上的重測為準。** 該重測在 G5 跑完後進行，結果會補進本節；在那之前，效果量沒有可引用的估計。

（原始 caveat：未訓練權重下 state 的影響偏小（同集合 16/20）。這是下界，不是效果量的估計；訓練後應以真實 slide 重做一次。）

### 訓練後重測（PI 裁定 2 的承諾，已兌現）

用 **G5 序列訓練後的最終模型**（seed 0，4 個 stage 全跑完）在 **279 張真實 test slide** 上重做同一個比較。

⚠️ **模型一致性驗證**：本檔為省時跳過了各 stage 的評估，因此先用 G5 的逐 slide 存檔逐筆比對 —— 279 筆中 279 筆選取完全相同、0 筆不同。✅ 模型與正式 G5 跑出來一致。

| 設定 | 選取集合相同 | 選取順序相同 | 平均重疊 |
|---|---|---|---|
| state OFF（主線） | 279/279 | 120/279 | 8.00/8 |
| state ON（G5） | 0/279 | 0/279 | 2.81/8 |

**訓練後**：state 開啟時 279/279（100.0%）的 slide 選取集合改變，未訓練 synthetic 的對應數字是 4/20（20%）。

⚠️ **這仍然不改變 G5 的落判** —— 落判看的是準確率判準，本節看的是「state 有沒有改變選取」。**改變選取而未改善準確率**，正是 G5 FAIL 的內容（DR-043）。

產物：`outputs/exp2/arch/noop_check.json`

## G4 前置：q_tau 是否真的進入計算

use_query=False 的實作是把 query 欄位填零（selector/model.py:44），而 run_exp2.Ctx.q0 = zeros(512)。因此只打開 use_query 而不接真正的 q_tau，結果與關閉時位元相同 —— G4 必須由 run_arch_completeness.wire_task_queries 接上 TaskQueryBank 才成立。

| use_query=True 時餵入的 q_tau | 與 use_query=False 的選取集合相同 | 判讀 |
|---|---|---|
| zeros(512)（run_exp2 現行 ctx.q0） | 20/20 | **位元相同 → 由構造保證的 null** |
| 真正的 task query（非零） | 16/20 | **非 no-op**，q_tau 確實進入計算 |

**判定：PASS** —— G4 必須先接上 TaskQueryBank 才成立。

## 主表

| 臂 | 說明 | seeds | task-IL final avg | class-IL final avg | 跨任務洩漏率 | selection Jaccard | group 配額 KL |
|---|---|---|---|---|---|---|---|
| base | hier-A5 基準（G1'） | 5 | 0.9089 ± 0.0077 | 0.8119 ± 0.0162 | 0.1220 ± 0.0237 | 0.1419 ± 0.0482 | 0.0191 ± 0.0031 |
| G5 | + E_t / B_t 狀態條件化 | 5 | 0.9039 ± 0.0228 | 0.8223 ± 0.0286 | 0.1032 ± 0.0243 | 0.1547 ± 0.0194 | 0.0429 ± 0.0265 |
| G4 | + q_tau 任務條件化 | 5 | 0.9120 ± 0.0214 | 0.8704 ± 0.0357 | 0.0627 ± 0.0314 | 0.2165 ± 0.0408 | 0.0284 ± 0.0248 |
| G3 | + group 層 L_sem (beta_g=0.1) | 5 | 0.9037 ± 0.0150 | 0.8234 ± 0.0038 | 0.0958 ± 0.0212 | 0.1297 ± 0.0048 | 0.0143 ± 0.0033 |

前兩欄是 pre-registered 的主要指標；後三欄為次要診斷。final avg 算全部 4 個 task；Jaccard 與配額 KL 只算前 3 個（CL 慣例）。

## 配對比較與落判

臂間比較一律**配對**（同 seed 相減）。win count 三級規則（DR-020）：5/5 = systematic、4/5 = directional inconclusive、≤3/5 = within noise。**不報 p 值**（DR-016）。

### G5　+ E_t / B_t 狀態條件化

共同 seeds：[0, 1, 2, 3, 4]（n=5）

| 指標 | 逐 seed 配對差值 | 配對 mean ± std | win count | 三級判讀 |
|---|---|---|---|---|
| task-IL final avg | -0.280, -2.720, +1.123, -2.731, +2.122 | -0.497 ± 2.206 pp | 2/5 | within noise |
| class-IL final avg | +4.489, +4.946, -1.295, -4.426, +1.512 | +1.045 ± 3.959 pp | 3/5 | within noise |
| 跨任務洩漏率 | -3.914, -8.268, +2.418, +0.291, +0.083 | -1.878 ± 4.242 pp | 2/5 | within noise |
| selection Jaccard | +0.052, -0.039, -0.022, +0.080, -0.007 | +0.013 ± 0.051 | 2/5 | within noise |
| group 配額 KL | +0.022, +0.016, -0.003, +0.065, +0.019 | +0.024 ± 0.025 | 1/5 | within noise |

**pre-registered 判準（原文，先於結果寫定）**：

- 通過 → win >= 4/5 且配對為正（任一準確率軸）→ stateful 成立，圖與 "Beyond HistoSelect" 保留該條，論文可寫 state-conditioned sequential acquisition（仍不得寫 "plan"，因為輪間 detach）
- 未通過 → <= 3/5 或為負 → 從架構圖移除 Panel E 與 "stateful" 一詞，改寫為 budgeted top-K selection under a shared frozen head，並在 limitation 說明：在此設定下狀態條件化未帶來可測增益。不得調參搶救。

**落判依據**：

- task-IL final avg：配對 -0.50 pp、win 2/5（within noise）→ 不滿足
- class-IL final avg：配對 +1.05 pp、win 3/5（within noise）→ 不滿足
- 落判規則：任一軸滿足即通過 → FAIL

**判定：FAIL** → <= 3/5 或為負 → 從架構圖移除 Panel E 與 "stateful" 一詞，改寫為 budgeted top-K selection under a shared frozen head，並在 limitation 說明：在此設定下狀態條件化未帶來可測增益。不得調參搶救。

### G4　+ q_tau 任務條件化

共同 seeds：[0, 1, 2, 3, 4]（n=5）

| 指標 | 逐 seed 配對差值 | 配對 mean ± std | win count | 三級判讀 |
|---|---|---|---|---|
| task-IL final avg | -3.279, -0.330, +0.789, +4.380, -0.001 | +0.312 ± 2.747 pp | 2/5 | within noise |
| class-IL final avg | +7.774, +8.795, +5.707, +7.089, -0.120 | +5.849 ± 3.520 pp | 4/5 | directional, inconclusive |
| 跨任務洩漏率 | -9.124, -10.791, -3.514, -5.779, -0.407 | -5.923 ± 4.189 pp | 5/5 | systematic |
| selection Jaccard | +0.125, -0.020, -0.008, +0.184, +0.092 | +0.075 ± 0.087 | 3/5 | within noise |
| group 配額 KL | +0.050, +0.019, -0.011, -0.009, -0.002 | +0.009 ± 0.025 | 3/5 | within noise |

**pre-registered 判準（原文，先於結果寫定）**：

- 通過 → win >= 4/5 且為正 → 圖上保留 q_tau，"task-conditioned" 一詞成立
- 未通過 → <= 3/5 → 圖上把 q_tau 標為 optional 或移除，論文寫：在跨器官任務序列中任務身分可由視覺特徵推得（S1，98.2/98.6%），顯式語意條件化在 patch 排序與群組配額兩個層級皆未提供增益。這是有機制解釋的 null，不是失敗。同器官設定列入 future work。

**落判依據**：

- task-IL final avg：配對 +0.31 pp、win 2/5（within noise）→ 不滿足
- class-IL final avg：配對 +5.85 pp、win 4/5（directional, inconclusive）→ 滿足
- 落判規則：兩軸皆須滿足 → FAIL
- ⚠️ **僅在 class-IL final avg 有效**，另一軸不動 —— 照實報告，不計為通過（PI 裁定 1）。

**判定：FAIL** → <= 3/5 → 圖上把 q_tau 標為 optional 或移除，論文寫：在跨器官任務序列中任務身分可由視覺特徵推得（S1，98.2/98.6%），顯式語意條件化在 patch 排序與群組配額兩個層級皆未提供增益。這是有機制解釋的 null，不是失敗。同器官設定列入 future work。

### G3　+ group 層 L_sem (beta_g=0.1)

共同 seeds：[0, 1, 2, 3, 4]（n=5）

| 指標 | 逐 seed 配對差值 | 配對 mean ± std | win count | 三級判讀 |
|---|---|---|---|---|
| task-IL final avg | -2.677, -0.538, +1.464, +1.650, -2.517 | -0.524 ± 2.079 pp | 2/5 | within noise |
| class-IL final avg | +3.113, +2.133, -1.558, +0.690, +1.408 | +1.157 ± 1.763 pp | 4/5 | directional, inconclusive |
| 跨任務洩漏率 | -5.461, -4.337, +3.021, -0.444, -5.855 | -2.615 ± 3.809 pp | 4/5 | directional, inconclusive |
| selection Jaccard | +0.008, -0.077, -0.046, +0.048, +0.006 | -0.012 ± 0.049 | 3/5 | within noise |
| group 配額 KL | -0.010, -0.000, -0.003, -0.008, -0.003 | -0.005 ± 0.004 | 5/5 | systematic |

**pre-registered 判準（原文，先於結果寫定）**：

- 通過 → win >= 4/5 且為正 → 報告為有效變體，論文寫「完整兩層 L_sem 帶來 __ pp 增益」
- 未通過 → <= 3/5 → 論文寫有數據的發現：在 classification 設定下 group 層語意先驗不提供增益，因為 q_tau 在任務內為常數，cos(g_j, q_tau) 退化為 8 個群組的靜態排序；HistoSelect 的 q 為逐問題變動，此差異來自 query 性質而非機制本身。

**落判依據**：

- task-IL final avg：配對 -0.52 pp、win 2/5（within noise）→ 不滿足
- class-IL final avg：配對 +1.16 pp、win 4/5（directional, inconclusive）→ 滿足
- 落判規則：兩軸皆須滿足 → FAIL
- ⚠️ **僅在 class-IL final avg 有效**，另一軸不動 —— 照實報告，不計為通過（PI 裁定 1）。

**判定：FAIL** → <= 3/5 → 論文寫有數據的發現：在 classification 設定下 group 層語意先驗不提供增益，因為 q_tau 在任務內為常數，cos(g_j, q_tau) 退化為 8 個群組的靜態排序；HistoSelect 的 q 為逐問題變動，此差異來自 query 性質而非機制本身。

## 總結

| 實驗 | 變因 | 判定 | 對架構圖的處置 |
|---|---|---|---|
| G5 | + E_t / B_t 狀態條件化 | **FAIL** | 移除 Panel E 與 "stateful"，改寫為 budgeted top-K selection |
| G4 | + q_tau 任務條件化 | **FAIL** | q_tau 標為 optional 或移除；寫成有機制解釋的 null |
| G3 | + group 層 L_sem (beta_g=0.1) | **FAIL** | 維持 patch-only；寫成有數據的發現 |

⚠️ G3 不論結果如何都**不回頭改主表** —— DR-007 pre-register 的是「用哪個 prior」，不是「用哪幾層」；本實驗是新增的消融維度。

逐 slide 預測：`outputs/exp2/arch/per_slide/*.json`（基準在 `outputs/exp2/hier2/per_slide/`）。

