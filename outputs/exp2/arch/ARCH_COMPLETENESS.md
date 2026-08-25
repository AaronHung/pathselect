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
| G5 | + E_t / B_t 狀態條件化 | 3 | 0.9050 ± 0.0150 | 0.8365 ± 0.0167 | 0.0924 ± 0.0272 | 0.1622 ± 0.0073 | 0.0291 ± 0.0132 |
| G4 | + q_tau 任務條件化 | — | — | — | — | — | — |
| G3 | + group 層 L_sem (beta_g=0.1) | — | — | — | — | — | — |

前兩欄是 pre-registered 的主要指標；後三欄為次要診斷。final avg 算全部 4 個 task；Jaccard 與配額 KL 只算前 3 個（CL 慣例）。

## 配對比較與落判

臂間比較一律**配對**（同 seed 相減）。win count 三級規則（DR-020）：5/5 = systematic、4/5 = directional inconclusive、≤3/5 = within noise。**不報 p 值**（DR-016）。

### G5　+ E_t / B_t 狀態條件化

共同 seeds：[0, 1, 2]（n=3）

| 指標 | 逐 seed 配對差值 | 配對 mean ± std | win count | 三級判讀 |
|---|---|---|---|---|
| task-IL final avg | -0.280, -2.720, +1.123 | -0.626 ± 1.945 pp | 1/3 | within noise |
| class-IL final avg | +4.489, +4.946, -1.295 | +2.713 ± 3.478 pp | 2/3 | directional, inconclusive |
| 跨任務洩漏率 | -3.914, -8.268, +2.418 | -3.255 ± 5.374 pp | 2/3 | directional, inconclusive |
| selection Jaccard | +0.052, -0.039, -0.022 | -0.003 ± 0.048 | 1/3 | within noise |
| group 配額 KL | +0.022, +0.016, -0.003 | +0.012 ± 0.013 | 1/3 | within noise |

**pre-registered 判準（原文，先於結果寫定）**：

- 通過 → win >= 4/5 且配對為正（任一準確率軸）→ stateful 成立，圖與 "Beyond HistoSelect" 保留該條，論文可寫 state-conditioned sequential acquisition（仍不得寫 "plan"，因為輪間 detach）
- 未通過 → <= 3/5 或為負 → 從架構圖移除 Panel E 與 "stateful" 一詞，改寫為 budgeted top-K selection under a shared frozen head，並在 limitation 說明：在此設定下狀態條件化未帶來可測增益。不得調參搶救。

**落判依據**：

- task-IL final avg：配對 -0.63 pp、win 1/3（within noise）→ 不滿足
- class-IL final avg：配對 +2.71 pp、win 2/3（directional, inconclusive）→ 不滿足
- 落判規則：任一軸滿足即通過 → FAIL

**判定：FAIL** → <= 3/5 或為負 → 從架構圖移除 Panel E 與 "stateful" 一詞，改寫為 budgeted top-K selection under a shared frozen head，並在 limitation 說明：在此設定下狀態條件化未帶來可測增益。不得調參搶救。

### G4　+ q_tau 任務條件化

⚠️ 尚未有資料。

### G3　+ group 層 L_sem (beta_g=0.1)

⚠️ 尚未有資料。

## 總結

| 實驗 | 變因 | 判定 | 對架構圖的處置 |
|---|---|---|---|
| G5 | + E_t / B_t 狀態條件化 | **FAIL** | 移除 Panel E 與 "stateful"，改寫為 budgeted top-K selection |
| G4 | + q_tau 任務條件化 | **PENDING** | — |
| G3 | + group 層 L_sem (beta_g=0.1) | **PENDING** | — |

⚠️ G3 不論結果如何都**不回頭改主表** —— DR-007 pre-register 的是「用哪個 prior」，不是「用哪幾層」；本實驗是新增的消融維度。

逐 slide 預測：`outputs/exp2/arch/per_slide/*.json`（基準在 `outputs/exp2/hier2/per_slide/`）。

