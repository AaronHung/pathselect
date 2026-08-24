# G3 / G4 / G5 —— 架構完整性

圖上有三個元件在所有主結果中都是關閉或未實作的：E_t/B_t 狀態迴圈、q_tau 任務條件化、group 層 L_sem。本檔把它們逐一測出來 —— **有效就保留，無效就從圖上移除並寫成有數據的 limitation**。

通用設定：架構 hier（per_budget）、|M|=512、B=8、c=1、order=reverse、5 seeds、epochs 5、lr 1e-3、prior=discriminative、beta_s=0.1、beta_u=0.1、λ 全 1.0。**每個實驗只動一個變因**，對照組沿用 G1' 已有的 hier-A5 存檔（不重跑）。

**判準讀法**：任一準確率軸（task-IL 或 class-IL final avg）滿足即算通過。G5 的原文明寫「任一準確率軸」；G4 / G3 只寫「為正」，本報告對三者採同一讀法，因為通用設定已把主要指標定義為task-IL 與 class-IL 兩個 final avg。

## G5 前置：no-op 檢查

判準：state 開啟後，c=1 八輪與 c=8 一輪的選取集合必須不同。

設定：synthetic slide + 未訓練的隨機初始化模型，20 組、B=8、arch=hier。比較 c=1 跑八輪 vs c=8 跑一輪。

| use_state | 選取集合相同 | 選取順序相同 | 判讀 |
|---|---|---|---|
| False（現行主線） | 20/20 | 0/20 | **no-op** —— 與 CLAIMS C-01 一致 |
| True（G5） | 16/20 | 0/20 | **非 no-op**，state 確實進入計算 |

**判定：PASS** —— G5 可以進行。

⚠️ 未訓練權重下 state 的影響偏小（同集合 16/20）。這是下界，不是效果量的估計；訓練後應以真實 slide 重做一次。

產物：`outputs/exp2/arch/noop_check.json`

## 主表

| 臂 | 說明 | seeds | task-IL final avg | class-IL final avg | 跨任務洩漏率 | selection Jaccard | group 配額 KL |
|---|---|---|---|---|---|---|---|
| base | hier-A5 基準（G1'） | 5 | 0.9089 ± 0.0077 | 0.8119 ± 0.0162 | 0.1220 ± 0.0237 | 0.1419 ± 0.0482 | 0.0191 ± 0.0031 |
| G5 | + E_t / B_t 狀態條件化 | — | — | — | — | — | — |
| G4 | + q_tau 任務條件化 | — | — | — | — | — | — |
| G3 | + group 層 L_sem (beta_g=0.1) | — | — | — | — | — | — |

前兩欄是 pre-registered 的主要指標；後三欄為次要診斷。final avg 算全部 4 個 task；Jaccard 與配額 KL 只算前 3 個（CL 慣例）。

## 配對比較與落判

臂間比較一律**配對**（同 seed 相減）。win count 三級規則（DR-020）：5/5 = systematic、4/5 = directional inconclusive、≤3/5 = within noise。**不報 p 值**（DR-016）。

### G5　+ E_t / B_t 狀態條件化

⚠️ 尚未有資料。

### G4　+ q_tau 任務條件化

⚠️ 尚未有資料。

### G3　+ group 層 L_sem (beta_g=0.1)

⚠️ 尚未有資料。

## 總結

| 實驗 | 變因 | 判定 | 對架構圖的處置 |
|---|---|---|---|
| G5 | + E_t / B_t 狀態條件化 | **PENDING** | — |
| G4 | + q_tau 任務條件化 | **PENDING** | — |
| G3 | + group 層 L_sem (beta_g=0.1) | **PENDING** | — |

⚠️ G3 不論結果如何都**不回頭改主表** —— DR-007 pre-register 的是「用哪個 prior」，不是「用哪幾層」；本實驗是新增的消融維度。

逐 slide 預測：`outputs/exp2/arch/per_slide/*.json`（基準在 `outputs/exp2/hier2/per_slide/`）。

