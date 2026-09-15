# DR-053 — 累積式頭下 hinge 的行為核查（只讀；第 3 項為注入式對照）

所有 run 都在 pod（CPU x86）；累積式 B5、固定頭 B6 與本卡的 hinge-8 對照同平台、同 seed。**Table 3 未動。**

## 1. 快照 C_old 與 U_new 遮罩逐筆比對（全長 run，Mac；`scripts/run_dr053_hinge8.py --trace`）

| run | continual_terms 呼叫數 | stored C_old == U_new 實際遮罩 == 由 entry.tau 推得的 C_old | 逐任務 entry 數（esca／rcc／brca） |
|---|---|---|---|
| B2 seed 0 | 10,765 | **10,765/10,765** | 4002／4776／1987 |
| A5 seed 0 | 10,765 | **10,765/10,765** | 4002／4776／1987 |

程式路徑：`selector/train.py::fill_memory` 以 stage 的 `class_mask` 算 u_old 並存 `class_mask_old`；`continual_terms` 以 `mask_logits(frozen_head(..., weighting="uniform"), entry.class_mask_old)` 算 U_new（同一個張量，無第二份拷貝）。逐筆 10,765/10,765 相同 → **無實作錯誤**。

## 2. hinge 觸發率（逐 stage；fold 1、seeds 0–4）

| 臂 | 頭 | stage 1 rcc | stage 2 brca | stage 3 lung |
|---|---|---|---|---|
| A5 | 累積式（B5） | 0.0100 ± 0.0061 | 0.0302 ± 0.0037 | 0.0605 ± 0.0107 |
| A5 | 固定頭 pod（B6） | 0.0169 ± 0.0106 | 0.0299 ± 0.0038 | 0.0727 ± 0.0041 |
| B2 | 累積式（B5） | 0.0349 ± 0.0071 | 0.0404 ± 0.0023 | 0.0875 ± 0.0067 |
| B2 | 固定頭 pod（B6） | 0.0329 ± 0.0149 | 0.0449 ± 0.0051 | 0.1121 ± 0.0055 |
| B2 | 累積式 + hinge 全 8 類（本卡第 3 項） | 0.0386 ± 0.0067 | 0.0523 ± 0.0062 | 0.1169 ± 0.0123 |

## 3. 快速對照：B2（只效用臂），hinge 的 U_old／U_new 改用全 8 類，其餘照累積式頭

| 設定 | class-IL | task-IL | 洩漏率 | Jaccard |
|---|---|---|---|---|
| 累積式（B5） | 64.42 ± 2.60 | 89.15 | 27.51 | 0.0361 |
| 累積式 + hinge 全 8 類（DR-053） | 80.97 ± 2.68 | 89.34 | 11.27 | 0.0434 |
| 固定頭 pod（B6） | 78.21 ± 3.20 | 90.08 | 13.92 | 0.0838 |

逐 seed：hinge-8 − 累積式 class-IL **+16.55 ± 5.08（5/5）**、洩漏率 -16.24；hinge-8 − 固定頭 pod +2.76 ± 5.59（3/5）。

## 4. 結論：**真實效應（協定後果），不是實作問題**

* 遮罩逐筆一致、hinge 只換成 8 類就回到 78–81 → 13.8 pp 的落差完全來自「U 的類別集合」。
* 機制：累積式頭下，任務 τ 的快照以 C_old（τ 時已見類別；esca 只有 2 類）量 U_old，之後 U_new 也遮罩到同一 C_old。hinge 因此只保住「證據在舊類別內部的鑑別力」（ESAD vs ESCC），對「證據是否仍能把舊任務與後來加入的類別分開」**沒有任何約束**——那正是 class-IL 的 8-way 混淆／洩漏。結果：task-IL 不變（89.2 vs 90.1）、洩漏率翻倍（13.9 → 27.5）、class-IL 掉 13.8 pp。
* A5 同理但有 replay（當前 C_t 的 CE）與 KD 補位，故只掉 5.3 pp；A3（無 hinge）不受影響（−1.6）。
* ⚠️ 第 3 項的 hinge-8 **不是協定合規的修法**：快照時用到未見類別的文字嵌入。合規的替代（建議另立 DR-054）：replay 時以**當前 C_t**重算 U_old —— `continual_terms` 已重載 Z，可由 `selected_from_entry` 取回 P_old，等權池化後在 C_t 上算 U_old(C_t)，與 U_new(C_t) 同口徑；不需未見類別、不改快照 schema。

來源：`outputs/exp3/dr053/TRACE_{B2,A5}_seed0.json`、`outputs/exp3/dr053_hinge8/per_slide/`（5 檔）、`outputs/exp3/ablation_{acc,fixed}/per_slide/`；`scripts/run_dr053_hinge8.py`（注入式，主線零改）。
