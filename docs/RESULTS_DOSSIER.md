# RESULTS DOSSIER — 證據總表

**2026-08-26 · Fable · 對照 `AaronHung/pathselect` main @ `626b923`**
**每個數字都是從 repo 產物檔讀出來核對的，不憑記憶；來源路徑附在各節末。**
**用途：PI 讀完一遍即可掌握全部思路。這不是論文，是論文背後的證據帳。**

> 這份與另一個 session 未鎖定的舊稿相比，修正了七處（列在 §8）。
> 核對過程中另發現九處 repo 治理文件彼此不一致（列在 §7），需要 Cursor 處理。

---

## 0. 怎麼讀這份文件

| 節 | 內容 | 讀它的目的 |
|---|---|---|
| §1 | 一頁摘要：論文站在九根柱子上 | 一頁看完全部主張與證據強度 |
| §2 | 方法的最終形狀（對照原始碼） | 論文 Method 節的事實基礎 |
| §3 | 基準設定與統計紀律 | 每個數字的口徑 |
| §4 | 證據鏈 4.1–4.11（依邏輯順序，非時間順序） | 每小節回答一個問題 |
| §5 | 寫作紅線（可宣稱 / 不可宣稱 / 必須誠實條款） | 寫作前最後一道閘 |
| §6 | 論文每一節的敘事骨架 | 預覽 |
| §7 | 核對時發現的 repo 不一致 → 交 Cursor | 治理層待辦 |
| §8 | 與另一稿的更正紀錄 | 為什麼要重核 |
| §9 | 尚未完成 | 下一步 |

**統計口徑全程只用三級（DR-020 / 憲法 §1.1）**：
**5/5 = systematic · 4/5 = directional, inconclusive · ≤3/5 = within noise。**
不報 p 值（DR-016）。臂間比較一律取 seed 配對後的 mean ± std。
3-seed 批次的 3/3 只能讀作「方向一致」，不能讀作「已定案」（§1.2）。

**符號**：J = 8（組織 group 數）｜|M| = Selection Memory 容量（主線 512）｜
B = 8（patch 預算）｜c = 1（每輪 chunk；無 state 時等價於一次選 8 個，見 C-01）｜
A1…A5 = CL 方法臂｜R1/R2 = 參照臂（不是 baseline）｜hier / flat = 階層 / 扁平選取器。

**三個口徑並報（DR-012）**：
- **task-IL**：argmax 只在該任務自己的兩個類別列（2-way，隨機 0.50）→ 任務內鑑別力
- **class-IL**：8-way argmax（隨機 0.125）→ 含跨任務混淆
- **跨任務洩漏率**：預測落到別的任務的類別列的比例。**head 全程凍結，洩漏 100% 可歸因於選取漂移**。

---

## 1. 一頁摘要：論文站在哪九根柱子上

| # | 主張 | 關鍵證據（數字｜win count） | 來源檔 |
|---|---|---|---|
| 1 | **學出來的選取遠勝隨機**；budget 曲線在 K=8 達峰 | 4 task × 7 K = **28/28 格** learned > random；各 task 平均 K=1 **+34.48 pp**、K=8 **+21.90**、K=64 **+20.14**；各 task 平均峰值 **0.8797 @ K=8**，K≥16 為 0.8738 | `outputs/exp0/BASELINES.md` |
| 2 | **序列訓練摧毀選取行為，三個軸一起崩**（SeqFT） | A1 臂（5 seeds）前三任務 class-IL forgetting 平均 **+51.93 pp**、task-IL **+11.09**；Jaccard **0.0015**（esca/rcc 低於隨機重疊的參照）；ΣU 由正轉負（esca 22.0 → −42.5）；雙 order 一致（S2） | `outputs/exp2/main/EXP2.md`、`outputs/exp2/seqft/SEQFT.md` |
| 3 | **frozen head 使遺忘可歸因、可量測**（洩漏率） | A1 洩漏率 **0.4408**（esca @T4 **0.7467**）；S2 中 esca 0.8889、rcc 0.6360 | `outputs/exp2/main/EXP2.md`、`outputs/exp2/seqft/TASK_IL.md` |
| 4 | **我方（A5）在階層下兩軸皆勝 replay-only（A3）** | task-IL **+3.28 ± 2.40（5/5）**、class-IL **+5.76 ± 3.42（5/5）**、洩漏率 **−2.42 ± 1.36（5/5）** | `outputs/exp2/hier2/HIER2.md` |
| 5 | **group-level distillation 有效，且與 patch-KD 分工不同** | hier-A5 − hier-A5nG：task-IL **+3.95 ± 2.50（5/5）**、class-IL **+3.71 ± 2.66（5/5）**；Jaccard +0.02（2/5，不動） | `outputs/exp2/hier2/HIER2.md` |
| 6 | **KD 與 replay 保存的是不同的東西** | B1（只 KD）Jaccard 0.0725 ≈ A3 0.0669，但洩漏率 **0.3231 = A5 的 3.2 倍**；A5 − B1 class-IL **+22.40（5/5）**、task-IL +6.21（4/5） | `outputs/exp2/ablation/B1_LANDING.md` |
| 7 | **記憶體：同容量全勝 + 4× 效率（下界）** | 5 個 \|M\| 全部 A5 > A3（task-IL），**4 個 systematic**（64:+4.90、128:+3.73、512:+3.28、1024:+2.49；256:+2.54 為 4/5）；A5@128 − A3@512 **+3.06 ± 2.80（5/5）** | `outputs/exp2/memory_hier/MEMORY_HIER.md` |
| 8 | **階層放大並穩定方法優勢** | A5 − A3 task-IL：flat-reverse +0.74（3/5）、flat-main +0.75 → hier **+3.28（5/5）**；A5 seed std task-IL ±1.41 → **±0.77** | `outputs/exp2/ORDER_DEPENDENCE.md`、`HIER2.md` |
| 9 | **兩個被移出主方法的元件各有一個 5/5 次要發現** | q_τ 使洩漏率 **−5.92 ± 4.19（5/5）**；group 層 L_sem 使配額 KL **−0.005 ± 0.004（5/5）**——但皆未轉化為準確率 | `outputs/exp2/arch/ARCH_COMPLETENESS.md` |

**方法學貢獻（貫穿全篇）**：四例「規格寫了、圖畫了，但從未生效」的元件（分屬架構 / 環境 / 實作 / 接線四層失效模式），全部靠「放大或替換元件、確認輸出改變」抓到（§4.11；CLAIMS C-26）。

**主張的整體形狀**：
> 「在哪裡看」是可學的（1）→ 序列學習會摧毀它（2），而且因為診斷頭凍結，這種摧毀是第一次可以被歸因而量測（3）→ 我們用階層選擇器 + Selection Memory + 兩層蒸餾與效用保存來保住它（4、5、6）→ 這比「把 replay 記憶體開大」有效也更穩（7、8）→ 而且我們誠實地把沒撐過判準的元件拿掉，同時報告它們留下的兩個 5/5 發現（9）。

---

## 2. 方法的最終形狀（對照 `selector/*.py` 核實）

**工作名稱**：*Continual Hierarchical Evidence Selection for Whole-Slide Images*
（正式方法名由 PI 定，但**不得**含 "Task-Conditioned"、"Stateful"、"Navigation"、"Zero-"、"Router"——C-01 / C-04 / DR-004）

**一句話**：一個共享的選擇器，在診斷任務序列到來時持續學習新任務該看哪裡，同時不忘記舊任務該看哪裡；**診斷模型全程凍結，任何效能變化只能歸因於選取行為**。

### 2.1 單一任務內的前向（推論路徑）

```
WSI patch 特徵 x_i（frozen CONCH，512-d，預抽；每張 slide 約 3,000–3,800 個 patch）
  ① 組織指派：8 條固定 tissue prompt → CONCH text 編碼 t_j；
     每個 patch 依 argmax cosine 指派到唯一一組 → 原型 g_j = 該組平均      （grouping.py）
  ② Group Selector F_g(g_j) → r_j                                            （model.py）
  ③ 配額：r 在非空組上 softmax × B → largest-remainder 取整，Σ b_j = B = 8
     ⚠️ 配額對「整個 budget」算（per_budget，DR-025），不是對 chunk        （allocation.py）
  ④ Patch Selector F_p(x_i) → s_i；每組內取 top-b_j                          （model.py）
  ⑤ 選出的 8 個 patch：softmax(top-K 分數) 加權池化 → L2 → 與 class-text {t_c}
     算 cosine → logits → argmax                                              （classifier.py）
```

- F_g、F_p 結構相同：`Linear(1537 → 256) → GELU → Linear(256 → 1)`。輸入是 `[feature ; q_τ ; e_t ; B̃_t]`，
  **主方法中 q_τ 與 (e_t, B̃_t) 兩個區塊填零**（維度不變、參數量不變）——這兩個閘門就是 G4 / G5 測的東西。
- top-K 用 straight-through：forward 是 hard mask，backward 走 softmax(s/T)。
- **沒有任何可訓練的診斷頭**：CONCH image/text encoder 與 class-text 全部凍結。可訓練的只有 F_g 與 F_p。
- 8 條組織 prompt 是 pre-registered 常數（tumor / stroma / lymphocyte / necrosis / normal epithelium / vessel / adipose / background），與任何 task 的 class label 無關，四個 task 共用。**沒有 k-means、沒有 clustering。**

### 2.2 跨任務（continual learning）

```
每個 task 只訓 LoRA（rank 4）掛在 F_g / F_p 上；task 結束後 merge：W_t = W_{t−1} + ΔW_t
→ 永遠只有一個 shared selector，推論不需要 task id，參數量不隨 task 數成長     （lora.py）

Selection Memory（|M| ≤ 512，reservoir）：每筆 = (τ, slide_id, e_t, B̃_t, r_old[J], cand_idx[≤256], s_old, u_old)
  不存 patch 特徵，只存 index，需要時從特徵檔重載                              （memory.py）

L_CL = λ_kd·L_KD + λ_eq·L_eq + λ_r·L_replay      （λ 全 1.0，從未調動）           （continual.py）
  L_KD     = KL(r_old ‖ r_new) + KL(s_old ‖ s_new)   ← 兩層蒸餾：group 層保配額、patch 層保 patch 身份
  L_eq     = hinge（新證據的效用 U 不得低於舊證據的 u_old，只罰退步）
  L_replay := L_diag 跑在從 M 取回的舊樣本上 —— replay 是資料機制，不是第三種 loss（DR-013）
```

### 2.3 完整目標

```
L = L_diag + β_s·L_sem + β_u·L_util + L_CL          β_s = β_u = 0.1
  L_diag  選出的證據要能分對（frozen head 上的 CE）
  L_sem   patch 層語意錨（discriminative：p_i = 1 − H(softmax(cos/T))/log C，DR-007）——弱正則
  L_util  以每個候選 patch 的 counterfactual gain u_i 為 anchor 的 KL（只在訓練時；需 label）
```

### 2.4 被證明移出主方法的三個元件（DR-043；判準寫死於 DR-039）

| 元件 | 原本圖上的位置 | 實驗 | 落判 | 處置 |
|---|---|---|---|---|
| E_t / B̃_t 狀態迴圈 | Panel E | G5 | FAIL（兩軸皆 ≤3/5） | 移除 Panel E；禁用 "stateful" / "sequential acquisition" |
| q_τ 任務條件化 | 兩個 selector 的輸入 | G4 | FAIL（task-IL 2/5；class-IL 4/5 單軸） | 移出主圖，標 optional / ablated；洩漏率發現要在正文有位置 |
| group 層 L_sem | "group & patch semantic IB" | G3 | FAIL（task-IL 2/5；class-IL 4/5 單軸） | 主方法維持 patch-only |

---

## 3. 基準設定與統計紀律

| 項目 | 設定 | 出處 |
|---|---|---|
| 任務 | TCGA **esca / rcc / brca / lung**（跨器官），各 2 類，8-way label space | `outputs/exp2/main/EXP2.md` |
| 資料量 | train 2273 張（esca 120、rcc 616、brca 763、lung 774）；test 279 張（**15 / 76 / 93 / 95**） | `outputs/exp1/diag/TASK_SEPARABILITY.md` |
| esca 解析度 | 一張 = **6.67 pp**；esca 上 < 6.67 pp 的差異一律標「不可區分」 | 憲法 §1.5 |
| 任務順序 | 主 order = **reverse**（esca → rcc → brca → lung），5 seeds；main order（lung → brca → rcc → esca）A3/A5 5 seeds、其餘 3 seeds | `outputs/exp2/ORDER_DEPENDENCE.md` |
| 操作點 | B = 8、c = 1（DR-005；budget 曲線峰值）；主線 \|M\| = 512 | `docs/ledger/DR-005.md` |
| 訓練 | epochs 5、lr 1e-3、β_s = β_u = 0.1、λ_kd = λ_eq = λ_r = 1.0、replay_k = 1 —— **全部從未調動** | 各 `EXP2.md` 表頭 |
| 聚合權重 | 主線 softmax（與訓練一致，DR-006）；等權保留為 selection-only ablation。⚠️ DR-006 明說「非依數值選定」（L3 在 B=8 的 softmax 0.8692 < 等權 0.8778，仍照選） | `outputs/exp1/stage1/RESULTS.md` |
| 指標口徑 | final avg 算全部 4 個 task；**forgetting / Jaccard / quota KL 只算前 3 個**（最後一個 task 兩時點相同，算進去只有稀釋作用）；洩漏率算全部 4 個 | 憲法 §3.1 |
| 統計 | 三級 win count；不報 p 值；n<5 批次自動加警語；跨批次配對只用共同 seeds（§1.3；兩次差點被共同子集騙到） | 憲法 §1 |
| 存檔 | 每個實驗逐 slide 存 `per_slide/*.json`（含 selected_idx、group_quota、pred_*、utility）；報告由腳本重算，不手抄 | 憲法 §2.1 |

**R1 / R2 不是 baseline（DR-011；Sol 獨立補充）**：R1 = per-task specialist（每 task 只用自己 120–774 張資料），A3/A5 經 replay 實質可及跨任務資料，所以 **R1 在 task-IL 上不是上界**；其參考意義在 class-IL（0.8777 全場最高）。R2 = joint offline reference（無順序），**不可用來論遺忘**（joint 只證 multi-task interference）。

---

## 4. 證據鏈

### 4.1 立足點：學出來的選取真的有用嗎？（Exp 0，B 曲線）

**問題**：如果 learned selector 贏不了隨機抽樣，整條線不存在。

**答案**：大幅贏。4 task × 7 個 K = **28 格無一輸**（最小差距 +8.42 pp）；對 grid 也是 28/28；對 similarity 27/28。

| K | random | grid | similarity | learned-flat | learned − random |
|---|---|---|---|---|---|
| 1 | 0.4868 | 0.2914 | 0.7699 | 0.8316 | **+34.48** |
| 8 | 0.6607 | 0.6269 | 0.8119 | **0.8797** | +21.90 |
| 16 | 0.6674 | 0.6951 | 0.8119 | 0.8738 | +20.64 |
| 64 | 0.6724 | 0.6824 | 0.7978 | 0.8738 | +20.14 |

（各 task 平均；random 為 5 seeds mean；learned-flat 用 v9 skill bank 直推論。）

**附帶發現一（budget 有最佳點）**：各 task 平均 **K=8 達峰 0.8797**，K=16/32/64 皆 0.8738 —— 超過 8 之後多加 patch 不再改善（略降 0.59 pp）。這證成 budgeted selection 本身，也定了操作點 B=8（DR-005）。HistoSelect 的 token 消融（5k 峰、10k 降）是同性現象。
⚠️ 誠實邊界：這條曲線來自 Exp 0 的 v9 skill bank；Exp 1 重訓的 L3 只掃到 B=16，其 B=8 → B=16 為 0.8692 → 0.8765（softmax），沒有出現下降。論文寫「K=8 為 Exp 0 掃出的峰值、其後不再改善」即可，不要寫成普遍定律。

**附帶發現二（軟預算）**：softmax 加權使 budget 是軟的。learned-flat 的 eff_K/K 從 K=1 的 1.000 掉到 K=64 的 **0.375**（similarity 的 softmax 在數值上接近等權，eff_K/K ≈ 1.000）→ 四條線中只有 learned 真在加權。重訓後的 L3 在 B=8 的 eff_K = **7.62–7.96**（近等權），所以操作點上「加權 vs 選取」的混淆很小；所有 learned 方法仍全報 softmax 與 uniform 兩欄。

**必須誠實條款**：rcc @K=1 是唯一 learned 輸 similarity 的格子（0.9079 vs 0.9211）；random / grid 只能等權（沒有分數），scored 與 unscored 的差距同時含「選得準」與「權重政策」兩個因子；grid 不是真正的 spatial uniform（特徵檔無座標）。

來源：`outputs/exp0/BASELINES.md`、`outputs/exp0/EFFECTIVE_K.md`、`outputs/exp1/stage1/RESULTS.md`、`outputs/exp1/stage1/DIAGNOSTICS.md`

### 4.2 任務結構：為什麼 task conditioning 在這裡沒有作用空間（S1 probe）

linear probe 預測 4-way task id（train 訓練、test 評估；多數類基準 0.3405）：

| 輸入表徵 | test accuracy |
|---|---|
| slide 平均 CONCH patch 特徵（512-d） | **0.9821** |
| 8 個 group prototype 串接（4096-d，= F_g 的實際輸入） | **0.9857** |
| 單一 patch 特徵（512-d；test 端全部 878,117 個 patch） | **0.8930**（slide 多數決 0.9713） |

**結論**：跨器官任務的身分從視覺特徵免費可得 → q_τ 提供的是模型已有的資訊。其例外是 **benchmark 的結構性質**，不是方法缺陷。這一發現：(a) 事後解釋了 Exp 1 Gate 1（各 task 平均 L4 − L3 ≈ −6.0 pp；DR-008 記為 −6.02，且該比較另有 per-task vs joint 訓練的混淆：L3 為 per-task 訓練、L4 為 joint）與後來 G4 的 null；(b) 給出下一篇的設定：同器官多任務 —— 老師原始的任務序列（GRAVEYARD G-05 / SEEDS S-02）；(c) 產出一個可能可移植的工具：任何 task-conditioned 工作都值得先做 20 分鐘 probe（S-15）。

來源：`outputs/exp1/diag/TASK_SEPARABILITY.md`、`outputs/exp1/stage1/RESULTS.md`

### 4.3 遺忘存在，而且是三個層次的（論文的動機實驗）

有兩份 SeqFT 資料，論文要分清楚用哪一份（DR-014）：
- **Exp 2 的 A1 臂**：5 seeds、reverse、β_u = 0.1（與所有 CL 臂同一個 within-task 目標）→ **這才是隔離 CL 貢獻的正式 baseline**，主表用它。
- **S2 / S3**：3 seeds、CE-only（β_u = 0）、**雙 order** → 標 preliminary，用來證明「雙 order 一致」。A1 在 β_u = 0 下與 S2 逐筆位元相同（1707 筆零差異，`scripts/verify_a1_matches_s2.py`）。

**A1 臂（5 seeds、reverse、逐 task、學完 T4 後）**

| task | class-IL forgetting | task-IL forgetting | 洩漏率 @T4 | Jaccard | 隨機重疊參照* | ΣU 學完 → @T4 |
|---|---|---|---|---|---|---|
| esca | **+54.67** | +13.33 | **0.7467** | 0.0000 | 0.00225 | 22.0 → **−42.5** |
| rcc | **+55.53** | +15.00 | 0.5500 | 0.0005 | 0.00220 | 142.7 → **−138.7** |
| brca | +45.59 | +4.95 | 0.4645 | 0.0041 | 0.00186 | 167.6 → **−147.7** |

\* 隨機參照 = 從 n 個 patch 隨機抽兩次 8 個的期望 Jaccard，逐 slide 算後平均（`TASK_IL.md` 口徑；`EXP2.md` 表中的 0.00106/0.00123/0.00129 是用 task 平均 n 算的，另一個口徑，見 §7-④）。esca / rcc 的 Jaccard **低於隨機參照** = 兩個時點選到的 patch 比隨機抽兩次還不重疊；brca 高於參照。

**三個軸各說一件事**：
- **A1 準確率**：class-IL 崩潰（+45 ~ +56 pp），但大部分是洩漏（下一節）；task-IL 的 +13 ~ +15 pp（esca/rcc）才是**真實的任務內遺忘**。
- **A2 選取行為**：Jaccard → 0，低於隨機重疊 → 選取焦點過去**完全無關**。
- **A3 效用**：ΣU 由正變負 → 新選的證據會讓預測**推向錯誤類別**，比什麼都不看還糟（反向支撐）。

**S2 / S3（3 seeds、CE-only）的雙 order 對照**：reverse 前三任務 class-IL forgetting 平均 **+54.82 pp**、main **+49.39 pp**；task-IL forgetting reverse esca +17.78 / rcc +22.37 / brca +2.87，main lung +18.25 / brca +6.09 / rcc +22.37；洩漏率 @T4 reverse esca **0.8889** / rcc 0.6360，main lung 0.5579 / rcc 0.6009。**事前預測寫入報告且對照後不改**（預測「accuracy 層 forgetting 可能較輕」，實際反而重 —— 我認為是可分離性讓隱式路由「可能」，但序列訓練不給「保留」任何誘因）。

**為什麼兩個口徑都要報**：只報 class-IL 會被批誇大（多為洩漏），只報 task-IL 會漏掉我們獨有的可測項 —— **frozen head 使洩漏 100% 歸因於選取漂移**，訓練分類器的 CL 工作做不到這點（DR-012；這要寫成 contribution，不是缺陷）。

⚠️ SEEDS S-03 寫的「brca 6/6 方向一致」（行為保留度預測準確率保留度）**在 repo 裡沒有對應產物**，本表不引用；要寫進論文須先重算並 commit（§7-⑤）。可安全寫的觀察：在兩個 order 中 brca 都是遺忘最小的 task，且其 Jaccard 都高於隨機參照。

來源：`outputs/exp2/main/EXP2.md`（A1 逐 task 表）、`outputs/exp2/seqft/SEQFT.md`、`outputs/exp2/seqft/TASK_IL.md`、`docs/ledger/DR-014.md`

### 4.4 CL 主表（flat 架構、reverse、5 seeds）—— 單調在這裡發生

| 臂 | task-IL | class-IL | 洩漏率 | Jaccard | 一句話 |
|---|---|---|---|---|---|
| A1 SeqFT | 0.8086 | 0.4774 | 0.4408 | 0.0015 | 無任何 CL 機制 |
| A2 + LoRA merge | 0.8105 | 0.4466 | 0.4756 | 0.0017 | **≈ A1**；merge 是 substrate，不是機制 |
| A3 + Replay | 0.9073 | 0.7778 | 0.1421 | 0.0669 | **真正的台階**；最強的簡單 baseline |
| A4 + Replay + KD | 0.8969 | 0.7972 | 0.1242 | **0.1365** | 中間消融 |
| **A5 Ours（Replay+KD+eq）** | **0.9147** | **0.8239** | **0.1005** | 0.1294 | |
| R1 per-task specialist | 0.9027 | 0.8777 | 0.0284 | 1.0 | 參照（不是上界，task-IL） |
| R2 joint offline | 0.8498 | 0.7789 | 0.1176 | 1.0 | 參照，不可論遺忘 |

**配對（逐 seed 相減）**：

| 對照 | task-IL | class-IL | 洩漏率 | Jaccard |
|---|---|---|---|---|
| A5 − A3 | +0.74 ± 1.93（**3/5，within noise**） | **+4.61 ± 2.29（5/5）** | **−4.16 ± 2.96（5/5）** | +0.06（4/5） |
| A5 − A1 | **+10.60（5/5）** | **+34.64（5/5）** | **−34.03（5/5）** | +0.13（5/5） |
| A4 − A3 | **−1.04 ± 0.15（0/5）** | +1.94（3/5） | −1.80（4/5） | **+0.07（5/5）** |
| A5 − A4 | +1.78（5/5） | +2.67（4/5） | −2.37（4/5） | −0.01（2/5） |

**單調（DR-015，flat 時代；後被階層版在 task-IL 上取代，見 4.6）**：replay 回復準確率，KD + eq 回復選取行為。flat 下 task-IL 不得宣稱勝出。
- **A4 − A3 是一個乾淨的 trade-off**：KD 有極穩定的小幅準確率代價（−1.04 ± 0.15，五個 seed 全在 −0.86 ~ −1.21 之間），換取行為保存（Jaccard 5/5）。（SEEDS S-07）
- **A2 − A1**：A2 − A1 reverse 在 5 seeds 為 −3.09 pp（3/5），逐 seed 橫跨 43 pp；曾被共同子集（seeds 0–2 皆正）誤判為「順序效應」，已撤回（DR-024；憲法 §1.3）。

**順序依賴（獨立成節，不當附註帶過）**：
- A3 − A1 **跨順序穩定**：task-IL +12.76 / +14.21、class-IL +36.90 / +34.28（reverse / main）→ 本專案最穩固的結果。
- A5 − A3 task-IL **跨順序穩定但微小**：+0.74 / +0.75；class-IL 是「reverse 有效、main 無效」（+4.61 / −0.28），不是翻轉。
- **A5 − A4 是唯一乾淨的翻轉**：task-IL +2.43 / −0.67，class-IL +2.04 / −1.88 → **eq 的貢獻非跨順序穩定**（論文必須寫；C-10 條件 2）。目前只有 flat 證據。

來源：`outputs/exp2/main/EXP2.md`、`outputs/exp2/ORDER_DEPENDENCE.md`、`docs/ledger/DR-015.md`、`DR-024.md`

### 4.5 元件消融：KD 與 replay 保存的是不同的東西（B1 / B2，5 seeds；E3，3 seeds）

| | B1 只 KD | B2 只 eq | A3 只 replay | A5 全開 |
|---|---|---|---|---|
| task-IL | 0.8525 | 0.8846 | 0.9073 | 0.9147 |
| class-IL | **0.5999 ± 0.1042** | 0.8089 | 0.7778 | 0.8239 |
| 洩漏率 | **0.3231** | **0.0906** | 0.1421 | 0.1005 |
| Jaccard | 0.0725 | 0.0874 | 0.0669 | 0.1294 |

（B1 / B2 都仍使用 replay 這個**資料機制**取回舊樣本，只是不在下游目標的損失上加 —— DR-013。）

**B1 的落點分析（機制證據，DR-033）**：seed 4 的 rcc 有 **59/76 張被推到 lung 的兩列**（53 LUAD + 6 LUSC），只有 17 張落在自己的類別列，該格 class-IL = 0.1711（低於 8 類隨機 0.125）**不是亂猜**（亂猜會散在 8 個），而同一批 slide 的 task-IL = 0.5526。→ **KD 保住了選取行為（Jaccard 0.0725 ≈ A3 的 0.0669），卻沒有保住證據的任務歸屬（洩漏 3.2× A5）**。任務內鑑別力大致還在（A5 − B1 task-IL +6.21，僅 4/5），跨任務歸屬崩了（class-IL +22.40，5/5）。→ 兩者不可互相取代：replay 補的正是任務歸屬。**必須**：B1 同時是最不穩的臂（class-IL ±0.1042，全場最大）。

**B2 的教訓（憲法 §1.3 第二例）**：3 seeds 時 B2 > A5（seeds 0–2 的 A5 − B2 class-IL 全在負），補到 5 seeds 翻盤（A5 − B2 = +1.49 ± 4.17，2/5）→ 差點用 3 seeds 把方法簡化掉一項（H2：B2 記憶體曲線，因此未跑）。**紀律的實地效果，一年不只一次防止錯誤。** A5 維持完整三項。另一觀察：B2 的 l_eq fire rate 0.1114 ≈ A5 0.0740 的 1.5 倍（觀察，不夠因果宣稱；`ORDER_DEPENDENCE.md` 寫 0.1142，疑為 3-seed 舊值，見 §7-⑦）。

**E3（β_u 消融，A5，3 seeds，seeds 0–2 配對）**：β_u = 0.1 → β_u = 0 在 task-IL **+1.12 ± 0.50（3/3）**；class-IL −0.87（1/3）、洩漏率 +0.81（1/3）、Jaccard −0.07（1/3）皆 within noise。→ L_util 定位為輔助項：只在 task-IL 有小幅、方向一致的效果，**3 seeds，不宜用力宣稱**。

來源：`outputs/exp2/ablation/EXP2.md`、`outputs/exp2/ablation/B1_LANDING.md`、`outputs/exp2/ablation/BETA_U.md`、`outputs/exp2/ablation_bu0/EXP2.md`、`scripts/decide_h2.py`

### 4.6 架構：階層如何被採用（G1 失敗 → G1' 成功）

**判準先寫死（DR-021）**：階層 ≥ flat 才在後續內採用階層；顯著劣於 flat 就撤掉。只動階層（不同時開 q_τ 與 state；Gate 1 教訓：同時開多件事就無法歸因）。

**G1（per_chunk 配額）失敗**：hier-A5 − flat-A5 class-IL = **−18.69 ± 10.41（0/5）**，當中叫停不是壞事。但結構性診斷顯示測到的不是階層：c=1 且無 state 使 r 逐輪不變，per_chunk 配額每輪只有一個名額、必然全給 argmax(r) → 退化為「單組內取該組 top-8」——**84.5% 的 slide 只用一組**（HIER.md 結構性診斷表：{1: 1179, 2: 193, 3: 20, 4: 3} / 1395 張 → 1179/1395 = 84.5%，平均 1.17 組）。⚠️ CLAIMS C-02 / 憲法 §3.6b 寫的是 88.6%，repo 裡沒有那個數字的產物，見 §7-⑨。CONTRACT-1 把「對 budget 配額」寫成「對 chunk 配額」是設計錯誤（DR-025）。產出憲法 §2.5：隔離單一變因前，先檢查在不變因條件下該機制是否退化。

**G1'（per_budget 配額；DR-021 判準原封不動沿用）**：單組比例 **2.4%**，平均用到 **4.31** 組（flat 為 2.37 組）→ 階層有作用空間，判準結果可採用。

| 配對（5 seeds） | task-IL | class-IL | 洩漏率 |
|---|---|---|---|
| hier-A5 − flat-A5 | −0.58 ± 1.63（1/5，within noise） | −1.20 ± 2.45（1/5，within noise） | **+2.15 ± 1.96（0/5；hier 洩漏系統性較高）** |
| hier-A3 − flat-A3 | **−3.11 ± 3.42**（1/5） | −2.35 ± 2.34（1/5） | +0.40（3/5） |
| **hier-A5 − hier-A3** | **+3.28 ± 2.40（5/5）** | **+5.76 ± 3.42（5/5）** | **−2.42 ± 1.36（5/5）** |

**三個補充**：
1. **階層採用為主線**（DR-029）。追加論據：hier-A5 的 seed 標準差變小（task-IL ±1.41 → **±0.77**、class-IL ±2.84 → ±1.62）→ **階層讓方法更穩定**。
2. **task-IL 主張解禁**（DR-029 supersede DR-015；DR-037）：flat +0.74 / +0.75（雙 order 一致但微小）→ hier **+3.28（5/5）**。**階層把穩定的微小優勢放大成 systematic** —— 這是採用階層最強的論據，優於 DR-021 原本的「可解釋配額」。
3. **誠實的來源句（必寫；C-10 條件 1）**：差距擴大同時來自 A5 更穩（正向）**與 replay-only 在階層下退化**（hier-A3 − flat-A3 = −3.11，負向）；A5 自身在階層下是 −0.58（within noise）。**不得只報前half。**

**兩個補釘的誠實條款**：
- hier-A5 的洩漏率比 flat-A5 **系統性高 +2.15 pp（0/5）**（12.20 vs 10.05）。階層在準確率上與 flat 相當、在洩漏率上略差 —— 論文報 hier 洩漏率時要用 hier 自己的數字，不能沿用 flat 的 10.05。
- hier-A3 − flat-A3 有一個口徑註記：flat 下 A3 的 F_g 無梯度、停在初始值，hier 下才訓練；兩者不是同一個 baseline，不得單獨拿來宣稱階層的效果（HIER2.md「跨模式比較的限制」）。

來源：`outputs/exp2/hier/HIER.md`（G1）、`outputs/exp2/hier2/HIER2.md`（G1'）、`outputs/exp2/ORDER_DEPENDENCE.md`、`docs/ledger/DR-021/025/029/037.md`

### 4.7 group-level distillation：首次有效驗證（G1'-b，DR-035）

**前史**：flat 下曾擾動 F_g 應證「放大 5 倍全隨機化權重網路，選取**位元相同**」，反向對照：同樣擾動在 hier 下會改變選取（排除「擾動本身無效」）→ L_KD 的 group 項在全部 flat 實驗中作用恰為零，**從未被測試過**。在退化階層（G1）下首測「未顯示效果」（DR-022）—— 那不奇怪：每張 slide 只用一組，配額分佈本來就沒東西可保存。

**在通過結構性把關的階層上重測**：hier-A5 − hier-A5nG（A5nG = `group_weight = 0`，group 項完全不計算）：

| 指標 | 配對 | win |
|---|---|---|
| task-IL | **+3.95 ± 2.50** | **5/5** |
| class-IL | **+3.71 ± 2.66** | **5/5** |
| 洩漏率 | −1.05 ± 2.66 | 3/5 |
| Jaccard | +0.02 ± 0.06 | 2/5 |

**兩層蒸餾的分工（架構圖 Panel I 的直接證據）**：group-KD 保**組織層配額分佈** → 顯現在準確率；patch-KD 保**具體 patch 身份** → 顯現在 Jaccard（A4 − A3 Jaccard 5/5，4.4）。若兩層保同一件事，拿掉一層應該同時傷兩個指標；實測拿掉 group 項只傷準確率、不動 Jaccard —— 配額變了，但選誰沒變。

來源：`outputs/exp2/hier2/HIER2.md`、`tests/test_arch_switch.py`、`docs/ledger/DR-022.md`（SUPERSEDED）、`DR-035.md`

### 4.8 L_sem：弱正則；選 discriminative 是為了避開 simple similarity（G2）

三臂（hier，5 seeds）：

| prior | task-IL | class-IL | 洩漏率 | Jaccard |
|---|---|---|---|---|
| none | 88.92 ± 2.35 | 81.20 ± 2.38 | 10.57 | 0.1387 |
| max_sim | 90.13 ± 0.52 | 82.27 ± 2.10 | **9.74** | 0.1657 |
| discriminative（主線） | 90.89 ± 0.77 | 81.19 ± 1.62 | 12.20 | 0.1419 |

配對：class-IL 三組全部 within noise（disc − none = **−0.02 ± 0.89，3/5**）；task-IL disc − max_sim = **+0.76 ± 0.75（5/5，但量級極小）**；disc − none = +1.98（4/5）。

**寫法（DR-036 → DR-038）**：
- ✅ 「**在階層架構下**，語意錨的移除不損害準確率。」（L_sem 作為弱正則，與 β_s = 0.1 刻意設小一致）。
- ❌ 不得宣稱 L_sem 改善準確率（C-25）。
- ❌ 已刪除「HistoSelect 的貢獻在於分組結構而非語意錨」—— 循環論證（我們正是在「分組結構壓過 patch 分數」的架構裡測 patch 層錨）。這句在 DR-038 裁定後仍殘留在 CLAIMS.md，2026-08-26 才被 Cursor 清掉 ——「裁定寫了但沒執行」的第一例。
- **必須**：max_sim 洩漏率最低（9.74 vs discriminative 12.20）；「只測了兩個軸」（group 層 L_sem 從未實作，見 4.11）；範圍限定「在階層架構下」（L_sem 只錨定 s；階層下配額由 r 決定，槓桿被稀釋；flat 無 prior 消融資料，不可外推）。
- 選 discriminative 的理由不變且獲新支持（DR-007）：max_sim 就是老師曾批評的 simple similarity；兩者效果相當，我們選了不是 similarity 的那個。

來源：`outputs/exp2/prior/PRIOR.md`、`docs/ledger/DR-007/036/038.md`

### 4.9 記憶體效率：論文的防禦主軸（階層版 E1）

最強的攻擊是「replay 做了全部的事，把 |M| 開大就好」。回答分主從（DR-042 修訂 A）：

**主要主張（同容量，一句話講完，證據最強）**：**在所有測試的記憶體預算（64–1024）下，A5 在 task-IL 上皆優於 replay-only；5 個容量中 4 個 systematic，含超出契約的 1024。**

| \|M\| | A3 task-IL | A5 task-IL | A5 − A3（配對） | win |
|---|---|---|---|---|
| 64 | 83.83 ± 3.29 | 88.73 ± 1.06 | **+4.90 ± 2.67** | 5/5 |
| 128 | 86.94 ± 2.30 | 90.67 ± 0.18 | **+3.73 ± 2.18** | 5/5 |
| 256 | 87.47 ± 2.71 | 90.01 ± 1.16 | +2.54 ± 3.35 | 4/5 |
| 512 | 87.61 ± 2.88 | 90.89 ± 0.77 | **+3.28 ± 2.40** | 5/5 |
| 1024 | 88.58 ± 1.30 | 91.08 ± 1.75 | **+2.49 ± 2.42** | 5/5 |

→ A3 拿到 1024（已超出 CONTRACT-3 的 512）在 task-IL 仍然輸；**稀缺端優勢最大**（64 格 +4.90）；A5 對預算穩健（task-IL 跨 |M| 標準差 A5 0.95 pp vs A3 1.81 pp）。結構性把關逐 |M| 皆過（單組比例 6.8%–34.5%，平均 3.15–4.14 組）。

**輔助主張（跨容量，配對而非均值）**：

| 比較 | task-IL 配對 | win | 倍數 | 支持？ |
|---|---|---|---|---|
| A5@64 − A3@1024 | +0.15 ± 1.68 | 3/5 | 16× | ❌ |
| A5@128 − A3@1024 | +2.09 ± 1.25 | 4/5 | 8× | ❌ |
| A5@64 − A3@512 | +1.12 ± 3.47 | 2/5 | 8× | ❌ |
| **A5@128 − A3@512** | **+3.06 ± 2.80** | **5/5** | **4×** | ✅ |

→ **測試範圍內 4× 記憶體效率**。引用時**三個限定必帶**：(a) 錨點 A3@512；對 A3@1024 只有 4/5，8× 不成立；(b) **A3 曲線未飽和**（task-IL 一路升到 88.58@1024，class-IL 到 1024 仍在升）→ 4× 是**下界**，不是 A3 需求的上界；(c) 限 task-IL；class-IL 在 1024 落入雜訊（+1.88 ± 5.69，2/5）。
**8× 已撤回**（DR-042）：原錨點 128 的 class-IL 配對只有 4/5 且 std(7.71) > mean(7.72)。

**class-IL 另報**：64 +4.59（4/5）、128 +7.72（4/5）、256 +5.84（4/5）、512 +5.76（5/5）、1024 +1.88（2/5）。

**flat 版（限適用；DR-019 已 SUPERSEDED）**：2× 效率（A5@128 class-IL 0.8253 ≈ A3 全域最佳 A3@256 0.8203）；A3 在 flat 有 256 → 512 的 systematic 下滑（+4.25 ± 2.27，5/5），已排除 replay 強度混淆（replay_k = 1 固定、與 |M| 無關、1:1 batch），這個原因是從未觸發（未解），故列為 open question（SEEDS S-01）。⚠️ flat 的「A3 在 256 後不再改善」防禦**在階層版不成立**，不得沿用。

來源：`outputs/exp2/memory_hier/MEMORY_HIER.md`、`outputs/exp2/memory/MEMORY.md`、`docs/ledger/DR-019.md`（SUPERSEDED）、`DR-042.md`

### 4.10 架構完整性（G3 / G4 / G5）：三個元件依事先判準移出主方法

判準先寫死（DR-039 + 修訂 A，皆早於任何結果）：G5 任一準確率軸 ≥4/5 且為正即通過（決定的是描述性用字 "stateful"）；G4 / G3 **兩軸皆須** ≥4/5 且為正（決定的是效能宣稱）；單軸通過而另一軸不動 → 照實報「僅在 X 軸有效」，不計為通過。對照組沿用 G1' 的 hier-A5 存檔；每個實驗只動一個變因。

| 實驗 | task-IL 配對 | class-IL 配對 | 判定 | 處置 |
|---|---|---|---|---|
| G5 + E_t/B̃_t 狀態 | −0.50 ± 2.21（2/5） | +1.05 ± 3.96（3/5） | **FAIL**（兩軸皆不足） | 移除 Panel E 與 "stateful"；改述 budgeted top-K selection under a shared frozen head |
| G4 + q_τ | +0.31 ± 2.75（2/5） | +5.85 ± 3.52（4/5，單軸） | **FAIL** | q_τ 移出主圖、標 optional；寫成有機制解釋的 null（S1） |
| G3 + group 層 L_sem（β_g = 0.1） | −0.52 ± 2.08（2/5） | +1.16 ± 1.76（4/5，單軸） | **FAIL** | 主方法維持 patch-only；寫成有數據的發現 |

**但兩個次要指標為 5/5，必須寫，而且是發現不是安慰（C-28 / C-29）**：
- **q_τ 使跨任務洩漏率 −5.92 ± 4.19 pp（5/5）**：條件化不改善準確率（任務身分已 98.6% 可讀），但系統性減少證據漏到別的任務 —— 它影響「選什麼」但未能轉為「解得好」。head 凍結，所以這 100% 是選取變了。
- **group 層 L_sem 使配額 KL −0.005 ± 0.004（5/5）**：同樣——保配額分佈、不動準確率（與 group-KD 同軸，4.7）。
- 兩者引用時**必須**帶「未能轉化為準確率增益」。「不能宣稱準確率增益」與「完全沒有效果」是兩件事，**不要把後者寫進 limitation**（C-04）。

**G5 的 no-op 檢查與訓練後重測**：state OFF 時 c=1 八輪 ≡ c=8 一輪（20/20 位元相同；C-01 的根據）；state ON 未訓練 synthetic 只 4/20 改變 —— PI 裁定這是**下界不是效果量**。訓練後模型（G5 seed 0）在 279 張真實 test slide 重做：state ON 有 **279/279（100%）選取集合改變，平均只剩 2.81/8 個 patch 重疊**（OFF 為 8.00/8）。**state 幾乎重寫了整個選取，準確率卻兩軸皆未達判準** —— 本專案對「機制生效 ≠ 機制有用」最強的一個實例；不改變落判，反而讓移除 "stateful" 更站得住。模型一致性：跳過各 stage 評估所訓練的模型與正式 G5 逐 slide 選取 279/279 相同。

**G4 差點成為構造性 null**：`run_exp2.Ctx.q0 = zeros(512)` 而 `use_query=False` 的實作是填零 → 只把開關打開而不接 `TaskQueryBank`，輸入與關閉時位元相同（實測 20/20）；接上的 q_τ 得 16/20。原因由 Cursor 在啟動前實測抓到（DR-040 / DR-041 → 憲法 §2.9 升格為必須）。既有結果不受影響（run_exp1 用真 query；run_exp2 從未開過 use_query）。

來源：`outputs/exp2/arch/ARCH_COMPLETENESS.md`、`outputs/exp2/arch/noop_check.json`、`outputs/exp2/arch/noop_trained.json`、`docs/ledger/DR-039/040/041/043.md`

### 4.11 四個「看起來在跑但其實沒有」的元件（方法學貢獻；CLAIMS C-26）

| 元件 | 失效層 | 怎麼發現 | 結局 |
|---|---|---|---|
| group-level KD | **架構**（flat 下無作用空間） | 擾動 F_g 為全隨機網路 → 選取位元相同；hier 下同樣對照有變 | 修配額口徑後**有效**（+3.95 / +3.71，5/5） |
| q_τ 資訊 | **環境**（跨器官可分離 98.2 / 98.6%） | S1 probe | 移出主方法；洩漏率 −5.92（5/5）保留為發現 |
| group 層 L_sem | **實作**（從未寫；`l_sem()` 沒有 r 參數、沒有第二個 KL） | 放大 g_j 五倍 → L_sem 位元不變（0.0226687789 → 0.0226687789）；放大 patch 特徵才變（→ 0.020234） | 補寫兩層版做 G3 消融（FAIL；主方法 patch-only） |
| q_τ 接線 | **接線**（有實作、有開關、開關也讀得到，但注入點餵零向量） | 開/關對照實跑：zeros 20/20 相同、真 q_τ 16/20 | G4 啟動前修正，無數字需撤回 |

**共同教訓**：架構圖與規格書不是實作的證據。前三層的 code 或設定**可能**發現，**第四層只有把開/關兩條路跑出來比對才會現形**。辨識武器只有一種 —— 放大或替換元件、確認輸出改變（憲法 §2.2 / §2.6 / §2.9）。而且三個元件（G3/G4/G5）**都通過了生效性檢查、都沒撐過效能判準**：§2.9 管「活著」，pre-registered 判準管「有用」，**不可互代**。四次都不是 PI 發現的（DR-041）。

來源：`docs/CLAIMS.md` C-26、`docs/PROJECT_NARRATIVE.md` §4、`tests/README.md`

---

## 5. 寫作紅線（依 `docs/CLAIMS.md` @ 626b923；ID 為 CLAIMS 條目）

**✅ 可宣稱（全部 5-seed 除非另註）**
- 學習式選取 ≫ random（28/28 格），budget 在 K=8 達峰 ｜ 4.1
- SeqFT 三軸遺忘、雙 order 一致（C-20）；洩漏 100% 歸因選取（C-22）｜ 4.3
- replay 大幅回復準確率且跨順序穩定（C-21）；replay 回復準確率、KD+eq 回復行為（flat 單調）｜ 4.4
- 階層下 A5 − A3 兩軸 systematic（+3.28 / +5.76）（C-10，附兩個條件）｜ 4.6
- group-KD 有效且與 patch-KD 分工（C-03）｜ 4.7
- KD 與 replay 保存不同對象（C-24，附 B1 最不穩）｜ 4.5
- 同容量全勝（4/5 容量 systematic）+ 4× 記憶體效率（附三限定）｜ 4.9
- q_τ 降洩漏 −5.92（C-28）、group L_sem 保配額 KL −0.005（C-29），均附「未轉化為準確率」｜ 4.10
- 「在階層架構下語意錨的移除不損害準確率」（C-25 替代說法）｜ 4.8

**❌ 不可宣稱**
- sequential / iterative acquisition、stateful、state-conditioned（C-01）
- task-conditioned 作為方法性質（q_τ 帶來準確率增益）（C-04）
- L_sem 改善準確率（C-25）；兩層 L_sem 進主方法（C-27）
- task-IL 於 **flat** 架構勝過 replay（DR-015 在 flat 仍有效）
- 8× / 16× 記憶體效率；4× 為 A3 需求的上界；「A3 加記憶體也沒用」（階層版不成立）
- eq 的貢獻跨順序穩定（A5 − A4 翻轉）
- partial observation / navigation / plan（G-07）；「真正特徵抽取」（特徵是預抽的）
- 任何 3-seed 批次（E3、main order 的 A1/A2/A4/R1/R2）的 3/3 當 systematic（C-12）
- 「規格寫了 / 圖畫了」的元件當作已驗證（C-26）

**⚠️ 必須誠實條款**
- 階層優勢的雙重來源（hier-A3 退化 −3.11）｜ hier-A5 洩漏率比 flat 高 +2.15（0/5）
- esca n=15，一張 = 6.67 pp
- A3 曲線未飽和；記憶體 class-IL 在 1024 落入雜訊；flat 版中段（128/256）class-IL 弱
- rcc @K=1 輸 similarity；random/grid 只能等權；grid 非真 spatial uniform
- L_sem「只測了兩個軸」+「在階層架構下」+ max_sim 洩漏率最低
- B1 最不穩（±0.1042）；A4 − A3 的準確率代價（−1.04，0/5）
- S2 為 preliminary（CE-only、3 seeds）；正式 SeqFT baseline 是 A1 臂

---

## 6. 論文每一節的敘事骨架（DR-018：不遷就頁數，venue 視圖另出）

1. **Intro**：「在哪裡看」是可學的（4.1）→ 序列學習摧毀它（三個軸，4.3）→ frozen head 使摧毀可歸因、可量測（洩漏率）→ 我們提出保存機制 → 三點貢獻：(i) 階層式持續證據選取 + 兩層蒸餾 + 效用保存；(ii) 行為遺忘的三軸量測與 frozen-head 歸因；(iii) 記憶體效率前沿（同容量全勝 + 4× 下界）。
2. **Method**：§2 的最終形狀；L_sem / L_util 弱正則定位誠實寫；replay 定義為資料機制；被移出的三個元件只在 ablation 出現。
3. **Experiments**：4.1 立足點與 budget → 4.3 動機（遺忘三軸）→ 4.4 主表（flat 單調）→ 4.6 / 4.7 階層與兩層蒸餾（主結果）→ 4.9 記憶體 → 4.5 / 4.8 消融。
4. **Analysis**：4.2 可分離性 + 4.10 兩個 5/5 次要發現 + 4.11 四層失效模式 + 順序依賴。
5. **Limitations**：跨器官使 conditioning 不可測 → 同器官 future（S-02）｜L_sem 範圍｜esca 解析度｜A3 未飽和｜eq 非跨順序穩定｜被移除的三個元件（有數據的 null）。
6. **Reproducibility 附錄**：pre-registration（DR-021 / DR-039）、位元相同測試、mutation check、決策帳（43 卡 + GRAVEYARD 11 + SEEDS 18）、逐 slide 全存檔、`verify_doc_numbers.py`。

---

## 7. 核對時發現的 repo 不一致 → 交 Cursor（全部是治理層，不影響任何數字）

| # | 問題 | 位置 | 建議處置 | 生效（Cursor，2026-08-26） |
|---|---|---|---|---|
| ① | **C-11 仍寫「2× 記憶體效率、A5@128 追 A3 全域最佳（A3@256）、各容量只用 64/512/1024 三格」**——這是 DR-019 的 flat 裁定，DR-019 已 SUPERSEDED-BY DR-042 | `docs/CLAIMS.md` C-11 | 改寫為 DR-042 修訂 A 的主從結構（同容量主、4× 輔、三限定）；2× 降為「flat 限適用」 | ✅ C-11 改寫為主從結構；2× 標為 flat 限適用 |
| ② | **C-12 仍寫「main order、元件消融、E3 都只有 3 seeds」**——B1/B2 已補到 5 seeds（DR-033），main order 的 A3/A5 已 5 seeds | `docs/CLAIMS.md` C-12 | 改為：3-seed 批次現為 E3 與 main order 的 A1/A2/A4/R1/R2 | ✅ C-12 改為 E3 與 main order 的 A1/A2/A4/R1/R2 |
| ③ | **`order_main/EXP2.md` 是 3-seed 的**（表頭 seeds [0,1,2]，A3 0.8869 / A5 0.8993），但 `per_slide/` 已有 A3/A5 seeds 3–4；`ORDER_DEPENDENCE.md` 用的是 5-seed（A3 89.59 / A5 90.35） | `outputs/exp2/order_main/EXP2.md` | 重跑 report 產生器（A3/A5 用 5 seeds、其餘 3 seeds 並加警語），或在檔頭註明以 ORDER_DEPENDENCE 為準 | ✅ 重產，加「n seeds」欄與 §1.2 警語；A3 0.8959 / A5 0.9035 與 ORDER_DEPENDENCE 一致 |
| ④ | **Jaccard 隨機參照口徑不一**：`run_exp2.py` 用 task 平均 n 算一個值（esca 0.00106），`run_seqft.py` / `recompute_task_il.py` 逐 slide 算後平均（esca 0.00225）。觀測 Jaccard 是逐 slide 平均，後者才是同口徑（結論方向不變：esca/rcc 皆低於參照） | `scripts/run_exp2.py:424-437` vs `scripts/recompute_task_il.py::jaccard_reference` | 統一為逐 slide 平均，重產 EXP2.md 的「隨機參照」欄；論文只用一個口徑 | ✅ 統一為逐 slide；11 份 EXP2.md 重產；**結論方向未變**；DR-044 |
| ⑤ | **SEEDS S-03「brca 6/6 方向一致」沒有 committed 產物**（憲法 §2.8） | `docs/ledger/SEEDS.md` S-03 | 要嘛用 seqft per_slide 重算「Jaccard 排序 vs A1 排序」逐 (order, seed) 並 commit 為小產物，要嘛刪去那句 | ⚠️ 重算為 **5/6，不是 6/6**；S-03 已改為實際數字；產物 BEHAVIOUR_VS_FORGETTING.md |
| ⑥ | DR-033 寫「59/76 張被判為 LUAD」；B1_LANDING 實際是 53 LUAD + 6 LUSC = 59 張落到 lung 兩列 | `docs/ledger/DR-033.md`（append-only，不改內文） | 在 DR-033 末尾補一行勘誤註記；論文用「lung 的兩列」 | ✅ DR-033 末尾補勘誤註記（內文未改） |
| ⑦ | README 寫測試 1049 條，現為 1077 | `README.md` | 順手改 | ✅ README 改為 1077 |
| ⑧ | B2 的 l_eq fire rate 兩處不同：`ablation/EXP2.md`（5 seeds）0.1114；`ORDER_DEPENDENCE.md` 0.1142 | `outputs/exp2/ORDER_DEPENDENCE.md` 末節 | 以 5-seed 值為準重產，或註明來源批次 | ✅ 改為從 5-seed 產物重算：0.1114（n=5） |
| ⑨ | G1 退化的單組比例兩個數字並存：**84.5%** 可從 `HIER.md` 的分佈表算出（1179/1395）；**88.6%** 出現在 CLAIMS C-02、憲法 §3.6b、`report_prior.py`、`tests/test_report_scripts.py`，但 repo 沒有任何產物寫著 88.6 | `docs/CLAIMS.md` C-02 等四處 | 統一為 84.5%（附 1179/1395 的算式），或補 commit 88.6% 的計算產物並說明口徑差異 | ✅ 88.6% **可重算**（全部 arm，3708/4185）；84.5% = 只算 A5。四處註明口徑；DR-045 |

---

## 8. 與另一稿（未鎖定）的更正紀錄

| 舊稿寫法 | 核對結果 | 更正 |
|---|---|---|
| 「learned ≫ random：20/20 格」 | 4 task × 7 K = 28 格 | **28/28**（對 grid 亦 28/28、對 similarity 27/28） |
| 「seed 4 的 rcc 59/76 張被判為 LUAD」 | 53 LUAD + 6 LUSC | 「59/76 張被推到 lung 的兩列」 |
| 「行為保留度預測準確率保留度（brca 6/6）」 | repo 無產物 | 移除，改為可查證的敘述（§4.3） |
| E3：「A5@β_u=0 0.9026 vs 0.9147」 | 混用 3-seed 與 5-seed 均值 | 改為同 seeds 配對 +1.12 ± 0.50（3/3） |
| 4.6 未提 hier-A5 洩漏率 +2.15（0/5） | HIER2.md 有此欄，方向對 hier 不利 | 補為必須誠實條款 |
| 3.9 沿用 flat 的「A5@256 std 0.60 vs A3 2.82」 | 那是 DR-019（flat）數字 | 階層版改用「跨 \|M\| 標準差 0.95 vs 1.81」 |
| Jaccard 隨機參照寫「≈0.0022」 | repo 有兩套口徑（0.00225 逐 slide vs 0.00106 task 平均 n） | 表中並列兩套、標明差異；論文用逐 slide 口徑（§7-④） |

---

## 9. 尚未完成

| 項目 | 誰 | 狀態 |
|---|---|---|
| 本 dossier 進 repo（`docs/RESULTS_DOSSIER.md`）並登錄 `verify_doc_numbers.py` Tier 2 | Cursor | prompt 已開（`PROMPT_Cursor_DOSSIER-FIGURES-20260826.md`） |
| §7 九項治理修正 | Cursor | 同上 |
| 圖表（5 張主圖 + 2 張小圖；`scripts/make_figures.py` → `figures/`） | Cursor | ✅ 2026-08-26 完成：7 張皆輸出 .pdf + .png（300 dpi），數值收在 `figures/figure_data.json`，與 report 逐項對照通過（見 `tests/test_make_figures.py`） |
| 機制生效性稽核 `outputs/verify/MECHANISM_AUDIT.md` | Cursor | 低優先（四個機制的消融差值已證據齊備，此份只是把證據集中成一張表） |
| 架構圖 v3 手繪（移除 Panel E；q_τ 移出主圖；L_sem 標 patch-level） | PI | 標籤定稿已凍結 |
| 論文晉稿 | Fable | PI 讀完本 dossier 給回饋後開工 |
| F3 要不要 | — | 建議正式取消（KD 已證是弱環節；gating 實作成本不低） |
