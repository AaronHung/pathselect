# CLAIMS — 可以宣稱什麼、不可以宣稱什麼

論文寫作前的最後一道閘。每一條都連到證據；**列在「不可宣稱」的，寫作時一律不得
出現，即使聽起來很順**。

> 相關政策見 [CONSTITUTION.md](CONSTITUTION.md)；決策脈絡見 [ledger/INDEX.md](ledger/INDEX.md)。

---

## ❌ 不可宣稱

### C-01 sequential / iterative acquisition（在目前所有已跑的實驗上）

**目前所有實驗的 chunked loop 在資訊上是 no-op。**

`use_state=False` 時 patch 分數 s 逐輪不變（分數重用），因此 c=1 跑八輪與
c=8 跑一輪選出的 patch **集合與順序都位元相同**。

> 實測：20 組（slide × 模型）中 **20/20 集合相同、20/20 順序也相同**。
> 對照組 `use_state=True`：只有 13/20 相同 —— 打破等價的是 **state**，不是 chunking。

**因此**：Exp 0 / S2 / S3 / Exp 2 / E1 / E2 / E3 / main order / 元件消融 / G1
全部不得宣稱 sequential acquisition、iterative refinement、或任何「逐步累積證據」
的敘事。它們在數學上等同一次性 top-K 選取。

**可以宣稱的替代說法**：budgeted top-K selection under a shared frozen head。

**解除條件**：`use_state=True` 的實驗（L6）跑出來，且證明其選取與 c=8 一次選取
不同。屆時 sequential 的宣稱只適用於那些實驗。

### ~~C-02 hierarchical selection~~ ✅ 已解除（G1'，DR-029）

G1 的 `per_chunk` 配額在 c=1 時退化為單組選取（88.6%），測到的不是階層。
**G1'（`per_budget`）已通過結構性把關**：單組比例 2.4%、平均用到 4.31 組。
**階層已採用為主線**，可正常宣稱。

### ~~C-03 group-level distillation~~ ✅ 已解除（G1'-b，DR-035）

在通過結構性把關的階層上（單組 2.4%）重測：**group-KD 有效**，
`hier-A5 − hier-A5nG` 於 task-IL **+3.95 ± 2.50（5/5）**、
class-IL **+3.71 ± 2.66（5/5）**，皆 systematic。DR-022 的結論已作廢。

⚠️ 宣稱時必須同時說明：**效果不顯示在 Jaccard**（+0.02，2/5）。
group-KD 保存的是**組織層配額分佈**、patch-KD 保存的是**具體 patch 身份**，
兩層分工不同 —— 這是架構圖 Panel I 兩層設計的直接證據。

### C-26 「規格寫了、架構圖畫了、但從未生效」的元件 ❌ 不可當作已驗證

**同一家族目前有三例。論文的 limitation 必須逐一列出。**

| 元件 | 規格 / 圖上有 | 實際狀態 |
|---|---|---|
| **group-level KD** | Panel I 的兩層蒸餾 | flat 下 F_g 輸出不影響選取，該項對所有指標影響恰為零；**階層版重測後有效**（DR-035） |
| **q_τ task conditioning** | 方法輸入的一部分 | 實作且有用到，但跨器官 benchmark 上 98.2/98.6% 線性可分 → 結構性冗餘（DR-008 / G-05） |
| **group-level semantic prior** | L_sem 原始規格是兩項：KL(B(r_j)‖B(p_j^sem)) + KL(B(s_i)‖B(p_i^sem)) | **只實作了 patch 項**。`l_sem()` 沒有 r 參數、沒有第二個 KL；訓練中從未計算 group prior。已用 mutation 實測確認：把 group prototype 擾動 5 倍，L_sem 數值**位元不變**（反向對照：擾動 patch 特徵會變） |

⚠️ **這三例的共同教訓**：架構圖與規格書不是實作的證據。
宣稱任何機制有效之前，必須用**替換或擾動**證明關掉它會改變輸出（憲法 §2.2 / §2.6）。

**2026-08-24 更新（DR-039）**：三例全部進入實測（G5 狀態、G4 q_tau、G3 group L_sem）。
在 `outputs/exp2/arch/ARCH_COMPLETENESS.md` 落判之前，本表的狀態欄仍然有效 ——
**「已實作」不等於「已驗證有效」**。特別注意：

- `selector/sem_loss.py` 的兩層 L_sem 是 **G3 的消融維度**，`beta_g=0` 時與
  `selector/train.py::l_sem` 位元相同。**主方法維持 patch-only**，主表不因 G3 改動。
- G5 的前置 no-op 檢查已通過（`outputs/exp2/arch/noop_check.json`）：state 打開後
  c=1 八輪與 c=8 一輪的選取集合不再恆等。但**通過 no-op 檢查只代表該元件進入計算，
  不代表它有用** —— 有用與否由 pre-registered 判準決定。

### C-04 task conditioning（q_τ）

S1 顯示跨器官任務 98.2 / 98.6% 線性可分，q_τ 在此 benchmark 結構性冗餘。
**解除條件**：同器官多任務 benchmark（SEEDS S-02）。

---

## ⚠️ 有限度宣稱

### C-10 A5 相對 A3 的優勢

**階層為主線後（DR-029），task-IL 與 class-IL 兩軸皆可宣稱**
（+3.28 / +5.76 pp，均 5/5 systematic）。

⚠️ 必須同時陳述兩件事，缺一即為選擇性報告：
1. **差距擴大有雙重來源** —— A5 更穩定（正向）**與 replay-only 在階層下退化**
   （hier-A3 − flat-A3 = −3.11 pp，負向）。不得只報前半。
2. **eq 的貢獻非跨順序穩定** —— A5 − A4 在 reverse 與 main 上方向相反
   （該對照目前只有 flat 版證據）。

flat 架構下的舊定調見 [DR-015](ledger/DR-015.md)（已 SUPERSEDED-BY DR-029），
其結論在 flat 上仍然成立。

### C-11 2× 記憶體效率

**跨容量陳述**：A5@128 達到 A3 全域最佳（A3@256）。同容量的機制主張只用
|M| = 64 / 512 / 1024 三格（DR-019）。中段 128/256 的證據弱，必須揭露。

### C-12 n < 5 的批次

main order、元件消融、E3 都只有 3 seeds。其 systematic 標籤只能讀作「方向一致」，
寫進論文前須回到 5-seed 確認（CONSTITUTION §1.2）。

---

## ✅ 可以宣稱

### C-20 catastrophic forgetting 存在且三軸皆崩

SeqFT 在 accuracy、selection 行為、utility 三個軸上都嚴重退化，雙 order 一致
（S2）。**這是本專案最穩固的結果之一。**

### C-21 replay 大幅回復準確率，且跨順序穩定

A3 − A1 在 reverse 與 main 上都是大幅改善，方向與量級一致
（class-IL +36.90 / +34.28 pp）。

### C-22 洩漏 100% 可歸因於選取漂移

head 是 frozen 的，所以跨任務洩漏只能來自選取改變（DR-012）。這是架構的直接後果。

### C-25 L_sem 改善準確率 ❌ 不可宣稱（DR-036）

G2 三臂（階層版 5 seeds）在 class-IL 上**全部落在雜訊內**
（discriminative − none = **−0.02 pp，3/5**）。task-IL 上
discriminative − max_sim = +0.76 pp（5/5）但量級極小。

**可以宣稱的替代說法**：semantic prior 作為**弱正則**，其移除不損害準確率
（與 β_s 刻意設小的設計一致）；HistoSelect 的貢獻在於**分組結構**而非語意先驗。
選 discriminative 的理由是**避免 simple similarity**（DR-007），不是效能。

⚠️ 必須同時報 **max_sim 的洩漏率最低（9.74）**。
⚠️ **範圍限定**：此結論**只適用階層架構**。L_sem 只錨定 patch 分數，
階層下 group 配額先決定名額、patch 分數只在組內排序，槓桿被稀釋；
flat 下 patch 分數單獨決定選取。**不可外推到 flat**（目前無 flat 的 prior 消融資料）。

### C-24 KD 與 replay 保存的是不同的東西（DR-033）

**KD 保住選取行為，replay 保住證據的任務歸屬，兩者不可互相取代。**

支撐：B1（只 KD）5-seed 落點分析 —— Jaccard 0.0725 與 A3（只 replay）的 0.0669
相當，但洩漏率 0.3231 是 A5 的 3.2 倍；A5 − B1 在 class-IL 為 systematic（5/5）
而 task-IL 僅 directional（4/5）。frozen head 使歸因封閉（DR-012）。

⚠️ 引用時必須同時報 **B1 是最不穩的一臂**（class-IL seed std ±0.1042，全場最大）。

證據檔：`outputs/exp2/ablation/B1_LANDING.md`

### C-23 per-task specialist 不是 task-IL 的上界

R1 每 task 只用自己的資料，replay 臂實質可及跨任務資料；R1 的參考意義在 class-IL
（DR-011）。
