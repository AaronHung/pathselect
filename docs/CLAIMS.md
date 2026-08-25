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

~~**解除條件**：`use_state=True` 的實驗（L6）跑出來，且證明其選取與 c=8 一次選取
不同。~~ **解除條件已消滅（G5 / DR-043）。**

G5（hier-A5 + `use_state=True`，5 seeds）確實打破了 no-op ——
前置檢查 16/20 集合相同，state 真的進入計算。但依 pre-registered 判準**落判 FAIL**：
task-IL −0.497 ± 2.206（2/5）、class-IL +1.045 ± 3.959（3/5），兩軸皆未達 4/5。

**因此「state 有進入計算」不等於「state 有用」** —— 前者只是機制生效性（憲法 §2.9），
後者要看判準。sequential / stateful 的宣稱**沒有任何實驗支撐**，本條不再有解除路徑。

**訓練後重測（2026-08-26）把這件事推到極端**：用 G5 訓練後的模型在 279 張真實
test slide 上重做，state 開啟時 **279/279（100%）** 的選取集合改變，平均只有
**2.81/8** 個 patch 與關閉時重疊。也就是說 state 幾乎**重寫了整個選取**，
準確率卻兩軸皆未達判準。**「大幅改變選取」與「改善結果」完全脫鉤** ——
這是本專案對「機制生效 ≠ 機制有用」最強的一個實例。
（`outputs/exp2/arch/noop_trained.json`；模型經 279/279 逐筆比對確認與正式 G5 一致。）

**用字禁令（DR-043）**：不得使用 **"stateful"、"state-conditioned"、
"sequential acquisition"** 描述本方法。架構圖移除 Panel E 與 E_t/B_t 輸入；
"Beyond HistoSelect" 移除 "stateful policy" 一條。

### C-27 兩層 L_sem 作為主方法

**主方法維持 patch-only L_sem，不加 group 項。**

G3 落判 FAIL（DR-043）：task-IL −0.524（2/5），僅 class-IL 單軸 4/5。
`selector/sem_loss.py` 的兩層版本是**消融維度**，不是主方法；
`beta_g=0` 時與 `selector/train.py::l_sem` 位元相同，主表不受影響。

⚠️ **但配額分佈的效果是可宣稱的** —— 見 [C-29](#c-29)。

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

**同一家族目前有四例。論文的 limitation 必須逐一列出。**

| 元件 | 規格 / 圖上有 | 實際狀態 |
|---|---|---|
| **group-level KD** | Panel I 的兩層蒸餾 | flat 下 F_g 輸出不影響選取，該項對所有指標影響恰為零；**階層版重測後有效**（DR-035） |
| **q_τ task conditioning** | 方法輸入的一部分 | 實作且有用到，但跨器官 benchmark 上 98.2/98.6% 線性可分 → 結構性冗餘（DR-008 / G-05） |
| **group-level semantic prior** | L_sem 原始規格是兩項：KL(B(r_j)‖B(p_j^sem)) + KL(B(s_i)‖B(p_i^sem)) | **只實作了 patch 項**。`l_sem()` 沒有 r 參數、沒有第二個 KL；訓練中從未計算 group prior。已用 mutation 實測確認：把 group prototype 擾動 5 倍，L_sem 數值**位元不變**（反向對照：擾動 patch 特徵會變） |
| **q_tau 接線** | 有實作、有開關，但注入點餵零向量 | `run_exp2.Ctx.q0 = zeros(512)`，而 `use_query=False` 的實作是填零 → 兩者輸入位元相同。實測 zeros vs 真 q_tau：與關閉時相同 **20/20 vs 16/20**。G4 啟動前修正（DR-040） |

⚠️ **這四例的共同教訓**：架構圖與規格書不是實作的證據。
宣稱任何機制有效之前，必須用**替換或擾動**證明關掉它會改變輸出（憲法 §2.2 / §2.6），
而且該實測必須在**實驗啟動之前**完成（憲法 §2.9 / DR-041）。

四層失效模式：**架構**（無作用空間）、**環境**（資訊冗餘）、**實作**（未寫）、
**接線**（寫了但沒接）。前三層看 code 或看設定**可能**發現，
**第四層只有實測才會現形** —— q_tau 有實作、有開關、開關也確實被讀取，
單看任何一處都正常，只有把開/關兩條路跑出來比對才看得到它們位元相同。

**2026-08-26 更新（G345 / DR-043）**：三例全部完成實測，**三者皆落判 FAIL**。

| 元件 | 實測 | 落判 | 處置 |
|---|---|---|---|
| E_t/B_t 狀態 | G5 | task-IL 2/5、class-IL 3/5 | 移除 Panel E 與 "stateful"（C-01） |
| q_tau | G4 | task-IL 2/5、class-IL 4/5（單軸） | 移出主圖、標 optional（C-04） |
| group 層 L_sem | G3 | task-IL 2/5、class-IL 4/5（單軸） | 主方法維持 patch-only（C-27） |

**這一輪最重要的教訓**：三個元件**都通過了機制生效性檢查**（打開後確實改變輸出），
但**都沒有通過效能判準**。「機制有生效」與「機制有用」是兩個不同的問題，
憲法 §2.9 管的是前者，pre-registered 判準管的是後者 —— **不可用前者代替後者**。

⚠️ 但 G4 與 G3 各有一個次要指標是 5/5 systematic（洩漏率、配額 KL），
必須報告，見 [C-28](#c-28) / [C-29](#c-29)。**落判 FAIL 不等於沒有效果。**

原始說明保留於下：

**2026-08-24 更新（DR-039）**：三例全部進入實測（G5 狀態、G4 q_tau、G3 group L_sem）。
在 `outputs/exp2/arch/ARCH_COMPLETENESS.md` 落判之前，本表的狀態欄仍然有效 ——
**「已實作」不等於「已驗證有效」**。特別注意：

- `selector/sem_loss.py` 的兩層 L_sem 是 **G3 的消融維度**，`beta_g=0` 時與
  `selector/train.py::l_sem` 位元相同。**主方法維持 patch-only**，主表不因 G3 改動。
- G5 的前置 no-op 檢查已通過（`outputs/exp2/arch/noop_check.json`）：state 打開後
  c=1 八輪與 c=8 一輪的選取集合不再恆等。但**通過 no-op 檢查只代表該元件進入計算，
  不代表它有用** —— 有用與否由 pre-registered 判準決定。
- **q_tau 接線是第四例（DR-040 / DR-041）**：`run_exp2.Ctx.q0` 是 `zeros(512)`，而
  `use_query=False` 的實作是把 query 欄位填零 —— 只打開開關而不接 `TaskQueryBank`，
  輸入與關閉時位元相同（實測 20/20）。**在 G4 啟動之前**發現並修正，
  所以它沒有污染任何數字；但它仍然計為第四例，因為失效本身是真的。
  ⚠️ 既有結果不受影響：`run_exp1.py` 用的是真 query，而 `run_exp2.py` 從未開啟
  use_query，零向量從未進入任何已發表的數字。

### C-04 task conditioning（q_τ）作為效能主張

S1 顯示跨器官任務 98.2 / 98.6% 線性可分，q_τ 在此 benchmark 結構性冗餘。

**2026-08-26 更新（G4 / DR-043）**：先前只有 flat 下的間接證據（L4 vs L3，
且該比較另有混淆）。G4 在**階層**下做了直接實測（q_tau 進 F_g，直接影響配額），
依 pre-registered 雙軸判準**落判 FAIL**：task-IL +0.312（2/5）、
class-IL +5.849（4/5，單軸不足）。

**用字禁令（DR-043）**：不得宣稱 q_tau 帶來準確率增益，不得用 **"task-conditioned"**
描述本方法。q_tau 移出主圖，標為 optional / ablated。

⚠️ **但 q_tau 對洩漏率的效果是可宣稱的** —— 見 [C-28](#c-28)。
「不能宣稱準確率增益」與「完全沒有效果」是兩回事，**不要把後者寫進 limitation**。

**解除條件**：同器官多任務 benchmark（SEEDS S-02）。

### C-25 L_sem 改善準確率（DR-036 → DR-038）

G2 三臂（階層版 5 seeds）在 class-IL 上**全部落在雜訊內**
（discriminative − none = **−0.02 pp，3/5**）。task-IL 上
discriminative − max_sim = +0.76 pp（5/5）但量級極小。

**可以宣稱的替代說法**（DR-038 裁定後的措辭）：
**「在階層架構下，語意先驗的移除不損害準確率。」**
semantic prior 作為**弱正則**（與 β_s 刻意設小的設計一致）。
選 discriminative 的理由是**避免 simple similarity**（DR-007），不是效能。

⚠️ **已刪除的措辭**：「HistoSelect 的貢獻在於分組結構而非語意先驗」——
DR-038 判定為**循環論證**（我們正是在「分組結構壓過 patch 分數」的階層架構裡
測 patch 層先驗），明令刪去。此處記錄刪除本身，避免它從別處回流。

⚠️ 必須同時報 **max_sim 的洩漏率最低（9.74）**。
⚠️ **範圍限定**：此結論**只適用階層架構**。L_sem 只錨定 patch 分數，
階層下 group 配額先決定名額、patch 分數只在組內排序，槓桿被稀釋；
flat 下 patch 分數單獨決定選取。**不可外推到 flat**（目前無 flat 的 prior 消融資料）。

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

### C-28 q_tau reduces cross-task evidence leakage

**q_tau 使跨任務洩漏率系統性降低 5.92 pp（−5.923 ± 4.189，5/5 systematic）。**

G4（hier-A5 + `use_query=True`，5 seeds，逐 seed 配對）。head 是 frozen 的，
所以洩漏率的變化 100% 可歸因於**選出的證據**（C-22）—— 顯式的任務語意讓選取
更集中在該任務自己的組織上。

⚠️ **必須同時附上條件：未轉化為準確率增益。** task-IL +0.312（2/5）、
class-IL +5.849（4/5，單軸不足以依雙軸判準宣稱）。
**不得**寫成「q_tau 改善了效能」。正確讀法：q_tau 影響「選什麼」，但那個改變
沒有變成準確率。

⚠️ 前置條件：`run_exp2.Ctx.q0` 是 `zeros(512)`，此結果來自
`run_arch_completeness.wire_task_queries` 接上真正的 `TaskQueryBank`（DR-040）。

證據檔：`outputs/exp2/arch/ARCH_COMPLETENESS.md`

### C-29 group-level semantic prior preserves quota distribution

**group 層語意先驗使 group 配額分佈的 KL 系統性降低 0.005（−0.005 ± 0.004，5/5）。**

G3（hier-A5 + `beta_g=0.1`，5 seeds，逐 seed 配對）。配額 KL 量的是「學完某個
task 時的組織層配額分佈」與「學完全部 task 後」的差距 —— 降低代表**組織層的
選取結構被保住**。這與 group-KD 的效果同軸（C-03：group-KD 保配額、patch-KD
保 patch 身份）。

⚠️ **必須同時附上條件：未轉化為準確率增益。** task-IL −0.524（2/5）、
class-IL +1.157（4/5，單軸不足）。主方法維持 patch-only（C-27）。

證據檔：`outputs/exp2/arch/ARCH_COMPLETENESS.md`

### C-20 catastrophic forgetting 存在且三軸皆崩

SeqFT 在 accuracy、selection 行為、utility 三個軸上都嚴重退化，雙 order 一致
（S2）。**這是本專案最穩固的結果之一。**

### C-21 replay 大幅回復準確率，且跨順序穩定

A3 − A1 在 reverse 與 main 上都是大幅改善，方向與量級一致
（class-IL +36.90 / +34.28 pp）。

### C-22 洩漏 100% 可歸因於選取漂移

head 是 frozen 的，所以跨任務洩漏只能來自選取改變（DR-012）。這是架構的直接後果。

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
