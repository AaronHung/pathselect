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

### C-03 group-level distillation 有效或無效

DR-022 的首次驗證是在退化的階層下做的，**兩個方向都不能宣稱**。
**解除條件**：G1'-b 重測。

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

### C-23 per-task specialist 不是 task-IL 的上界

R1 每 task 只用自己的資料，replay 臂實質可及跨任務資料；R1 的參考意義在 class-IL
（DR-011）。
