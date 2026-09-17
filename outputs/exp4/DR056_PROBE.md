# DR-056 第 7 步 — 候選級 replay 的 probe（|M| = 128、fold 1、seed 1）

A5、hier、`--head accumulating`、`--uold current`、B = 8、c = 1、epochs 5。
四個 run **同批同機**（見 [MACHINE.md](MACHINE.md)），四路平行、每 run 8 執行緒。

⚠️ 這是**第二批**。第一批的兩個候選級 run 因共用側倉目錄互相覆寫候選特徵而
作廢（證據與修法見 [DR-056](../../docs/ledger/DR-056.md) 的 2026-09-18 補記）。

⚠️ **這不是「同方法換 buffer」，而是換了 replay 的定義**（DR-056）：分組原型改由
候選子集計算，配額隨之改變。差值不得寫成無損壓縮的誤差。

⚠️ **exp3 的數字只能當參考，不得與本批相減** —— 不同機器，baseline 未逐位元對齊。

## 主表

| 順序 | 設定 | ACC（class-IL） | Masked ACC（task-IL） | Forgetting ↓ | BWT | wall-clock |
|---|---|---|---|---|---|---|
| reverse | 讀整張（對照） | 0.7651 | 0.9124 | 0.1902 | -0.1902 | 3,903 s = 65.0 min |
| reverse | 只讀候選 | 0.7299 | 0.8678 | 0.2264 | -0.2264 | 1,712 s = 28.5 min |
| forward（--order main） | 讀整張（對照） | 0.8075 | 0.9071 | 0.0728 | -0.0728 | 3,911 s = 65.2 min |
| forward（--order main） | 只讀候選 | 0.7949 | 0.8945 | 0.1307 | -0.1307 | 1,699 s = 28.3 min |

## 配對差值（只讀候選 − 讀整張，同批同機、逐折相減）

| 對照 | 順序 | ΔACC (pp) | ΔMasked (pp) | ΔForgetting (pp) | Δwall-clock |
|---|---|---|---|---|---|
| run 2 − run 1 | reverse | -3.52 | -4.46 | +3.62 | -2,191 s（-36.5 min） |
| run 4 − run 3 | forward（--order main） | -1.27 | -1.26 | +5.78 | -2,212 s（-36.9 min） |

## 判準（PI 於 2026-09-17 凍結，先於看到數字）

| 判準 | 實測 | 結果 |
|---|---|---|
| ACC ≥ -2.0 pp（reverse） | -3.52 pp | ❌ 未過 |
| Masked ≥ -2.0 pp（reverse） | -4.46 pp | ❌ 未過 |
| Forgetting ≤ +3.0 pp（reverse） | +3.62 pp | ❌ 未過 |
| ACC ≥ -2.0 pp（forward（--order main）） | -1.27 pp | ✅ 通過 |
| Masked ≥ -2.0 pp（forward（--order main）） | -1.26 pp | ✅ 通過 |
| Forgetting ≤ +3.0 pp（forward（--order main）） | +5.78 pp | ❌ 未過 |
| 方向一致性：reverse 與 forward 的 ΔACC 不得一好一壞 | reverse -3.52 pp / forward -1.27 pp | ✅ 通過 |

**整體：❌ 有判準未過**

⚠️ **fold 1 單折、無變異數** —— probe 只能排除崩掉，**不能確認有效**。
要宣稱效果必須擴到完整十折／二十折，且對照組與實驗組同批同機重跑。

## 同機決定性核對（第一批 vs 第二批的「讀整張」）

「讀整張」的路徑從不碰側倉，兩批之間唯一的差別是執行時間。同一台機器、
同一份程式、同一個 seed，per_slide 應**逐位元相同**；若不同，代表這台機器
上的結果不可重現，整個配對比較都要重新檢討。

| 順序 | 第一批檔案 | 第二批檔案 | 逐位元相同？ |
|---|---|---|---|
| reverse | A5_reverse_seed1_M128_hier_acc_ucur.json | A5_reverse_seed1_M128_hier_acc_ucur.json | ✅ 相同（sha256 aa27dddab141…） |
| forward（--order main） | A5_main_seed1_M128_hier_acc_ucur.json | A5_main_seed1_M128_hier_acc_ucur.json | ✅ 相同（sha256 e7eb25d5669e…） |

## 資料量實測

### 單一 replay 步驟讀入的 bytes

候選級 replay 的每一步只開**一個**側倉檔；現行 replay 的每一步開該 slide 的
**完整特徵檔**。下表是同一批 slide 的逐筆配對實測。

| 量 | 側倉（只讀候選） | 完整特徵檔（讀整張） | 倍率 |
|---|---|---|---|
| min | 65,877 B | 78,571 B | 1.0× |
| mean | 526,212 B | 6,590,862 B | 12.5× |
| max | 528,469 B | 23,274,219 B | 44.0× |

側倉每筆幾乎是定值（k = 256 時 k×512×4 B ＋ 標頭 ＋ cand_idx 憑據 k×8 B）；
完整特徵檔隨 slide 的 patch 數變動，最大的一張是側倉的 44 倍。

### 整段訓練需要的特徵總量

| 口徑 | 筆數 | 總量 |
|---|---|---|
| **方法真正需要保留**（|M| = 128，reverse） | 128 | **64.2 MiB** |
| 側倉如實寫出（現行實作） | 2273 | 1.1139 GiB |
| 同一批 slide 的完整特徵檔 | 2273 | 13.9522 GiB |

保留集合的 task 分佈：{'tcga_brca': 48, 'tcga_esca': 4, 'tcga_lung': 49, 'tcga_rcc': 27}。

**64.2 MiB ÷ 13.95 GiB = 0.45%** ——
候選級 replay 在整個訓練期間對**舊任務**資料的需求就只有這些位元組。
哪 |M| 筆是無模型重放 `ReservoirSampling(seed=0)` 算出的**確切**集合，
不是用平均估（`tests/test_candidate_replay.py` 與真正的 SelectionMemory 逐筆比對）。

⚠️ 兩個口徑的落差是**實作細節**，不是方法性質：現行 `fill_memory` 對每一張
快照過的 slide 都寫側倉，而且寫入發生在 reservoir 汰換**之前**，所以磁碟上的
檔數等於看過的 slide 數。汰換時刪檔即可收斂到第一列，本探針未做這件事。

⚠️ 當前任務的特徵檔仍然要讀：那是 forward pass 的輸入，不是 replay 的需求。
第 8 步以「舊 task 特徵檔設為不可讀」直接驗證自足性（腳本已備妥，尚未執行）。

## 結論

**第 7 步未通過凍結判準**（4/7 條未過）。
依第 8 步的前提「僅在第 7 步結果可接受時做」，**不啟動第 8 步**（腳本已備妥，等 PI 裁示）。

未過的判準：

* ACC ≥ -2.0 pp（reverse）
* Masked ≥ -2.0 pp（reverse）
* Forgetting ≤ +3.0 pp（reverse）
* Forgetting ≤ +3.0 pp（forward（--order main））

代價與收益都要照報：

* **準確率**：ACC 降 3.52 pp（reverse）／1.27 pp（forward）；Forgetting 升 3.62／5.78 pp。
* **資料量**：單一 replay 步驟平均少讀 12.5 倍；舊任務需求 64.2 MiB 對 13.95 GiB（0.45%）。
* **時間**：每個 run 快約 37 分鐘（28 分鐘對 65 分鐘）。

**這個降幅不是壓縮誤差，是換了 replay 的定義的後果**（DR-056）：分組原型改由
候選子集計算，`run_rounds` 的配額隨之改變。把它寫成「buffer 變小、效果幾乎
不變」是不誠實的。

⚠️ fold 1 單折、無變異數。這個 probe 能說的只有「沒有崩掉，但代價可量測」；
要判定這個代價可不可接受，必須擴到完整十折／二十折，對照組與實驗組同批同機。
