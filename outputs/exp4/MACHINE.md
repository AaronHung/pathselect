# outputs/exp4 的執行環境（exp/candidate-replay 探針）

| 項目 | 值 |
|---|---|
| 機器 | RunPod，GPU **RTX 4090**（本實驗不使用 GPU），CPU **AMD EPYC 7702 64-Core** |
| 容器 CPU 配額 | 72.25 核（cgroup v1：7,225,000 / 100,000） |
| RAM | 251 GB |
| Python / torch | 3.11.10 / **2.8.0+cu128**（容器內建為 2.4.1+cu124，刻意在 venv ps4 裝回 2.8.0 以對齊產生 outputs/exp3 的環境） |
| 每 run 執行緒 | OMP/MKL/torch = 8 |
| branch / commit | exp/candidate-replay / 6111ce4e771659726acd12c68abd3e47dfb4b95d |
| 日期 | 2026-09-17 |

⚠️ **跨機器的數字不得相減**：outputs/exp3 是在另一台 pod（RTX 5090 機型）產生的。
本目錄內的配對（run 2 − run 1、run 4 − run 3）都在**同一台、同一批**完成；
與 outputs/exp3 的比較只能當**參考**，不得作為配對統計。

## baseline 對數（run 1，現行路徑）與 PI 裁定 A（2026-09-17）

以現行路徑（`candidate_only=False`）在本機重跑 exp3 的 A5 fold 1：

| 項目 | 值 |
|---|---|
| 本批最終 class-IL | 0.73538483 |
| outputs/exp3 既有紀錄 | 0.80506508 |
| 差 | 6.97e-2（**未逐位元對齊**） |
| `selected_idx` 不同的筆數 | 553 / 569 |
| stage 0（無 replay）不同筆數 | **0 / 15**（權重已有 1e-6 量級差） |
| 單 run wall-clock（實測） | **4,466 s = 74.4 分鐘** |

歸因：stage 0 不跑 replay、本探針改動的函式根本不會執行，該 stage 選片逐筆相同，
分歧自 stage 1 起全面出現 → 平台浮點差異經 top-k 離散決策放大，機制與 DR-052
記錄的 Mac/pod 分歧相同，**非改動所致**。PI 裁定 **A：往下跑 probe**。

⚠️ 因此 **exp3 的數字在本批只能當參考，不得與本批相減**。

工時外推（實測基礎，非估算基礎）：若擴到二十折，40 run 在這台以 10 路平行
是**四波約 5 小時**，不是先前估的 2.5 小時。
