# Codebase map

## `selector/` — 方法本體

| 檔案 | 負責 |
|---|---|
| `text_encoder.py` | CONCH text tower；`f_txt` = 各類別 prompt 嵌入的 L2-normalized 平均，`logit_scale = 56.3477` |
| `classifier.py` | frozen head（CONTRACT-4）：選中的 patch → 分數加權池化 → L2 normalize → CONCH 類別文字 logits。**沒有可訓練的診斷頭** |
| `model.py` | `GroupSelector` / `PatchSelector`；`use_query` / `use_state` 關掉時是**填零而不縮短維度** |
| `rounds.py` | chunked 迴圈（CONTRACT-1）；`allocation` ∈ {`per_budget`（主線）, `per_chunk`} |
| `state.py` | 證據狀態（CONTRACT-2）：`e_t = mean(E_t)`、`B_tilde = B_t/B_0`，輪間 detach |
| `grouping.py` | 組織型別分組（8 組）與 group prototype |
| `allocation.py` | 名額分配（largest remainder） |
| `priors.py` | `semantic_prior`：`none` / `max_sim` / `discriminative`（主線） |
| `train.py` | `l_diag` / `l_sem`（patch 層）/ `evidence_loss` / `train_step` / `fill_memory` / `continual_terms` |
| `sem_loss.py` | 兩層 L_sem（G3 消融維度）。`beta_g=0` 時與 `train.l_sem` **位元相同** |
| `continual.py` | `l_kd` / `l_eq` / `l_util` / `differentiable_utility` |
| `memory.py` | Selection Memory（CONTRACT-3）：\|M\| ≤ 512、reservoir sampling、**不存 patch 特徵** |
| `lora.py` | LoRA 與序列 merge（`W_eff = W + BA·scale`，merge 前後位元相同） |
| `utility.py` | counterfactual gain |
| `baselines.py` / `flat_selector.py` / `multiround.py` / `evaluate.py` / `task_query.py` / `device.py` | 基線、flat 版選取器、評估與工具 |

## `scripts/` — 實驗與報告

跑實驗：`run_exp0_baselines.py`（Random/Grid）、`run_exp1.py`（L3–L6 階梯）、
`run_seqft.py`（SeqFT）、`run_exp2.py`（**主實驗**，臂 A1–A5/B1/B2/R1/R2）、
`run_arch_completeness.py`（G3/G4/G5）。

產報告：`report_*.py`，每個對應一個 `outputs/**/**.md`。
守門：`check_batch_products.py`（產物齊全）、`job_status.py`（長 job 的存活訊號）、
`check_state_noop.py`（機制生效性實測，憲法 §2.9）。

`pipeline_*.sh` 是批次排程，一律 `set -euo pipefail` + heartbeat + 產物檢查。

## `outputs/` — 結果

| 目錄 | 內容 |
|---|---|
| `exp0/` | Random / Grid 基線、`EFFECTIVE_K.md` |
| `exp1/stage1/` | L3–L6 消融階梯 `RESULTS.md` |
| `exp2/seqft/` | SeqFT 遺忘曲線 `SEQFT.md` / `TASK_IL.md`（+ `ckpt/`） |
| `exp2/main/`, `order_main/` | 主表與順序依賴 `ORDER_DEPENDENCE.md` |
| `exp2/hier/`, `hier2/` | G1 / G1'（階層；`hier2` 是 per_budget 修正後的主線） |
| `exp2/memory/`, `memory_hier/` | E1 記憶體曲線（flat / 階層）`MEMORY.md` / `MEMORY_HIER.md` |
| `exp2/prior/`, `ablation/`, `ablation_bu0/` | G2 先驗、B1 落點、beta_u |
| `exp2/arch/` | G3/G4/G5 架構完整性 `ARCH_COMPLETENESS.md`、`noop_check.json` |

每個實驗都有 `per_slide/*.json` 逐 slide 存檔（憲法 §2.1），報告一律由腳本從這些
存檔重算，不手抄數字。

## `docs/` — 治理

`CONSTITUTION.md`（不隨單次實驗改動的規則）、`CLAIMS.md`（**不可宣稱**的清單）、
`ledger/INDEX.md` + `DR-*.md`（決策紀錄，append-only）、`ledger/GRAVEYARD.md`（廢案）、
`ledger/SEEDS.md`（未解釋的現象）。

## `reference/` — 唯讀存檔

前一版（v9）的產物，附 `SHA256SUMS.txt`。**不得修改**，只用於回歸比對。

## `tests/` — 守門

`pytest tests/ -q`。除了一般單元測試，還有幾類專門的守門：
`test_no_banned_deps.py`（與舊專案脫鉤）、`test_classifier_identical.py`（train/eval 位元相同）、
`test_cli_entrypoints.py`（每個腳本 `--help` 必須跑得起來）、
`test_report_scripts.py`（報告腳本要能用真實資料跑完並產出必要章節）、
`test_ledger.py`（DR 編號無缺口、status 與索引一致）。

寫法紀律見 `tests/README.md` —— 每條新斷言都要用 mutation check 證明它抓得到違規。
