# Codebase map（完整版）

> 本檔為結構性地圖：目錄、入口、資料流（含張量形狀）、損失、開關、凍結範圍、
> 測試與產物。決策脈絡見 [`ledger/INDEX.md`](ledger/INDEX.md)；
> 專案演進敘事見 [`PROJECT_NARRATIVE.md`](PROJECT_NARRATIVE.md)。
> 撰寫基準：commit 52593fc（2026-08-25）。

---

## 1. 目錄地圖

| 目錄 | 放什麼 | 誰會用到 |
|---|---|---|
| `configs/` | 唯一設定檔 `pathselect.yaml`（操作點 B=8/c=1、task 序、資料路徑、CONCH 權重路徑） | 所有 scripts 經 `selector.text_encoder.load_config` 讀取 |
| `data/` | 資料載入（`wsi_dataset.py` 的 `WSIClf`、`table_utils.py` 的表格/split/特徵讀取）與 `class_prompts.json`（類別 prompt ensemble） | `selector/evaluate.py`、`selector/text_encoder.py` |
| `selector/` | 方法本體：兩層選取器、chunked loop、狀態、分組、配額、prior、loss、記憶體、LoRA、評估 | 所有 run_* 腳本 |
| `scripts/` | 實驗 runner（`run_*`）、報告產生器（`report_*`）、守門（`check_*`、`job_status`）、批次排程（`pipeline_*.sh`）、驗證與煙霧測試 | 人與 pipeline |
| `tests/` | 1049 條 pytest（`README.md:35`），含紅線測試與 mutation check 紀律（`tests/README.md`） | CI / 提交前 |
| `docs/` | 治理層：`CONSTITUTION.md`（跨實驗規則）、`CLAIMS.md`（可/不可宣稱）、`ledger/`（DR-001..042、GRAVEYARD、SEEDS） | 寫作與決策 |
| `outputs/` | 所有實驗產物：逐 slide JSON、報告 md、log、`_status/` job 狀態、`cache/` 文字特徵快取 | 報告腳本重算來源 |
| `reference/` | v9 唯讀存檔（`SHA256SUMS.txt` 凍結）：skill bank、eval JSON、conch 相關 | `run_exp0_baselines.py`、`verify_v9_delta.py`（只讀） |
| `sota/` | DR-048 的 SOTA 協定線：指標（`metrics.py`）、正交投影合併（`opcm.py`）、zero-shot 參照線（`zeroshot.py`）、主表（`report_sota.py`）。**與 DR-046 的產物分表**，協定不同不可混讀 | `scripts/sota_queue.sh` |
| `third_party/` | vendored 的 CONCH 推論程式（`conch/`） | `selector/text_encoder.py` |

---

## 2. 執行入口（scripts/）

### 實驗 runner

| 腳本 | 做什麼 | 主要 CLI 參數（預設） | 讀 | 寫 |
|---|---|---|---|---|
| `run_exp0_baselines.py` | Exp 0：random/grid/similarity/learned-flat 四條無 CL 基線的 K 曲線 | `--max-eval 0`、`--tag ""`；常數 `KS=(1..64)`、`RANDOM_SEEDS=(0..4)` | test split 特徵；`reference/v9/skill_bank_reverse_f1.pt` | `outputs/exp0/baselines_reverse_f1.json`、`BASELINES.md`、`EFFECTIVE_K.md` |
| `run_exp1.py` | Exp 1：L3–L6 消融階梯（單任務/joint） | `--stage 1`、`--levels ""`（stage1→L3,L4）、`--seeds 0,1,2`、`--epochs 5`、`--lr 1e-3`、`--beta-s 0.1`、`--prior discriminative`、`--chunk 1` | train/test split | `outputs/exp1/stage{n}/per_slide/`、`RESULTS.md`、`DIAGNOSTICS.md`、`results.json` |
| `run_seqft.py` | S2：SeqFT 序列訓練遺忘（無任何 CL 機制、L3b flat） | `--orders reverse,main`、`--seeds 0,1,2`、`--epochs 5`、`--budget 8`、`--chunk 1`、`--report-only` | train/test split | `outputs/exp2/seqft/per_slide/`、`ckpt/`、`SEQFT.md`、`curves.json`、`results.json` |
| `run_exp2.py` | **主實驗**：CL 方法臂 A1–A5/A5nG/B1/B2/R1/R2 | `--arms 全部`、`--order reverse`、`--seeds 0,1,2,3,4`、`--epochs 5`、`--lr 1e-3`、`--beta-s 0.1`、`--beta-u 0.1`、`--prior discriminative`、`--budget 8`、`--chunk 1`、`--rank 4`、`--lambda-kd/eq/replay 1.0`、`--replay-k 1`、`--mem-capacity None(=512)`、`--allocation per_budget`、`--arch flat`、`--tag main`、`--report-only` | train/test split | `outputs/exp2/<tag>/per_slide/{arm}_{order}_seed{s}{suffix}.json`、`EXP2.md` |
| `run_arch_completeness.py` | G3/G4/G5：以 runtime injection 在 run_exp2 上一次只開一個變因（`exp ∈ {g5,g4,g3}`） | 位置參數 `exp`、`--seeds 0,1,2,3,4`、`--beta-g 0.1`；內部固定 `--tag arch`、`--arch hier_state/hier_query/hier` | 同 run_exp2；G4 另建 `TaskQueryBank` | `outputs/exp2/arch/per_slide/` |
| `selector/train.py`（也有 CLI） | within-task 訓練迴圈（per_task / joint），主要供早期階段與被 import | `--mode joint`、`--budget 8`、`--chunk 1`、`--prior discriminative`、`--beta-s 0.1`、`--beta-u 0.1`、`--lr 1e-3`、`--epochs 1`、`--group-grad ste_allocation` | train split | 不落檔（stdout） |

### 報告產生器（全部只讀 per_slide 存檔重算，不重跑訓練）

| 腳本 | 產出 | 參數 |
|---|---|---|
| `report_hier.py` | `outputs/exp2/hier/HIER.md`（G1：flat vs hier，per_chunk 舊口徑） | 無 |
| `report_hier2.py` | `outputs/exp2/hier2/HIER2.md`（G1'：per_budget，含結構性把關） | 無 |
| `report_hier2_structural.py` | stdout：每張 slide 用到幾組（可在批次跑完前先看；>50% 單組即非零退出） | 無 |
| `report_memory.py` | `outputs/exp2/memory/MEMORY.md`（E1 flat 版） | 無 |
| `report_memory_hier.py` | `outputs/exp2/memory_hier/MEMORY_HIER.md`（E1 階層版；章節順序由 `SECTION_ORDER`（`report_memory_hier.py:431`）固定：主張 → 跨容量配對 → DR-019 重驗） | 無 |
| `report_prior.py` | `outputs/exp2/prior/PRIOR.md`（G2 三臂 prior） | 位置參數 arch（預設 `hier`，`report_prior.py:78`） |
| `report_order_dependence.py` | `outputs/exp2/ORDER_DEPENDENCE.md`（reverse vs main，逐 arm 共同 seeds） | 無 |
| `report_b1_landing.py` | `outputs/exp2/ablation/B1_LANDING.md` + 冪等注入 `ablation/EXP2.md` 同一節 | 無 |
| `report_arch_completeness.py` | `outputs/exp2/arch/ARCH_COMPLETENESS.md`（判準為字面常數 `PRE_REGISTERED`；落判規則 `VERDICT_RULE = {"G5":"any","G4":"both","G3":"both"}`，`report_arch_completeness.py:77`） | 無 |
| `recompute_task_il.py` | `outputs/exp2/seqft/TASK_IL.md`（S3：從已存 selected_idx+weights 無損重建 logits 後算 task-IL 與洩漏率） | 無 |

### 守門與驗證

| 腳本 | 做什麼 |
|---|---|
| `check_batch_products.py` | 憲法 §3.5 產物存在性檢查：`--tag/--arms/--seeds/--order reverse/--suffix/--also`；缺檔或空檔即非零退出 |
| `check_state_noop.py` | 憲法 §2.9 生效性實測：state 開/關、query 零向量/真 q_tau 各 20 組 synthetic 比對；寫 `outputs/exp2/arch/noop_check.json` |
| `job_status.py` | 憲法 §3.7 存活訊號：`--job/--state running|done|failed/--stage/--note` → `outputs/_status/<job>.json` |
| `decide_h2.py` | H1 判準：B2 ≥ A5（class-IL、5 seeds、win≥4/5）才跑 H2；exit 0 = 跑 |
| `verify_a1_matches_s2.py` | 驗證 Exp2-A1（beta_u=0）與 S2 逐筆位元一致（1707 筆，DR-014） |
| `verify_v9_delta.py` | V2+P-A：v9 skill bank 在新 pipeline 的偏移對照 → `outputs/verify/DELTA_v9.md`、`FLIPS_v9.md`、`per_slide_v9.json` |
| `v9_reference.py` | 唯讀翻譯 v9 存檔 key（全 repo 唯一允許出現舊方法名的程式檔） |
| `diag_task_separability.py` | S1：task 可分離性 probe → `outputs/exp1/diag/TASK_SEPARABILITY.md` |
| `smoke_cl.py` / `smoke_rounds.py` | 煙霧測試：CL 全 pipeline / chunked loop 契約，不落正式結果檔 |
| `pipeline_20260823.sh` | 四段批次（G1'-b → G2 → main order 補 seeds → E1 階層版）；`set -euo pipefail` + heartbeat + 產物檢查 |
| `pipeline_stage4_20260824.sh` | 只跑上面第 4 段（E1 階層版 40 輪） |
| `pipeline_g345_20260824.sh` | G5 → G4 → G3 → 報告；等 stage4 退出後才開始 |

---

## 3. 核心資料流

共用符號：一張 slide 的 patch 特徵 `Z [n, 512]`（n 依 slide 約 3,100–3,800，
見 `outputs/exp2/seqft/SEQFT.md` 附註 A2 的平均 n）；類別文字特徵
`f_txt [8, 512]`（4 task × 2 類，8-way label space）；`logit_scale = 56.3477`
（取自 CONCH checkpoint，`selector/text_encoder.py:12`）；tissue 文字特徵
`t_tissue [J=8, 512]`；`q_tau [512]`；操作點 `B=8`、`c=1`（`configs/pathselect.yaml:23-24`）。

前置（一次性）：
1. `text_encoder.build_f_txt(task)`（`selector/text_encoder.py:132`）：
   `data/class_prompts.json` → CONCH text tower 編碼 `[P, 512]` → 每類 prompt 平均
   → L2 normalize → `f_txt [C=2, 512]`，四個 task 串接為 `[8, 512]`；快取
   `outputs/cache/f_txt_{task}.pt`。
2. `grouping.tissue_text_features`（`selector/grouping.py:57`）：8 組固定 tissue prompt
   → `t_tissue [8, 512]`，快取 `outputs/cache/f_tissue.pt`。
3. `grouping.assign_groups(Z, t_tissue)`（`selector/grouping.py:66`）：
   cos `[n, 8]` → argmax 指派 `assignment [n]`；group prototype
   `prototypes [8, 512]`（組內平均，空組零向量）；`mask [8]`、`sizes [8]`。
4. `task_query.encode_task_query`（`selector/task_query.py:30`）：
   `normalize(mean_c f_txt[c])` → `q_tau [512]`（定義凍結，DR-008）。

### (a) flat 架構的推論路徑

主實驗 flat 組態為 `use_query=False, use_state=False, hierarchy=False`
（`scripts/run_exp2.py:60`），入口 `run_exp2.evaluate`（`scripts/run_exp2.py:214`）：

1. `rounds.run_rounds(Z, grouping, q_tau=zeros(512), f_group, f_patch, budget=8, chunk=1, hierarchy=False, ...)`（`selector/rounds.py:88`）。
2. 每輪先取狀態特徵 `state.feature() = [e_t ; B_tilde_t] [513]`
   （`selector/state.py:97`；t=0 時 e_t 全零）。
3. `f_patch.score(Z, q_tau, state_feat)`（`selector/model.py:69`）→
   `build_input`（`selector/model.py:24`）拼成 `[n, 1537] = [x_i ; q_tau ; e_t ; B_tilde]`
   （`use_query=False`/`use_state=False` 時對應區塊**填零而不縮短維度**，
   `selector/model.py:44-46`）→ `Linear(1537→256) → GELU → Linear(256→1)` → `s [n]`。
   `use_state=False` 時 s 逐輪不變，只算一次並快取重用（`selector/rounds.py:128-136`），
   數值與逐輪重算完全相同。
4. flat 分支（`selector/rounds.py:178-186`）：`topk_indices(s, c=1, mask=available)` 取
   本輪 1 個 patch，`b [8]` 僅記錄落點。
5. `state.update(z_new [1,512], idx [1])`（`selector/state.py:45`）：累加 e_t、
   把該 patch 從候選移除、B_t 減 1。跑 8 輪 → `selected [8]`。
   （因 s 不變，這在資訊上等同一次 top-8；CLAIMS C-01。）
6. 分類：`classifier.softmax_weights(s, idx)` → `w [8]`（`selector/classifier.py:84`）；
   `conch_classify(Z[idx] [8,512], w, f_txt, logit_scale)`（`selector/classifier.py:19`）：
   加權和 → L2 normalize → `Z_w [1,512]` → `logits [1,8]`。
7. 預測：class-IL = 8-way argmax；task-IL = 該 task 兩列上的 2-way argmax
   （`scripts/run_exp2.py:237-238`）。

### (b) hier 架構的推論路徑

hier 組態 `use_query=False, use_state=False, hierarchy=True`（`scripts/run_exp2.py:61`），
`allocation="per_budget"`（主線，DR-025）：

1–3. 與 flat 相同，另外多算 group 分數：
   `f_group.score(prototypes [8,512], q_tau, state_feat)` → `r [8]`
   （`selector/rounds.py:131-132`；輸入同樣是 `[8, 1537]`）。
4. 配額（`selector/rounds.py:140-161`）：`allocation.allocate(r, B=8, mask, capacity)`
   （`selector/allocation.py:27`）—— 非空組上 `softmax(r)` × 8 → largest-remainder
   （Hare-Niemeyer）→ `quota [8]`，`sum(quota)=8`；溢出依 r 大小回流有餘裕的組。
   逐輪追蹤各組已取數 `taken [8]`，本輪從「`taken_j < quota_j` 且 r_j 最高」的組取
   1 個 patch；全滿額時放寬配額限制（`selector/rounds.py:153-161`）。
5. 組內選取（`selector/rounds.py:187-197`）：`topk_indices(s, b_j, mask=組內可選)`。
6. 之後同 flat（state.update ×8 輪 → selected [≤8] → frozen head → logits [1,8]）。

`allocation="per_chunk"`（舊口徑，僅 ablation）：每輪對 chunk c 配額
`allocate(r, c_this, ...)`（`selector/rounds.py:162-163`）；c=1 時 largest-remainder
只有一個名額、必然給 argmax(r)，r 又逐輪不變 → 退化為單組選取（DR-025）。

### (c) 訓練時多出來的部分

入口 `train.train_step`（`selector/train.py:117`），被
`run_exp2.train_stage`（`scripts/run_exp2.py:180`）呼叫：

1. **straight-through mask**：每輪 `straight_through_topk(s, k, mask)`
   （`selector/model.py:84`）回傳 `hard + (soft − soft.detach()) [n]` ——
   forward 是硬 0/1，backward 走 `softmax(s/T)`。各輪 `ste_mask` 累加成 `ste [n]`
   （`selector/train.py:140-142`）。
2. **F_g 的梯度路徑（group_grad="ste_allocation"，DR-009）**：配額取整不可微，
   故把 `a_j = softmax(r)_j` 以 `m_j * (a_j / a_j.detach())` 注入該組 mask
   （`selector/rounds.py:194-196`）—— forward 恆等於 1（head 數值不變），
   backward 讓 r_j 收到梯度。`"none"` 模式下 F_g 完全收不到梯度（僅 ablation）。
3. **frozen head（CONTRACT-4）**：`frozen_head(Z, s_last, ste, f_txt, logit_scale)`
   （`selector/train.py:51`）→ `w = ste ⊙ exp(s−max)` 正規化 → 加權池化
   `[1,512]` → L2 normalize → `logits [1,8]`。梯度同時經分數與選取決策流回 F_p。
4. **L_evidence**（`selector/train.py:91`）：
   - `l_diag`：`CE(logits [1,8], label)`（`selector/train.py:72`）。
   - `l_sem`：`KL(softmax(prior/τ) ‖ softmax(s/τ))`，兩者皆為 patch 上長度 n 的分布
     （`selector/train.py:78`）；`prior [n]` 來自
     `priors.semantic_prior(Z, f_txt, kind)`（`selector/priors.py:43`，no_grad）。
   - `l_util`（beta_u≠0 時）：`cand_idx = top_candidates(s, available, 256) [≤256]`
     （`selector/utility.py:29`）；`utility.counterfactual_gain`（`selector/utility.py:56`）
     以 rank-1 更新一次算完 `u [≤256]`（`E_cand = (S+X)/(|E_t|+1) [N,512]` →
     `logits_cand [N,8]` → `u_i = loss(now) − loss(cand)`）；
     `l_util = KL(softmax(u) ‖ softmax(s_cand))`（`selector/continual.py:87`）。
   - 合成：`L_evidence = L_diag + β_s·L_sem + β_u·L_util`（`selector/train.py:96-112`；
     `utility=None` 或 `beta_u=0` 時 L_util 完全不計算，位元等同未接上）。
5. **CL 三項**（`train.continual_terms`，`selector/train.py:314`；每步從
   `memory.sample(replay_k=1)` 取舊 entry）：
   - `reload_features(entry)`（`selector/memory.py:151`）依 slide_id 從特徵檔重載
     `Z [n,512]`（entry 不存 patch feature，CONTRACT-3），重新 `assign_groups`、
     重跑 `run_rounds`。
   - `L_KD = l_kd(r_old [8], r_new [8], s_old [≤256], s_new[cand_idx] [≤256])`
     （`selector/continual.py:37`）= `group_weight·KL(r_old‖r_new) + KL(s_old‖s_new)`；
     `group_weight=0`（A5nG 臂）時 group 項**完全不計算**、r_new 不進計算圖。
   - `L_eq = l_eq(U_new, u_old)`（`selector/continual.py:72`）：
     `U_new = log C − CE(uniform-head logits, label)`（`differentiable_utility`，
     `selector/continual.py:60`）、`u_old` 為 entry 內選中 patch 的 gain 加總；
     hinge：只在退步時罰。
   - `L_replay = l_diag(frozen_head(...), label)` —— replay 是資料機制，
     這一項就是一般任務損失跑在舊資料上（DR-013；`selector/train.py:346-348`）。
6. **總損失**：`total_loss = L_evidence + λ_kd·L_KD + λ_eq·L_eq + λ_r·L_replay`
   （`selector/train.py:351`、`selector/continual.py:102`；三項全 None 時回傳恰好
   為零的張量，與 SeqFT 位元相同）。`loss.backward(); opt.step()`
   （`scripts/run_exp2.py:205`；Adam，lr 1e-3、weight_decay 1e-4）。
7. **LoRA 更新與 merge**（有 LoRA 的臂 A2–A5/A5nG/B1/B2）：optimizer 只看到
   `lora_parameters`（`scripts/run_exp2.py:155-157`）；forward 每次物化
   `W_eff = W + (B@A)·(α/r)` 走單一 `F.linear`（`selector/lora.py:65-69`，
   保證 merge 前後位元相同）；每個 task stage 結束呼叫 `merge_lora`
   （`scripts/run_exp2.py:289-290`）：`W ← W_eff` 後 A 重新隨機初始化、B 歸零。
8. **記憶體填充**（`train.fill_memory`，`selector/train.py:277`；stage 結束後、
   `no_grad`）：對該 task train split 每張 slide 跑一次 run_rounds，
   `make_entry(tau, slide_id, e_t [512], B_tilde, r_old [8], cand_idx [≤256],
   s_old [≤256], u_old [≤256])`（`selector/memory.py:135`）→
   `SelectionMemory.add`（reservoir sampling，容量預設 512）。

註：evaluate 時 `run_exp2` 一律傳 `ctx.q0 = torch.zeros(512)` 當 q_tau
（`scripts/run_exp2.py:120`）；因 `use_query=False` 是填零實作，這對所有既有結果
無影響，但 G4 必須由 `run_arch_completeness.wire_task_queries`
（`scripts/run_arch_completeness.py:110`）接上真正的 `TaskQueryBank`（DR-040）。

---

## 4. 損失函數清單

| 函式 | 簽名（節錄） | 輸入 → 輸出 | 目前被誰呼叫 |
|---|---|---|---|
| `train.l_diag`（`selector/train.py:72`） | `(logits, label)` | `[1,8]`, int → scalar CE | `evidence_loss`（train.py:103）、`continual_terms` 的 replay 項（train.py:348） |
| `train.l_sem`（`selector/train.py:78`） | `(patch_score, prior, tau=1.0)` | `[n]`,`[n]` → scalar KL | `evidence_loss`（train.py:104） |
| `train.evidence_loss`（`selector/train.py:91`） | `(logits, label, patch_score, prior, *, beta_s, beta_u, utility, cand_idx)` | → `(scalar, parts)` | `train_step`（train.py:162）、`run_exp1.train_one`（run_exp1.py:150）、`run_seqft.train_task`（run_seqft.py:127） |
| `continual.l_kd`（`selector/continual.py:37`） | `(r_old, r_new, s_old, s_new, tau=1.0, group_weight=1.0)` | `[8]×2`,`[≤256]×2` → scalar | `continual_terms`（train.py:337） |
| `continual.l_eq`（`selector/continual.py:72`） | `(u_new, u_old, mode="hinge")` | scalar, float → scalar | `continual_terms`（train.py:345） |
| `continual.l_util`（`selector/continual.py:87`） | `(patch_score, utility, tau=1.0)` | `[≤256]×2` → scalar KL | `evidence_loss`（train.py:109），僅 beta_u≠0 且訓練期 |
| `continual.differentiable_utility`（`selector/continual.py:60`） | `(logits_uniform, label)` | `[1,8]` → scalar `log C − CE` | `continual_terms`（train.py:345） |
| `continual.continual_loss`（`selector/continual.py:102`） | `(kd, eq, replay, *, λ×3)` | → `(scalar, parts)`；全 None 時恰好為零 | `train.total_loss`（train.py:354） |
| `train.total_loss`（`selector/train.py:351`） | `(l_evidence, kd, eq, replay, *, λ×3)` | → `(scalar, parts)` | `run_exp2.train_stage`（run_exp2.py:202）、`smoke_cl.py` |
| `sem_loss.l_sem`（`selector/sem_loss.py:48`） | `(patch_score, patch_prior, group_score, group_prior, beta_g=0.0)` | 兩層 KL；beta_g=0 時與 `train.l_sem` 位元相同 | **未被任何實驗程式呼叫**（只有 `tests/test_sem_loss.py`）。G3 實際走的是 wrapper 加項，見下 |
| `sem_loss.group_sem_term`（`selector/sem_loss.py:36`） | `(group_score [J'], group_prior [J'], tau)` | → scalar KL | `run_arch_completeness.wrap_train_step`（run_arch_completeness.py:89）——G3 在既有 loss 上加 `beta_g · group_sem_term` |
| `continual.is_disabled`（`selector/continual.py:127`） | `(kd, eq, replay)` | → bool | **未被主程式使用**（只有 tests/test_continual.py） |
| `utility.counterfactual_gain_loop`（`selector/utility.py:79`） | 逐一計算參考實作 | | **只供單元測試比對**（tests/test_utility.py），正式路徑用向量化版 |

實際傳入永遠是預設值的參數：

- `l_eq` 的 `mode`：實作有 `"hinge"/"l2"` 兩種（`selector/continual.py:24`），
  但 `continual_terms` 的 `eq_mode` 沒有任何呼叫端覆寫，**"l2" 從未在實驗中使用**。
- `l_kd` / `l_sem` / `l_util` 的 `tau`：一律 1.0，無呼叫端覆寫。
- `run_rounds` 的 `temperature`（STE softmax）：一律 1.0。
- `train.py` CLI 的 `--group-grad`：`run_exp1/run_seqft/run_exp2` 都不傳此參數，
  一律吃預設 `ste_allocation`；`"none"` 只出現在 tests。
- `memory.FIFO` 汰換策略：實作存在（`selector/memory.py:75`）但所有實驗都用預設
  reservoir sampling；FIFO 只有 tests 用。
- `lora.PerTaskLoRABank`：只被 `smoke_cl.py` 與 tests 使用，不進任何正式實驗
  （oracle 上界之實作，`selector/lora.py:144`）。
- `flat_selector.py` / `multiround.py`（EvidenceSelector、SequentialBudgetedObserver
  等）：只服務 Exp 0 與 v9 對照（`run_exp0_baselines.py`、`verify_v9_delta.py`）
  與 `evaluate.top_k_select`，**不在主方法路徑上**。

---

## 5. 開關與旗標總表

| 旗標 | 預設 | 影響範圍 | 用了非預設值的實驗 |
|---|---|---|---|
| `--arch`（run_exp2） | `flat` | `ARCH` 組態（`run_exp2.py:59-62`）：只切 hierarchy，q_tau/state 一律關 | `hier`：G1（hier）、G1'（hier2）、G2（prior）、E1 階層版（memory_hier）、G345（arch）。注入組態 `hier_state`/`hier_query` 由 `run_arch_completeness.py:35-38` 加入 |
| `--allocation` | `per_budget`（`selector/rounds.py:58`） | 階層配額口徑；`per_chunk` 在 c=1 退化單組（DR-025） | `per_chunk`：僅 G1（`outputs/exp2/hier/`，其存檔缺 `allocation` 欄位＝舊語意）。DR-025 之後所有批次顯式傳 `per_budget` |
| `use_query` | run_exp2 硬編 `False` | selector 輸入的 q_tau 區塊（關=填零，`model.py:44`） | `True`：Exp1 L4（真 query，`run_exp1.py:60`）、G4（`hier_query`＋`wire_task_queries` 接線） |
| `use_state` | run_exp2 硬編 `False` | e_t/B̃ 區塊；False 時分數逐輪重用 | `True`：G5（`hier_state`）；Exp1 L6 定義存在（`run_exp1.py:64`）但 stage2 未跑（見產物：exp1 只有 stage1） |
| `--prior` | `discriminative`（`selector/priors.py:20`） | L_sem 的 anchor | `none`/`max_sim`：G2（`outputs/exp2/prior/`，各 5 seeds、hier） |
| `--beta-s` | 0.1 | L_sem 權重 | 無實驗改過（消融走 prior=none，非 beta_s=0） |
| `--beta-u` | 0.1 | L_util 是否計算（0 = 完全不算，位元等同未接上） | `0`：E3（`outputs/exp2/ablation_bu0/`，3 seeds）；S2 是在 L_util 接上前跑的 CE-only（DR-014） |
| `--lambda-kd/eq/replay` | 各 1.0，全程不調 | CL 三項權重；臂的開關由 ARMS spec 控制（關=該項是 None，非 λ=0） | 無（DR-039 也固定 λ 全 1.0） |
| `kd_group_weight`（ARMS spec） | 1.0 | L_KD 的 group 項係數；0 = 完全不計算 | `0.0`：A5nG 臂（`run_exp2.py:84-85`，DR-022/035） |
| `--group-grad` | `ste_allocation`（`selector/rounds.py:47`） | F_g 梯度路徑；`none` = F_g 是固定隨機函數 | 無實驗用過 `none`（僅 ablation 保留＋tests） |
| `--budget` / `--chunk` | 8 / 1（`configs/pathselect.yaml:23-24`，DR-005） | 操作點；c=8 即 one-shot | Exp0 K∈{1..64}；Exp1 budget 曲線 B∈{1,2,4,8,16}（`run_exp1.py:51`） |
| `--mem-capacity` | `None` → 512（CONTRACT-3，`selector/memory.py:21`） | Selection Memory 上限；>512 需 `allow_over_contract=True`（`memory.py:89-100`） | E1 兩版：{64,128,256,1024}（memory、memory_hier 各 tag） |
| `--replay-k` | 1 | 每步取幾筆舊 entry | 無（smoke_cl 用 2，非正式） |
| `--mem-slides` | 0（=該 task 全部） | fill_memory 每 task 寫入幾張 | 無 |
| `--rank` | 4（`selector/lora.py:28`） | LoRA rank；alpha=r → scale=1.0 | 無 |
| `--epochs` / `--lr` | 5 / 1e-3（run_exp2/run_seqft；train.py CLI 的 epochs 預設 1） | | 無（正式批次皆 5 / 1e-3） |
| `--order` | `reverse`（esca→rcc→brca→lung） | 任務序 | `main`（lung→brca→rcc→esca）：S2、order_main（A3/A5 5 seeds、其餘 3 seeds） |
| `--seeds` | run_exp2 預設 `0,1,2,3,4` | | 3 seeds：S2、E3、order_main 部分臂、Exp1（`run_exp1.py:53`） |
| `weighting` | `softmax`（主線，訓練一致性，DR-006） | frozen head / 評估聚合權重 | `uniform` = selection-only ablation；random/grid 基線只能 uniform |
| `candidate_size` | 256（`selector/utility.py:26`） | 候選集合 / memory cand_idx | 無 |
| `beta_g`（G3 wrapper） | 0.1（`run_arch_completeness.py:43`；`sem_loss.l_sem` 本身預設 0.0） | group 層 L_sem | G3 用 0.1；beta_g=0 為位元相同性檢查模式 |
| `--tag` | `main` | 產物子目錄 | 各批次：hier/hier2/prior/memory/memory_hier/order_main/ablation/ablation_bu0/arch/verify_a1 |

---

## 6. 凍結 vs 可訓練

**凍結（永不進 optimizer）**

- CONCH 全部：只用 text tower 建 `f_txt`/`t_tissue`/`q_tau`，全程 `torch.no_grad`
  （`selector/text_encoder.py:83`），結果落 cache 後連 tower 都不再載入。
  vision 特徵是離線抽好的檔案。
- frozen head（CONTRACT-4）：`frozen_head` / `conch_classify` **沒有任何可訓練參數**
  （純張量運算，`selector/train.py:51`、`selector/classifier.py:19`）。
- `semantic_prior` / `counterfactual_gain`：`@torch.no_grad`
  （`selector/priors.py:43`、`selector/utility.py:56`）。
- LoRA 臂中的 base weight/bias：`requires_grad=False`
  （`selector/lora.py:48-50`）。
- `e_t` 輪間 detach（`selector/state.py:61`）；memory entry 全部 detach 到 CPU
  （`selector/memory.py:138`）。

**可訓練**

- `GroupSelector`（F_g）與 `PatchSelector`（F_p）：各為
  `Linear(1537→256) → GELU → Linear(256→1)`（`selector/model.py:51-62`）。
  參數量（實測）：**每個 selector 393,985、兩個共 787,970**。
- 進 optimizer 的參數（`run_exp2.trainable`，`scripts/run_exp2.py:155-157`）：
  - 無 LoRA 的臂（A1、R1、R2；及 run_exp1/run_seqft 全部）：F_g+F_p 全參數
    787,970，Adam(lr 1e-3, weight_decay 1e-4)。
  - LoRA 臂（A2–A5、A5nG、B1、B2）：只有 `lora_A [r,in]`、`lora_B [out,r]`
    （`selector/lora.py:106-112`）。r=4 時**可訓練 16,400**（實測；
    第一層 4×1537+256×4=7,172、第二層 4×256+1×4=1,028，每個 selector 8,200×2），
    base 787,970 凍結。

**LoRA 掛哪裡、merge 何時發生**

- `apply_lora` 遞迴把模組下**所有 nn.Linear** 換成 `LoRALinear`
  （`selector/lora.py:91-99`）—— 即 F_g 與 F_p 各自的兩層 Linear，共 4 層。
- merge：每個 task stage 訓練結束時 `merge_lora(*models)` 一次
  （`scripts/run_exp2.py:289-290`）；`merge_()` 做 `W ← W_eff` 後 LoRA 歸零
  （`selector/lora.py:73-77`），forward 位元不變（W_eff 物化設計，
  `selector/lora.py:7-15`）。最終永遠只有一個 shared selector，推論不需 task id。

---

## 7. 測試地圖

`python -m pytest tests/ -q`，README 記錄目前 1049 條（`README.md:35`）。
🔴 = 紅線 / 治理類守門。

| 檔案 | 測什麼 |
|---|---|
| `test_allocation.py` | CONTRACT-1 largest-remainder 配額 + 溢出回流 |
| `test_allocation_mode.py` | DR-025 per_budget vs per_chunk 口徑（含 c=1 退化行為） |
| `test_arch_completeness.py` | G3/G4/G5 runner：注入必須生效、beta_g=0 位元相同、零向量 query 的 no-op 實證 |
| `test_arch_report.py` | G345 報告：pre-registered 判準常數與 `verdict()` 落判邏輯（mutation check 9 項） |
| `test_arch_switch.py` | 🔴 G1：flat 與 hier 之間只有一個開關不同；含 F_g 替換 mutation check（G0） |
| `test_baselines.py` | Exp 0 random/grid 選取政策性質 |
| `test_classifier_identical.py` | 🔴 訓練與評估共用同一分類器，相同輸入位元相同 |
| `test_cli_entrypoints.py` | 🔴 每個 CLI 腳本 `--help` 必須走完 argparse（DR-026） |
| `test_continual.py` | l_kd / l_eq / l_util / continual_loss（含全關=零張量） |
| `test_grouping.py` | argmax 指派、prototype、空組 mask |
| `test_label_space_alignment.py` | 🔴 8-way label space 疊放順序與 reference/v9 對齊 |
| `test_leakage.py` | 🔴 q_tau 不得洩漏 label（簽名檢查 + 同 task 跨 label 位元相同） |
| `test_ledger.py` | 🔴 DR 編號無缺口、status 合法、INDEX 一致（規則 3） |
| `test_lora.py` | LoRA merge 前後位元相同、參數隔離 |
| `test_memory.py` | CONTRACT-3 容量上限、reservoir、entry schema、allow_over_contract |
| `test_memory_hier_report.py` | DR-042：倍數必須來自 5/5 systematic、章節順序（8 項 mutation） |
| `test_model_ste.py` | build_input 填零語意、straight-through forward/backward |
| `test_no_banned_deps.py` | 🔴 禁字紅線：Zero*/Router/QPMIL 不得回流（DR-002/004，含逐檔窄例外表） |
| `test_per_slide_records.py` | 🔴 每次評估必須落逐 slide JSON（憲法 §2.1） |
| `test_priors.py` | 三種 prior + `assert_full_class_space`（label leakage 防線） |
| `test_report_scripts.py` | 🔴 報告腳本用 fixture 資料實際跑完 main()（憲法 §3.6 / §3.6b） |
| `test_rounds.py` | CONTRACT-1 行為：sum b=c、不重選、group_grad 三態（預設有梯度/none 無/forward 位元同） |
| `test_sem_loss.py` | G3 兩層 L_sem：beta_g=0 位元相同（以計算圖斷言，見 SEEDS S-18 的陷阱） |
| `test_state.py` | CONTRACT-2 EvidenceState：e_t/B̃/detach |
| `test_train_losses.py` | frozen head + evidence_loss（beta_u=0 位元等同未接上） |
| `test_utility.py` | 向量化 counterfactual gain == 迴圈參考實作 |

寫法紀律見 `tests/README.md`：每條新斷言先以人造違例證明會 FAIL（§1）；
科學主張也要以替換/擾動實測（§1b）。

---

## 8. 產物地圖（outputs/）

| 目錄 | 實驗 | 主要檔案 | 引用它的 DR |
|---|---|---|---|
| `cache/` | 文字特徵快取 | `f_txt_{task}.pt` ×4、`f_tissue.pt` | — |
| `_status/` | 長 job 存活訊號 | `pipeline.json`（failed 3/4）、`pipeline_stage4.json`（done）、`pipeline_g345.json`（running） | DR-032 |
| `verify/` | v9 偏移對照 | `DELTA_v9.md`、`FLIPS_v9.md`、`per_slide_v9.json` | DR-006 |
| `exp0/` | Random/Grid/similarity/learned-flat K 曲線 | `BASELINES.md`（K=8 峰值 0.8797）、`EFFECTIVE_K.md`（D2 eff_K）、`baselines_reverse_f1.json` | DR-005、DR-006 |
| `exp1/stage1/` | L3/L4 消融（Gate 1） | `RESULTS.md`、`DIAGNOSTICS.md`、`per_slide/`（L3/L4 × 3 seeds） | DR-010 |
| `exp1/diag/` | S1 可分離性 probe | `TASK_SEPARABILITY.md`（98.21/98.57%） | DR-008、G-05 |
| `exp2/seqft/` | S2/S3 SeqFT | `SEQFT.md`、`TASK_IL.md`、`per_slide/`（2 orders × 3 seeds × 4 stages）、`ckpt/`（24 檔 72MB） | DR-011、DR-012 |
| `exp2/main/` | 主表（flat、reverse、5 seeds、七臂） | `EXP2.md`、`per_slide/`（7 arms × 5） | DR-014、DR-015、DR-016 |
| `exp2/order_main/` | main order（A3/A5 5 seeds、其餘 3） | `EXP2.md`、`per_slide/` | DR-024、DR-037 |
| `exp2/` 根 | 順序依賴 | `ORDER_DEPENDENCE.md` | DR-024、DR-037 |
| `exp2/ablation/` | B1/B2 元件消融 + B1 落點 | `EXP2.md`、`B1_LANDING.md`、`BETA_U.md`、`per_slide/`（A3/A4/A5/B1/B2 × 5） | DR-023、DR-033 |
| `exp2/ablation_bu0/` | E3 beta_u=0 對照（3 seeds） | `EXP2.md`、`per_slide/` | DR-023 |
| `exp2/hier/` | G1（per_chunk，退化紀錄，保留不刪） | `HIER.md`、`per_slide/`（A3/A5/A5nG ×5 hier） | DR-021、DR-025 |
| `exp2/hier2/` | G1'（per_budget，**階層主線**） | `HIER2.md`、`per_slide/`（A3/A5/A5nG ×5 hier） | DR-029、DR-035、DR-037 |
| `exp2/memory/` | E1 flat 版記憶體曲線 | `MEMORY.md`、`per_slide/`（A3/A5 × 5 |M| × 5 seeds） | DR-017、DR-019、DR-020 |
| `exp2/memory_hier/` | E1 階層版（方案 A） | `MEMORY_HIER.md`、`per_slide/`（同上，_hier） | DR-031、DR-042 |
| `exp2/prior/` | G2 三臂 prior（hier、5 seeds） | `PRIOR.md`、`per_slide/`（none/max_sim；discriminative 沿用 hier2） | DR-036、DR-038 |
| `exp2/arch/` | G3/G4/G5 架構完整性 | `ARCH_COMPLETENESS.md`（2026-08-25 完成：G5/G4/G3 **全部 FAIL**，依 pre-registered 判準自動落判）、`noop_check.json`、`per_slide/`（hier_state ×5、hier_query ×5、A5g ×5） | DR-039、DR-040、DR-041 |
| `exp2/verify_a1/` | A1↔S2 位元一致性驗證（beta_u=0、3 seeds） | `EXP2.md`、`per_slide/` | DR-014 |
| `exp2/*.log` | 各批次原始 log（`pipeline.log`、`g1prime.log`、`h_series.log`…） | | DR-026、DR-027、DR-032 的證據 |

檔名規約（`run_exp2.py:368-374`）：
`{arm}_{order}_seed{s}[_M{cap}][_{arch}][_{prior}].json`；缺後綴代表舊語意
（flat、512、discriminative）—— 跨 tag 蒐集必須過濾所有語意欄位（憲法 §3.6b）。
