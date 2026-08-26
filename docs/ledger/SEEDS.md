# SEEDS — 研究種子（L3 柴火的另一半）

GRAVEYARD 收的是**被砍掉**的方向（有復活條件）；SEEDS 收的是**跑出來但沒解釋**
的現象。兩者都是下一篇論文的原料，差別在墓園是「決定不做」，種子是「還不知道」。

固定欄位：`種子` / `證據鉤子` / `可能形態`（S-01 另附來源 DR 與完整證據）。

**維護規則**
- 入苗圃 = 加一列。
- **被本篇採用** = 從本檔移除，並在對應 DR 卡註記。
- **被證據砍掉** = 移入 `GRAVEYARD.md`，附復活條件。

---

## S-01  replay 記憶體容量與 task 組成的交互作用

- **種子**：replay buffer 的**容量**與其**task 組成**之間存在交互作用，
  容量變大不必然更好 —— 存在一個窄的甜蜜點，出了區間反而退化。
- **證據鉤子**：
  - A3（LoRA + replay）的 class-IL 在 |M| = 256 → 512 為 **systematic decline**
    （配對 +4.25 ± 2.27 pp，5/5）。
  - **已排除 replay 強度混淆**：`replay_k` 固定為 1、與 |M| 無關，
    L_replay 與 current-task 的 batch 比例固定 1:1，
    `SelectionMemory.sample()` 的退回分支從未觸發（|M| 全程 ≥ 64）。
  - **對照組 A5 沿同一區間平穩**（四個指標 1/5、2/5、2/5、1/5，全部 within noise）
    —— 現象只發生在純 replay 臂。
  - 資料：`outputs/exp2/memory/MEMORY.md`；逐 slide 於 `outputs/exp2/memory/per_slide/`。
- **可能形態**：CL 中 **replay buffer 組成的診斷研究** —— 在固定取樣強度下，
  拆解 buffer 的 task 比例、時間跨度與樣本難度對可塑性/穩定性的影響。
  reservoir sampling 在多 task 序列下的組成漂移是最直接的切入點。
- **來源 DR**：[DR-017](DR-017.md)（E1 記憶體曲線）、[DR-020](DR-020.md)（定性與三級規則）


---

# 苗圃（S-02 起）

S-01 在上方單獨列出，因為它有完整的實驗證據鉤子；以下為一行式登錄。

| S | 種子 | 證據鉤子 | 可能形態 |
|---|---|---|---|
| S-02 | 同器官多任務 CL suite（subtype / grade / receptor） | S1：跨器官 98.2/98.6% 線性可分，故本 benchmark 測不出 task conditioning；[GRAVEYARD](GRAVEYARD.md) 中 G-05 q_τ 主張的復活條件 | CVPR 主實驗候選 |
| S-03 | 行為遺忘作為通用 CL 指標（frozen-evaluator 洩漏率） | frozen head 使洩漏 100% 歸因選取漂移；Jaccard 低於隨機參照；**行為↔遺忘方向一致 5/6**（2026-08-26 重算，非原記的 6/6；`outputs/exp2/seqft/BEHAVIOUR_VS_FORGETTING.md`） | 獨立 metric / benchmark 論文 |
| S-04 | 記憶體效率前沿作為 CL 評估軸 | E1 的 \|M\| 掃描設計與 2× 判準 | 評估協定提案 |
| S-05 | utility-gated KD（牌一） | 已實作 + 位元相同測試；u_old 在 M 中現成 | 方法升級第一格 |
| S-06 | Selection Memory 多樣性取樣（牌二） | reservoir 介面可替換；group 配額可當覆蓋度量 | 「記憶什麼」而非「記多少」 |
| S-07 | KD 的準確率–行為 trade-off | A4−A3 task-IL 0/5 −1.04±0.15，Jaccard 5/5 +0.07 | 分析節或短文 |
| S-08 | LoRA sequential-merge 動力學 | A2≈A1（merge 本身不解遺忘）；merge 前後位元相同實作 | 連 model-merging 文獻 |
| S-09 | VLM 本體的 CL（放開 CONCH frozen） | 本篇 frozen 是刻意的歸因設計 | LoRA for VLM CL |
| S-10 | 真 navigation：獨立 coarse view | [GRAVEYARD](GRAVEYARD.md) 中 G-07 partial observation 的復活條件 | 正名之作 |
| S-11 | VLN ↔ evidence acquisition 形式對應 | E_t / B_t 狀態、reveal 語意、greedy 措辭紀律 | position / 短文 |
| S-12 | Agent scaffold 整合（PathAgent loop + 我們的 CL selector） | 架構圖外框已畫過相容介面 | agent 論文 |
| S-13 | VQA 橋接（HistoSelect setting + 我們的 CL） | L_sem / 分組與 HistoSelect 同構 | 與 S-02 可合併 |
| S-14 | eff_K / 軟預算理論 | D2：eff_K/K 隨 K 降至 0.375；B=8 時 ≈ 等權 | 分析短文或附錄升格 |
| S-15 | 可分離性 probe 作為 benchmark 設計工具 | S1 probe 20 分鐘判定 conditioning 可測性 | 方法學貢獻：task-conditioned 工作的標準前測 |
| S-16 | stateful 選取的 per-round utility 監督 | last-round-only 只覆蓋 1/8 決策點 | L6 完整版先決修正 |
| S-17 | budget 曲線峰值現象（K=8 後下降） | Exp 0 四 task 平均峰值在 K=8；esca 七個 K 全平 | 證據稀釋 vs 冗餘的分析 |
| S-18 | 研究用程式碼的 mutation-checked 斷言實務（**含科學主張**） | 本專案 68 條 ledger 測試；一次靜默失敗的字串替換（commit 8675021 前）被自身疏失發現；G0 以「替換 F_g 為擾動網路 → 選取位元相同」把「零影響」從推論變成證據，並以 hier 下的反向對照排除「替換本身無效」；另發現「數值不可分辨的違例」（0·kl+patch ≡ patch）會讓 mutation 假通過，須改以計算圖斷言 | research engineering 的方法學短文或附錄 |

---

## 更正紀錄

**S-03（2026-08-26）**：原寫「brca 6/6 方向一致」，但 repo 裡沒有對應產物（憲法 §2.8）。
依 PROMPT DOSSIER-FIGURES-20260826 §B-⑤ 重算：定義先寫死（Jaccard 最高的 task 是否
即 forgetting 最小的 task，逐 (order, seed)），實際為 **5/6**，唯一不一致的批次是 `reverse` order 的 seed 0
（Jaccard 最高是 brca，但 forgetting 最小的是 rcc）。**沒有為了湊 6/6 改定義。**
產物：`outputs/exp2/seqft/BEHAVIOUR_VS_FORGETTING.md`（`scripts/report_behaviour_vs_forgetting.py`）。
