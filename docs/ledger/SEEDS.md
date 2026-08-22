# SEEDS — 研究種子（L3 柴火的另一半）

GRAVEYARD 收的是**被砍掉**的方向（有復活條件）；SEEDS 收的是**跑出來但沒解釋**
的現象。兩者都是下一篇論文的原料，差別在墓園是「決定不做」，種子是「還不知道」。

固定欄位：`種子` / `證據鉤子` / `可能形態` / `來源 DR`。

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
