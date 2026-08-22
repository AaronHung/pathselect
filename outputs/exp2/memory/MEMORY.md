# E1 — 記憶體效率曲線

目的：反駁「replay 做了全部的事，把 |M| 開大就好」。

arms = ['A3', 'A5']、|M| ∈ [512]、reverse order、seeds [0, 1, 2, 3, 4]。其餘設定與主表完全相同（B=8、c=1、epochs 5、lr 1e-3、beta_s 0.1、beta_u 0.1、λ 全 1.0 不調、replay_k=1）。

⚠️ **|M| = 1024 超出 CONTRACT-3 的 |M| ≤ 512**，是刻意探測契約之外的診斷點，需在程式中顯式 opt-in （`SelectionMemory(..., allow_over_contract=True)`）。
⚠️ |M| = 512 的 A3 / A5 直接沿用主表存檔（同設定的決定性重跑）。

## task-IL final avg（越大越好）

| \|M\| | A3 + Replay | A5 Ours (Replay+KD+eq) | A5 − A3（配對） |
|---|---|---|---|
| 512 | 90.73 ± 1.82 | 91.47 ± 1.41 | +0.74 ± 1.93（3/5） |

## class-IL final avg（越大越好）

| \|M\| | A3 + Replay | A5 Ours (Replay+KD+eq) | A5 − A3（配對） |
|---|---|---|---|
| 512 | 77.78 ± 1.48 | 82.39 ± 2.84 | +4.61 ± 2.29（5/5） |

## 跨任務洩漏率（越小越好）

| \|M\| | A3 + Replay | A5 Ours (Replay+KD+eq) | A5 − A3（配對） |
|---|---|---|---|
| 512 | 14.21 ± 1.80 | 10.05 ± 2.55 | -4.16 ± 2.96（5/5） |

## selection Jaccard（越大越好）

| \|M\| | A3 + Replay | A5 Ours (Replay+KD+eq) | A5 − A3（配對） |
|---|---|---|---|
| 512 | 0.0669 ± 0.0147 | 0.1294 ± 0.0617 | +0.06 ± 0.06（4/5） |

## 追平點

A3 在 |M|=512 的 class-IL final avg = **0.7778**。

→ **A5 在 |M|=512 時追平 A3 在 |M|=512 的 class-IL 表現。**

反向檢查（PI 指定的負面資訊）：A5 在 |M|=512 的 class-IL = **0.8239**；A3 在所有測試的 |M| 都沒有追上它。

逐 slide 預測：`outputs/exp2/memory/per_slide/*.json`

