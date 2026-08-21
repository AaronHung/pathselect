# Exp 0 — Random / Grid baseline

論文 Table 1。四條線走**完全相同**的下游路徑：`selected patches → 權重 → L2 normalize → conch_classify → argmax`，唯一的差別是「選哪些 patch」與「用什麼權重」。

reverse order、fold 1、8-way label space、K ∈ {8, 16, 32, 64}、random 跑 5 seeds (0..4) 報 mean ± std、grid 跑 1 次。learned-flat 用 `reference/v9/skill_bank_reverse_f1.pt`，**純推論不重訓**。

## ⚠️ 權重政策不對稱（無法避免）

| policy | 選法 | 權重 |
|---|---|---|
| random | 均勻隨機 K 個，不重複 | **等權** |
| grid | 依特徵原順序等距，stride = n // K | **等權** |
| similarity | frozen CONCH patch-text 相似度 top-K | softmax(top-K 分數) |
| learned-flat | per-task selector top-K | softmax(top-K 分數) |

random 與 grid **沒有分數**，因此只能等權；這不是設計選擇，是這兩條線的
本質限制。scored 與 unscored 之間的差距同時包含「選得比較準」與「權重政策不同」兩個因素，**不要當成純粹的選取能力差**。
若要單獨看選取能力，請比較 selection-only ablation（見 DELTA_v9.md）。

⚠️ **grid 不是真正的 spatial uniform**：特徵檔是 `[n, 512]` 純張量，資料集裡沒有 patch 座標，所以只能沿特徵的原始掃描序等距抽樣。

## 結果

| task | K | random (mean ± std) | grid | similarity | learned-flat |
|---|---|---|---|---|---|
| tcga_esca | 8 | 0.5333 ± 0.0471 | 0.4000 | 0.8000 | 0.8000 |
| tcga_esca | 16 | 0.5333 ± 0.0667 | 0.6667 | 0.8000 | 0.8000 |
| tcga_esca | 32 | 0.5333 ± 0.0471 | 0.5333 | 0.7333 | 0.8000 |
| tcga_esca | 64 | 0.5200 ± 0.0298 | 0.5333 | 0.7333 | 0.8000 |
| tcga_rcc | 8 | 0.8605 ± 0.0072 | 0.8553 | 0.9474 | 0.9737 |
| tcga_rcc | 16 | 0.8763 ± 0.0220 | 0.8816 | 0.9474 | 0.9605 |
| tcga_rcc | 32 | 0.8711 ± 0.0216 | 0.8684 | 0.9474 | 0.9605 |
| tcga_rcc | 64 | 0.8632 ± 0.0150 | 0.8684 | 0.9474 | 0.9605 |
| tcga_brca | 8 | 0.5333 ± 0.0328 | 0.4946 | 0.7527 | 0.8925 |
| tcga_brca | 16 | 0.5484 ± 0.0132 | 0.5269 | 0.7527 | 0.8925 |
| tcga_brca | 32 | 0.5570 ± 0.0140 | 0.5699 | 0.7634 | 0.8925 |
| tcga_brca | 64 | 0.5484 ± 0.0076 | 0.5699 | 0.7419 | 0.8925 |
| tcga_lung | 8 | 0.7158 ± 0.0268 | 0.7579 | 0.7474 | 0.8526 |
| tcga_lung | 16 | 0.7116 ± 0.0264 | 0.7053 | 0.7474 | 0.8421 |
| tcga_lung | 32 | 0.7432 ± 0.0120 | 0.7474 | 0.7789 | 0.8421 |
| tcga_lung | 64 | 0.7579 ± 0.0182 | 0.7579 | 0.7684 | 0.8421 |

### 四 task 平均

| K | random | grid | similarity | learned-flat | learned − random (pp) |
|---|---|---|---|---|---|
| 8 | 0.6607 | 0.6269 | 0.8119 | 0.8797 | +21.90 |
| 16 | 0.6674 | 0.6951 | 0.8119 | 0.8738 | +20.64 |
| 32 | 0.6761 | 0.6798 | 0.8058 | 0.8738 | +19.76 |
| 64 | 0.6724 | 0.6824 | 0.7978 | 0.8738 | +20.14 |

逐筆結果：`outputs/exp0/baselines_reverse_f1.json`
（欄位 task / task_name / K / policy / seed / acc / n_slides）

