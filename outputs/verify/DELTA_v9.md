# DELTA_v9 — 拔掉舊方法造成的偏移

純推論診斷，不重訓。`reference/v9/skill_bank_reverse_f1.pt` 的 per-task selector，budget=64、單次 top-K（不用 multiround）、reverse order、fold 1、8-way label space（疊放順序已由 `tests/test_label_space_alignment.py` 釘死）。

**權重政策：主線用 softmax(top-K selector 分數)，依訓練一致性原則選定，非依數值選定。** 等權（selection-only）在下方單獨列為 ablation。

| task | v9 accuracy | new accuracy | delta (pp) |
|---|---|---|---|
| tcga_esca | 0.8667 | 0.8000 | -6.67 |
| tcga_rcc | 0.9474 | 0.9605 | +1.31 |
| tcga_brca | 0.8925 | 0.8925 | -0.00 |
| tcga_lung | 0.9053 | 0.8421 | -6.32 |
| **mean** | **0.9030** | **0.8738** | **-2.92** |

## 對照條件

| | v9 | new |
|---|---|---|
| f_txt | QPMIL CFE 增強類別特徵 | CONCH text tower 原生 |
| 分類器 | QPMIL 完整前向 (aggregate_and_predict) | `conch_classify` |
| selector | reference/v9 skill bank | 同左（未重訓） |
| 聚合權重 | QPMIL 內部 bag aggregation | softmax(top-K 分數) |
| budget / 選法 | 64 / 單次 top-K | 同左 |
| slide 集合 | fold 1 test split | 同左 |

v9 欄取自 `reference/v9/eval/task*_reverse_f1.json` 的 `lambda_0.00.zeronav_oneshot`（one-shot 不受 λ 影響，各 λ 皆同值）。

## slide 數對照

| task | v9 n_slides | new n_slides |
|---|---|---|
| tcga_esca | 15 | 15 |
| tcga_rcc | 76 | 76 |
| tcga_brca | 93 | 93 |
| tcga_lung | 95 | 95 |

## Ablation：selection-only（等權聚合）

同一組 top-K、同一個 `conch_classify`，只把權重換成等權。**這是 ablation，不是候選主線** —— 主線已依訓練一致性定為 softmax。

| task | 主線 softmax | selection-only 等權 |
|---|---|---|
| tcga_esca | 0.8000 | 0.8667 |
| tcga_rcc | 0.9605 | 0.9605 |
| tcga_brca | 0.8925 | 0.8925 |
| tcga_lung | 0.8421 | 0.8316 |
| **mean** | **0.8738** | **0.8878** |

## 判定

所有 task 的 |delta| 都在 10 pp 以內。

重跑：`python scripts/verify_v9_delta.py`
