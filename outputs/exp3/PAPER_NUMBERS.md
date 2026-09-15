# 論文數字對照表（DR-052 之後的口徑）

**主結果口徑**：累積式類別頭（任務 t 只在已見類別 C_t 上訓練與評估）、單層選擇器（flat）、比較協定十折（seed = fold）、平台 pod CPU x86（同平台配對見 outputs/exp3/EXP3.md）。標「fixed-vocabulary setting」者為固定 8 類頭的既有數字（E2／E3／seen-class、零樣本 top-8），不重跑。本檔只放數字與來源；判讀在 DR-052／DR-053 與 dossier §11。

## 1. Table 1 — 與發表結果並列（兩順序）

### reverse（ESCA→RCC→BRCA→LUNG，Tab. 2）

| 方法 | ACC | Forgetting | Masked ACC | 來源 |
|---|---|---|---|---|
| JointTrain (Upper) | 0.908±0.022 | — | 0.937±0.022 | QPMIL-VL Tab. 2（發表值） |
| FineTune (Lower) | 0.234±0.008 | 0.927±0.023 | 0.803±0.060 | QPMIL-VL Tab. 2（發表值） |
| EWC | 0.235±0.011 | 0.928±0.020 | 0.833±0.069 | QPMIL-VL Tab. 2（發表值） |
| LwF | 0.236±0.016 | 0.908±0.030 | 0.900±0.041 | QPMIL-VL Tab. 2（發表值） |
| A-GEM (buffer 30) | 0.536±0.047 | 0.527±0.067 | 0.872±0.026 | QPMIL-VL Tab. 2（發表值） |
| ER-ACE (buffer 30) | 0.703±0.049 | 0.281±0.062 | 0.889±0.041 | QPMIL-VL Tab. 2（發表值） |
| DER++ (buffer 30) | 0.684±0.055 | 0.310±0.069 | 0.910±0.043 | QPMIL-VL Tab. 2（發表值） |
| ER (buffer 30) | 0.644±0.028 | 0.387±0.050 | 0.901±0.035 | QPMIL-VL Tab. 2（發表值） |
| ConSlide (buffer 30) | 0.499±0.025 | 0.058±0.032 | 0.854±0.039 | QPMIL-VL Tab. 2（發表值） |
| AttriCLIP | 0.694±0.058 | 0.207±0.063 | 0.861±0.019 | QPMIL-VL Tab. 2（發表值） |
| MI-Zero | 0.839±0.034 | — | 0.909±0.019 | QPMIL-VL Tab. 2（發表值） |
| QPMIL-VL | 0.859±0.032 | 0.064±0.031 | 0.925±0.018 | QPMIL-VL Tab. 2（發表值） |
| QPMIL-VL（reproduced, released code；reverse, mini-batch 16（前輪，設定未對齊）） | 0.814 ± 0.065 | 0.147 | 0.908 | `outputs/exp2/sota/repro_qpmil/pod`（10 折） |
| QPMIL-VL（reproduced, released code；reverse, mini-batch 8（其論文反向設定）） | 0.868 ± 0.046 | 0.064 | 0.935 | `outputs/exp2/sota/repro_qpmil/pod_reverse_b8`（10 折） |
| **PathSelect（ours；accumulating head，flat）** | **0.803 ± 0.038** | **0.134** | **0.922** | `outputs/exp3/sota_acc`（10 折） |

### forward（LUNG→BRCA→RCC→ESCA，Tab. 1）

| 方法 | ACC | Forgetting | Masked ACC | 來源 |
|---|---|---|---|---|
| JointTrain (Upper) | 0.908±0.022 | — | 0.937±0.022 | QPMIL-VL Tab. 1（發表值） |
| FineTune (Lower) | 0.308±0.045 | 0.841±0.054 | 0.844±0.038 | QPMIL-VL Tab. 1（發表值） |
| EWC | 0.309±0.051 | 0.836±0.064 | 0.840±0.035 | QPMIL-VL Tab. 1（發表值） |
| LwF | 0.378±0.081 | 0.731±0.116 | 0.916±0.031 | QPMIL-VL Tab. 1（發表值） |
| A-GEM (buffer 30) | 0.436±0.058 | 0.670±0.072 | 0.879±0.045 | QPMIL-VL Tab. 1（發表值） |
| ER-ACE (buffer 30) | 0.666±0.049 | 0.075±0.057 | 0.917±0.023 | QPMIL-VL Tab. 1（發表值） |
| DER++ (buffer 30) | 0.749±0.055 | 0.219±0.059 | 0.904±0.029 | QPMIL-VL Tab. 1（發表值） |
| ER (buffer 30) | 0.790±0.040 | 0.186±0.044 | 0.894±0.036 | QPMIL-VL Tab. 1（發表值） |
| ConSlide (buffer 30) | 0.659±0.022 | 0.076±0.030 | 0.861±0.017 | QPMIL-VL Tab. 1（發表值） |
| ER (buffer 100) | 0.827±0.028 | 0.143±0.034 | 0.918±0.028 | QPMIL-VL Tab. 1（發表值） |
| AttriCLIP | 0.616±0.056 | 0.285±0.059 | 0.844±0.026 | QPMIL-VL Tab. 1（發表值） |
| MI-Zero | 0.839±0.034 | — | 0.909±0.019 | QPMIL-VL Tab. 1（發表值） |
| QPMIL-VL | 0.890±0.021 | 0.027±0.014 | 0.930±0.018 | QPMIL-VL Tab. 1（發表值） |
| QPMIL-VL（reproduced, released code；forward（其 Tab. 1，原生設定）） | 0.884 ± 0.033 | 0.038 | 0.926 | `outputs/exp2/sota/repro_qpmil/pod_forward`（10 折） |
| **PathSelect（ours；accumulating head，flat）** | **0.835 ± 0.031** | **0.077** | **0.906** | `outputs/exp3/sota_acc`（10 折） |

## 2. Table 2 — 鏈（兩順序；累積式頭，flat 除「＋兩層選擇器」）

### reverse（ESCA→RCC→BRCA→LUNG，Tab. 2）

| 列 | ACC | Forgetting | Masked ACC | 來源 |
|---|---|---|---|---|
| 無保存（sequential FT） | 0.481 ± 0.057 | 0.584 | 0.808 | `outputs/exp3/sota_acc` A1 flat（10 折） |
| ＋replay | 0.819 ± 0.042 | 0.121 | 0.915 | `outputs/exp3/sota_acc` A3 flat（10 折） |
| ＋蒸餾＋效用（PathSelect） | 0.803 ± 0.038 | 0.134 | 0.922 | `outputs/exp3/sota_acc` A5 flat（10 折） |
| ＋兩層選擇器（hier） | 0.798 ± 0.018 | 0.137 | 0.918 | `outputs/exp3/sota_acc` A5 hier（10 折） |
| 零樣本 top-8（fixed-vocabulary setting；與 stage 無關） | 0.812 ± 0.024 | 0.000 | 0.899 | `outputs/exp2/sota` ZS-top8（10 折） |
| QPMIL-VL（發表） | 0.859±0.032 | 0.064±0.031 | 0.925±0.018 | QPMIL-VL Tab. 2 |
| QPMIL-VL（reproduced；reverse, mini-batch 16（前輪，設定未對齊）） | 0.814 ± 0.065 | 0.147 | 0.908 | `pod` |
| QPMIL-VL（reproduced；reverse, mini-batch 8（其論文反向設定）） | 0.868 ± 0.046 | 0.064 | 0.935 | `pod_reverse_b8` |

逐折配對（reverse；A − B；較佳折數依 ACC／Masked ↑、Forgetting ↓）

| 配對 | ACC | Masked ACC | Forgetting |
|---|---|---|---|
| A3 − A1（replay − 無保存） | +0.3380（10/10） | +0.1069（10/10） | -0.4629（10/10） |
| A5 − A3（蒸餾＋效用 − replay） | -0.0158（2/10） | +0.0067（7/10） | +0.0124（4/10） |
| hier − flat（A5） | -0.0050（6/10） | -0.0041（4/10） | +0.0036（5/10） |

### forward（LUNG→BRCA→RCC→ESCA，Tab. 1）

| 列 | ACC | Forgetting | Masked ACC | 來源 |
|---|---|---|---|---|
| 無保存（sequential FT） | 0.549 ± 0.099 | 0.478 | 0.783 | `outputs/exp3/sota_acc` A1 flat（10 折） |
| ＋replay | 0.831 ± 0.044 | 0.078 | 0.909 | `outputs/exp3/sota_acc` A3 flat（10 折） |
| ＋蒸餾＋效用（PathSelect） | 0.835 ± 0.031 | 0.077 | 0.906 | `outputs/exp3/sota_acc` A5 flat（10 折） |
| ＋兩層選擇器（hier） | 0.830 ± 0.035 | 0.069 | 0.912 | `outputs/exp3/sota_acc` A5 hier（10 折） |
| 零樣本 top-8（fixed-vocabulary setting；與 stage 無關） | 0.812 ± 0.024 | 0.000 | 0.899 | `outputs/exp2/sota` ZS-top8（10 折） |
| QPMIL-VL（發表） | 0.890±0.021 | 0.027±0.014 | 0.930±0.018 | QPMIL-VL Tab. 1 |
| QPMIL-VL（reproduced；forward（其 Tab. 1，原生設定）） | 0.884 ± 0.033 | 0.038 | 0.926 | `pod_forward` |

逐折配對（main；A − B；較佳折數依 ACC／Masked ↑、Forgetting ↓）

| 配對 | ACC | Masked ACC | Forgetting |
|---|---|---|---|
| A3 − A1（replay − 無保存） | +0.2825（10/10） | +0.1260（10/10） | -0.4003（10/10） |
| A5 − A3（蒸餾＋效用 − replay） | +0.0037（5/10） | -0.0038（3/10） | -0.0012（5/10） |
| hier − flat（A5） | -0.0048（4/10） | +0.0067（6/10） | -0.0072（5/10） |

## 3. Table 3 — fold 1、五 seed、累積式頭、flat、reverse（`outputs/exp3/ablation_acc`）

| 臂 | class-IL | task-IL | 洩漏率 | Jaccard | ΔU(M1, C_t) | ΔU(M1, all8) |
|---|---|---|---|---|---|---|
| A2 LoRA merge only | 44.45 ± 11.36 | 81.41 ± 3.39 | 47.95 ± 13.00 | 0.0021 ± 0.0026 | -3.971 ± 1.268 | -4.360 ± 1.253 |
| A3 + replay | 76.99 ± 1.40 | 90.17 ± 1.85 | 15.04 ± 1.31 | 0.0556 ± 0.0221 | -0.271 ± 0.202 | -0.640 ± 0.297 |
| A4 + replay + KD | 75.62 ± 3.31 | 90.56 ± 1.82 | 16.19 ± 2.75 | 0.1645 ± 0.0582 | -0.389 ± 0.243 | -0.762 ± 0.321 |
| A5 full (PathSelect) | 75.90 ± 4.37 | 91.31 ± 1.28 | 16.33 ± 4.15 | 0.0894 ± 0.0470 | -0.367 ± 0.244 | -0.738 ± 0.382 |
| B1 KD only | 57.34 ± 12.37 | 84.55 ± 5.25 | 34.82 ± 13.27 | 0.0494 ± 0.0381 | -2.615 ± 1.350 | -2.995 ± 1.457 |
| B2 utility hinge only | 64.42 ± 2.60 | 89.15 ± 3.15 | 27.51 ± 0.51 | 0.0361 ± 0.0125 | -2.266 ± 0.863 | -2.647 ± 0.972 |

| 配對（A − B；勝 = A 較佳） | class-IL | task-IL | 洩漏率 | Jaccard | ΔU(all8) |
|---|---|---|---|---|---|
| A5 − A3 | -1.10（1/5） | +1.15（4/5） | +1.29（1/5） | +0.0338（4/5） | -0.099（3/5） |
| A5 − A4 | +0.28（4/5） | +0.75（3/5） | +0.14（2/5） | -0.0751（0/5） | +0.023（3/5） |
| A5 − B1 | +18.55（5/5） | +6.76（5/5） | -18.49（5/5） | +0.0400（4/5） | +2.256（5/5） |
| A5 − B2 | +11.48（5/5） | +2.17（4/5） | -11.18（5/5） | +0.0533（4/5） | +1.909（5/5） |

## 4. 稿內數字句（新口徑；每句附來源）

| 句 | 來源 |
|---|---|
| Unprotected sequential selection keeps 0.48 ACC in the reverse order; replay recovers it to 0.82 | sota_acc A1/A3 flat reverse（B7） |
| PathSelect reaches 0.803 ACC with 0.134 forgetting (reverse) and 0.835 / 0.077 (forward) | sota_acc A5 flat（B1／B3） |
| It trails the strongest published result by 0.056 (reverse, 0.859) and 0.055 (forward, 0.890) | QPMIL-VL Tab. 2／Tab. 1 發表值 − sota_acc A5 flat |
| Masked ACC 0.922 (reverse) / 0.906 (forward) | sota_acc A5 flat |
| Distillation plus utility preservation over replay alone: -0.016 ACC (2/10 folds, reverse), +0.004 (5/10, forward) | 配對 A5 − A3（B7／B1／B3） |
| The hierarchical selector changes ACC by -0.005 (6/10, reverse) and -0.005 (4/10, forward) | 配對 hier − flat（B1／B3） |
| The zero-shot top-8 reference reaches 0.812 ACC (fixed-vocabulary setting) vs PathSelect 0.803 (reverse) / 0.835 (forward) | outputs/exp2/sota ZS-top8（固定頭）、sota_acc A5 flat |
| On fold 1 (five seeds) full preservation vs replay-only: -1.10 pp class-IL (1/5) | ablation_acc（B5） |

## 5. 固定 8 類頭的既有結果（fixed-vocabulary setting；不重跑）

* E2 門控對照（A5ce vs A5，fold 1 五 seed）：`outputs/exp2/dr051/E2_GATED_CONTROL.md` —— fixed-vocabulary setting
* E3 選片 vs 加權拆解：`outputs/exp2/dr051/E3_DECOMPOSITION.md` —— fixed-vocabulary setting
* seen-class 檢查：`outputs/exp2/sota/SEEN_CLASS_CHECK.md` —— fixed-vocabulary setting（事後限制到已見類別的評估；與累積式頭的訓練口徑不同）
* 零樣本 top-8：`outputs/exp2/sota/per_slide/ZS-top8_*`（無參數、與 stage 無關）—— fixed-vocabulary setting
* E0 hinge 觸發率／ΔU 口徑：`outputs/exp2/dr051/E0*.md` —— fixed-vocabulary setting；累積式頭下的觸發率見 `outputs/exp3/DR053_HINGE.md`

