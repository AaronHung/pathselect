# 順序依賴（獨立成節）

CL 的結論不必然跨任務順序成立。本檔把翻轉的部分獨立列出，**不當雜訊帶過**（CONSTITUTION §3.2）。

reverse = esca → rcc → brca → lung；main = lung → brca → rcc → esca。
配對只用兩個 order **共同有的 seeds [0, 1, 2]**（reverse 另有 seeds 3,4，此處不納入，以免混入不同樣本）。

## 各臂在兩個 order 上的表現

### task-IL final avg

| 臂 | reverse | main | main − reverse（配對） | win | 判定 |
|---|---|---|---|---|---|
| A1 | 77.24 | 74.48 | -2.76 ± 1.47 | 0/3 | **systematic** |
| A2 | 82.26 | 88.31 | +6.05 ± 4.64 | 3/3 | **systematic** |
| A3 | 90.00 | 88.69 | -1.31 ± 3.12 | 1/3 | within noise |
| A4 | 88.96 | 90.60 | +1.64 ± 2.42 | 2/3 | within noise |
| A5 | 91.39 | 89.93 | -1.46 ± 1.49 | 1/3 | within noise |

### class-IL final avg

| 臂 | reverse | main | main − reverse（配對） | win | 判定 |
|---|---|---|---|---|---|
| A1 | 41.04 | 45.77 | +4.72 ± 8.73 | 2/3 | within noise |
| A2 | 48.97 | 64.65 | +15.69 ± 16.29 | 2/3 | within noise |
| A3 | 77.95 | 80.05 | +2.10 ± 3.59 | 2/3 | within noise |
| A4 | 79.05 | 81.88 | +2.83 ± 5.84 | 1/3 | within noise |
| A5 | 81.09 | 80.00 | -1.08 ± 4.03 | 1/3 | within noise |

### 跨任務洩漏率

| 臂 | reverse | main | main − reverse（配對） | win | 判定 |
|---|---|---|---|---|---|
| A1 | 50.90 | 39.46 | -11.45 ± 14.80 | 2/3 | within noise |
| A2 | 43.47 | 27.65 | -15.82 ± 15.30 | 2/3 | within noise |
| A3 | 13.34 | 12.10 | -1.24 ± 3.03 | 2/3 | within noise |
| A4 | 12.77 | 10.65 | -2.12 ± 5.84 | 2/3 | within noise |
| A5 | 11.05 | 11.85 | +0.80 ± 4.42 | 1/3 | within noise |

## 跨順序的穩定性

方向相反才算**翻轉**；若一側的效果量小於 0.5 pp（幾乎為零），標為「一邊有效、一邊無效」而非翻轉 —— 兩者的論文含義不同。

| 對照 | 指標 | reverse | main | 判定 |
|---|---|---|---|---|
| A5 − A4 | task-IL final avg | +2.43 pp | -0.67 pp | **翻轉** |
| A5 − A4 | class-IL final avg | +2.04 pp | -1.88 pp | **翻轉** |
| A5 − A3 | task-IL final avg | +1.38 pp | +1.24 pp | 跨順序穩定 |
| A5 − A3 | class-IL final avg | +3.14 pp | -0.05 pp | reverse 有效、main 無效 |
| A3 − A1 | task-IL final avg | +12.76 pp | +14.21 pp | 跨順序穩定 |
| A3 − A1 | class-IL final avg | +36.90 pp | +34.28 pp | 跨順序穩定 |

### 讀法

- **唯一乾淨的翻轉是 A5 − A4**（task-IL 與 class-IL 皆翻轉）：reverse 上 eq 有正貢獻、main 上是負的。**eq 的貢獻非跨順序穩定。**這是本檔最主要的發現。
- **A5 − A3 是「reverse 有效、main 無效」，不是翻轉**：main 側的 class-IL 差值幾乎為零（量級小於 0.5 pp），把它寫成翻轉會誇大。
- **A3 − A1 跨順序穩定**：replay 相對 SeqFT 在兩個 order 上都是大幅改善，方向與量級都一致。這是本專案最穩固的結果。

這是 CL 的真實現象（任務難度與順序位置交互作用），不是實作瑕疵。

### ⚠️ A2 − A1 不是順序效應，是 seed 變異

上表用共同 seeds 0–2 時，A2 − A1 在 reverse 是 +7.92 pp、main 是 +18.89 pp，看起來像「main 上 LoRA merge 更有效」。**但那是取樣造成的假象。**

reverse 的**全 5 seeds** 逐筆差值：`+1.84, +16.31, +5.61, −11.91, −27.29` pp，平均 **−3.09 pp**、win **3/5（within noise）**，幅度橫跨 43 pp。共同的 seeds 0–2 剛好是正的那三個。

**因此 A2 − A1 不得列為順序依賴的例證** —— 它在 reverse 上根本就是 within noise，跨 order 的差異被 seed 變異淹沒。

**方法學教訓**：兩個 order 的 seed 數不同時（reverse 5、main 3），只用共同子集做配對雖然統計上正確，但**子集可能不代表母體**。任何跨 order 的宣稱都必須同時檢查該對照在各自 order 的全 seeds 上是否成立。

## 觀察：l_eq fire rate

B2（只 eq）的 `l_eq_fire_rate` 為 0.1142，A5（三項全開）為 0.0740 —— **B2 的 hinge 被觸發的比例約為 A5 的 1.5 倍**。

⚠️ **這是觀察，不做因果宣稱。** 可能的讀法包括「A5 的 replay 與 KD 已經把 utility 撐住、使 hinge 較少觸發」，也可能只是兩臂訓練軌跡不同的副產物。要區分需要另一組診斷，不在本輪範圍。

逐 slide 預測：`outputs/exp2/main/per_slide/`（reverse）、`outputs/exp2/order_main/per_slide/`（main）

