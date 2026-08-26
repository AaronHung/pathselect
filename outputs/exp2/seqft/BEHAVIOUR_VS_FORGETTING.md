# 行為保留度是否預測準確率保留度？（SEEDS S-03 的重算）

S-03 寫「brca 6/6 方向一致」，但 repo 裡沒有對應產物。本檔重算。

**定義（先寫死，不因結果調整）**：每個 (order, seed) 取三個非最後學的 task；
「一致」= Jaccard 最高的 task 與 forgetting 最小的 task 是同一個。
Jaccard 為逐 slide 算後平均；forgetting 為 class-IL accuracy 的差（pp，正 = 退步）。

## 結果：**5/6** 一致

| order | seed | Jaccard 最高 | forgetting 最小 | 一致？ | 逐 task（Jaccard｜forgetting pp） |
|---|---|---|---|---|---|
| reverse | 0 | brca | rcc | ❌ | esca 0.0000｜+73.33；rcc 0.0009｜+19.74；brca 0.0051｜+50.54 |
| reverse | 1 | brca | brca | ✅ | esca 0.0000｜+66.67；rcc 0.0000｜+73.68；brca 0.0057｜+21.51 |
| reverse | 2 | brca | brca | ✅ | esca 0.0000｜+73.33；rcc 0.0000｜+85.53；brca 0.0668｜+29.03 |
| main | 0 | brca | brca | ✅ | lung 0.0007｜+58.95；brca 0.0030｜+25.81；rcc 0.0026｜+69.74 |
| main | 1 | rcc | rcc | ✅ | lung 0.0000｜+50.53；brca 0.0014｜+47.31；rcc 0.0083｜+34.21 |
| main | 2 | brca | brca | ✅ | lung 0.0014｜+64.21；brca 0.0080｜+16.13；rcc 0.0044｜+77.63 |

其中 Jaccard 最高者為 **brca** 的批次：5/6。

⚠️ **S-03 的「6/6 方向一致」不成立** —— 實際為 **5/6**。S-03 該句應改為實際數字（憲法 §2.4）。

⚠️ 這是 **3-seed × 2 order 的觀察**，n 太小，不足以支撐「行為保留度預測準確率保留度」的一般性宣稱（憲法 §1.2）。可安全寫的是逐批次的事實。

