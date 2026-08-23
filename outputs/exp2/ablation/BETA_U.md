# E3 — beta_u 消融（L_util 的權重）

arm = A5、reverse order、|M| = 512、seeds [0, 1, 2]、其餘設定與消融表相同。
beta_u = 0 表示 **L_util 完全不計算也不相加**（與未接上該項位元相同）。

| 指標 | beta_u = 0 | beta_u = 0.1（主線） | 配對 0.1 − 0 | win | 判定 |
|---|---|---|---|---|---|
| task-IL final avg | 90.26 ± 1.50 | 91.39 ± 1.98 | +1.12 ± 0.50 | 3/3 | **systematic** |
| class-IL final avg | 81.96 ± 4.17 | 81.09 ± 2.24 | -0.87 ± 6.32 | 1/3 | within noise |
| 跨任務洩漏率 | 10.24 ± 3.69 | 11.05 ± 2.57 | +0.81 ± 5.53 | 1/3 | within noise |
| selection Jaccard | 0.1583 ± 0.0401 | 0.0910 ± 0.0435 | -0.07 ± 0.08 | 1/3 | within noise |

三級規則見 DR-020。**不報 p 值**（DR-016）。

⚠️ **本批只有 3 seeds，三級規則是為 n=5 校準的。** 3/3 的證據強度明顯低於 5/5，此處的 systematic 應讀作「方向一致」而非「已定案」；要寫進論文須回到 5-seed 批次確認。

逐 slide 預測：`outputs/exp2/ablation_bu0/per_slide/`（beta_u=0）、`outputs/exp2/ablation/per_slide/`（beta_u=0.1）

