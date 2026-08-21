# Exp 1 診斷紀錄

1. 每個 (task, level) 的 eff_K（沿用 D2 的算法 1 / sum(w_i^2)）
2. group 配額分佈：各 task 把 B 個名額分給哪些 tissue group
3. L6 每一輪選中的 group（看 e_t 更新後有沒有換組）

## 1. eff_K @ B=8（逐 slide 後平均 ± std，跨 seed 合併）

| task | L3 | L4 |
|---|---|---|
| tcga_esca | 7.62 ± 0.56 | 6.89 ± 1.05 |
| tcga_rcc | 7.63 ± 0.48 | 7.55 ± 0.58 |
| tcga_brca | 7.96 ± 0.05 | 7.62 ± 0.36 |
| tcga_lung | 7.73 ± 0.39 | 6.93 ± 1.00 |

## 2. group 配額分佈 @ B=8（各 task 選中的 patch 落在哪些 tissue group，跨 slide 與 seed 加總後的比例）

### L3 Flat learned selector

| task | tumor | stroma | lymphocyte | necrosis | normal_epithelium | vessel | adipose | background |
|---|---|---|---|---|---|---|---|---|
| tcga_esca | 0.575 | 0.064 | 0.133 | 0.072 | 0.094 | 0.008 | 0.000 | 0.053 |
| tcga_rcc | 0.677 | 0.122 | 0.089 | 0.016 | 0.000 | 0.049 | 0.024 | 0.024 |
| tcga_brca | 0.030 | 0.375 | 0.166 | 0.009 | 0.000 | 0.000 | 0.005 | 0.414 |
| tcga_lung | 0.827 | 0.022 | 0.006 | 0.095 | 0.002 | 0.000 | 0.000 | 0.048 |

### L4 + task conditioning q_tau

| task | tumor | stroma | lymphocyte | necrosis | normal_epithelium | vessel | adipose | background |
|---|---|---|---|---|---|---|---|---|
| tcga_esca | 0.469 | 0.108 | 0.139 | 0.058 | 0.061 | 0.114 | 0.000 | 0.050 |
| tcga_rcc | 0.565 | 0.058 | 0.098 | 0.014 | 0.001 | 0.134 | 0.013 | 0.118 |
| tcga_brca | 0.067 | 0.214 | 0.184 | 0.017 | 0.000 | 0.075 | 0.135 | 0.308 |
| tcga_lung | 0.684 | 0.074 | 0.060 | 0.055 | 0.012 | 0.019 | 0.001 | 0.095 |

