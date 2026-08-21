# FLIPS_v9 — 翻轉分析（P-A）

## ⚠️ 能算什麼、不能算什麼

`reference/v9/` 只封存了**每個 task 的彙總 accuracy**（`eval/*.json` 是 λ × policy 的 accuracy，另一個彙總檔是 4×4 cross-task 矩陣），**沒有逐 slide 預測**。因此「v9 對→new 錯」這種 gross flip 無法從存檔算出來，只能從淨值推出區間。要拿到真值，唯一辦法是把 v9 的 backbone 重跑一次推論（需要 timm/torchvision 與舊方法的模型程式），那已超出目前授權範圍。

下面提供兩件**算得出來**的事：
1. 對 v9 的 gross flip **區間**（由淨值與正確張數推得）。
2. 主線 softmax ↔ selection-only 等權之間的**逐張 gross flip**（真值）。

## 1. 對 v9 的 gross flip 區間

令 `f_cw` = v9 對→new 錯，`f_wc` = v9 錯→new 對，則 `f_wc - f_cw = new_correct - v9_correct`（淨值）。
單靠淨值無法定出 gross，只能給上界：`f_cw ≤ min(v9_correct, new_wrong)`、`f_wc ≤ min(v9_wrong, new_correct)`。

| task | n | v9 correct | new correct | 淨 | f_cw 上界 | f_wc 上界 |
|---|---|---|---|---|---|---|
| tcga_esca | 15 | 13 | 12 | -1 | 3 | 2 |
| tcga_rcc | 76 | 72 | 73 | +1 | 3 | 4 |
| tcga_brca | 93 | 83 | 83 | +0 | 10 | 10 |
| tcga_lung | 95 | 86 | 80 | -6 | 15 | 9 |

所以「esca / brca 淨值為 0」**不能**推論成零翻轉：以 brca 為例，最多可能有 10 張互相抵消。這個問題目前無解，需要 v9 逐 slide 預測。

## 2. 主線 softmax ↔ selection-only 等權：逐張 gross flip（真值）

同一組 top-K、同一個分類器，只換權重政策。

| task | n | softmax 對 | 等權 對 | 淨 | 等權對→softmax錯 | 等權錯→softmax對 | gross |
|---|---|---|---|---|---|---|---|
| tcga_esca | 15 | 12 | 13 | -1 | 1 | 0 | 1 |
| tcga_rcc | 76 | 73 | 73 | +0 | 0 | 0 | 0 |
| tcga_brca | 93 | 83 | 83 | +0 | 1 | 1 | 2 |
| tcga_lung | 95 | 80 | 79 | +1 | 2 | 3 | 5 |

### 翻轉 slide 明細

#### tcga_esca

| slide id | true | 等權 pred | softmax pred | 方向 |
|---|---|---|---|---|
| TCGA-2H-A9GN-01Z-00-DX1.2373BEFC-931F-460D-8F6C-49C424600930 | 0 | 0 | 1 | 等權對 → softmax錯 |

#### tcga_brca

| slide id | true | 等權 pred | softmax pred | 方向 |
|---|---|---|---|---|
| TCGA-BH-A0H5-01Z-00-DX1.28F24D4D-EE80-4EDA-BC30-A194E22FD61C | 4 | 4 | 5 | 等權對 → softmax錯 |
| TCGA-A8-A07O-01Z-00-DX1.3D657129-F2A7-4BC1-A910-805FBDCE2212 | 4 | 0 | 4 | 等權錯 → softmax對 |

#### tcga_lung

| slide id | true | 等權 pred | softmax pred | 方向 |
|---|---|---|---|---|
| TCGA-55-8511-01Z-00-DX1.8EDFB05B-5B59-46EA-973C-1048B1E284D2 | 6 | 6 | 7 | 等權對 → softmax錯 |
| TCGA-86-7955-01Z-00-DX1.ef4f4d94-5efb-4a07-97cf-b0ed69085827 | 6 | 6 | 7 | 等權對 → softmax錯 |
| TCGA-66-2768-01Z-00-DX1.02fadcc4-9d05-4b37-9114-d8e80c09ef1a | 7 | 0 | 7 | 等權錯 → softmax對 |
| TCGA-35-4122-01Z-00-DX1.2ac022e4-e796-49e5-9a24-f0ff3f76a527 | 6 | 7 | 6 | 等權錯 → softmax對 |
| TCGA-44-3917-01Z-00-DX1.cf5b5b49-de5e-4f2e-90b4-b138f55560a9 | 6 | 0 | 6 | 等權錯 → softmax對 |

## 3. 預測落點（P-B 的經驗佐證）

每個 task 佔 8-way label space 的第 2p、2p+1 列。若疊放順序錯位，預測會大量落到別的 task 的列上。

| task | 自己的列 | 落在自己列 | 落到別的 task |
|---|---|---|---|
| tcga_esca | [0, 1] | 14/15 | 1 |
| tcga_rcc | [2, 3] | 76/76 | 0 |
| tcga_brca | [4, 5] | 93/93 | 0 |
| tcga_lung | [6, 7] | 95/95 | 0 |

逐 slide 原始預測：`outputs/verify/per_slide_v9.json`

