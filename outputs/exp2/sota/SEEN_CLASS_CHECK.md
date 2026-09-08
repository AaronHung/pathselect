# 已見類別限制版指標（seen-class check）

PathSelect（`arm=A5`）十折，四個設定。把 class-IL 的 argmax 從**固定八類**
改成**每階段只在已見類別上**取，重算三個指標並列出差值。

## 為什麼要做這個檢查

本方法的診斷頭是固定的八類 cosine 頭，自第一個任務起就涵蓋全部類別、
不隨任務累積（`docs/CLAIMS.md` C-30）。外部累積式協定在中間階段只在**已見**
類別上預測。固定八類在早期階段更嚴、峰值較低，**可能使 Forgetting 偏小** ——
本檔量化這個差多少。

## 結果

| 設定 | 指標 | 固定八類（現行） | 只在已見類別 | 差 |
|---|---|---|---|---|
| reverse ／ flat | class-IL ACC ↑ | 0.8295 ± 0.029 | 0.8295 ± 0.029 | **+0.0000** |
| reverse ／ flat | Forgetting ↓ | 0.0899 ± 0.043 | 0.0967 ± 0.039 | **+0.0067** |
| reverse ／ flat | BWT ↑ | -0.0821 ± 0.050 | -0.0847 ± 0.051 | **-0.0026** |
| reverse ／ hier | class-IL ACC ↑ | 0.8542 ± 0.044 | 0.8542 ± 0.044 | **+0.0000** |
| reverse ／ hier | Forgetting ↓ | 0.0605 ± 0.041 | 0.0629 ± 0.041 | **+0.0024** |
| reverse ／ hier | BWT ↑ | -0.0561 ± 0.038 | -0.0590 ± 0.038 | **-0.0029** |
| forward ／ flat | class-IL ACC ↑ | 0.8406 ± 0.043 | 0.8406 ± 0.043 | **+0.0000** |
| forward ／ flat | Forgetting ↓ | 0.0549 ± 0.043 | 0.0574 ± 0.046 | **+0.0025** |
| forward ／ flat | BWT ↑ | -0.0452 ± 0.045 | -0.0492 ± 0.048 | **-0.0040** |
| forward ／ hier | class-IL ACC ↑ | 0.8111 ± 0.029 | 0.8111 ± 0.029 | **+0.0000** |
| forward ／ hier | Forgetting ↓ | 0.0612 ± 0.040 | 0.0661 ± 0.040 | **+0.0049** |
| forward ／ hier | BWT ↑ | -0.0528 ± 0.038 | -0.0584 ± 0.039 | **-0.0056** |

## 三點讀法

**① class-IL ACC 的 `+0.0000` 是恆等，不是巧合。** 最終階段的「已見類別」
就是全部八類，兩種取法在該階段逐筆相同；class-IL ACC 只看最終階段。
**因此 `docs/SOTA_TABLE.md` 裡我們與外部方法的 ACC 比較不受此設定差異影響。**

**② 方向 4/4 一致，量級很小。** 改成只在已見類別上取 argmax 後，早期階段
準確率上升、峰值抬高，Forgetting 四個設定**全部變大**、BWT 全部變小 ——
與預期方向相符。最大位移 **0.0067**（reverse／flat），其餘三個 0.0024–0.0049，
**不到 Forgetting 逐折標準差（0.038–0.046）的五分之一**。

**③ 對外部比較的結論不變。** 以反向為例：我們的 Forgetting 0.0899 → 0.0967，
外部重現值 0.064，差距由 +0.026 變成 +0.033 —— 方向與量級的判讀都不改變。

## 方法

逐 slide 存檔（`outputs/exp2/sota/per_slide/*.json`）**沒有存 logits**，
只有 `pred_class_il`（八類 argmax）與 `pred_task_il`。重算的做法是
用存下來的 `selected_idx` 與 `weights_softmax`，經**與評估完全相同的路徑**
（`selector.classifier.conch_classify` + `Ctx.f_txt` + `Ctx.logit_scale`）
重建 logits，再在已見類別的欄位子集上取 argmax。

stage `i` 的已見類別 = 該 order 前 `i+1` 個 task 各自的兩個全域類別欄位
（全域索引固定：esca 0–1、rcc 2–3、brca 4–5、lung 6–7）。

⚠️ **重建忠實度已驗證**：本次抽驗 1600 筆，重建的八類 argmax
與存檔的 `pred_class_il` **全數一致** —— 不一致會直接中止，不會產表。

⚠️ **最終階段不重算**：已見類別即全部八類，直接沿用存檔值。

產生：`python scripts/report_seen_class_check.py`。

