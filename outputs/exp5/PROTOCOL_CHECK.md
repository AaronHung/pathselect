# exp5 協定核對

* 本批：`outputs/exp5/chk_m512_f1/per_slide/A5_reverse_seed1_M512_hier_acc_ucur.json`（569 筆）
* 基準：`/workspace/pathselect-exp5/outputs/exp3/sota_ucur/per_slide/A5_reverse_seed1_hier_acc_ucur.json`（569 筆）
* 共同的 (stage, task, slide)：569

每個 stage 的 n 是**該時點評估的所有任務**加總（學完 t 個任務就評估 t 個），
class-IL 也是同一個口徑。

| stage | 評估的任務 | n | 本批 class-IL | 基準 class-IL | selected_idx 不同 | pred 不同 |
|---|---|---|---|---|---|---|
| 0 | esca | 15 | 1.00000000 | 1.00000000 | 0 | 0 |
| 1 | esca, rcc | 91 | 0.96703297 | 0.96703297 | 0 | 0 |
| 2 | brca, esca, rcc | 184 | 0.86413043 | 0.86413043 | 0 | 0 |
| 3 | brca, esca, lung, rcc | 279 | 0.83512545 | 0.83512545 | 0 | 0 |

## 判讀

✅ **全部逐位元相同** → 本批與 exp3 同平台，數字可直接對讀。

