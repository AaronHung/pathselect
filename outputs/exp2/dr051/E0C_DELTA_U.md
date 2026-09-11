# E0c — ΔU 口徑

commit `a2734ee155e67bf55b142caeb549a89f61ca65f9`。以下引文的行號在產生本檔時逐條自檢（該行必須含所引片段）。

## (i) 現行腳本的口徑

Table 3 的 ΔU 由 `scripts/report_dr046.py::delta_utility` 產生（`scripts/report_dr046.py:166`、`scripts/report_dr046.py:168`）：對每個舊任務取 `sum_u_at_end − sum_u_at_learn`，再對舊任務取 `statistics.mean`。而 `sum_u_at_*` 是**該任務 test 切片的 `utility_total` 加總**（`scripts/run_exp2.py:620`）。

**答：兩者都不是 —— 是「各任務先對切片加總、再跨任務平均」。**不是逐切片平均，也不是所有舊任務切片一起平均；片數多的任務（brca 93）主導量級。

## (ii) 兩種逐切片平均（fold 1、reverse、seeds 0–4、flat；mean ± sd over seeds）

* **S（現行）**：mean_task Σ_slide ΔU
* **M1（各任務先平均再跨任務平均）**：mean_task mean_slide ΔU
* **M2（所有舊任務切片一起）**：mean over all old-task slides of ΔU

ΔU 逐切片 = `utility_total`(final stage) − `utility_total`(own stage)；`utility_total` = 等權證據下 log C − CE（telescoped counterfactual gain，`selector/utility.py:98`）。

| 臂 | S 現行 | M1 逐任務平均 | M2 全切片平均 | 逐任務 M1（esca / rcc / brca） |
|---|---|---|---|---|
| A1 flat | -220.3837 ± 89.4526 | -3.7961 ± 1.5483 | -3.5932 ± 1.4585 | -4.295 / -3.703 / -3.390 |
| A2 flat | -300.8434 ± 104.4499 | -4.7045 ± 1.4451 | -4.9051 ± 1.7030 | -3.832 / -6.537 / -3.745 |
| A3 flat | -21.2192 ± 11.4455 | -0.7775 ± 0.2471 | -0.3460 ± 0.1866 | -1.911 / -0.247 / -0.175 |
| A4 flat | -31.9359 ± 19.3159 | -0.9500 ± 0.5312 | -0.5207 ± 0.3149 | -2.104 / -0.304 / -0.443 |
| A5 flat | -16.8175 ± 11.9393 | -0.7429 ± 0.3094 | -0.2742 ± 0.1947 | -1.942 / -0.314 / +0.028 |
| A5 hier（補充） | -14.4816 ± 8.3782 | -0.5713 ± 0.2249 | -0.2361 ± 0.1366 | -1.421 / -0.298 / +0.006 |

自檢：S 與 M1 的 A1／A2／A5 皆重現 audit C1 §4（−220.38／−300.84／−16.82 與 −3.796／−4.705／−0.743）✅。
test 片數 esca 15／rcc 76／brca 93；M2 以片數加權（brca 佔 93/184），M1 三任務等權。

## (iii) CE 的 temperature／logit scale

* 沒有另設 temperature。所有 CE 都是 `F.cross_entropy` 直接吃 `logit_scale × cos`：
  * counterfactual gain／`utility_total`：`_ce` 定義於 `selector/utility.py:44`；logits 由 `selector/utility.py:71`（候選）與 `selector/utility.py:53`（當前）產生。
  * hinge 的 U_new：`selector/continual.py:68`，其 `logits_uniform` 來自 `frozen_head` `selector/train.py:67`。
* `logit_scale` 取自 CONCH checkpoint（已 exp），不自訂常數（`selector/text_encoder.py:12`；載入於 `selector/text_encoder.py:142`）。實際值（`outputs/cache/f_txt_*.pt`）= **56.3477**。

## (iv) flat 是否存並蒸餾 r_old

**是。** `r = f_group.score(...)` 在 `run_rounds` 內**不分架構**都會算（`selector/rounds.py:131`）；`fill_memory` 把 `last.r` 存進 entry 作 `r_old`（`selector/train.py:308`）；`continual_terms` 以 `l_kd(entry.r_old, last.r, ...)` 蒸餾（`selector/train.py:337`），group 項係數 `kd_group_weight` 預設 1.0（`scripts/run_exp2.py:233`），只有 A5nG 臂設 0。
因此 flat 的 L_KD group 項是活的（audit C1 §1：flat 的 F_g 只從此項收梯度）；但 flat 下 r 不進入選取、也不進入 head（audit C1 §1），故此項不改變任何輸出。

