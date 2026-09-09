# Audit C1 — 唯讀程式碼稽核

commit `ccc83a8f0727d8875f0e19eb281c09c826c92f19`（工作區乾淨）。
**未修改任何程式碼、未啟動任何訓練。** 所有數字皆來自既有產物或對其重算。

---

## 1. F_g 的梯度路徑

**答：兩種架構的答案不同。**

* **`--arch flat`（主線的 flat 列）**：`softmax(r_j)` **完全不進入** pooled 表示
  `z`、不進入 patch 分數、不進入任何可微損失。F_g **只從 `L_KD` 的 group 層 KL
  收到梯度**；因此**在 task 1（記憶體為空、不進 `continual_terms`）F_g 收到的
  task-relevant 梯度為零**。
* **`--arch hier`**：透過 straight-through 注入，`softmax(r_j)` 以 forward 恆為 1 的
  乘法因子進入 `ste_mask` → pooled `z` → logits → `L_diag`，F_g 因此收到梯度。

### 存在 straight-through，但只接在 hierarchy 分支

`selector/rounds.py::run_rounds`，L168–198：

```python
        a_soft = None
        if group_grad == "ste_allocation":
            active = grouping.mask & (cap > 0)
            a_soft = torch.zeros_like(r)
            if bool(active.any()):
                idx_a = active.nonzero(as_tuple=False).reshape(-1)
                a_soft = a_soft.index_copy(
                    0, idx_a, torch.softmax(r.index_select(0, idx_a), dim=0))

        picks, ste = [], torch.zeros_like(s)
        if not hierarchy:
            flat_idx = topk_indices(s, c_this, mask=state.available_mask)
            picks.append(flat_idx)
            ste = ste + straight_through_topk(s, c_this, temperature=temperature,
                                              mask=state.available_mask)
```

`a_soft` 雖然在 flat 下也被算出來，但**只在下面這個迴圈裡被使用**，而該迴圈的
range 在 flat 下是空的（`for j in (range(...) if hierarchy else ())`），L186–196：

```python
        for j in (range(grouping.num_groups) if hierarchy else ()):
            k = int(b[j])
            if k <= 0:
                continue
            member = (grouping.assignment == j) & state.available_mask     # [n]
            picks.append(topk_indices(s, k, mask=member))
            m_j = straight_through_topk(s, k, temperature=temperature, mask=member)
            if a_soft is not None:
                # forward 恆等於 1，只把梯度注入 r_j；head 的數值完全不變
                m_j = m_j * (a_soft[j] / a_soft[j].detach().clamp_min(1e-12))
            ste = ste + m_j
```

`ste` 進入 head 的路徑，`selector/train.py::frozen_head` L60–66：

```python
    e = (torch.exp(s - s.max().detach()) if weighting == "softmax"
         else torch.ones_like(s))
    w_un = ste_mask * e
    w = w_un / w_un.sum().clamp_min(EPS)
    pooled = F.normalize((w.unsqueeze(-1) * Z).sum(0, keepdim=True), dim=-1)
    return logit_scale * (pooled @ f_txt.to(pooled.dtype).t())
```

### 逐一損失項對 F_g 的梯度

| 損失 | 觸及的張量 | 到得了 F_g？ |
|---|---|---|
| `L_diag` | `logits` ← `ste_mask`, `s`, `Z` | **hier 可**（經 `a_soft`）／**flat 不可** |
| `L_sem` | `patch_score`（= `s`）與 prior，`train.py:103` | **否** —— 不含 `r` |
| `L_util` | `patch_score` 的候選切片，`train.py:110` | **否** —— 不含 `r` |
| `L_KD` | `_kl(r_old, r_new)` + `_kl(s_old, s_new)`，`continual.py::l_kd` | **是**（group 項，`group_weight != 0` 時） |
| `L_eq` | `logits_uniform` ← `ste_mask`, `Z`（`weighting="uniform"`，`s` 被 `ones_like` 取代） | **hier 可**／**flat 不可** |

`GROUP_GRAD_MODES` 與其設計理由記於 `selector/rounds.py` L36–48（`"ste_allocation"`
為主線預設，`"none"` 僅供 ablation）。

---

## 2. 池化權重

**(a) head 用 softmax(s) 加權池化 —— 是。** `selector/train.py::frozen_head`
L50–66（程式碼見上），`weighting="softmax"` 為預設；`w_un = ste_mask * exp(s - max)`
再正規化。

**(b) counterfactual teacher `u_i` 用等權平均 —— 是。**
`selector/utility.py::counterfactual_gain` L68–70：

```python
    E_cand = (evidence_sum.reshape(1, -1) + X_cand) / (n_selected + 1)   # [N, D]
    E_cand = F.normalize(E_cand, dim=-1)
    logits_cand = logit_scale * (E_cand @ f_txt.t())                     # [N, C]
```

`evidence_sum` 為純和（`selector/state.py::evidence_sum` 回傳 `self._sum`），
除以 `n_selected + 1` 即等權平均。該近似在 `utility.py` 模組 docstring 明列為
pre-registered。

**(c) hinge 裡的 `U(P)` 用等權。** `selector/train.py::continual_terms` L340–345：

```python
    if use_eq:
        _idx, pos = selected_from_entry(entry, budget)
        u_old = float(entry.u_old.index_select(0, pos).sum())
        logits_uniform = frozen_head(Z, last.s, ste, f_txt, logit_scale,
                                     weighting="uniform")
        eq = l_eq(differentiable_utility(logits_uniform, label), u_old, mode=eq_mode)
```

`differentiable_utility` = `log C − CE(logits_uniform, y)`
（`selector/continual.py`）。因此 hinge 的兩側（`u_old` 與 `U_new`）**都是等權**，
口徑一致。

---

## 3. `U_old` 的來源

**答：讀取記憶體 entry 的既存欄位，不重算、不需要舊 selector 快照。**

`selector/train.py::continual_terms` L341–342（見上）：
`u_old = float(entry.u_old.index_select(0, pos).sum())`

* `pos` 由 `selector/memory.py::selected_from_entry` 取得 —— 在候選中依 `s_old`
  取 top-`budget` 的位置（entry 不直接記錄「選了誰」，用 `s_old` 還原）。
* `entry.u_old` 於寫入記憶體時算好：`selector/train.py::fill_memory` L305–307
  呼叫 `counterfactual_gain(res.state.evidence_sum(), res.state.n_selected, ...)`。
* `float(...)` 使其成為**常數**，不帶梯度。

因此 `U_old` 是**寫入時凍結的純量**，`U_new` 才是當前 selector 算的。

---

## 4. Table 3 的 ΔU 單位

**產生器**：`scripts/report_dr046.py::delta_utility` L163–168：

```python
def delta_utility(M_seed, tasks: list[str]) -> float:
    """ΔUtility = 各舊 task 的 (U_final − U_own) 平均。越高越好，負值 = 退化。"""
    per = M_seed.get("per_task", {})
    d = [per[t]["sum_u_at_end"] - per[t]["sum_u_at_learn"]
         for t in tasks[:-1] if t in per]
    return statistics.mean(d) if d else float("nan")
```

`sum_u_at_*` 定義於 `scripts/run_exp2.py::arm_metrics` L620–621：

```python
            "sum_u_at_learn": sum(r["utility_total"] for r in at_i),
            "sum_u_at_end": sum(r["utility_total"] for r in at_e),
```

| 問項 | 答 |
|---|---|
| sum 還是 mean | **對 slide 加總**，再對「舊 task」取平均（`tasks[:-1]`，即最後一個任務不計） |
| 哪個 test set | 各 task **自己的 test split**；fold 1 的片數 esca **15**／rcc **76**／brca **93** |
| CE 內的 logit scale | CONCH 的 `logit_scale`（`Ctx.logit_scale`，由 `build_f_txt` 取得），未另設溫度 |
| 加權 head 還是等權 surrogate | **等權 surrogate** —— `utility_total` 由 `sequential_utility_total` 產生，其內部呼叫 `counterfactual_gain`（等權，見第 2(b) 項） |

### 重算為逐 slide 平均

（fold 1、seeds 0–4、flat；先對 slide 取平均再對舊 task 取平均）

| 臂 | 稿內值（逐 task 加總後平均） | 逐 slide 平均 |
|---|---|---|
| Sequential fine-tuning（A1） | −220.38 ± 89.45 | **−3.796 ± 1.548** |
| LoRA merge only（A2） | −300.84 ± 104.45 | **−4.705 ± 1.445** |
| **Full preservation（A5, flat）** | **−16.82 ± 11.94** | **−0.743 ± 0.309** |

⚠️ 兩種口徑的**相對關係不變**（A2 < A1 < A5），但絕對量級差約 60–70 倍，
因為各 task 的 test 片數不同（15／76／93），加總口徑會讓片數多的 task 主導。

---

## 5. Hinge 觸發率

**有記錄，不需重跑。** 記錄點在 `scripts/run_exp2.py` L237–239：

```python
                if eq is not None:
                    eq_seen += 1
                    eq_fired += int(float(eq.detach()) > 0.0)
```

寫入 per_slide 的 `l_eq_fire_rate` / `l_eq_steps` 欄位。
A5 flat、fold 1、seeds 0–4 的平均：

| stage | 任務 | `max(0, U_old − U_new) > 0` 的比例 | replay 步數 |
|---|---|---|---|
| 0 | tcga_esca | **無**（記憶體為空，不進 `continual_terms`） | 0 |
| 1 | tcga_rcc | **0.0169** | 3080 |
| 2 | tcga_brca | **0.0305** | 3815 |
| 3 | tcga_lung | **0.0740** | 3870 |

⚠️ hinge 在 **1.7%–7.4%** 的 replay 步驟才啟動，且隨任務數遞增。

---

## 6. Replay 與 hinge 的重疊

**同一筆重放 slide、同一步 —— 是。** `selector/train.py::continual_terms`
L340–348，兩者共用同一次 `run_rounds` 的 `Z`、`last.s`、`ste`：

```python
    if use_eq:
        _idx, pos = selected_from_entry(entry, budget)
        u_old = float(entry.u_old.index_select(0, pos).sum())
        logits_uniform = frozen_head(Z, last.s, ste, f_txt, logit_scale,
                                     weighting="uniform")
        eq = l_eq(differentiable_utility(logits_uniform, label), u_old, mode=eq_mode)
    if use_replay:
        replay = l_diag(frozen_head(Z, last.s, ste, f_txt, logit_scale), label)
```

**但「兩個梯度都是 dCE_new/dθ」的說法要修正 —— 部分成立。** 兩者確實都是同一張
重放 slide 上的 cross-entropy 梯度，但**經過不同的池化**：

* `replay` = `CE(frozen_head(..., weighting="softmax"))` —— softmax(s) 加權池化
* hinge 啟動時 `eq = u_old − (log C − CE_uniform)`，故 `d(eq)/dθ = +d(CE_uniform)/dθ`
  —— **等權**池化（`weighting="uniform"` 時 `e = ones_like(s)`，分數 `s` 只影響
  「選誰」不影響「權重」）

因此兩者的梯度方向高度相關但**不相同**：softmax 分支的梯度會流經 `s` 的權重，
等權分支不會。`u_old` 是常數（見第 3 項），不貢獻梯度。

---

## 7. 正向順序軌跡

輸出於 [`docs/audit_C1_forward_trajectory/`](audit_C1_forward_trajectory/)，
資料源 `outputs/exp2/sota/per_slide/A5_main_*.json`（10 折，**未重跑**）：

| 檔案 | 內容 |
|---|---|
| `A5_flat_forward_accuracy_matrix.csv` | 100 列 = 10 折 × 10 個 (stage, eval_task) 格 |
| `A5_hier_forward_accuracy_matrix.csv` | 同上 |
| `A5_flat_forward_group_quota.csv` | 逐折逐 (stage, task) 的 `b_j` 平均（8 組） |
| `A5_hier_forward_group_quota.csv` | 同上 |

欄位：`fold, stage, stage_task, eval_task, class_il, task_il, n_slides`
與 `fold, stage, eval_task, b_0..b_7`。

⚠️ **`b_j` 是評估時的實際配額**（來自 per_slide 的 `group_quota`，即該 slide 被
選中的 patch 在 8 個組上的分佈），**不是訓練期的配額紀錄** —— 訓練期的 `b_j`
未落檔。flat 列的 `b_j` 由 `run_exp2.evaluate` 事後統計得出，並非 `allocate()`
的輸出（flat 不走分組配額）。

---

## NOT FOUND

* **訓練期的 group 配額 `b_j`**：`RoundRecord.b` 存在於記憶體中，但**未寫入任何
  產物**。第 7 項的 `b_j` 因此只能取自評估期。
* **`L_util` 的 hinge 觸發率**：只有 `L_eq` 有 `l_eq_fire_rate`，`L_util` 無對應欄位。
