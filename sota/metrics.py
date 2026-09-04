"""DR-048 B4：外部基準協定的四個持續學習指標。

    ACC（average accuracy，↑）、Masked ACC（↑）、Forgetting（↓）、BWT（↑）

全部從**準確率矩陣** `A[i][j]` = 「訓練完第 i 階段後，在第 j 個任務測試集上的
準確率」導出（`j <= i`，上三角為 None）。矩陣本身由 `outputs/exp2/**/per_slide/*.json`
的逐 slide 預測重算，不另存中間檔。

出處與可信度 —— 逐條標明是**核對過的**還是**推得的**（DR-048 稽核，
基準方法的名稱見 `docs/DR048_PROTOCOL_AUDIT.md`，本檔依 `test_no_banned_deps`
的規定不指名）：

* **ACC**（核對過）：官方 `manager/manager.py` 的
  `average_test_acc = self.sum_test_acc / len(self.cfg['dataset_names'])`
  —— 即「最終階段對各任務測試準確率的**未遮罩**平均」，等同本 repo 的
  class-IL final avg。

* **Masked ACC**（核對過）：官方 `manager/manager.py::_eval_masked_metrics` 的
  `y_hat = y_hat[:, label_shift : label_shift + dataset_subtype_num[...]]`
  （原註解 `# mask irrelevant logits`）—— 先把 logits 切到該任務自己的類別再
  argmax，等同本 repo 的 task-IL。基準論文明講它是 "for reference only"。

* **Forgetting / BWT**（**推得的，非逐字引用**）：論文正文只寫「依
  (Lopez-Paz and Ranzato 2017; Hayes et al. 2018; Fini et al. 2022)」，
  **沒有印出公式**，官方 repo 也只記錄 forgetting 的**過程**（逐 epoch 在舊任務
  val 上的預測），沒有算純量。因此採用被引文獻的標準式：

      BWT        = mean_{j < T-1} ( A[T-1][j] - A[j][j] )
      Forgetting = mean_{j < T-1} ( max_{l in [j, T-2]} A[l][j] - A[T-1][j] )

  **支持這組式子的證據**（`docs/DR048_PROTOCOL_AUDIT.md` 記錄）：從論文表中抽出
  28 組 (Forgetting, BWT)，**每一組都滿足 `Forgetting >= |BWT|`，零反例**；其中
  多數恰好相等，少數明顯分離（最極端者 0.058 vs −0.021）。這正是上式的指紋 ——
  兩者只差在基準點（`A[j][j]` vs `max_{l>=j} A[l][j]`），故恆有
  `Forgetting >= -BWT`，並在「學完任務 j 之後準確率還會再上升」時分離。
  這同時**否證**了「Forgetting 直接定義為 −BWT」的簡化讀法。

  ⚠️ 這是推論。若日後取得作者的公式或原始碼，須回來核對本檔。

* **Upper-bound Ratio**：需要 JointTrain 上界，本 repo 沒有跑，
  一律回報 `None`（不猜、不用別的東西代替）。

Forgetting / BWT 在 `T < 2` 時分母為 0，回報 `None`。
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence

#: per_slide 記錄裡的預測欄位。ACC 用未遮罩，Masked ACC 用遮罩後。
CLASS_IL = "pred_class_il"
TASK_IL = "pred_task_il"


def accuracy_matrix(records: Iterable[dict], tasks: Sequence[str], *,
                    key: str = CLASS_IL) -> list[list[float | None]]:
    """把逐 slide 預測摺成 `A[i][j]`（第 i 階段之後、第 j 個任務的準確率）。

    `records` 必須已經篩到**單一** (arm, order, seed, arch)；本函式不做篩選，
    混入別的臂會靜默算出錯的矩陣。`tasks` 是該 order 的任務順序。
    """
    T = len(tasks)
    idx = {t: j for j, t in enumerate(tasks)}
    hit: list[list[int]] = [[0] * T for _ in range(T)]
    tot: list[list[int]] = [[0] * T for _ in range(T)]
    for r in records:
        i, j = r["stage"], idx.get(r["task"])
        if j is None or not 0 <= i < T:
            continue
        tot[i][j] += 1
        hit[i][j] += int(r[key] == r["true"])
    return [[(hit[i][j] / tot[i][j] if tot[i][j] else None) for j in range(T)]
            for i in range(T)]


def _final_row(A: Sequence[Sequence[float | None]]) -> list[float]:
    row = A[len(A) - 1]
    vals = [v for v in row if v is not None]
    if len(vals) != len(row):
        raise ValueError(f"最終階段缺少任務的評估（{len(vals)}/{len(row)}）—— "
                         "ACC 是對全部任務平均，缺一格就不能算")
    return vals


def average_accuracy(A: Sequence[Sequence[float | None]]) -> float:
    """ACC：最終階段對**全部**任務準確率的平均。"""
    vals = _final_row(A)
    return sum(vals) / len(vals)


def bwt(A: Sequence[Sequence[float | None]]) -> float | None:
    """BWT = mean_{j<T-1} ( A[T-1][j] − A[j][j] )。越大越好。"""
    T = len(A)
    if T < 2:
        return None
    d = []
    for j in range(T - 1):
        final, own = A[T - 1][j], A[j][j]
        if final is None or own is None:
            raise ValueError(f"BWT 需要 A[{T - 1}][{j}] 與 A[{j}][{j}]，有缺格")
        d.append(final - own)
    return sum(d) / len(d)


def forgetting(A: Sequence[Sequence[float | None]]) -> float | None:
    """Forgetting = mean_{j<T-1} ( max_{l in [j, T-2]} A[l][j] − A[T-1][j] )。越小越好。

    峰值取在**最終階段之前**的所有階段（含剛學完 j 的那一階段），
    這是它與 −BWT 唯一的差別。
    """
    T = len(A)
    if T < 2:
        return None
    d = []
    for j in range(T - 1):
        peak = [A[l][j] for l in range(j, T - 1) if A[l][j] is not None]
        final = A[T - 1][j]
        if not peak or final is None:
            raise ValueError(f"Forgetting 需要第 {j} 欄 [{j}, {T - 2}] 的峰值與 A[{T - 1}][{j}]")
        d.append(max(peak) - final)
    return sum(d) / len(d)


def upper_bound_ratio(A: Sequence[Sequence[float | None]]) -> None:
    """Upper-bound Ratio：需要 JointTrain 上界，本 repo 沒有跑 → 恆為 `None`。"""
    return None


def all_metrics(records: Iterable[dict], tasks: Sequence[str]) -> dict[str, float | None]:
    """一次算完四個指標。ACC / Forgetting / BWT 走未遮罩，Masked ACC 走遮罩。"""
    recs = list(records)
    A = accuracy_matrix(recs, tasks, key=CLASS_IL)
    Am = accuracy_matrix(recs, tasks, key=TASK_IL)
    return {"acc": average_accuracy(A),
            "masked_acc": average_accuracy(Am),
            "forgetting": forgetting(A),
            "bwt": bwt(A),
            "upper_bound_ratio": upper_bound_ratio(A)}
