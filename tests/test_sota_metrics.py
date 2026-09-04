"""DR-048 B4：`sota/metrics.py` 的四個指標。

重點在**區分 Forgetting 與 −BWT** —— 兩者只差在基準點取 `A[j][j]` 還是
`max_{l>=j} A[l][j]`。若測試資料裡「學完某任務後準確率不再上升」，兩式會恰好
相等，測試就分不出實作有沒有寫錯。因此本檔刻意同時放了會分離與不會分離的
兩種案例，並附反向對照（把 forgetting 換成 −bwt 必須測得出來）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_exp2 import DEFAULT_ARCH, ORDERS, acc                       # noqa: E402
from sota.metrics import (accuracy_matrix, all_metrics,              # noqa: E402
                          average_accuracy, bwt, forgetting, upper_bound_ratio)

TASKS = ["t0", "t1", "t2"]
PER_SLIDE = ROOT / "outputs" / "exp2" / "main" / "per_slide"


def rec(stage: int, task: str, true: int, cls: int, tsk: int) -> dict:
    return {"stage": stage, "task": task, "true": true,
            "pred_class_il": cls, "pred_task_il": tsk}


def matrix_from(cells: dict[tuple[int, int], list[bool]]):
    """`{(stage, task_idx): [每片是否答對]}` → 記錄串。"""
    out = []
    for (i, j), oks in cells.items():
        out += [rec(i, TASKS[j], 1, 1 if ok else 0, 1 if ok else 0) for ok in oks]
    return out


# ── accuracy_matrix ─────────────────────────────────────────────────────────

def test_matrix_shape_and_values():
    recs = matrix_from({(0, 0): [True, True, False, False],      # 0.50
                        (1, 0): [True, True, True, False],       # 0.75
                        (1, 1): [True, False]})                  # 0.50
    A = accuracy_matrix(recs, TASKS)
    assert A[0][0] == 0.5 and A[1][0] == 0.75 and A[1][1] == 0.5
    assert A[0][1] is None and A[0][2] is None and A[2] == [None, None, None]


def test_matrix_ignores_unknown_task_and_out_of_range_stage():
    recs = matrix_from({(0, 0): [True, True]})
    recs.append(rec(0, "not_a_task", 1, 1, 1))
    recs.append(rec(99, "t0", 1, 0, 0))
    assert accuracy_matrix(recs, TASKS)[0][0] == 1.0


def test_masked_key_reads_task_il_column():
    """兩個欄位不同時，`key` 必須真的切換 —— 否則 Masked ACC 是假的。"""
    recs = [rec(0, "t0", 1, 0, 1)] * 4          # class-IL 全錯、task-IL 全對
    assert accuracy_matrix(recs, TASKS, key="pred_class_il")[0][0] == 0.0
    assert accuracy_matrix(recs, TASKS, key="pred_task_il")[0][0] == 1.0


# ── ACC ─────────────────────────────────────────────────────────────────────

def test_acc_is_mean_over_tasks_not_over_slides():
    """任務片數不同時，逐任務平均 ≠ 逐片平均。基準協定用的是逐任務平均。"""
    A = [[1.0, None], [0.0, 1.0]]
    assert average_accuracy(A) == 0.5


def test_acc_rejects_incomplete_final_row():
    with pytest.raises(ValueError, match="最終階段"):
        average_accuracy([[1.0, None], [1.0, None]])


# ── Forgetting vs BWT：核心區辨 ─────────────────────────────────────────────

def test_monotone_case_forgetting_equals_minus_bwt():
    """學完就只降不升 → 兩式恰好相等（論文多數列的情形）。"""
    A = [[0.9, None, None], [0.7, 0.8, None], [0.5, 0.6, 0.7]]
    assert forgetting(A) == pytest.approx(-bwt(A))
    assert bwt(A) == pytest.approx(((0.5 - 0.9) + (0.6 - 0.8)) / 2)


def test_post_learning_rise_separates_forgetting_from_minus_bwt():
    """學完 t0 之後 t0 還上升（0.9 → 0.95）→ 峰值不在對角線，兩式分離。

    這是唯一能抓出「把 Forgetting 寫成 −BWT」的案例。
    """
    A = [[0.9, None, None], [0.95, 0.8, None], [0.5, 0.6, 0.7]]
    assert bwt(A) == pytest.approx(((0.5 - 0.9) + (0.6 - 0.8)) / 2)      # 基準 = 對角線
    assert forgetting(A) == pytest.approx(((0.95 - 0.5) + (0.8 - 0.6)) / 2)
    assert forgetting(A) > -bwt(A), "峰值高於對角線時 Forgetting 必須嚴格大於 −BWT"


def test_forgetting_peak_excludes_final_stage():
    """峰值只取到 T−2；若誤含最終階段，單調上升的欄位會得到 0 而非負值。"""
    A = [[0.5, None], [0.9, 0.8]]
    assert forgetting(A) == pytest.approx(0.5 - 0.9)     # 含最終階段的話會是 0.0


def test_single_task_has_no_forgetting_or_bwt():
    assert forgetting([[0.9]]) is None and bwt([[0.9]]) is None


def test_upper_bound_ratio_is_none_not_guessed():
    assert upper_bound_ratio([[0.9, None], [0.8, 0.7]]) is None


@pytest.mark.parametrize("fn", [forgetting, bwt])
def test_missing_cell_raises_not_silently_skipped(fn):
    with pytest.raises(ValueError):
        fn([[None, None], [0.8, 0.7]])


# ── 與既有程式交叉核對 ──────────────────────────────────────────────────────

def _real(arm: str = "A5", seed: int = 0):
    p = PER_SLIDE / f"{arm}_reverse_seed{seed}.json"
    if not p.is_file():
        pytest.skip(f"缺 {p}")
    return [r for r in json.loads(p.read_text())
            if (r.get("arch") or DEFAULT_ARCH) == "flat"]


def test_matrix_final_row_agrees_with_run_exp2_acc():
    """同一格用 repo 既有的 `run_exp2.acc` 再算一次，兩者必須相同。"""
    recs, tasks = _real(), ORDERS["reverse"]
    A = accuracy_matrix(recs, tasks)
    last = len(tasks) - 1
    for j, t in enumerate(tasks):
        sub = [r for r in recs if r["stage"] == last and r["task"] == t]
        assert A[last][j] == pytest.approx(acc(sub, "pred_class_il"))


def test_real_data_exercises_the_separating_case():
    """釘住一件事實：真實資料**確實**落在兩式分離的區間。

    否則 `test_post_learning_rise_...` 只是人造案例，無法保證我們回報的
    Forgetting 在真資料上不等於 −BWT（那樣兩個欄位就是重複的）。
    """
    m = all_metrics(_real(), ORDERS["reverse"])
    assert m["forgetting"] > -m["bwt"], "真實資料未觸發峰值高於對角線的情形"


@pytest.mark.parametrize("arm", ["A1", "A2", "A3", "A5"])
def test_forgetting_never_below_minus_bwt_on_real_data(arm):
    """論文 28 組數字的不變量：`Forgetting >= |BWT|`，零反例。"""
    for seed in range(5):
        p = PER_SLIDE / f"{arm}_reverse_seed{seed}.json"
        if not p.is_file():
            continue
        recs = [r for r in json.loads(p.read_text())
                if (r.get("arch") or DEFAULT_ARCH) == "flat"]
        m = all_metrics(recs, ORDERS["reverse"])
        assert m["forgetting"] >= -m["bwt"] - 1e-12, f"{arm} seed{seed} 違反不變量"


def test_masked_acc_is_strictly_higher_than_acc_on_real_data():
    """task-IL 只在自己的類別裡選 → 不低於 class-IL；真實資料上是**嚴格**高。

    用 `>` 不用 `>=`：`>=` 在「Masked ACC 誤讀成 class-IL 欄位」時仍會通過
    （兩個相等），抓不到 `all_metrics` 把 key 傳錯的情形。
    """
    m = all_metrics(_real(), ORDERS["reverse"])
    assert m["masked_acc"] > m["acc"]


def test_all_metrics_routes_each_key_to_the_right_column():
    """合成案例直接釘住路由：class-IL 全錯、task-IL 全對。"""
    recs = [rec(0, "t0", 1, 0, 1), rec(1, "t0", 1, 0, 1), rec(1, "t1", 1, 0, 1)]
    m = all_metrics(recs, TASKS[:2])
    assert m["acc"] == 0.0 and m["masked_acc"] == 1.0
