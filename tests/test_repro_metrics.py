"""DR-048：外部基準自身輸出的指標解析（`sota/repro_metrics.py`）。

重點是**不重新推導**：他們的 `test_acc.txt` 就是準確率矩陣，我們只把它餵進
與自家表格相同的 `bwt` / `forgetting`。因此測試要盯的是「有沒有讀對、
有沒有把殘缺的折當成完整的」，而不是重算公式（那有 test_sota_metrics 守）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from sota.metrics import bwt, forgetting                              # noqa: E402
from sota.repro_metrics import (EXPECTED_TASKS, find_folds,           # noqa: E402
                                fold_metrics, read_masked, read_matrix)

#: 一份下三角矩陣：第一個任務學完 0.9，之後掉到 0.5
TRI = "[[0.9], [0.8, 0.7], [0.5, 0.6, 0.4]]"
MASK = "[0.95, 0.85, 0.75]"


def _mk(tmp_path: Path, acc=TRI, mask=MASK, fold=1) -> Path:
    d = tmp_path / f"train-data_split_seed_{fold}" / "metrics"
    d.mkdir(parents=True)
    (d / "test_acc.txt").write_text(acc)
    (d / "test_mask_acc.txt").write_text(mask)
    return d


def test_matrix_is_padded_to_square_with_none(tmp_path):
    A = read_matrix(_mk(tmp_path))
    assert A == [[0.9, None, None], [0.8, 0.7, None], [0.5, 0.6, 0.4]]


def test_non_triangular_input_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="下三角"):
        read_matrix(_mk(tmp_path, acc="[[0.9], [0.8, 0.7, 0.6]]"))


def test_masked_is_read_verbatim(tmp_path):
    assert read_masked(_mk(tmp_path)) == [0.95, 0.85, 0.75]


def test_metrics_reuse_the_same_definitions_as_our_own_table(tmp_path):
    """關鍵：Forgetting / BWT 必須是**我們統一的定義**套在他們的矩陣上，
    不是他們或我另外算一套。"""
    m = fold_metrics(_mk(tmp_path), expected_tasks=3)
    A = [[0.9, None, None], [0.8, 0.7, None], [0.5, 0.6, 0.4]]
    assert m["forgetting"] == pytest.approx(forgetting(A))
    assert m["bwt"] == pytest.approx(bwt(A))
    assert m["acc"] == pytest.approx((0.5 + 0.6 + 0.4) / 3)
    assert m["masked_acc"] == pytest.approx((0.95 + 0.85 + 0.75) / 3)


def test_fold_that_died_midway_is_rejected_not_averaged(tmp_path):
    """跑到 task 2 就掛掉的折會留下一個**形狀完全合法**的 2×2 下三角矩陣。

    不對照預期任務數的話，它會被當成「完整的 2 任務實驗」平均進十折 ——
    數字看起來很正常，結果悄悄失真。這是本檔最重要的一條。
    """
    d = _mk(tmp_path, acc="[[0.9], [0.8, 0.7]]", mask="[0.95, 0.85]")
    with pytest.raises(ValueError, match="沒跑完"):
        fold_metrics(d, expected_tasks=4)


def test_masked_length_must_also_match(tmp_path):
    d = _mk(tmp_path, acc="[[0.9], [0.8, 0.7], [0.5, 0.6, 0.4]]", mask="[0.95, 0.85]")
    with pytest.raises(ValueError, match="masked"):
        fold_metrics(d, expected_tasks=3)


def test_expected_tasks_defaults_to_the_benchmark_size():
    assert EXPECTED_TASKS == 4


def test_find_folds_picks_up_fold_number_from_directory_name(tmp_path):
    for k in (1, 3, 10):
        _mk(tmp_path, fold=k)
    assert sorted(find_folds(tmp_path)) == [1, 3, 10]


def test_find_folds_skips_runs_without_metrics(tmp_path):
    _mk(tmp_path, fold=1)
    (tmp_path / "train-data_split_seed_2" / "metrics").mkdir(parents=True)  # 空的
    assert sorted(find_folds(tmp_path)) == [1], "沒有 metrics 的折不該被撿進來"
