"""DR-048 B6/B7：zero-shot 選擇器與 SOTA 主表的彙總。

不碰資料集 —— 這裡只測「怎麼選」與「怎麼彙總」，模型前向與資料載入由
`tests/test_sota_metrics.py` 那條與真實產物交叉核對的路徑負責。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from sota.report_sota import agg, cell, load_runs                 # noqa: E402
from sota.zeroshot import VARIANTS, select                        # noqa: E402

TASKS = ["t0", "t1"]


# ── zero-shot 的選擇 ────────────────────────────────────────────────────────

def test_meanpool_takes_every_patch():
    assert torch.equal(select("meanpool", 37, 8, torch.Generator()), torch.arange(37))


def test_rand8_respects_the_budget_and_stays_in_range():
    g = torch.Generator().manual_seed(0)
    idx = select("rand8", 500, 8, g)
    assert idx.numel() == 8 and idx.min() >= 0 and idx.max() < 500
    assert idx.unique().numel() == 8, "隨機抽樣不得重複"


def test_rand8_falls_back_to_all_patches_when_slide_is_smaller_than_budget():
    idx = select("rand8", 3, 8, torch.Generator().manual_seed(0))
    assert sorted(idx.tolist()) == [0, 1, 2]


def test_rand8_is_reproducible_for_the_same_generator_seed():
    a = select("rand8", 200, 8, torch.Generator().manual_seed(7))
    b = select("rand8", 200, 8, torch.Generator().manual_seed(7))
    assert torch.equal(a, b)


def test_rand8_actually_varies_with_the_seed():
    """反向對照：不同種子必須選到不同的 patch，否則上一條沒有意義。"""
    a = select("rand8", 200, 8, torch.Generator().manual_seed(7))
    b = select("rand8", 200, 8, torch.Generator().manual_seed(8))
    assert not torch.equal(a, b)


def test_variant_labels_do_not_collide_with_training_arms():
    from run_exp2 import ARMS
    assert not (set(VARIANTS.values()) & set(ARMS)), "參照線的臂名撞到訓練臂"


# ── 主表彙總 ────────────────────────────────────────────────────────────────

def _rec(arm, fold, seed, stage, task, ok, arch="flat"):
    return {"arm": arm, "order": "reverse", "seed": seed, "fold": fold,
            "arch": arch, "stage": stage, "task": task, "true": 1,
            "pred_class_il": 1 if ok else 0, "pred_task_il": 1 if ok else 0}


def _write(tmp_path: Path, name: str, recs) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(recs))
    return p


def test_load_runs_keys_by_arm_arch_and_separates_folds(tmp_path):
    _write(tmp_path, "a.json", [_rec("A5", 1, 1, 0, "t0", True)])
    _write(tmp_path, "b.json", [_rec("A5", 2, 2, 0, "t0", True)])
    _write(tmp_path, "c.json", [_rec("A5", 1, 1, 0, "t0", True, arch="hier")])
    runs = load_runs(tmp_path, "reverse")
    assert set(runs) == {("A5", "flat"), ("A5", "hier")}
    assert set(runs[("A5", "flat")]) == {(1, 1), (2, 2)}, "不同 fold 必須是不同的 run"


def test_load_runs_filters_by_order(tmp_path):
    r = _rec("A5", 1, 1, 0, "t0", True)
    r["order"] = "main"
    _write(tmp_path, "a.json", [r])
    assert load_runs(tmp_path, "reverse") == {}


def test_agg_averages_over_runs_not_over_slides(tmp_path):
    """兩個 run，ACC 分別 1.0 與 0.0 → 平均 0.5，即使片數不同。"""
    runs = {(1, 1): [_rec("A5", 1, 1, s, t, True)
                     for s in range(2) for t in TASKS[:s + 1]],
            (2, 2): [_rec("A5", 2, 2, s, t, False)
                     for s in range(2) for t in TASKS[:s + 1]] * 5}
    a = agg(runs, TASKS)
    assert a["n"] == 2 and a["acc"][0] == pytest.approx(0.5)


def test_agg_skips_incomplete_runs_and_reports_them():
    """最終階段少一個任務的 run 不能被算進去，也不能被靜默丟掉。"""
    good = [_rec("A5", 1, 1, s, t, True) for s in range(2) for t in TASKS[:s + 1]]
    bad = [_rec("A5", 2, 2, 0, "t0", True)]          # 只跑到 stage 0
    a = agg({(1, 1): good, (2, 2): bad}, TASKS)
    assert a["n"] == 1 and a["skipped"] == [(2, 2)]


def test_agg_reports_zero_sd_for_a_single_run():
    runs = {(1, 1): [_rec("A5", 1, 1, s, t, True) for s in range(2) for t in TASKS[:s + 1]]}
    assert agg(runs, TASKS)["acc"] == (1.0, 0.0)


def test_cell_renders_missing_metric_as_dash():
    assert cell(None) == "—" and cell((0.5, 0.01)) == "0.500 ± 0.010"
