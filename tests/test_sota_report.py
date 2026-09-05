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

from sota.report_sota import agg, cell, load_runs, paired         # noqa: E402
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


def test_load_runs_keys_by_arm_arch_order_and_separates_folds(tmp_path):
    _write(tmp_path, "a.json", [_rec("A5", 1, 1, 0, "t0", True)])
    _write(tmp_path, "b.json", [_rec("A5", 2, 2, 0, "t0", True)])
    _write(tmp_path, "c.json", [_rec("A5", 1, 1, 0, "t0", True, arch="hier")])
    runs = load_runs(tmp_path, "reverse")
    assert set(runs) == {("A5", "flat", "reverse"), ("A5", "hier", "reverse")}
    assert set(runs[("A5", "flat", "reverse")]) == {(1, 1), (2, 2)}, \
        "不同 fold 必須是不同的 run"


def test_load_runs_filters_by_order(tmp_path):
    r = _rec("A5", 1, 1, 0, "t0", True)
    r["order"] = "main"
    _write(tmp_path, "a.json", [r])
    assert load_runs(tmp_path, "reverse") == {}


def test_load_runs_keeps_orders_apart_when_both_requested(tmp_path):
    """同一個臂的 forward 與 reverse 是**兩個實驗** —— 併在一起算會錯位，
    因為兩者的 stage→task 對應不同（reverse 的 stage 0 是 ESCA，forward 是 LUNG）。
    """
    fwd = _rec("A5", 1, 1, 0, "t0", True)
    fwd["order"] = "main"
    _write(tmp_path, "a.json", [_rec("A5", 1, 1, 0, "t0", True)])
    _write(tmp_path, "b.json", [fwd])
    runs = load_runs(tmp_path, ["reverse", "main"])
    assert set(runs) == {("A5", "flat", "reverse"), ("A5", "flat", "main")}


# ── 逐折配對 ────────────────────────────────────────────────────────────────

def _run(arm, fold, ok_at_learn, ok_at_end, arch="flat"):
    """一個完整的四階段 run。

    ⚠️ 必須用**真實的 task 名**（`ORDERS["reverse"]`）—— `paired` 是拿
    `ORDERS[order]` 去對 `all_metrics`，合成的 `t0/t1` 對不上，會靜默算不出指標。

    第一個任務在最終階段的對錯由 `ok_at_end` 控制 → 直接決定 Forgetting 的正負。
    """
    from run_exp2 import ORDERS
    tasks = ORDERS["reverse"]
    out = []
    for s in range(len(tasks)):
        for j, task in enumerate(tasks[:s + 1]):
            ok = ok_at_learn if (j == 0 and s == 0) else True
            if j == 0 and s == len(tasks) - 1:
                ok = ok_at_end
            out.append(_rec(arm, fold, fold, s, task, ok, arch))
    return out


def test_paired_counts_higher_is_better_metrics_by_positive_difference(tmp_path):
    for f in (1, 2, 3):
        _write(tmp_path, f"a{f}.json", _run("A5", f, True, True))
        _write(tmp_path, f"b{f}.json", _run("A3", f, True, False))
    runs = load_runs(tmp_path, "reverse")
    rows = {r["label"]: r for r in
            paired(runs, ("A5", "flat", "reverse"), ("A3", "flat", "reverse"))}
    assert rows["ACC"]["better"] == 3 and rows["ACC"]["mean"] > 0


def test_paired_counts_forgetting_the_other_way_round(tmp_path):
    """Forgetting 越小越佳 —— 差值為**負**才算 A 較佳。

    這是最容易寫錯的一條：三軸若一律數正號，Forgetting 的勝場會完全顛倒。
    """
    for f in (1, 2, 3):
        _write(tmp_path, f"a{f}.json", _run("A5", f, True, True))    # 不遺忘
        _write(tmp_path, f"b{f}.json", _run("A3", f, True, False))   # 遺忘
    runs = load_runs(tmp_path, "reverse")
    rows = {r["label"]: r for r in
            paired(runs, ("A5", "flat", "reverse"), ("A3", "flat", "reverse"))}
    assert rows["Forgetting"]["mean"] < 0, "A5 遺忘較少 → 差值應為負"
    assert rows["Forgetting"]["better"] == 3, "負差值必須算成 A 較佳"


def test_paired_uses_only_folds_present_on_both_sides(tmp_path):
    for f in (1, 2, 3):
        _write(tmp_path, f"a{f}.json", _run("A5", f, True, True))
    for f in (2, 3):
        _write(tmp_path, f"b{f}.json", _run("A3", f, True, False))
    runs = load_runs(tmp_path, "reverse")
    rows = paired(runs, ("A5", "flat", "reverse"), ("A3", "flat", "reverse"))
    assert all(r["n"] == 2 for r in rows), "只有兩折兩邊都有"
    assert all([f for f, _x in r["per_fold"]] == [2, 3] for r in rows)


def test_paired_returns_nothing_when_one_side_is_missing(tmp_path):
    _write(tmp_path, "a.json", _run("A5", 1, True, True))
    runs = load_runs(tmp_path, "reverse")
    assert paired(runs, ("A5", "flat", "reverse"), ("A3", "flat", "reverse")) == []


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


# ── 端到端：每一列必須用自己 order 的 task 序 ────────────────────────────────

def test_forward_row_is_computed_with_forward_task_order(tmp_path):
    """forward 列必須用 **forward 的 task 序**算。

    誤用 reverse 的序不會靜默算錯，而是讓整個 run **算不出來被跳過** ——
    reverse 的第一個任務（ESCA）在 forward 記錄裡只出現在最後一階段，
    Forgetting 需要的「最終階段之前的峰值」是空的 → `all_metrics` 拋 ValueError
    → `agg` 把該 run 列入 skipped。所以症狀是 forward 列變成 n=0 而不是數字有誤。
    """
    from run_exp2 import ORDERS
    from sota.report_sota import agg, load_runs, main as report_main

    fwd = ORDERS["main"]
    recs = []
    for s in range(len(fwd)):
        for j, task in enumerate(fwd[:s + 1]):
            ok = not (j == 0 and s == len(fwd) - 1)      # 只有首個任務在最後答錯
            r = _rec("A5", 1, 1, s, task, ok)
            r["order"] = "main"
            recs.append(r)
    _write(tmp_path, "fwd.json", recs)

    runs = load_runs(tmp_path, ["main"])
    key = ("A5", "flat", "main")
    right = agg(runs[key], ORDERS["main"])
    wrong = agg(runs[key], ORDERS["reverse"])
    assert right["n"] == 1 and right["forgetting"] is not None
    assert wrong["n"] == 0 and wrong["forgetting"] is None, \
        "這份合成資料分不出兩種 task 序，測試無效"

    out = tmp_path / "T.md"
    assert report_main(["--src", str(tmp_path), "--out", str(out),
                        "--order", "reverse"]) == 0
    line = [l for l in out.read_text().split("\n") if "forward" in l and "`A5`" in l]
    assert line, "報表裡找不到 forward 那一列"
    assert f"{right['forgetting'][0]:.3f}" in line[0], (
        f"forward 列的 Forgetting 應為 {right['forgetting'][0]:.3f}\n{line[0]}")
    assert "| 1 |" in line[0], f"forward 列的 n runs 應為 1（誤用 reverse 序會變 0）\n{line[0]}"
