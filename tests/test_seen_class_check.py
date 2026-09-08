"""`scripts/report_seen_class_check.py` —— 已見類別限制版指標的產生器。

守兩件事：**唯讀**（除了那一份報表誰都不准動），以及**拒算殘缺折**
（跑到一半的折若被當成完整的平均進去，十折均值會悄悄失真）。

指標本身的公式不在這裡測 —— 它直接沿用 `sota/metrics.py` 的 `forgetting` / `bwt`，
那邊有 `tests/test_sota_metrics.py` 守著。
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

SRC = ROOT / "scripts" / "report_seen_class_check.py"

import report_seen_class_check as R                                   # noqa: E402


# ── 唯讀 ────────────────────────────────────────────────────────────────────

def test_only_writes_the_single_report():
    """靜態檢查：整支腳本只有一處寫檔，且寫的是那份報表。"""
    tree = ast.parse(SRC.read_text())
    writes = [n for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
              and n.func.attr in ("write_text", "write_bytes", "writelines")]
    assert len(writes) == 1, f"預期只有 1 處寫檔，實際 {len(writes)} 處"


def test_no_destructive_or_dataset_writing_calls():
    """不得有刪除／覆寫資料的呼叫。

    只禁**呼叫**，不禁字串 —— 說明文字裡提到 `per_slide/` 是正當的，
    真正要防的是動到它。唯一那處寫檔由上一條測試盯著。
    """
    tree = ast.parse(SRC.read_text())
    banned = {"remove", "unlink", "rmtree", "rmdir", "save", "copy", "move"}
    hits = [n.func.attr for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr in banned]
    assert not hits, f"出現破壞性呼叫：{hits}"


def test_output_path_is_the_declared_artifact():
    assert R.OUT == ROOT / "outputs" / "exp2" / "sota" / "SEEN_CLASS_CHECK.md"


# ── 拒算殘缺折 ──────────────────────────────────────────────────────────────

def test_incomplete_fold_aborts_instead_of_being_averaged(tmp_path, monkeypatch):
    """最終階段缺任務的折必須中止，不得靜默平均進十折。"""
    import json
    from run_exp2 import ORDERS
    tasks = ORDERS["reverse"]
    recs = []
    for s in range(len(tasks) - 1):                 # 少跑最後一個 stage
        for t in tasks[:s + 1]:
            recs.append({"arm": "A5", "order": "reverse", "seed": 1, "fold": 1,
                         "arch": "flat", "stage": s, "task": t, "true": 1,
                         "slide_id": "x", "pred_class_il": 1, "pred_task_il": 1,
                         "selected_idx": [0], "weights_softmax": [1.0]})
    p = tmp_path / "A5_reverse_seed1.json"
    p.write_text(json.dumps(recs))
    with pytest.raises(SystemExit, match="沒跑完"):
        R.fold_matrices("reverse", "flat", 1, p, verify=0)


def test_completeness_is_checked_before_touching_features(tmp_path):
    """殘缺折必須**在讀特徵之前**就被擋下 —— 否則要先花數分鐘才發現不能用。

    合成記錄用的是不存在的 slide_id；若檢查在後面，會先撞 KeyError 而不是
    我們要的 SystemExit。
    """
    import json
    from run_exp2 import ORDERS
    recs = [{"arm": "A5", "order": "reverse", "seed": 1, "fold": 1, "arch": "flat",
             "stage": 0, "task": ORDERS["reverse"][0], "true": 1,
             "slide_id": "不存在的-slide", "pred_class_il": 1, "pred_task_il": 1,
             "selected_idx": [0], "weights_softmax": [1.0]}]
    p = tmp_path / "A5_reverse_seed1.json"
    p.write_text(json.dumps(recs))
    with pytest.raises(SystemExit, match="沒跑完"):
        R.fold_matrices("reverse", "flat", 1, p, verify=0)


# ── 已見類別的欄位子集 ──────────────────────────────────────────────────────

def test_seen_columns_grow_by_two_per_stage():
    ls = ["tcga_esca", "tcga_rcc", "tcga_brca", "tcga_lung"]
    assert R.seen_columns("reverse", 0, ls) == [0, 1]
    assert R.seen_columns("reverse", 1, ls) == [0, 1, 2, 3]
    assert R.seen_columns("reverse", 3, ls) == list(range(8))


def test_seen_columns_follow_the_order_not_the_label_space():
    """forward 的第一個任務是 lung（全域欄位 6–7），不是 esca。

    寫錯成「照 label_space 前 2(i+1) 欄」會在 reverse 下巧合正確、
    在 forward 下全錯 —— 這條專門守它。
    """
    ls = ["tcga_esca", "tcga_rcc", "tcga_brca", "tcga_lung"]
    assert R.seen_columns("main", 0, ls) == [6, 7]
    assert R.seen_columns("main", 1, ls) == [4, 5, 6, 7]


def test_final_stage_sees_every_class_in_both_orders():
    """class-IL ACC 差值恆為 0 的根據：最終階段兩種取法涵蓋同一組類別。"""
    ls = ["tcga_esca", "tcga_rcc", "tcga_brca", "tcga_lung"]
    for order in ("reverse", "main"):
        assert R.seen_columns(order, 3, ls) == list(range(8))


def test_mismatched_reconstruction_aborts(tmp_path):
    """重建的八類 argmax 與存檔不符時必須中止。

    這是整份分析的**安全前提**：若重建不忠實，所有差值都沒有意義。
    做法是拿真實記錄、把 `pred_class_il` 改成錯的值，重建就對不上。
    """
    import json
    from run_exp2 import ORDERS
    src = ROOT / "outputs" / "exp2" / "sota" / "per_slide" / "A5_reverse_seed1.json"
    if not src.is_file():
        pytest.skip("缺 A5_reverse_seed1.json")
    tasks = ORDERS["reverse"]
    real = json.loads(src.read_text())
    # 最終階段每個 task 各留一筆（通過前置的完整性檢查）
    keep = []
    for t in tasks:
        keep += [r for r in real if r["stage"] == len(tasks) - 1 and r["task"] == t][:1]
    bad = dict([r for r in real if r["stage"] == 0][0])
    bad["pred_class_il"] = (bad["pred_class_il"] + 1) % 8      # 故意改錯
    p = tmp_path / "A5_reverse_seed1.json"
    p.write_text(json.dumps(keep + [bad]))
    with pytest.raises(SystemExit, match="重建不符"):
        R.fold_matrices("reverse", "flat", 1, p, verify=5)
