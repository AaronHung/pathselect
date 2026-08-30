"""DR-046 Phase 0 報表的自檢必須有牙齒（規格 C-8）。

⚠️ 為什麼需要這幾條：在**正確**資料上，把容差放寬到 1.0、或把「不符就中止」
   改成 pass，腳本一樣跑得過 —— 光靠執行一次抓不到。這兩件事只能用測試釘住。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import report_dr046 as R                                             # noqa: E402


def test_tolerance_is_the_spec_value():
    """容差是規格寫死的 5e-4，不得放寬。"""
    assert R.TOL == 5e-4


def test_self_check_aborts_when_numbers_disagree(monkeypatch):
    """比對不符時必須 SystemExit，不得只印警告後照樣產表。"""
    real = R.dossier_values()
    assert {"A1", "A2", "A5"} <= set(real), "RESULTS_DOSSIER §4.4 缺對照臂"
    bad = dict(real)
    bad["A5"] = (real["A5"][0] + 0.01, real["A5"][1])       # 偏離遠大於容差
    monkeypatch.setattr(R, "dossier_values", lambda: bad)
    recs = R.load_records()
    from run_exp2 import ORDERS
    from selector.text_encoder import load_config
    with pytest.raises(SystemExit):
        R.self_check(recs, ORDERS[R.ORDER], list(load_config()["tasks"]))


def test_self_check_passes_on_the_real_numbers():
    recs = R.load_records()
    from run_exp2 import ORDERS
    from selector.text_encoder import load_config
    lines = R.self_check(recs, ORDERS[R.ORDER], list(load_config()["tasks"]))
    assert sum("✅" in ln for ln in lines) == 6, "應有 3 臂 × 2 指標皆通過"
    assert not any("❌" in ln for ln in lines)


def test_dossier_values_parses_the_main_table():
    v = R.dossier_values()
    # 只斷言結構與範圍，不寫死數字（數字的正確性由自檢本身負責）
    for arm in ("A1", "A2", "A3", "A4", "A5", "R1", "R2"):
        assert arm in v, f"§4.4 少了 {arm}"
        ti, ci = v[arm]
        assert 0.0 <= ci <= ti <= 1.0, f"{arm} 的 (task-IL, class-IL) 不合理：{v[arm]}"


def test_reference_arms_have_no_forgetting_column():
    """R1 / R2 的 Forgetting / Jaccard / Utility 必須標「—」（DR-011）。"""
    md = (ROOT / "docs" / "DR046_TABLE.md")
    if not md.exists():
        pytest.skip("尚未產生 DR046_TABLE.md")
    text = md.read_text(encoding="utf-8")
    # 只看指標表；最後的「略過的 slide 數」只有兩欄，不適用本規則
    metric_part = text.split("## Utility Retention 略過的 slide 數")[0]
    # ⚠️ 期望值寫死，**不從 R.NOT_APPLICABLE 取** —— 那樣把它改成空集合會讓
    #    迴圈不執行、測試空洞地通過（與 verify_doc_numbers 踩過的同一個坑）。
    assert R.NOT_APPLICABLE == {"R1", "R2"}, "參照臂的清單被改動"
    for arm in ("R1", "R2"):
        rows = [ln for ln in metric_part.splitlines() if ln.startswith(f"| {arm} |")]
        assert rows, f"表中找不到 {arm}"
        for ln in rows:
            assert "—" in ln, f"{arm} 該列沒有標「—」：{ln}"


def test_table_a_final_matches_dossier_main_table():
    """彙總表的 A_Final 必須等於 §4.4 的 class-IL（同一個量）。"""
    md = ROOT / "docs" / "DR046_TABLE.md"
    if not md.exists():
        pytest.skip("尚未產生 DR046_TABLE.md")
    text = md.read_text(encoding="utf-8")
    block = text.split("## 彙總（mean ± sd，class-IL）")[1].split("## 彙總")[0]
    ref = R.dossier_values()
    seen = 0
    for ln in block.splitlines():
        m = re.match(r"^\| (A[1-5]|R[12]) \| \d+ \| ([\d.]+) ±", ln)
        if not m:
            continue
        arm, got = m.group(1), float(m.group(2))
        assert abs(got - ref[arm][1]) <= R.TOL, (
            f"{arm} 的 A_Final {got} 與 §4.4 的 class-IL {ref[arm][1]} 不符")
        seen += 1
    assert seen == 7, f"只比對到 {seen} 臂"


# ── 指標定義本身（用手造的矩陣釘死，不依賴真實資料）──────────────────────

#  a[s][j]：第 s 階段評第 j 個 task。下三角，None = 尚未學到。
#     j=0    j=1    j=2
#  s=0 0.90   —      —
#  s=1 0.70  0.80    —
#  s=2 0.50  0.60   0.70
A = [[0.90, None, None],
     [0.70, 0.80, None],
     [0.50, 0.60, 0.70]]


def test_a_final_is_the_last_row_mean():
    assert R.a_final(A) == pytest.approx((0.50 + 0.60 + 0.70) / 3)


def test_plasticity_is_the_diagonal_mean():
    """對角線 = 剛學完該 task 當下 —— 不是最後一列。"""
    assert R.plasticity(A) == pytest.approx((0.90 + 0.80 + 0.70) / 3)
    assert R.plasticity(A) != pytest.approx(R.a_final(A)), "對角線不該等於最後一列"


def test_forgetting_uses_max_over_stages_not_just_the_learning_stage():
    """max_s a[s][j] − a[final][j]，只算舊 task（j < final）。

    ⚠️ 這裡刻意讓 j=0 的最大值出現在 s=0（0.90），若實作只看某一個 stage
    就會算錯。
    """
    want = ((0.90 - 0.50) + (0.80 - 0.60)) / 2
    assert R.forgetting(A) == pytest.approx(want)


def test_forgetting_ignores_the_last_task():
    """最後一個 task 的「學完」與「最終」是同一時點，算進去只會稀釋。"""
    only_last = [[None, None, None], [None, None, None], [None, None, 0.70]]
    assert R.forgetting(only_last) != R.forgetting(only_last) or True  # nan 或空
    v = R.forgetting(only_last)
    assert v != v, "沒有舊 task 時應為 nan，而不是 0"


def test_forgetting_is_positive_when_performance_drops():
    up = [[0.50, None], [0.90, 0.80]]          # j=0 從 0.50 升到 0.90
    assert R.forgetting(up) == pytest.approx(0.0), "沒退步時不應為正"
    down = [[0.90, None], [0.50, 0.80]]
    assert R.forgetting(down) == pytest.approx(0.40)
