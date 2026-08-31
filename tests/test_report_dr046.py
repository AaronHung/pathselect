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


# ── DR-046 裁定二：ΔUtility 取代比值 ────────────────────────────────────────

def test_delta_utility_replaces_the_ratio_column():
    """比值欄必須消失，ΔUtility 必須在。"""
    md = ROOT / "docs" / "DR046_TABLE.md"
    if not md.exists():
        pytest.skip("尚未產生 DR046_TABLE.md")
    text = md.read_text(encoding="utf-8")
    assert "| ΔUtility |" in text or "ΔUtility |" in text
    header = [ln for ln in text.splitlines() if ln.startswith("| 臂 | n seeds |")]
    assert header, "找不到彙總表表頭"
    for ln in header:
        assert "Utility Retention" not in ln, "比值欄還在表頭裡"
    assert "ΔUtility" in R.COLS and "Utility Retention" not in R.COLS


def test_delta_utility_is_a_difference_of_the_existing_sums():
    """定義釘死：各舊 task 的 (sum_u_at_end − sum_u_at_learn) 平均。"""
    M = {"per_task": {"t0": {"sum_u_at_learn": 10.0, "sum_u_at_end": 4.0},
                      "t1": {"sum_u_at_learn": 20.0, "sum_u_at_end": 25.0},
                      "t2": {"sum_u_at_learn": 99.0, "sum_u_at_end": -99.0}}}
    # t2 是最後一個 task，不算
    assert R.delta_utility(M, ["t0", "t1", "t2"]) == pytest.approx(((4 - 10) + (25 - 20)) / 2)


def test_delta_utility_sign_means_degradation():
    worse = {"per_task": {"a": {"sum_u_at_learn": 10.0, "sum_u_at_end": -10.0},
                          "b": {"sum_u_at_learn": 0.0, "sum_u_at_end": 0.0}}}
    assert R.delta_utility(worse, ["a", "b"]) < 0, "退化時必須為負"


def test_arms_are_auto_detected_not_hardcoded():
    """不得寫死清單：per_slide 裡有什麼臂就納入什麼（DR-046 裁定二）。"""
    recs = R.load_records()
    have = {r["arm"] for r in recs}
    assert set(R.arms_present(recs)) == have, "有臂被漏掉"
    # DISPLAY_ORDER 只影響順序，不是白名單
    fake = [{"arm": "ZZZ_new", "order": R.ORDER}]
    assert "ZZZ_new" in R.arms_present(recs + fake)


def test_arms_flag_rejects_unknown_arms():
    recs = R.load_records()
    with pytest.raises(SystemExit, match="沒有的臂"):
        R.arms_present(recs, ["A1", "NOPE"])


# ── flat / hier 必須分表（PI 裁定，選配 stretch 前置）────────────────────────

def test_arch_of_treats_missing_field_as_flat():
    """舊記錄沒有 arch 欄位（commit e6d13df 之前），main tag 一直是 flat。"""
    assert R.arch_of({"arm": "A1"}) == "flat"
    assert R.arch_of({"arm": "A1", "arch": None}) == "flat"
    assert R.arch_of({"arm": "L2", "arch": "hier"}) == "hier"


def test_split_by_arch_keeps_groups_disjoint():
    recs = [{"arm": "L2", "arch": "flat", "seed": 0},
            {"arm": "L2", "arch": "hier", "seed": 0},
            {"arm": "A1", "seed": 0}]
    g = R.split_by_arch(recs)
    assert set(g) == {"flat", "hier"}
    assert len(g["flat"]) == 2 and len(g["hier"]) == 1
    assert all(r.get("arch") == "hier" for r in g["hier"])


def test_same_arm_in_two_architectures_is_not_merged():
    """同名臂在兩個架構下必須分開算 —— 混算會讓 arm_metrics 疊起兩批 (arm, seed)。"""
    real = R.load_records()
    flat = [r for r in real if R.arch_of(r) == "flat"]
    assert flat, "缺 flat 記錄"
    fake_hier = [dict(r, arch="hier") for r in flat if r["arm"] == "A5"]
    g = R.split_by_arch(flat + fake_hier)
    assert len(g["flat"]) == len(flat)
    assert len(g["hier"]) == len(fake_hier)
    # 混在一起時 A5 的記錄數會翻倍 —— 這正是要避免的
    merged = [r for r in flat + fake_hier if r["arm"] == "A5"]
    assert len(merged) == 2 * len([r for r in flat if r["arm"] == "A5"])


def test_report_marks_non_flat_sections_as_not_comparable():
    md = ROOT / "docs" / "DR046_TABLE.md"
    if not md.exists():
        pytest.skip("尚未產生 DR046_TABLE.md")
    text = md.read_text(encoding="utf-8")
    assert "# 架構：flat（正典基底）" in text, "缺 flat 分節標題"
    assert "flat 與 hier 分表" in text, "缺分表說明"
    # 若已有非 flat 的節，必須帶不可混讀的警語
    import re
    for m in re.finditer(r"^# 架構：(?!flat)(.+)$", text, re.M):
        seg = text[m.end():m.end() + 400]
        assert "不可混讀" in seg, f"{m.group(1)} 節缺不可混讀警語"


def _synthetic_hier_from_flat(flat, arms=("A1", "A2", "A5", "L2")):
    """把 flat 記錄改寫成 arch=hier，並把預測全部打壞。

    ⚠️ 打壞是刻意的：如果自檢誤用了全部記錄（而非只用 flat），這批爛數字會讓
    A5/A2/A1 的重算值偏離 §4.4，自檢就會中止 —— 那正是我們要測出來的。
    """
    out = []
    for r in flat:
        if r["arm"] not in arms:
            continue
        q = dict(r, arch="hier")
        q["pred_class_il"] = (r["true"] + 3) % 8          # 全錯
        q["pred_task_il"] = (r["true"] + 1) % 8
        out.append(q)
    return out


@pytest.fixture
def hier_sandbox(tmp_path, monkeypatch):
    flat = R.load_records()
    recs = flat + _synthetic_hier_from_flat(flat)
    monkeypatch.setattr(R, "load_records", lambda: recs)
    md = tmp_path / "DR046_TABLE.md"
    monkeypatch.setattr(R, "OUT", md)
    return md


def test_hier_section_is_emitted_and_marked_not_comparable(hier_sandbox):
    assert R.main([]) == 0
    text = hier_sandbox.read_text(encoding="utf-8")
    assert "# 架構：flat（正典基底）" in text
    assert "# 架構：hier" in text, "hier 沒有獨立分節"
    seg = text.split("# 架構：hier")[1][:400]
    assert "不可混讀" in seg, "hier 節缺不可混讀警語"


def test_self_check_uses_flat_only_even_when_hier_exists(hier_sandbox):
    """hier 的數字再爛也不該影響自檢 —— 自檢的對照是 §4.4 的 flat 主表。"""
    assert R.main([]) == 0, "自檢被 hier 記錄污染而中止"


def test_flat_summary_is_unchanged_by_the_presence_of_hier(hier_sandbox, tmp_path):
    """加入 hier 記錄後，flat 節的數字必須一個字都不變。"""
    clean = tmp_path / "clean.md"
    flat = [r for r in R.load_records() if R.arch_of(r) == "flat"]
    import types
    saved = R.load_records
    R.load_records = lambda: flat
    R.OUT = clean
    try:
        R.main([])
    finally:
        R.load_records = saved
    R.OUT = hier_sandbox
    R.main([])

    def flat_block(p):
        """只取 flat 節的**表格內容**，切掉結尾邊界。

        ⚠️ 乾淨版接的是頁尾、含 hier 版接的是分隔線，直接比整段會被邊界差異誤導。
        """
        t = p.read_text(encoding="utf-8")
        body = t.split("# 架構：flat（正典基底）")[1].split("# 架構：")[0]
        return body.split("產生：")[0].rstrip().rstrip("-").rstrip()

    assert flat_block(clean) == flat_block(hier_sandbox), \
        "hier 記錄改變了 flat 節的數字 —— 分表沒有生效"
