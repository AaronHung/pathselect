"""E1 階層版報告的跨容量配對與 DR-019 重驗（DR-042）。

⚠️ 效率倍數是論文會寫進 abstract 的數字。這裡把「倍數只能來自 5/5 systematic 的
   跨容量配對」這條規則釘死，事後放寬會被抓到。
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import report_memory_hier as R                                        # noqa: E402

SEEDS = [0, 1, 2, 3, 4]


def fake(diffs_by_pair):
    """造一個 M：{(arm, cap): {seed: {metric: value}}}。"""
    M = {}
    for (arm, cap), vals in diffs_by_pair.items():
        M[(arm, cap)] = {s: {"per_task": {"t": 1}, **{k: v[i] for k, v in vals.items()}}
                         for i, s in enumerate(SEEDS)}
    return M


def const(val):
    return [val] * 5


# ── 跨容量配對 ─────────────────────────────────────────────────────────────

def test_cross_pairs_are_the_four_the_pi_specified():
    assert R.CROSS == [("A5", 64, "A3", 1024), ("A5", 128, "A3", 1024),
                       ("A5", 64, "A3", 512), ("A5", 128, "A3", 512)]


def test_paired_is_per_seed_not_mean_difference():
    """配對 = 同 seed 相減。均值相同但逐 seed 方向不同時，win count 必須反映出來。"""
    M = fake({("A5", 128): {"final_task_il": [0.10, 0.02, 0.02, 0.02, 0.02]},
              ("A3", 512): {"final_task_il": [0.13, 0.01, 0.01, 0.01, 0.01]}})
    d, mean, _sd, w, n = R.paired(M, SEEDS, "A5", 128, "A3", 512, "final_task_il", True)
    assert n == 5 and w == 4, "一個 seed 為負必須反映在 win count"
    assert mean == pytest.approx(statistics.mean(d))


def test_paired_direction_flips_for_lower_is_better_metrics():
    M = fake({("A5", 128): {"mean_leak": const(0.10)},
              ("A3", 512): {"mean_leak": const(0.20)}})
    assert R.paired(M, SEEDS, "A5", 128, "A3", 512, "mean_leak", False)[3] == 5
    assert R.paired(M, SEEDS, "A5", 128, "A3", 512, "mean_leak", True)[3] == 0


def test_paired_returns_none_when_a_cell_is_missing():
    M = fake({("A5", 128): {"final_task_il": const(0.9)}})
    assert R.paired(M, SEEDS, "A5", 128, "A3", 512, "final_task_il", True) is None


# ── 倍數只能來自 5/5 systematic ─────────────────────────────────────────────

def _M(task_il):
    """task_il: {(arm, cap): [5 個值]}"""
    return fake({k: {"final_task_il": v, "final_class_il": v,
                     "mean_leak": const(0.1), "mean_jaccard": const(0.1)}
                 for k, v in task_il.items()})


def test_multiplier_requires_five_of_five():
    """4/5 directional 不得產生倍數宣稱。"""
    M = _M({("A5", 128): [0.90, 0.90, 0.90, 0.90, 0.80],
            ("A3", 512): const(0.85), ("A3", 1024): const(0.99),
            ("A5", 64): const(0.50)})
    text = "\n".join(R.efficiency_section(M, SEEDS, [64, 128, 512, 1024]))
    assert "沒有任何跨容量配對達到 5/5" in text
    assert "× 記憶體效率" not in text.split("### class-IL")[0].replace(
        "記憶體效率主張", "")


def test_multiplier_takes_the_largest_supported_ratio():
    """兩個比較都 5/5 時取倍數較大者。"""
    M = _M({("A5", 64): const(0.95), ("A5", 128): const(0.95),
            ("A3", 512): const(0.85), ("A3", 1024): const(0.85)})
    text = "\n".join(R.efficiency_section(M, SEEDS, [64, 128, 512, 1024]))
    assert "在測試範圍內達 16× 記憶體效率" in text


def test_efficiency_section_always_states_the_unsaturated_caveat():
    M = _M({("A5", 64): const(0.95), ("A5", 128): const(0.95),
            ("A3", 512): const(0.85), ("A3", 1024): const(0.85)})
    text = "\n".join(R.efficiency_section(M, SEEDS, [64, 128, 512, 1024]))
    assert "未飽和" in text and "測試範圍內的下界" in text and "不是 A3 需求的上界" in text


def test_efficiency_section_records_the_retraction():
    M = _M({("A5", 128): const(0.9), ("A3", 512): const(0.8)})
    text = "\n".join(R.efficiency_section(M, SEEDS, [128, 512]))
    assert "「8×」宣稱，該宣稱已撤回" in text
    assert "std(7.71) > mean(7.72)" in text


def test_efficiency_is_built_on_task_il_not_class_il():
    """class-IL 再漂亮也不能產生倍數。"""
    M = fake({("A5", 128): {"final_task_il": [0.9, 0.9, 0.9, 0.9, 0.5],
                            "final_class_il": const(0.99),
                            "mean_leak": const(0.1), "mean_jaccard": const(0.1)},
              ("A3", 512): {"final_task_il": const(0.85),
                            "final_class_il": const(0.10),
                            "mean_leak": const(0.1), "mean_jaccard": const(0.1)}})
    text = "\n".join(R.efficiency_section(M, SEEDS, [128, 512]))
    assert "沒有任何跨容量配對達到 5/5" in text


# ── DR-019 逐條重驗 ────────────────────────────────────────────────────────

def _M19(a3_class, a5_class, a3_task, a5_task):
    caps = sorted(a3_class)
    return fake({**{("A3", c): {"final_class_il": const(a3_class[c]),
                                "final_task_il": const(a3_task[c]),
                                "mean_leak": const(0.1), "mean_jaccard": const(0.1)}
                    for c in caps},
                 **{("A5", c): {"final_class_il": const(a5_class[c]),
                                "final_task_il": const(a5_task[c]),
                                "mean_leak": const(0.1), "mean_jaccard": const(0.1)}
                    for c in caps}})


CAPS = [64, 128, 256, 512, 1024]


def test_claim1_fails_when_a3_still_improves_after_256():
    M = _M19({64: .60, 128: .70, 256: .75, 512: .74, 1024: .80},
             {c: .85 for c in CAPS},
             {c: .80 for c in CAPS}, {c: .90 for c in CAPS})
    text = "\n".join(R.dr019_section(M, SEEDS, CAPS))
    assert "仍在改善" in text
    assert "| ① A3 在 256 之後不再改善，含 1024 皆未超過 A5@128 | ❌ **不成立** |" in text


def test_claim1_holds_when_a3_plateaus_and_stays_below_a5_128():
    M = _M19({64: .60, 128: .70, 256: .75, 512: .74, 1024: .73},
             {64: .70, 128: .80, 256: .80, 512: .80, 1024: .80},
             {c: .80 for c in CAPS}, {c: .90 for c in CAPS})
    text = "\n".join(R.dr019_section(M, SEEDS, CAPS))
    assert "不再改善" in text
    assert "| ① A3 在 256 之後不再改善，含 1024 皆未超過 A5@128 | ✅ 成立 |" in text


def test_claim3_compares_variability_across_capacities():
    steady = {c: .90 for c in CAPS}
    jumpy = {64: .70, 128: .95, 256: .70, 512: .95, 1024: .70}
    text = "\n".join(R.dr019_section(
        _M19({c: .5 for c in CAPS}, {c: .5 for c in CAPS}, jumpy, steady),
        SEEDS, CAPS))
    assert "| ③ A5 對記憶體預算穩健而 A3 不穩 | ✅ 成立 |" in text
    text2 = "\n".join(R.dr019_section(
        _M19({c: .5 for c in CAPS}, {c: .5 for c in CAPS}, steady, jumpy),
        SEEDS, CAPS))
    assert "| ③ A5 對記憶體預算穩健而 A3 不穩 | ❌ **不成立** |" in text2


def test_claim4_requires_the_scarce_end_to_be_the_largest_gap():
    a3 = {c: .80 for c in CAPS}
    big_at_64 = {64: .95, 128: .85, 256: .84, 512: .83, 1024: .82}
    text = "\n".join(R.dr019_section(
        _M19({c: .5 for c in CAPS}, {c: .5 for c in CAPS}, a3, big_at_64), SEEDS, CAPS))
    assert "| ④ 稀缺端優勢最大 | ✅ 成立 |" in text
    big_at_1024 = {64: .82, 128: .83, 256: .84, 512: .85, 1024: .95}
    text2 = "\n".join(R.dr019_section(
        _M19({c: .5 for c in CAPS}, {c: .5 for c in CAPS}, a3, big_at_1024),
        SEEDS, CAPS))
    assert "| ④ 稀缺端優勢最大 | ❌ **不成立** |" in text2


def test_claim2_is_always_deferred_to_the_paired_recompute():
    M = _M19({c: .5 for c in CAPS}, {c: .9 for c in CAPS},
             {c: .5 for c in CAPS}, {c: .9 for c in CAPS})
    assert "| ② 2× 記憶體效率 | **改由本檔重裁** |" in "\n".join(
        R.dr019_section(M, SEEDS, CAPS))


def test_dr019_section_states_that_flat_results_do_not_transfer():
    M = _M19({c: .5 for c in CAPS}, {c: .9 for c in CAPS},
             {c: .5 for c in CAPS}, {c: .9 for c in CAPS})
    text = "\n".join(R.dr019_section(M, SEEDS, CAPS))
    assert "hier-A3 − flat-A3 = −3.11 pp" in text
    assert "不成立的照實報" in text


def test_claim1_fails_when_a3_plateaus_but_overtakes_a5_128():
    """①有兩半：「256 後不再改善」**且**「含 1024 皆未超過 A5@128」。

    ⚠️ 只測第一半的話，把判定式寫成 `not improves` 也會過 —— 這裡讓 A3 飽和
    但超過 A5@128，兩種寫法的答案才會分開。
    """
    M = _M19({64: .60, 128: .70, 256: .90, 512: .88, 1024: .89},   # 256 後不再改善
             {c: .80 for c in CAPS},                                # A5@128 = .80
             {c: .80 for c in CAPS}, {c: .90 for c in CAPS})
    text = "\n".join(R.dr019_section(M, SEEDS, CAPS))
    assert "不再改善" in text
    assert "**高於** A5@128" in text
    assert "| ① A3 在 256 之後不再改善，含 1024 皆未超過 A5@128 | ❌ **不成立** |" in text


def test_unsaturated_caveat_is_the_full_sentence_not_just_the_word():
    """⚠️ 只斷言「未飽和」會被誤放行 —— 該詞在本節開頭的理由 2 也出現過。"""
    M = _M({("A5", 64): const(0.95), ("A5", 128): const(0.95),
            ("A3", 512): const(0.85), ("A3", 1024): const(0.85)})
    text = "\n".join(R.efficiency_section(M, SEEDS, [64, 128, 512, 1024]))
    assert "**A3 的曲線在測試範圍內未飽和**" in text
    assert "class-IL 到 1024 仍在上升" in text
