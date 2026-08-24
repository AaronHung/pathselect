"""G3/G4/G5 報告 —— 判準邏輯與端到端產出。

⚠️ 判準是 pre-registered 的字面常數；本檔把「落判規則」釘死，讓事後改動會被抓到。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import report_arch_completeness as A                                  # noqa: E402


# ── 三級規則 ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("wins,n,want", [
    (5, 5, "systematic"), (4, 5, "directional, inconclusive"),
    (3, 5, "within noise"), (0, 5, "within noise"), (3, 3, "systematic"),
])
def test_tier(wins, n, want):
    assert A.tier(wins, n) == want


# ── 落判 ────────────────────────────────────────────────────────────────────

def test_verdict_passes_on_five_of_five_positive():
    v, _ = A.verdict({"final_task_il": (0.02, 5, 5), "final_class_il": (-0.01, 1, 5)})
    assert v == "PASS"


def test_verdict_passes_on_four_of_five_positive():
    v, _ = A.verdict({"final_task_il": (0.01, 4, 5), "final_class_il": (0.0, 2, 5)})
    assert v == "PASS"


def test_verdict_fails_on_three_of_five():
    v, _ = A.verdict({"final_task_il": (0.05, 3, 5), "final_class_il": (0.05, 3, 5)})
    assert v == "FAIL"


def test_verdict_fails_when_win_count_is_high_but_mean_is_negative():
    """判準是「win >= 4/5 **且** 配對為正」—— 兩個條件都要。"""
    v, _ = A.verdict({"final_task_il": (-0.03, 4, 5)})
    assert v == "FAIL"


def test_verdict_either_accuracy_axis_suffices():
    v, _ = A.verdict({"final_task_il": (-0.02, 0, 5), "final_class_il": (0.02, 5, 5)})
    assert v == "PASS"


def test_verdict_ignores_secondary_metrics():
    """次要指標再漂亮也不能讓實驗通過。"""
    v, _ = A.verdict({"final_task_il": (0.001, 2, 5), "final_class_il": (0.001, 2, 5),
                      "mean_leak": (-0.5, 5, 5), "mean_jaccard": (0.5, 5, 5)})
    assert v == "FAIL"


@pytest.mark.parametrize("wins,n", [(3, 3), (4, 4), (2, 2)])
def test_verdict_fails_when_seeds_are_fewer_than_five(wins, n):
    """三級規則是為 n=5 校準的（憲法 §1.2）；n<5 不得逕自判 PASS。

    ⚠️ (4, 4) 這格是關鍵：它同時滿足 win>=4 與 mean>0，只有 n>=5 這個守門擋得住。
    """
    v, _ = A.verdict({"final_task_il": (0.05, wins, n)})
    assert v == "FAIL"


# ── pre-registered 常數 ─────────────────────────────────────────────────────

def test_every_experiment_has_both_branches_pre_registered():
    keys = {k for k, *_ in A.EXPERIMENTS}
    assert keys == set(A.PRE_REGISTERED), "實驗與判準對不上"
    for k, spec in A.PRE_REGISTERED.items():
        assert spec["pass"].strip() and spec["fail"].strip(), k


def test_experiments_change_exactly_one_thing_from_the_baseline():
    """G5 / G4 換 arch、G3 換 arm —— 每個實驗只有一個維度與基準不同。"""
    _, _, _, b_arm, b_arch = A.BASELINE
    for k, _n, _root, arm, arch in A.EXPERIMENTS:
        diff = (arm != b_arm) + (arch != b_arch)
        assert diff == 1, f"{k} 與基準差了 {diff} 個維度"


def test_primary_metrics_are_the_two_accuracy_axes():
    assert [k for k, _ in A.PRIMARY] == ["final_task_il", "final_class_il"]


# ── 端到端：合成資料要能落出非 PENDING 的判定 ────────────────────────────────

TASKS = ["tcga_esca", "tcga_rcc", "tcga_brca", "tcga_lung"]


def records(arm, arch, seed, correct):
    """造一批 per-slide 記錄。correct 控制學完 T4 後判對的比例。"""
    out = []
    for stage in range(4):
        for i, t in enumerate(TASKS[:stage + 1]):
            lo = 2 * ["tcga_esca", "tcga_rcc", "tcga_brca", "tcga_lung"].index(t)
            for k in range(10):
                hit = k < (10 if stage < 3 else correct)
                out.append({
                    "arm": arm, "order": "reverse", "seed": seed, "stage": stage,
                    "task": t, "slide_id": f"{t}_{k}", "true": lo,
                    "pred_class_il": lo if hit else (lo + 2) % 8,
                    "pred_task_il": lo if hit else lo + 1,
                    "selected_idx": list(range(k, k + 8)),
                    "group_quota": [1] * 8, "n_patch": 3000, "B": 8,
                    "utility_total": 1.0, "arch": arch, "allocation": "per_budget",
                })
    return out


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    base_dir = tmp_path / "hier2" / "per_slide"
    exp_dir = tmp_path / "arch" / "per_slide"
    base_dir.mkdir(parents=True); exp_dir.mkdir(parents=True)
    monkeypatch.setattr(A, "OUT_DIR", tmp_path / "arch")
    monkeypatch.setattr(A, "BASELINE", ("base", "基準", base_dir, "A5", "hier"))
    monkeypatch.setattr(A, "EXPERIMENTS", [
        ("G5", "+state", exp_dir, "A5", "hier_state"),
        ("G4", "+q_tau", exp_dir, "A5", "hier_query"),
        ("G3", "+group L_sem", exp_dir, "A5g", "hier"),
    ])
    return base_dir, exp_dir, tmp_path / "arch" / "ARCH_COMPLETENESS.md"


def test_end_to_end_emits_pass_and_fail(sandbox):
    base_dir, exp_dir, md = sandbox
    for s in range(5):
        (base_dir / f"A5_{s}.json").write_text(json.dumps(records("A5", "hier", s, 5)))
        # G5：每個 seed 都更好 → 5/5 且為正 → PASS
        (exp_dir / f"g5_{s}.json").write_text(
            json.dumps(records("A5", "hier_state", s, 7)))
        # G4：每個 seed 都更差 → FAIL
        (exp_dir / f"g4_{s}.json").write_text(
            json.dumps(records("A5", "hier_query", s, 3)))
        # G3：與基準相同 → win 0/5 → FAIL
        (exp_dir / f"g3_{s}.json").write_text(json.dumps(records("A5g", "hier", s, 5)))
    assert A.main() == 0
    text = md.read_text(encoding="utf-8")
    for h in ("## G5 前置：no-op 檢查", "## G4 前置：q_tau 是否真的進入計算",
              "## 主表", "## 配對比較與落判", "## 總結"):
        assert h in text, f"缺章節 {h}"
    assert "PENDING" not in text, "合成資料齊全時不該出現 PENDING"
    assert "| G5 | +state | **PASS** |" in text
    assert "| G4 | +q_tau | **FAIL** |" in text
    assert "| G3 | +group L_sem | **FAIL** |" in text
    # 判準原文必須逐字出現在產物裡
    assert A.PRE_REGISTERED["G5"]["pass"] in text
    assert A.PRE_REGISTERED["G4"]["fail"] in text


def test_end_to_end_marks_missing_experiments_as_pending(sandbox):
    base_dir, _exp_dir, md = sandbox
    for s in range(5):
        (base_dir / f"A5_{s}.json").write_text(json.dumps(records("A5", "hier", s, 5)))
    assert A.main() == 0
    assert md.read_text(encoding="utf-8").count("PENDING") >= 3


def test_report_refuses_to_run_without_the_baseline(sandbox):
    with pytest.raises(SystemExit, match="找不到基準資料"):
        A.main()


# ── §3.6b：fixture 必須涵蓋腳本實際遍歷的所有維度 ──────────────────────────

def test_paired_uses_only_seeds_present_in_both_arms(sandbox):
    """憲法 §1.3 / DR-034：seed 數不齊時只能用共同子集。

    ⚠️ 這個 fixture 刻意讓實驗臂多一個 seed 5，且該 seed 表現極好 —— 若腳本沒有
    取共同子集，win count 的分母會錯，且會拿沒有配對的 seed 灌水。
    """
    base_dir, exp_dir, md = sandbox
    for s in range(5):
        (base_dir / f"A5_{s}.json").write_text(json.dumps(records("A5", "hier", s, 5)))
        (exp_dir / f"g5_{s}.json").write_text(
            json.dumps(records("A5", "hier_state", s, 5)))
    (exp_dir / "g5_5.json").write_text(json.dumps(records("A5", "hier_state", 5, 10)))
    assert A.main() == 0
    text = md.read_text(encoding="utf-8")
    assert "共同 seeds：[0, 1, 2, 3, 4]（n=5）" in text
    assert "| G5 | +state | **FAIL** |" in text, "多出來的 seed 5 不該影響落判"


def test_baseline_excludes_degenerate_per_chunk_records(sandbox):
    """per_chunk 在 c=1 下退化為單組（DR-025）—— 不得混進基準。

    ⚠️ report_prior.py 就是栽在這裡：混入退化記錄後產生了假結論。
    """
    base_dir, exp_dir, md = sandbox

    def base_row():
        return [ln for ln in md.read_text(encoding="utf-8").splitlines()
                if ln.startswith("| base |")][0]

    for s in range(5):
        (base_dir / f"A5_{s}.json").write_text(json.dumps(records("A5", "hier", s, 5)))
        (exp_dir / f"g5_{s}.json").write_text(
            json.dumps(records("A5", "hier_state", s, 7)))
    assert A.main() == 0
    clean = base_row()
    assert "| G5 | +state | **PASS** |" in md.read_text(encoding="utf-8")

    # 混入 per_chunk 退化記錄（同 arm / 同 arch / 同 seed，只有 allocation 不同）
    for s in range(5):
        bad = records("A5", "hier", s, 0)
        for r in bad:
            r["allocation"] = "per_chunk"
        (base_dir / f"A5_{s}_perchunk.json").write_text(json.dumps(bad))
    assert A.main() == 0
    assert base_row() == clean, "基準列被 per_chunk 退化記錄改變了"
