"""圖表的數字必須等於報告的數字（PROMPT DOSSIER-FIGURES-20260826 §C）。

⚠️ 圖與報告算出不同數字是最糟的結果 —— reviewer 一定會發現。
`make_figures.py` 沿用 `run_exp2.arm_metrics`，本檔再從產物端把關一次。
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "figures" / "figure_data.json"
OUT = ROOT / "outputs"

#: 圖上的文字不得出現的敘事字（沿用 test_no_banned_deps 的精神）。
FIGURE_BANNED = ["stateful", "sequential", "iterative", "navigation",
                 "task-conditioned", "task conditioned", "zero-", "router"]


def data():
    if not DATA.exists():
        pytest.skip("尚未產生 figures/figure_data.json")
    return json.loads(DATA.read_text(encoding="utf-8"))


# ── 與報告逐項對照 ──────────────────────────────────────────────────────────

MEM_EXPECT = {"64": ("+4.90", 5), "128": ("+3.73", 5), "256": ("+2.54", 4),
              "512": ("+3.28", 5), "1024": ("+2.49", 5)}


@pytest.mark.parametrize("cap", sorted(MEM_EXPECT))
def test_fig5_matches_memory_hier_report(cap):
    """五個容量的 A5 − A3 task-IL 配對 mean 與 win count 都要與 MEMORY_HIER.md 相符。"""
    d = data()
    md = (OUT / "exp2" / "memory_hier" / "MEMORY_HIER.md").read_text(encoding="utf-8")
    p = d["fig5_memory_hier"]["paired"][cap]["final_task_il"]
    mean_s, wins = MEM_EXPECT[cap]
    assert f"{p['mean'] * 100:+.2f}" == mean_s, f"|M|={cap} 圖與期望值不符"
    assert p["wins"] == wins and p["n"] == 5
    # 報告裡也要真的有這個數字（不是只跟寫死的期望值比）
    assert re.search(rf"\|\s*{cap}\s*\|.*{re.escape(mean_s)}", md), \
        f"MEMORY_HIER.md 的 |M|={cap} 那列沒有 {mean_s}"


@pytest.mark.parametrize("key,expect", [("final_task_il", "+3.28"),
                                        ("final_class_il", "+5.76"),
                                        ("mean_leak", "-2.42")])
def test_fig4_matches_hier2_report(key, expect):
    d = data()
    md = (OUT / "exp2" / "hier2" / "HIER2.md").read_text(encoding="utf-8")
    p = d["fig4_main_hier"]["paired"]["A5-A3"][key]
    assert f"{p['mean'] * 100:+.2f}" == expect
    assert p["wins"] == 5 and p["n"] == 5
    assert expect.replace("-", "−") in md or expect in md


@pytest.mark.parametrize("name,key,expect,wins", [
    ("G5", "final_task_il", "-0.497", 2),
    ("G4", "final_class_il", "+5.849", 4),
    ("G4", "mean_leak", "-5.923", 5),
])
def test_figS2_matches_arch_report(name, key, expect, wins):
    d = data()
    md = (OUT / "exp2" / "arch" / "ARCH_COMPLETENESS.md").read_text(encoding="utf-8")
    p = d["figS2_arch_completeness"][name][key]
    assert f"{p['mean'] * 100:+.3f}" == expect
    assert p["wins"] == wins and p["n"] == 5
    assert expect in md


def test_fig2_peak_matches_baselines_report():
    d = data()
    md = (OUT / "exp0" / "BASELINES.md").read_text(encoding="utf-8")
    f2 = d["fig2_budget_curve"]
    assert f2["peak_K"] == 8
    assert f"{f2['peak_value']:.4f}" == "0.8797"
    assert "0.8797" in md


# ── 版面與紀律 ──────────────────────────────────────────────────────────────

def test_every_figure_has_pdf_and_png():
    d = data()
    for name, blk in d.items():
        paths = blk.get("paths", [])
        exts = {Path(p).suffix for p in paths}
        assert exts == {".pdf", ".png"}, f"{name} 缺向量或點陣輸出：{paths}"
        for p in paths:
            assert (ROOT / p).exists() and (ROOT / p).stat().st_size > 1000, p


def test_figure_text_has_no_banned_narrative_words():
    """圖上的文字不得出現已被證偽的敘事字（C-01 / C-04 / DR-043）。"""
    src = (ROOT / "scripts" / "make_figures.py").read_text(encoding="utf-8")
    # 只看會畫到圖上的字串：set_*label / set_title / annotate / label=
    drawn = re.findall(r"(?:set_[a-z]*label|set_title|annotate)\(([^\n]*)", src)
    drawn += re.findall(r'label="([^"]*)"', src)
    blob = " ".join(drawn).lower()
    hit = [w for w in FIGURE_BANNED if w in blob]
    assert not hit, f"圖上文字出現被禁的敘事字：{hit}"


def test_no_p_values_or_significance_stars():
    src = (ROOT / "scripts" / "make_figures.py").read_text(encoding="utf-8")
    assert "p-value" not in src.lower() and "pvalue" not in src.lower()
    assert not re.search(r'["\']\s*\*{1,3}\s*["\']', src), "不得標星號"


def test_arm_colours_are_fixed_and_colorblind_safe():
    import importlib.util as u
    spec = u.spec_from_file_location("mf", ROOT / "scripts" / "make_figures.py")
    m = u.module_from_spec(spec); spec.loader.exec_module(m)
    okabe_ito = {"#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442",
                 "#0072B2", "#D55E00", "#CC79A7", "#4D4D4D", "#B0B0B0"}
    for arm in ("A1", "A2", "A3", "A4", "A5", "A5nG", "random"):
        assert m.C[arm].upper() in okabe_ito, f"{arm} 的顏色不在 Okabe-Ito 內"
    assert m.C["A5"] == m.C["A5nG"], "A5nG 應與 A5 同色，靠 hatch 區分"
    assert "A5nG" in m.HATCH


# ── §3.6 smoke ─────────────────────────────────────────────────────────────

def test_smoke_mode_runs_on_fixture():
    """fixture 涵蓋兩個 order 與 seed 數不齊（§3.6b），且不覆蓋真實產物。"""
    before = DATA.read_bytes() if DATA.exists() else None
    r = subprocess.run([sys.executable, "scripts/make_figures.py", "--smoke"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, f"smoke 失敗：\n{r.stdout[-2000:]}\n{r.stderr[-1000:]}"
    assert "跳過 0" in r.stdout, f"smoke 有圖沒產出：\n{r.stdout}"
    if before is not None:
        assert DATA.read_bytes() == before, "smoke 不得覆蓋真實的 figure_data.json"
