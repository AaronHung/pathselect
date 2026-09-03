"""憲法 §3.6 —— 報告腳本必須能用真實資料實際跑完 main()。

`--help` 測試走完 argparse，但擋不住 main() 內部的 NameError —— 那需要真實資料
才會觸發。同類錯誤已發生三次（ALLOCATION_MODES、diag、以及更早的替換未命中），
前兩層防護都擋不住第三次。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

#: 腳本 → (需要存在的輸入目錄, 產物, 產物中必須出現的章節標題)
REPORTS = {
    "report_hier.py": ("outputs/exp2/hier/per_slide",
                       "outputs/exp2/hier/HIER.md",
                       ["## 主表", "## 配對比較", "## 結構性診斷",
                        "## Pre-registered 判準"]),
    "report_hier2.py": ("outputs/exp2/hier2/per_slide",
                        "outputs/exp2/hier2/HIER2.md",
                        ["## 主表", "## 結構性診斷", "## ⚠️ 結構性把關",
                         "## Pre-registered 判準"]),
    "report_memory.py": ("outputs/exp2/memory/per_slide",
                         "outputs/exp2/memory/MEMORY.md",
                         ["## F1", "## F2", "## 記憶體效率主張", "## 結論"]),
    "report_memory_hier.py": ("outputs/exp2/memory_hier/per_slide",
                              "outputs/exp2/memory_hier/MEMORY_HIER.md",
                              ["## 結構性把關", "## 記憶體主張",
                               "### 主要主張（同容量，不需跨容量比較）",
                               "### 輔助主張（跨容量效率倍數）",
                               "**引用時必須同時寫出三個限定：**",
                               "### class-IL 另報", "## 跨容量配對比較",
                               "## DR-019 的四條可宣稱在階層版是否成立"]),
    "report_order_dependence.py": ("outputs/exp2/order_main/per_slide",
                                   "outputs/exp2/ORDER_DEPENDENCE.md",
                                   ["## 跨順序的穩定性", "### 讀法"]),
    # 輸入是 G1' 的基準存檔；G3/G4/G5 自己的資料未齊時報告仍須能跑（標 PENDING）
    "report_arch_completeness.py": ("outputs/exp2/hier2/per_slide",
                                    "outputs/exp2/arch/ARCH_COMPLETENESS.md",
                                    ["## G5 前置：no-op 檢查",
                                     "## G4 前置：q_tau 是否真的進入計算",
                                     "## 主表", "## 配對比較與落判", "## 總結"]),
    "report_b1_landing.py": ("outputs/exp2/ablation/per_slide",
                             "outputs/exp2/ablation/B1_LANDING.md",
                             ["## KD 與 replay 保存的是不同的東西",
                              "### 預測落點", "### seed 4 的 rcc：逐筆落點",
                              "### ⚠️ B1 同時是最不穩的一臂"]),
}


@pytest.mark.parametrize("script", sorted(REPORTS), ids=lambda s: s)
def test_report_script_runs_and_emits_sections(script):
    src, out, sections = REPORTS[script]
    if not (REPO_ROOT / src).is_dir() or not any((REPO_ROOT / src).glob("*.json")):
        pytest.skip(f"缺輸入資料：{src}")
    r = subprocess.run([sys.executable, f"scripts/{script}"],
                       capture_output=True, text=True, cwd=REPO_ROOT)
    assert r.returncode == 0, f"{script} 執行失敗：\n{r.stderr[-2000:]}"
    text = (REPO_ROOT / out).read_text(encoding="utf-8")
    assert len(text) > 500, f"{out} 產物過短（{len(text)} 字元）"
    missing = [h for h in sections if h not in text]
    assert not missing, f"{out} 缺章節：{missing}"


# ── §3.6b：fixture 必須涵蓋腳本實際遍歷的所有維度 ──────────────────────────

def test_order_dependence_handles_uneven_seed_counts():
    """§3.6b：兩個 order 的各 arm seed 數不齊時也要能跑。

    ⚠️ 這正是 2026-08-24 的 KeyError 現場：main order 只有 A3/A5 補到 5 seeds，
    其餘仍 3 seeds；共同 seeds 若在 order 層取聯集，A1 就會被要求提供不存在的 seed。
    單 order 的 fixture 走不到這行。
    """
    import json
    from collections import defaultdict

    counts = defaultdict(set)
    for tag, order in (("main", "reverse"), ("order_main", "main")):
        d = REPO_ROOT / "outputs" / "exp2" / tag / "per_slide"
        if not d.is_dir():
            pytest.skip(f"缺資料：{tag}")
        for f in d.glob("*.json"):
            for r in json.loads(f.read_text()):
                if (r["order"] == order and r.get("arch", "flat") == "flat"
                        and r.get("mem_capacity", 512) == 512):
                    counts[(order, r["arm"])].add(r["seed"])
    sizes = {len(v) for v in counts.values()}
    assert len(sizes) > 1, f"fixture 未涵蓋 seed 數不齊的情況：{sizes}"

    r = subprocess.run([sys.executable, "scripts/report_order_dependence.py"],
                       capture_output=True, text=True, cwd=REPO_ROOT)
    assert r.returncode == 0, f"seed 數不齊時失敗：\n{r.stderr[-1500:]}"
    text = (REPO_ROOT / "outputs" / "exp2" / "ORDER_DEPENDENCE.md").read_text()
    assert "(n=" in text, "表格未標出各 arm 的實際 n"


def test_prior_report_excludes_degenerate_hierarchy_records():
    """§3.6b 附帶：跨 tag 蒐集必須過濾 allocation。

    `hier` tag 是 G1 的 per_chunk 紀錄（階層退化 88.6% 單組；全部 arm 口徑，DR-045），
    `hier2` 才是 per_budget 主線。不過濾就會把退化紀錄當成主線臂。
    """
    from scripts.report_prior import collect

    groups = collect("hier")
    if not groups.get("discriminative"):
        pytest.skip("缺 prior 資料")
    bad = [r for r in groups["discriminative"]
           if r.get("allocation", "per_chunk") != "per_budget"]
    assert not bad, f"discriminative 混進了 {len(bad)} 筆非 per_budget 的紀錄"


# ── 治理文件的數字必須可從產物溯源（DR-043 之後的常設守門）──────────────────

def test_doc_numbers_are_traceable_to_artifacts():
    """`verify_doc_numbers.py` 必須通過。

    它做兩件事：從 per_slide **重算**配對統計後比對文件的指定段落（Tier 1），
    以及確認文件引用的數字**同時**存在於被引用的產物裡（Tier 2）。
    ⚠️ 第一版的兩個條件都寫鬆了（整份文件子字串搜尋、`(not in_doc) or in_art`），
    六個 mutation 全數漏抓；現行版本已逐一驗證抓得到。
    """
    r = subprocess.run([sys.executable, "scripts/verify_doc_numbers.py"],
                       capture_output=True, text=True, cwd=REPO_ROOT)
    assert r.returncode == 0, f"數字無法溯源：\n{r.stdout[-3000:]}"


# ── DR-046 凍結：論文稿的數值溯源 ───────────────────────────────────────────

def test_paper_tolerance_is_the_spec_value():
    """容差是規格寫死的 5e-3，不得放寬。

    ⚠️ 放寬容差不會讓掃描在正確稿子上失敗，跑一次抓不到 —— 只能用測試釘住。
    """
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import verify_doc_numbers as V
    assert V.PAPER_TOL == 5e-3


def test_paper_scan_flags_a_changed_number(tmp_path, monkeypatch):
    """稿內數字改一位 → 掃描必須報出來。"""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import verify_doc_numbers as V
    src = (REPO_ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
    assert "$+3.06$" in src or "+3.06" in src, "稿內找不到用來擾動的錨點數字"
    fake = tmp_path / "main.tex"
    # ⚠️ 擾動值要挑**不可能在產物裡碰撞**的。掃描是「與產物數值池比對」，
    #    改成 +9.06 之類仍會命中池子裡別的數字（例如某個標準差），
    #    看起來像通過 —— 這是本掃描的已知弱點，見 scan_paper 的說明。
    fake.write_text(src.replace("+3.06", "+77.77"), encoding="utf-8")
    monkeypatch.setattr(V, "PAPER", fake)
    bad = V.scan_paper()
    assert any("77.77" in b for b in bad), f"改掉的數字沒被抓到：{bad}"


def test_paper_scan_flags_a_pending_usage(tmp_path, monkeypatch):
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import verify_doc_numbers as V
    src = (REPO_ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
    fake = tmp_path / "main.tex"
    fake.write_text(src + "\n\\pending{something unfinished}\n", encoding="utf-8")
    monkeypatch.setattr(V, "PAPER", fake)
    bad = V.scan_paper()
    assert any("pending" in b for b in bad), f"塞回的 \\pending 沒被抓到：{bad}"


def test_paper_scan_passes_on_the_real_manuscript():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import verify_doc_numbers as V
    bad = V.scan_paper()
    assert not bad, "稿件溯源未通過：\n" + "\n".join(bad)


def test_paper_scan_skips_latex_lengths():
    """`p{4.75cm}` / `0.48\\textwidth` 是排版參數，不該被要求溯源。

    ⚠️ 沒有這道略過，v0.71 的表格欄寬會逼人把版面數字塞進「可溯源產物」。
    """
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import verify_doc_numbers as V
    got = [t for _i, t, _c in V.paper_numbers(
        r"\begin{tabular}{p{1.25cm}p{4.75cm}} 0.48\textwidth 0.9147 3.5pt")]
    assert got == ["0.9147"], f"排版參數沒被略過：{got}"


def test_paper_scan_still_sees_real_numbers_next_to_units():
    """略過規則不能誤殺真數字 —— 只有緊接單位的才算排版參數。"""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import verify_doc_numbers as V
    got = [t for _i, t, _c in V.paper_numbers(r"gains $+3.28$ pp and $0.8239$ accuracy")]
    assert got == ["+3.28", "0.8239"], got
