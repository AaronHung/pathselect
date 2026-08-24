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
    "report_order_dependence.py": ("outputs/exp2/order_main/per_slide",
                                   "outputs/exp2/ORDER_DEPENDENCE.md",
                                   ["## 跨順序的穩定性", "### 讀法"]),
    # 輸入是 G1' 的基準存檔；G3/G4/G5 自己的資料未齊時報告仍須能跑（標 PENDING）
    "report_arch_completeness.py": ("outputs/exp2/hier2/per_slide",
                                    "outputs/exp2/arch/ARCH_COMPLETENESS.md",
                                    ["## G5 前置：no-op 檢查", "## 主表",
                                     "## 配對比較與落判", "## 總結"]),
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

    `hier` tag 是 G1 的 per_chunk 紀錄（階層退化 88.6% 單組），
    `hier2` 才是 per_budget 主線。不過濾就會把退化紀錄當成主線臂。
    """
    from scripts.report_prior import collect

    groups = collect("hier")
    if not groups.get("discriminative"):
        pytest.skip("缺 prior 資料")
    bad = [r for r in groups["discriminative"]
           if r.get("allocation", "per_chunk") != "per_budget"]
    assert not bad, f"discriminative 混進了 {len(bad)} 筆非 per_budget 的紀錄"
