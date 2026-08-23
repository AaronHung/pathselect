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
