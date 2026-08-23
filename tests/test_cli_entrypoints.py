"""每個 CLI 腳本的 argparse 都必須跑得起來。

⚠️ 為什麼需要這條：`import scripts.run_exp2` **不會**執行 main()，所以 main() 裡的
NameError（例如 argparse 用到未 import 的常數）不會被任何既有測試抓到 ——
2026-08-25 就發生過：檔案壞了、671 條測試全綠，直到排隊中的 job 撞上去才爆炸，
而且 G1' 因此整批被靜默跳過。`--help` 會走完整個 argparse 設定，是最便宜的守門。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = sorted(p for p in (REPO_ROOT / "scripts").glob("*.py")
                 if not p.name.startswith("_"))


def _has_argparse(path: Path) -> bool:
    return "argparse.ArgumentParser" in path.read_text(encoding="utf-8")


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_script_imports_cleanly(path: Path):
    """模組層必須可 import（抓 import 期的錯）。"""
    r = subprocess.run(
        [sys.executable, "-c",
         f"import sys; sys.path.insert(0, {str(REPO_ROOT)!r}); "
         f"import importlib.util as u; "
         f"spec = u.spec_from_file_location('m', {str(path)!r}); "
         f"m = u.module_from_spec(spec); spec.loader.exec_module(m)"],
        capture_output=True, text=True, cwd=REPO_ROOT)
    assert r.returncode == 0, f"{path.name} import 失敗：\n{r.stderr[-1500:]}"


@pytest.mark.parametrize("path", [p for p in SCRIPTS if _has_argparse(p)],
                         ids=lambda p: p.name)
def test_script_help_runs(path: Path):
    """`--help` 會走完 argparse 設定，抓 main() 內的 NameError。"""
    r = subprocess.run([sys.executable, str(path), "--help"],
                       capture_output=True, text=True, cwd=REPO_ROOT)
    assert r.returncode == 0, f"{path.name} --help 失敗：\n{r.stderr[-1500:]}"
    assert "usage:" in r.stdout.lower()
