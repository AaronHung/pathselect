"""逐 slide 產物的單檔大小上限（DR-048 Prompt 6-0）。

PI 要求「無任何單檔 >5 MB 進入 git」。這條把一次性的檢查變成**常設守門** ——
`.gitignore` 不適合做這件事：它會讓超大產物**靜默消失**而不是被擋下，
反而更難發現寫錯了什麼。

起因是 `ZS-mean` 的第一版：`selected_idx` 存的就是 `range(n)`（n 最大 8466）、
權重是 n 份相同的數，**單折 57 MB** 全是零資訊量的冗餘。
逐 slide 產物一旦超過幾 MB，幾乎都代表存了不該存的東西。
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
LIMIT_MB = 5.0

#: 早於本規則、且已在版控歷史裡的大檔。**只放既有的**，不得用來放行新產物 ——
#: DR 卡的 evidence 引用 commit hash，改寫歷史會破壞可追溯性
#: （與 `.gitignore` 開頭那段註解同一個理由）。
GRANDFATHERED = {
    "outputs/exp2/main/dr046_deltas_seed0.pt",
    "outputs/exp2/main/dr046_deltas_seed1.pt",
    "outputs/exp2/main/dr046_deltas_seed2.pt",
    "outputs/exp2/main/dr046_deltas_seed3.pt",
    "outputs/exp2/main/dr046_deltas_seed4.pt",
    "outputs/exp1/stage1/results.json",
}


def _git_visible_files() -> list[str]:
    """已追蹤 ＋ 未被 ignore 的未追蹤檔 —— 也就是「會進 git 的」。"""
    r = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"],
                       cwd=REPO_ROOT, capture_output=True, text=True)
    return r.stdout.split("\n") if r.returncode == 0 else []


def test_no_oversized_per_slide_product():
    """逐 slide 產物一律 <= 5 MB。"""
    bad = []
    for rel in _git_visible_files():
        if "/per_slide" not in rel or not rel.endswith(".json"):
            continue
        f = REPO_ROOT / rel
        if not f.is_file():
            continue
        mb = f.stat().st_size / 1024 / 1024
        if mb > LIMIT_MB:
            bad.append(f"{rel}: {mb:.1f} MB")
    assert not bad, (
        f"逐 slide 產物超過 {LIMIT_MB} MB —— 幾乎都代表存了不該存的東西"
        "（例如把 range(n) 當成 selected_idx）：\n  " + "\n  ".join(bad))


def test_no_new_oversized_file_anywhere():
    """整個 repo 都不得有新的 >5 MB 檔案；既有的逐一列在 GRANDFATHERED。"""
    bad = []
    for rel in _git_visible_files():
        f = REPO_ROOT / rel
        if not f.is_file() or rel in GRANDFATHERED:
            continue
        mb = f.stat().st_size / 1024 / 1024
        if mb > LIMIT_MB:
            bad.append(f"{rel}: {mb:.1f} MB")
    assert not bad, (
        f"新增了 >{LIMIT_MB} MB 的檔案：\n  " + "\n  ".join(bad) +
        "\n若確實必要，請在 PI 核可後加進 GRANDFATHERED 並說明理由。")


def test_grandfathered_entries_still_exist():
    """放行清單不得留下已不存在的項目 —— 否則會逐漸變成無效的擋箭牌。"""
    missing = [p for p in sorted(GRANDFATHERED) if not (REPO_ROOT / p).is_file()]
    assert not missing, f"GRANDFATHERED 裡有不存在的檔案，請移除：{missing}"


@pytest.mark.parametrize("rel", sorted(GRANDFATHERED))
def test_grandfathered_entries_are_actually_oversized(rel: str):
    """放行清單只准放**真的**超標的既有檔案，不得拿來放行普通檔案。"""
    f = REPO_ROOT / rel
    if not f.is_file():
        pytest.skip(f"{rel} 不存在")
    assert f.stat().st_size / 1024 / 1024 > LIMIT_MB, (
        f"{rel} 沒有超過 {LIMIT_MB} MB，不該出現在 GRANDFATHERED")
