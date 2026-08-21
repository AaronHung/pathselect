"""PI 裁定 C — 每一次評估都必須落一份逐 slide 預測。

v9 只存了彙總 accuracy，害 flip 分析只能給區間。這條測試掃過 outputs/ 下所有
逐 slide JSON，確認欄位齊全，避免同樣的問題再發生一次。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
REQUIRED = {"slide_id", "task", "true"}
#: 選取類評估另外必須帶的欄位
REQUIRED_SELECTION = {"selected_idx", "weights_softmax", "weights_uniform"}


def _per_slide_files() -> list[Path]:
    root = REPO_ROOT / "outputs"
    if not root.is_dir():
        return []
    return sorted(list(root.glob("**/per_slide/*.json"))
                  + list(root.glob("**/per_slide_*.json")))


def test_there_is_at_least_one_per_slide_dump():
    if not (REPO_ROOT / "outputs").is_dir():
        pytest.skip("尚未產生任何 outputs")
    assert _per_slide_files(), "outputs/ 下找不到任何逐 slide 預測檔（裁定 C）"


@pytest.mark.parametrize("path", _per_slide_files(),
                         ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_per_slide_records_have_required_fields(path: Path):
    records = json.loads(path.read_text())
    assert isinstance(records, list) and records, f"{path} 是空的"
    for i, r in enumerate(records[:200]):
        missing = REQUIRED - set(r)
        assert not missing, f"{path}[{i}] 缺欄位 {missing}"
        assert isinstance(r["slide_id"], str) and r["slide_id"]
        assert any(k.startswith("pred") for k in r), f"{path}[{i}] 沒有任何預測欄位"


@pytest.mark.parametrize("path", [p for p in _per_slide_files() if "exp1" in str(p)],
                         ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_selection_dumps_record_indices_and_weights(path: Path):
    records = json.loads(path.read_text())
    for i, r in enumerate(records[:200]):
        missing = REQUIRED_SELECTION - set(r)
        assert not missing, f"{path}[{i}] 缺欄位 {missing}"
        assert len(r["selected_idx"]) == len(r["weights_softmax"]) == \
               len(r["weights_uniform"]), f"{path}[{i}] 長度不一致"
        assert abs(sum(r["weights_softmax"]) - 1.0) < 1e-3, f"{path}[{i}] softmax 權重未歸一"
