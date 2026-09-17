"""DR-056 第 8 步的隔離／還原機制。

這是整個探針裡唯一會動到**資料集檔案**的程式碼，所以還原必須可證：
manifest 先寫後移、restore 完整還原、行程被 kill 後可用 --restore-only 救回、
拒絕覆蓋同名檔。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_dr056_step8 import Quarantine, restore_only        # noqa: E402


def _tree(tmp_path, n=5):
    d = tmp_path / "feats-l1-s256_CONCH"
    d.mkdir()
    files = []
    for i in range(n):
        f = d / f"slide{i}.pt"
        f.write_bytes(bytes([i]) * (10 + i))
        files.append(f)
    return d, files


def test_hide_then_restore_is_exact(tmp_path):
    d, files = _tree(tmp_path)
    before = {f.name: f.read_bytes() for f in files}
    q = Quarantine(tmp_path / "q", tmp_path / "m.json")
    assert q.hide(files) == len(files)
    assert [f for f in files if f.exists()] == []          # 真的不見了
    with pytest.raises(FileNotFoundError):                 # root 也讀不到：ENOENT
        open(files[0], "rb")
    assert q.restore() == len(files)
    assert {f.name: f.read_bytes() for f in files} == before


def test_manifest_is_written_before_the_move(tmp_path):
    """行程在移動當下被 kill，manifest 也已經記錄了那一筆。"""
    d, files = _tree(tmp_path, 3)
    mf = tmp_path / "m.json"
    q = Quarantine(tmp_path / "q", mf)
    seen = []
    real_flush = q._flush

    def spy():
        real_flush()
        seen.append(len(json.loads(mf.read_text())["moved"]))
    q._flush = spy
    q.hide(files)
    assert seen == [1, 2, 3]                               # 每移一筆前都先落地


def test_restore_only_recovers_after_a_kill(tmp_path):
    d, files = _tree(tmp_path, 4)
    before = {f.name: f.read_bytes() for f in files}
    mf = tmp_path / "m.json"
    q = Quarantine(tmp_path / "q", mf)
    q.hide(files)
    del q                                                   # 模擬行程消失，沒跑 finally
    assert [f for f in files if f.exists()] == []
    assert restore_only(mf) == 0
    assert {f.name: f.read_bytes() for f in files} == before
    assert json.loads(mf.read_text())["moved"] == []


def test_refuses_to_overwrite_in_quarantine(tmp_path):
    d, files = _tree(tmp_path, 2)
    qdir = tmp_path / "q"
    q = Quarantine(qdir, tmp_path / "m.json")
    q.hide(files[:1])
    clash = qdir / f"{d.name}__{files[0].name}"
    assert clash.is_file()
    files[0].write_bytes(b"new")                            # 同名檔又出現
    with pytest.raises(RuntimeError, match="拒絕覆蓋"):
        q.hide(files[:1])


def test_restore_never_clobbers_an_existing_original(tmp_path):
    d, files = _tree(tmp_path, 2)
    q = Quarantine(tmp_path / "q", tmp_path / "m.json")
    q.hide(files)
    files[0].write_bytes(b"someone put it back")            # 原位已有檔
    q.restore()
    assert files[0].read_bytes() == b"someone put it back"  # 不覆蓋
    assert files[1].exists()
