#!/usr/bin/env python3
"""DR-056 第 8 步：舊 task 特徵檔不可讀的情況下重跑 run 2（自足性驗證）。

PI 的第 8 步原文是「設為不可讀」。**`chmod 000` 在 pod 上無效** —— 容器以 root
執行，root 有 CAP_DAC_OVERRIDE，會繞過權限位元照樣讀到。因此改用等價但對 root
也成立的作法：把檔案**改名移進隔離目錄**，`open()` 直接 ENOENT。

流程：每學完一個 task（`fill_memory` 回傳後），把該 task **train split** 的特徵檔
移進隔離目錄。test split 不動（評估要用）。結束時無條件還原。

還原保證三層：
  1. 移動**之前**先把 manifest 寫到磁碟（含每一筆的來源與去處）。
  2. try/finally ＋ atexit 都呼叫還原。
  3. 即使行程被 kill，`--restore-only <manifest>` 可獨立還原。

用法（正常）：
  python scripts/run_dr056_step8.py -- --arms A5 --seeds 1 ... --replay-candidate-only
用法（救援）：
  python scripts/run_dr056_step8.py --restore-only logs/exp4/step8_manifest.json
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

MANIFEST_DEFAULT = ROOT / "logs" / "exp4" / "step8_manifest.json"


class Quarantine:
    """把檔案移進隔離目錄，並保證能還原。manifest 先寫、後移動。"""

    def __init__(self, qdir: Path, manifest: Path):
        self.qdir = qdir
        self.manifest = manifest
        self.moved: list[tuple[str, str]] = []
        qdir.mkdir(parents=True, exist_ok=True)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        self._flush()

    def _flush(self) -> None:
        self.manifest.write_text(json.dumps(
            {"quarantine": str(self.qdir), "moved": self.moved}, indent=1), encoding="utf-8")

    def hide(self, paths) -> int:
        n = 0
        for p in paths:
            p = Path(p)
            if not p.is_file():
                continue
            dst = self.qdir / f"{p.parent.name}__{p.name}"
            if dst.exists():
                raise RuntimeError(f"隔離目錄已有同名檔，拒絕覆蓋：{dst}")
            self.moved.append((str(p), str(dst)))
            self._flush()                      # ⚠️ 先記錄，再移動
            os.rename(p, dst)
            n += 1
        return n

    def restore(self) -> int:
        n = 0
        for src, dst in reversed(self.moved):
            if Path(dst).is_file() and not Path(src).exists():
                os.rename(dst, src)
                n += 1
        self.moved = []
        self._flush()
        return n


def restore_only(manifest: Path) -> int:
    d = json.loads(manifest.read_text())
    n = 0
    for src, dst in reversed(d.get("moved", [])):
        if Path(dst).is_file() and not Path(src).exists():
            os.rename(dst, src)
            n += 1
    d["moved"] = []
    manifest.write_text(json.dumps(d, indent=1), encoding="utf-8")
    print(f"已還原 {n} 檔")
    return 0


def train_feature_files(cfg, task: str, task_pos: int) -> list[Path]:
    """該 task **train split** 的特徵檔（test split 不碰 —— 評估要用）。"""
    from selector.evaluate import slide_dataset
    ds, _shift = slide_dataset(cfg, task, task_pos, "train")
    feat_dir = Path(cfg["dataset_root_dir"] +
                    cfg["path_feat"].format(task, cfg["conch_path_feat"]))
    out = []
    for sid in ds.sids:
        out += sorted(feat_dir.glob(f"{sid}*"))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], add_help=False)
    ap.add_argument("--restore-only", default=None)
    ap.add_argument("--quarantine", default=str(ROOT / "logs" / "exp4" / "quarantine"))
    ap.add_argument("--manifest", default=str(MANIFEST_DEFAULT))
    ap.add_argument("-h", "--help", action="help")
    a, rest = ap.parse_known_args(argv if argv is not None else sys.argv[1:])
    if a.restore_only:
        return restore_only(Path(a.restore_only))
    if rest and rest[0] == "--":
        rest = rest[1:]

    import run_exp2

    q = Quarantine(Path(a.quarantine), Path(a.manifest))
    atexit.register(q.restore)
    orig_fill = run_exp2.fill_memory

    def fill_and_hide(memory, models, task, cfg, *args, **kw):
        added = orig_fill(memory, models, task, cfg, *args, **kw)
        pos = list(cfg["tasks"]).index(task)
        n = q.hide(train_feature_files(cfg, task, pos))
        print(f"       [step8] {task} 的 train 特徵檔已隔離 {n} 檔 "
              f"（累計 {len(q.moved)}）", flush=True)
        return added

    run_exp2.fill_memory = fill_and_hide
    try:
        sys.argv = ["run_exp2.py"] + rest
        return run_exp2.main()
    finally:
        run_exp2.fill_memory = orig_fill
        print(f"       [step8] 還原 {q.restore()} 檔", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
