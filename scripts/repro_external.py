#!/usr/bin/env python3
"""DR-048 Prompt 10-5：外部基準方法的**重現檢查**。

流程（每一步都寫進 log，失敗一律記錄原因，不靜默略過）：

  1. clone 到 repo **之外**（預設 `~/research/ext/`），已存在就重用、不覆寫
  2. 把設定檔複製一份並改成本機路徑（**不改對方原檔**）
  3. smoke：reverse、fold 1、1 epoch，**帶硬性逾時**（過夜跑不能卡住）
  4. smoke 通過且單折推估 <= 預算 → 以對方**預設設定**跑 reverse 十折
     smoke 不通過 → 寫下不可行紀錄（含逐字錯誤訊息）並正常結束

⚠️ 這台機器**沒有 CUDA**（只有 MPS），而對方設定是 `cuda_id: 0`。
   這是 smoke 最可能的失敗點 —— 正因如此才要先 smoke，而不是直接排十折。

⚠️ 外部方法的名稱、URL、產物目錄名一律從 `scripts/repro_baseline_coords.py`
   取得（該模組是唯讀且有窄例外）；本檔**不含任何禁用識別字**。
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import repro_baseline_coords as C                                  # noqa: E402
from selector.text_encoder import load_config                      # noqa: E402

LOGDIR = ROOT / "logs" / "sota"
OUT_ROOT = ROOT / "outputs" / "exp2" / "sota"


def log(fh, msg: str) -> None:
    line = f"[{time.strftime('%F %T')}] {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def infeasible(reason: str, detail: str = "") -> int:
    """寫下不可行紀錄。**正常結束（return 0）** —— 這是預期中的結果之一，
    不是佇列失敗；讓後續步驟照跑。"""
    LOGDIR.mkdir(parents=True, exist_ok=True)
    p = LOGDIR / C.INFEASIBLE_NAME
    body = [f"時間：{time.strftime('%F %T')}",
            f"結論：**重現檢查不可行** —— {reason}", "",
            "本機環境：",
            f"  platform     {sys.platform}",
            f"  python       {sys.version.split()[0]}", ""]
    try:
        import torch
        body += [f"  torch        {torch.__version__}",
                 f"  cuda         {torch.cuda.is_available()}",
                 f"  mps          {torch.backends.mps.is_available()}", ""]
    except Exception:                                   # noqa: BLE001
        body += ["  torch        （無法載入）", ""]
    body += ["逐字錯誤／診斷：", detail or "（無）", ""]
    p.write_text("\n".join(body))
    print(f"→ {p}", flush=True)
    return 0


def clone(fh, ext_dir: Path) -> Path | None:
    dest = ext_dir / C.CLONE_DIRNAME
    if (dest / ".git").is_dir():
        log(fh, f"已存在，重用：{dest}")
        return dest
    ext_dir.mkdir(parents=True, exist_ok=True)
    log(fh, f"clone → {dest}")
    r = subprocess.run(["git", "clone", "--depth", "1", C.CLONE_URL, str(dest)],
                       capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        log(fh, f"❌ clone 失敗：{r.stderr[-800:]}")
        return None
    return dest


def patched_config(fh, repo: Path, work: Path, *, smoke: bool) -> Path | None:
    """複製一份設定檔並改成本機路徑與 reverse 順序。**不動對方原檔。**"""
    src = repo / C.CONFIG_REL
    if not src.is_file():
        log(fh, f"❌ 找不到設定檔 {src}")
        return None
    cfg = load_config()
    text = src.read_text()
    subs = {
        "dataset_root_dir": cfg["dataset_root_dir"],
        "conch_ckpt_path": cfg["conch_ckpt_path"],
        "result_dir": str(work),          # 產物落到我們的目錄，不寫進對方 repo
    }
    for key, val in subs.items():
        text, n = re.subn(rf"^{key}:.*$", f"{key}: {val}", text, count=1, flags=re.M)
        if not n:
            log(fh, f"⚠️ 設定檔沒有 `{key}`，未覆寫")
        else:
            log(fh, f"  {key} → {val}")
    # reverse 順序（其 Tab. 2）—— 除了 config，還必須換掉 class ensemble 的鍵序
    text, n = re.subn(r"^class_ensemble_path:.*$",
                      f"class_ensemble_path: {C.CLASS_ENSEMBLE_REVERSE}",
                      text, count=1, flags=re.M)
    log(fh, f"  class_ensemble_path → {C.CLASS_ENSEMBLE_REVERSE}"
            f"{'' if n else '（⚠️ 未覆寫）'}")
    for key, val in (("dataset_names", C.REVERSE_DATASET_NAMES),
                     ("dataset_label_shift", C.REVERSE_LABEL_SHIFT),
                     ("dataset_subtype_num", C.REVERSE_SUBTYPE_NUM)):
        text = re.sub(rf"^{key}:.*$", f"{key}: {list(val)}", text, count=1, flags=re.M)
    if smoke:
        text = re.sub(r"^epochs:.*$", f"epochs: {list(C.SMOKE_EPOCHS)}",
                      text, count=1, flags=re.M)
    work.mkdir(parents=True, exist_ok=True)
    out = work / ("config_smoke.yaml" if smoke else "config_full.yaml")
    out.write_text(text)
    log(fh, f"設定檔已產生：{out}")
    return out


def run_entry(fh, repo: Path, cfg_path: Path, extra: list[str], timeout: int,
              python: str | None = None):
    """跑對方的進入點。回傳 (returncode, 秒數, 尾端輸出)。

    `--time` 是對方的**必填**旗標（`main.py --help` 確認），這裡自動帶上。
    """
    env = dict(os.environ)
    env["WANDB_MODE"] = "offline"          # 過夜不可能有人去登入
    env["WANDB_SILENT"] = "true"
    cmd = [python or sys.executable, C.ENTRY, "--config", str(cfg_path),
           "--time", time.strftime("%Y%m%d-%H%M%S"), *extra]
    log(fh, f"執行：{' '.join(cmd)}  （cwd={repo}, timeout={timeout}s）")
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=repo, env=env, capture_output=True,
                           text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, time.time() - t0, f"逾時（{timeout}s）未結束"
    tail = ((r.stdout or "")[-1500:] + "\n" + (r.stderr or "")[-2500:]).strip()
    return r.returncode, time.time() - t0, tail


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ext-dir", default=str(Path.home() / "research" / "ext"),
                    help="clone 的落點（**必須在 repo 之外**）")
    ap.add_argument("--smoke-timeout", type=int, default=5400,
                    help="smoke 的硬性逾時秒數（預設 90 分）")
    ap.add_argument("--fold-timeout", type=int, default=7200)
    ap.add_argument("--folds", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--python", default=None,
                    help="跑對方程式的直譯器（獨立環境的 venv）；預設用本行程的")
    ap.add_argument("--fold-budget-min", type=float, default=float(C.FOLD_BUDGET_MIN),
                    help="單折預算（分）。超過就只跑 --folds-over-budget 折")
    ap.add_argument("--folds-over-budget", type=int, default=3)
    ap.add_argument("--skip-smoke", type=float, default=None, metavar="EST_MIN",
                    help="跳過 smoke，直接用給定的單折推估（分）。"
                         "只在**同一輪已經量過** smoke、要接續跑其餘折時使用；"
                         "值必須是實測推導的，不得憑空填。")
    ap.add_argument("--deadline", default=None,
                    help="不再**開始**新的一折的時刻，格式 YYYY-MM-DDTHH:MM。"
                         "已開始的那一折會跑完。")
    args = ap.parse_args(argv)

    ext_dir = Path(args.ext_dir).expanduser().resolve()
    if ROOT in ext_dir.parents or ext_dir == ROOT:
        return infeasible("clone 落點在 repo 之內，違反紅線", f"ext_dir={ext_dir}")

    LOGDIR.mkdir(parents=True, exist_ok=True)
    work = OUT_ROOT / C.OUT_SUBDIR
    with (LOGDIR / "repro_external.log").open("a") as fh:
        log(fh, "═══ 外部基準重現檢查開始 ═══")
        repo = clone(fh, ext_dir)
        if repo is None:
            return infeasible("clone 失敗", "見 logs/sota/repro_external.log")

        rc, _s, tail = run_entry(fh, repo, Path("--help-probe"), ["--help"], 120,
                                 args.python)
        log(fh, f"`--help` returncode={rc}；輸出尾端：\n{tail[:900]}")

        cfg_s = patched_config(fh, repo, work, smoke=True)
        if cfg_s is None:
            return infeasible("無法產生設定檔", "見 log")

        if args.skip_smoke is not None:
            est = args.skip_smoke * 60
            log(fh, f"⏭ 依 --skip-smoke 跳過 smoke，沿用先前實測推導的單折推估 "
                    f"{args.skip_smoke:.0f} 分")
        else:
            log(fh, "── smoke：reverse、fold 1、1 epoch ──")
            rc, secs, tail = run_entry(fh, repo, cfg_s, ["--seed", "1"],
                                       args.smoke_timeout, args.python)
            log(fh, f"smoke returncode={rc}，耗時 {secs / 60:.1f} 分")
            if rc != 0:
                log(fh, "❌ smoke 未通過")
                return infeasible(
                    f"smoke 未通過（returncode={rc}，耗時 {secs / 60:.1f} 分）", tail)
            est = secs * 12                  # 1 epoch → 預設 12 epoch 的線性外推
            log(fh, f"單折推估 {est / 60:.0f} 分（1 epoch 實測 {secs / 60:.1f} 分 × 12）")
        folds = [int(x) for x in args.folds.split(",")]
        if est / 60 > args.fold_budget_min:
            folds = folds[:args.folds_over_budget]
            log(fh, f"⚠️ 單折推估 {est / 60:.0f} 分 > 預算 {args.fold_budget_min:.0f} 分 "
                    f"→ 依裁定只跑前 {len(folds)} 折 {folds}")
        else:
            log(fh, f"單折推估在預算內 → 跑 {len(folds)} 折")

        cfg_f = patched_config(fh, repo, work, smoke=False)
        if cfg_f is None:
            return infeasible("無法產生正式設定檔", "見 log")
        deadline = None
        if args.deadline:
            deadline = time.mktime(time.strptime(args.deadline, "%Y-%m-%dT%H:%M"))
            log(fh, f"截止：{args.deadline}（之後不再開始新的一折）")
        bad, done = [], []
        for k in folds:
            if deadline and time.time() >= deadline:
                log(fh, f"⏹ 已過截止時刻，停止；完成 {len(done)} 折 {done}")
                break
            log(fh, f"── fold {k} ──")
            rc, secs, tail = run_entry(fh, repo, cfg_f, ["--seed", str(k)],
                                       args.fold_timeout, args.python)
            log(fh, f"fold {k} returncode={rc}，耗時 {secs / 60:.1f} 分")
            if rc != 0:
                bad.append(k)
                log(fh, f"❌ fold {k} 失敗：{tail[-600:]}")
            else:
                done.append(k)
        log(fh, f"═══ 結束：完成 {len(done)} 折 {done}；失敗 {len(bad)} 折 "
                f"{bad if bad else ''} ═══")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
