#!/usr/bin/env python3
"""憲法 §3.7 —— 長 job 的存活訊號。

    python scripts/job_status.py --job pipeline --state running --stage 1/4 --note "G1'-b"

寫入 `outputs/_status/<job>.json`：{state, stage, note, updated_at}。
PI 只要看這一個檔就知道 job 是活著、完成、還是死了。
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STATUS_DIR = REPO_ROOT / "outputs" / "_status"
STATES = ("running", "done", "failed")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--state", required=True, choices=STATES)
    ap.add_argument("--stage", default="")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    path = STATUS_DIR / f"{args.job}.json"
    prev = json.loads(path.read_text()) if path.is_file() else {}
    blob = {
        "job": args.job,
        "state": args.state,
        "stage": args.stage or prev.get("stage", ""),
        "note": args.note,
        "updated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "started_at": prev.get("started_at") or datetime.now(timezone.utc)
                      .astimezone().isoformat(timespec="seconds"),
    }
    path.write_text(json.dumps(blob, ensure_ascii=False, indent=1) + "\n")
    print(f"[status] {args.job}: {args.state} stage={blob['stage']} {args.note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
