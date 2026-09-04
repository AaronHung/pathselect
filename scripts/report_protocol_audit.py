#!/usr/bin/env python3
"""把協定稽核的結果寫成 `docs/DR048_PROTOCOL_AUDIT.md`（DR-048 A1）。

事實蒐集與措辭全在 `scripts/audit_benchmark_protocol.py`（唯讀模組）；
本檔只負責落檔。兩者分開是 `test_only_read_only_scripts_may_be_exempt` 的要求：
申請禁用字例外的腳本必須唯讀，所以寫檔的部分不能待在同一個檔裡。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_benchmark_protocol import build_report_lines               # noqa: E402

OUT = ROOT / "docs" / "DR048_PROTOCOL_AUDIT.md"


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(build_report_lines()) + "\n")
    print(f"\u2192 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
