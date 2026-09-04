#!/usr/bin/env python3
"""DR-048 B7：SOTA 主表 → `docs/SOTA_TABLE.md`。

**只有數字，不寫解讀** —— 與 `scripts/report_dr046_gates.py` 同一個規矩。

## 為什麼要有一張獨立的表

DR-046 的表（`docs/DR046_TABLE.md`）是 **fold 1 上的 5 個 seed**；本表是
**10 折、每折一個 run、seed = 折號**（PI 裁定 2）。兩者的變異來源不同 ——
前者只含初始化與資料順序的隨機性，後者含資料切分的隨機性 ——
**不可混讀**，也不可把兩張表的 ± 放在一起比。

`OPCM-Merge (adapted)` 是唯一的例外：它吃的是 C 臂的 delta 快取，只在
**fold 1、seed 0–4（DR-046 協定）**存在。表中逐列標明每一列的協定。

外部方法的數字（基準論文 Tab. 2）**本檔一律留空**，由 PI 逐列填入並註明出處；
腳本不從論文抄數字進來，避免產生「看起來像是我們重算的」欄位。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_exp2 import ARMS, DEFAULT_ARCH, ORDERS                      # noqa: E402
from sota.metrics import all_metrics                                 # noqa: E402

OUT = ROOT / "docs" / "SOTA_TABLE.md"

#: 顯示名稱。沒列到的臂沿用 run_exp2 的 ARMS 說明。
LABEL = {"OPCM": "OPCM-Merge (adapted)",
         "ZS-mean": "Zero-shot, mean-pool (all patches)",
         "ZS-rand8": "Zero-shot, random-B patches"}

#: 每一列的協定註記。key 是臂名。
PROTOCOL = {"OPCM": "DR-046 協定（fold 1、seed 0–4）—— **不是** 10 折"}
DEFAULT_PROTOCOL = "SOTA 協定（10 折，seed = 折號）"

#: 外部方法：只列名稱與出處欄，數字留白給 PI 填。
EXTERNAL = ["JointTrain (Upper)", "FineTune (Lower)"]

METRICS = [("acc", "ACC ↑"), ("masked_acc", "Masked ACC ↑"),
           ("forgetting", "Forgetting ↓"), ("bwt", "BWT ↑")]


def load_runs(src: Path, order: str) -> dict:
    """`{(arm, arch): {(fold, seed): [records]}}`。一個檔案 = 一個 run。"""
    runs: dict = defaultdict(lambda: defaultdict(list))
    for f in sorted(src.glob("*.json")):
        for r in json.loads(f.read_text()):
            if r.get("order") != order:
                continue
            key = (r["arm"], r.get("arch") or DEFAULT_ARCH)
            runs[key][(r.get("fold", 1), r["seed"])].append(r)
    return runs


def agg(runs: dict, tasks) -> dict:
    """每個 run 各算一次指標，再對 run 取 mean ± sd。"""
    per_run = {}
    for rk, recs in sorted(runs.items()):
        try:
            per_run[rk] = all_metrics(recs, tasks)
        except ValueError as exc:                    # 未跑完的 run：跳過並記錄
            per_run[rk] = {"_error": str(exc)}
    good = {k: v for k, v in per_run.items() if "_error" not in v}
    out = {"n": len(good), "runs": sorted(good), "skipped": sorted(set(per_run) - set(good))}
    for key, _lab in METRICS:
        vals = [v[key] for v in good.values() if v.get(key) is not None]
        out[key] = ((statistics.mean(vals),
                     statistics.stdev(vals) if len(vals) > 1 else 0.0)
                    if vals else None)
    return out


def cell(v) -> str:
    return "—" if v is None else f"{v[0]:.3f} ± {v[1]:.3f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", default="sota")
    ap.add_argument("--order", default="reverse", choices=list(ORDERS))
    args = ap.parse_args(argv)

    src = ROOT / "outputs" / "exp2" / args.tag / "per_slide"
    if not src.is_dir():
        print(f"⚠️ 沒有 {src} —— 佇列還沒跑出東西")
        return 1
    tasks = ORDERS[args.order]
    runs = load_runs(src, args.order)
    if not runs:
        print(f"⚠️ {src} 裡沒有 order = {args.order} 的記錄")
        return 1

    L = ["# SOTA 主表（DR-048）", "",
         f"任務順序：`{args.order}`＝ {' → '.join(t.replace('tcga_', '').upper() for t in tasks)}"
         "（對應基準論文 Tab. 2）。", "",
         "⚠️ **不可與 `docs/DR046_TABLE.md` 混讀。** 該表是 **fold 1 上的 5 個 seed**；"
         "本表是 **10 折、每折一個 run、seed = 折號**。兩者的 ± 量的是不同的東西"
         "（前者只含初始化／順序的隨機性，後者含資料切分的隨機性），"
         "把兩張表的數字並排比較會得到沒有意義的結論。", "",
         "⚠️ **本檔只放數字，不寫解讀。** 判讀的文字在 "
         "[`docs/ledger/DR-048.md`](ledger/DR-048.md) 與論文裡。", "",
         "指標定義見 [`sota/metrics.py`](../sota/metrics.py) 的模組說明 —— "
         "ACC 與 Masked ACC 是從官方程式核對過的，Forgetting 與 BWT 是**推得的**"
         "（論文沒有印出公式），推導依據與反證都記在那裡。", "",
         "訓練設定維持本 repo 的：**5 epoch、lr 1e-3、rank 4**"
         "（與基準論文的 12 epoch 不同，屬方法設定，PI 裁定 3）。", "",
         "| 方法 | 架構 | " + " | ".join(lab for _k, lab in METRICS) +
         " | n runs | 協定 | 溯源 |",
         "|---|---|" + "---|" * len(METRICS) + "---|---|---|"]

    for (arm, arch), rr in sorted(runs.items()):
        a = agg(rr, tasks)
        name = LABEL.get(arm) or (ARMS[arm]["name"] if arm in ARMS else arm)
        prov = ", ".join(f"fold{f}/seed{s}" for f, s in a["runs"]) or "—"
        L.append(f"| {name}（`{arm}`） | `{arch}` | " +
                 " | ".join(cell(a[k]) for k, _lab in METRICS) +
                 f" | {a['n']} | {PROTOCOL.get(arm, DEFAULT_PROTOCOL)} | `{prov}` |")
        if a["skipped"]:
            L.append(f"| ↳ ⚠️ 未完成、未計入 | | | | | | {len(a['skipped'])} | | "
                     f"`{', '.join(f'fold{f}/seed{s}' for f, s in a['skipped'])}` |")

    L += ["", "## 外部方法（基準論文 Tab. 2）", "",
          "⚠️ **數字留空，待 PI 逐列填入並註明出處。** "
          "本腳本刻意不從論文抄數字 —— 抄進來的欄位會與上表「我們自己重算的」"
          "欄位長得一模一樣，讀者無法分辨。", "",
          "| 方法 | " + " | ".join(lab for _k, lab in METRICS) + " | 出處 |",
          "|---|" + "---|" * len(METRICS) + "---|"]
    for name in EXTERNAL:
        L.append(f"| {name} | " + " | ".join("" for _ in METRICS) + " |  |")

    L += ["", "---", "",
          f"產生：`python sota/report_sota.py --tag {args.tag} --order {args.order}`。"
          f"資料源：`outputs/exp2/{args.tag}/per_slide/*.json`。", ""]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n")
    print(f"→ {OUT}")
    for (arm, arch), rr in sorted(runs.items()):
        a = agg(rr, tasks)
        acc_, f_ = a["acc"], a["forgetting"]
        print(f"  {arm:9s} {arch:5s} n={a['n']}  ACC={cell(acc_)}  Forgetting={cell(f_)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
