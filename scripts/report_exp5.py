#!/usr/bin/env python3
"""exp5（M = 64 備份實驗）的報表。只讀 outputs/exp5，不碰其他實驗的產物。

⚠️ 本批所有數字都在同一台 5090 pod、同一批次產生。**不得與 exp3／exp4／H200 相減**
（跨機器浮點分歧可達 2.8 pp，與真實效應同量級 —— DR-052）。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_exp2 import ORDERS                                   # noqa: E402
from sota.metrics import all_metrics                          # noqa: E402

ORDER_LABEL = {"reverse": "reverse", "main": "forward"}
METRICS = [("acc", "ACC"), ("masked_acc", "Masked ACC"),
           ("forgetting", "Forgetting"), ("bwt", "BWT")]


def load_tag(src: Path, order: str) -> dict[int, dict]:
    """{fold: metrics}。一個檔案一個 run；order 進 key —— 不同順序是不同實驗。"""
    out = {}
    if not src.is_dir():
        return out
    for f in sorted(src.glob("*.json")):
        recs = json.loads(f.read_text())
        if not recs or recs[0].get("order") != order:
            continue
        fold = recs[0].get("fold", 1)
        try:
            out[fold] = all_metrics(recs, ORDERS[order])
        except ValueError:
            pass                                   # 未跑完的 run：跳過
    return out


def msd(v, nd=3):
    v = [x for x in v if x is not None]
    if not v:
        return "—"
    sd = statistics.stdev(v) if len(v) > 1 else 0.0
    return f"{statistics.mean(v):.{nd}f} ± {sd:.{nd}f}"


def paired(a: dict, b: dict, key: str):
    """同折相減。缺折就不配對 —— 不同折的數字不可相減。"""
    ks = sorted(set(a) & set(b))
    if not ks:
        return None
    d = [a[k][key] - b[k][key] for k in ks]
    better = sum(1 for x in d if x > 0) if key != "forgetting" else sum(1 for x in d if x < 0)
    return {"mean": statistics.mean(d), "n": len(ks), "better": better,
            "sd": statistics.stdev(d) if len(d) > 1 else 0.0}


def table(rows: list[tuple[str, dict]], title: str) -> list[str]:
    L = [f"### {title}", "",
         "| 設定 | 順序 | n | " + " | ".join(m[1] for m in METRICS) + " |",
         "|---|---|---|" + "---|" * len(METRICS)]
    for label, per_order in rows:
        for order in ("reverse", "main"):
            per = per_order.get(order, {})
            if not per:
                continue
            cells = [msd([m[k] for m in per.values()]) for k, _ in METRICS]
            L.append(f"| {label} | {ORDER_LABEL[order]} | {len(per)} | " + " | ".join(cells) + " |")
    return L + [""]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", default=str(ROOT / "outputs" / "exp5"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "exp5" / "EXP5_TABLES.md"))
    ap.add_argument("--buffer-json", default=None, help="measure_exp5_buffer.py 的輸出")
    a = ap.parse_args(argv)
    src = Path(a.src)

    # tag → 顯示名稱（Table 2 的鏈 ＋ Table 1 那一列）
    CHAIN = [("m64_replay", "＋replay（A3, flat）"),
             ("m64_distutil", "＋蒸餾＋效用下限（A5 flat, uold=current）"),
             ("m64_full", "＋group 層預算＝本方法（A5 hier, uold=current）")]

    data = {}
    for tag, _lab in CHAIN:
        data[tag] = {"reverse": load_tag(src / f"{tag}_rev" / "per_slide", "reverse"),
                     "main": load_tag(src / f"{tag}_fwd" / "per_slide", "main")}

    L = ["# exp5 —— |M| = 64 備份實驗的結果表", "",
         "來源 `outputs/exp5/`，機器與環境見 [MACHINE.md](MACHINE.md)，預註冊見 `docs/ledger/DR-057.md`。", "",
         "⚠️ **本批所有數字都在同一台 5090 pod、同一批次產生。不得與 exp3／exp4／H200 相減。**",
         "⚠️ **兩套數字絕不混用**：H200 那套若在 9/23 16:00 前完整完成，論文全部用 H200；否則整套用本批。", ""]

    L += table([(lab, data[tag]) for tag, lab in CHAIN], "Table 2 的鏈（|M| = 64，十折，seed = fold）")

    # 逐折配對：本方法 − 只有 replay
    L += ["### 逐折配對（同折相減，同批同機）", "",
          "| 配對 | 順序 | ΔACC | ΔMasked | ΔForgetting ↓ | n |", "|---|---|---|---|---|---|"]
    for order in ("reverse", "main"):
        for lab, hi, lo in (("本方法 − 只有 replay", "m64_full", "m64_replay"),
                            ("本方法 − ＋蒸餾＋效用下限", "m64_full", "m64_distutil"),
                            ("＋蒸餾＋效用下限 − 只有 replay", "m64_distutil", "m64_replay")):
            A, B = data[hi][order], data[lo][order]
            cells = []
            n = 0
            for k, _ in METRICS[:3]:
                r = paired(A, B, k)
                if r is None:
                    cells.append("—")
                else:
                    n = r["n"]
                    cells.append(f"{r['mean']:+.4f}（{r['better']}/{r['n']}）")
            if n:
                L.append(f"| {lab} | {ORDER_LABEL[order]} | " + " | ".join(cells) + f" | {n} |")
    L += ["",
          "「較佳折數」對 ACC／Masked 是差值為正的折數，對 Forgetting 是差值為負的折數。",
          "⚠️ 三級 win-count 規則（5/5 systematic 等）是為 model seed 校準的，**未對 fold 層級的",
          "變異來源重新校準**，所以這裡只列勝負折數，不做等級判讀（PI 裁示，見 docs/SOTA_TABLE.md）。", ""]

    # buffer 實測
    if a.buffer_json and Path(a.buffer_json).is_file():
        B = json.loads(Path(a.buffer_json).read_text())
        L += ["### replay 實際需要的特徵量（實測）", "",
              "replay 會重新載入被記憶庫留下的那些切片的**完整特徵檔**。下表是留在記憶庫裡的",
              f"那 |M| 筆所對應的特徵檔大小總和（無模型重放 reservoir 汰換算出確切的保留集合）。", "",
              "| \\|M\\| | 順序 | 保留筆數 | 特徵量 | 占訓練特徵庫 |", "|---|---|---|---|---|"]
        for e in B.get("entries", []):
            L.append(f"| {e['capacity']} | {ORDER_LABEL.get(e['order'], e['order'])} | {e['found']} | "
                     f"**{e['total_MiB']:.1f} MiB** | {e['pct_of_train']:.2f}% |")
        L += ["", f"訓練特徵庫總量：{B.get('train_total_GiB', 0):.2f} GiB（{B.get('train_files', 0)} 檔）。", ""]

    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"→ {a.out}")
    for tag, lab in CHAIN:
        for order in ("reverse", "main"):
            n = len(data[tag][order])
            print(f"  {lab[:28]:<30} {ORDER_LABEL[order]:<8} {n}/10 折")
    return 0


if __name__ == "__main__":
    sys.exit(main())
