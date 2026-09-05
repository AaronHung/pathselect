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
from sota.external_baselines import (CAVEATS, CITATION,              # noqa: E402
                                     ORDER_NOTE, ROWS)
from sota.metrics import all_metrics                                 # noqa: E402

OUT = ROOT / "docs" / "SOTA_TABLE.md"

#: 顯示名稱。沒列到的臂沿用 run_exp2 的 ARMS 說明。
LABEL = {"OPCM": "OPCM-Merge (adapted, paper Alg. 1)",
         "OPCM-nomask": "OPCM (released code, no-op mask)",
         "ZS-mean": "Zero-shot, mean-pool (all patches)",
         "ZS-rand8": "Zero-shot, random-B patches"}

#: 每一列的協定註記。key 是臂名。
_OPCM_PROTO = "DR-046 協定（fold 1、seed 0–4）—— **不是** 10 折"
PROTOCOL = {"OPCM": _OPCM_PROTO, "OPCM-nomask": _OPCM_PROTO}
DEFAULT_PROTOCOL = "SOTA 協定（10 折，seed = 折號）"
def protocol_of(arm: str, order: str) -> str:
    base = PROTOCOL.get(arm, DEFAULT_PROTOCOL)
    return f"{base}；{ORDER_NOTE.get(order, order)}"

METRICS = [("acc", "ACC ↑"), ("masked_acc", "Masked ACC ↑"),
           ("forgetting", "Forgetting ↓"), ("bwt", "BWT ↑")]


def load_runs(src: Path, orders) -> dict:
    """`{(arm, arch, order): {(fold, seed): [records]}}`。一個檔案 = 一個 run。

    ⚠️ `order` 進 key —— 不同任務順序是**不同的實驗**，
    而且 `all_metrics` 要用該 order 自己的 task 序（forward 是
    LUNG→BRCA→RCC→ESCA），混在一起算出來的準確率矩陣會錯位。
    """
    want = {orders} if isinstance(orders, str) else set(orders)
    runs: dict = defaultdict(lambda: defaultdict(list))
    for f in sorted(src.glob("*.json")):
        for r in json.loads(f.read_text()):
            o = r.get("order")
            if o not in want:
                continue
            key = (r["arm"], r.get("arch") or DEFAULT_ARCH, o)
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


def _provenance(runs: list[tuple[int, int]]) -> str:
    """把 (fold, seed) 清單壓成一行可讀的溯源說明。

    十折逐一列出會佔掉整個欄寬，反而看不出重點；但**不能只寫個數** ——
    讀者要能確認是「哪十折」。所以連續且 seed = fold 時壓成區間，
    其餘情形照舊逐一列出。
    """
    if not runs:
        return "—"
    folds = [f for f, _s in runs]
    if all(s == f for f, s in runs) and folds == list(range(min(folds), max(folds) + 1)):
        return f"fold {min(folds)}–{max(folds)}（seed = fold）"
    if len({f for f, _s in runs}) == 1:
        f = folds[0]
        seeds = sorted(s for _f, s in runs)
        if seeds == list(range(min(seeds), max(seeds) + 1)):
            return f"fold {f}，seed {min(seeds)}–{max(seeds)}"
    return ", ".join(f"fold{f}/seed{s}" for f, s in runs)


#: 逐折配對比較（PI 指定，Prompt 7-4）。(名稱, A 的 key, B 的 key)
#: key = (arm, arch, order)。一律報 **A − B**。
PAIRS = [("hier − flat（A5, reverse）", ("A5", "hier", "reverse"), ("A5", "flat", "reverse")),
         ("A5 − A3（flat, reverse）",   ("A5", "flat", "reverse"), ("A3", "flat", "reverse")),
         ("A5 − A1（flat, reverse）",   ("A5", "flat", "reverse"), ("A1", "flat", "reverse"))]

#: 配對只報這三軸（PI 指定）。(key, 顯示名, 越大越好)
PAIR_METRICS = [("acc", "ACC", True), ("masked_acc", "Masked ACC", True),
                ("forgetting", "Forgetting", False)]


def paired(runs: dict, ka, kb) -> list[dict]:
    """同折配對：只取兩邊都有的 fold，逐折相減。"""
    ra, rb = runs.get(ka, {}), runs.get(kb, {})
    ta, tb = ORDERS[ka[2]], ORDERS[kb[2]]
    common = sorted(set(ra) & set(rb))
    out = []
    for key, lab, higher in PAIR_METRICS:
        d = []
        for rk in common:
            try:
                va = all_metrics(ra[rk], ta)[key]
                vb = all_metrics(rb[rk], tb)[key]
            except ValueError:
                continue
            if va is not None and vb is not None:
                d.append((rk[0], va - vb))
        if not d:
            continue
        vals = [x for _f, x in d]
        better = sum((x > 0) if higher else (x < 0) for x in vals)
        out.append({"label": lab, "n": len(vals), "better": better,
                    "mean": statistics.mean(vals),
                    "per_fold": d, "higher": higher})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", default="sota")
    ap.add_argument("--order", default="reverse", choices=list(ORDERS),
                    help="主順序；forward（`main`）的臂會另列一行，不需分開跑")
    ap.add_argument("--src", default=None,
                    help="改讀別的 per_slide 目錄（測試用；預設由 --tag 決定）")
    ap.add_argument("--out", default=None,
                    help="改寫到別的路徑（測試用；預設 docs/SOTA_TABLE.md）")
    args = ap.parse_args(argv)

    src = Path(args.src) if args.src else ROOT / "outputs" / "exp2" / args.tag / "per_slide"
    out = Path(args.out) if args.out else OUT
    if not src.is_dir():
        print(f"⚠️ 沒有 {src} —— 佇列還沒跑出東西")
        return 1
    runs = load_runs(src, list(ORDERS))
    if not runs:
        print(f"⚠️ {src} 裡沒有任何可用記錄")
        return 1

    main_tasks = ORDERS[args.order]
    L = ["# SOTA 主表（DR-048）", "",
         f"主順序：`{args.order}`＝ "
         f"{' → '.join(t.replace('tcga_', '').upper() for t in main_tasks)}"
         "（對應基準論文 Tab. 2）。**forward 順序的臂另列一行**，"
         "協定欄逐列標明是哪一個順序。", "",
         "⚠️ **不可與 `docs/DR046_TABLE.md` 混讀。** 該表是 **fold 1 上的 5 個 seed**；"
         "本表是 **10 折、每折一個 run、seed = 折號**。兩者的 ± 量的是不同的東西"
         "（前者只含初始化／順序的隨機性，後者含資料切分的隨機性），"
         "把兩張表的數字並排比較會得到沒有意義的結論。", "",
         "⚠️ **本檔只放數字，不寫解讀。** 判讀的文字在 "
         "[`docs/ledger/DR-048.md`](ledger/DR-048.md) 與論文裡。", "",
         "指標定義見 [`sota/metrics.py`](../sota/metrics.py) 的模組說明："
         "**ACC** 與 **Masked ACC** 從基準論文的官方程式逐行核對；"
         "**BWT** 依 Lopez-Paz & Ranzato (2017)、**Forgetting** 依 "
         "Chaudhry et al. (2018) 的 max-based 定義（PI 裁定，Prompt 6-3）。", "",
         "訓練設定維持本 repo 的：**5 epoch、lr 1e-3、rank 4**"
         "（與基準論文的 12 epoch 不同，屬方法設定，PI 裁定 3）。", "",
         "OPCM 有**兩列**：`OPCM` 照論文 Alg. 1（零對角遮罩生效）、"
         "`OPCM-nomask` 重現官方釋出程式的實際行為"
         "（`Tensor.diag()` 回傳副本，那行遮罩是 no-op，`G(·)` 退化成恆等）。"
         "詳見 [`sota/opcm.py`](../sota/opcm.py) 與 "
         "[`docs/ledger/DR-048.md`](ledger/DR-048.md)。", "",
         "| 方法 | 架構 | " + " | ".join(lab for _k, lab in METRICS) +
         " | n runs | 協定 | 溯源 |",
         "|---|---|" + "---|" * len(METRICS) + "---|---|---|"]

    #: reverse 在前、forward 在後；其餘照臂名排序
    order_rank = {args.order: 0}
    for (arm, arch, order), rr in sorted(
            runs.items(), key=lambda kv: (order_rank.get(kv[0][2], 1), kv[0][0], kv[0][1])):
        a = agg(rr, ORDERS[order])          # ⚠️ 用該 order 自己的 task 序
        name = LABEL.get(arm) or (ARMS[arm]["name"] if arm in ARMS else arm)
        L.append(f"| {name}（`{arm}`） | `{arch}` | " +
                 " | ".join(cell(a[k]) for k, _lab in METRICS) +
                 f" | {a['n']} | {protocol_of(arm, order)} | `{_provenance(a['runs'])}` |")
        if a["skipped"]:
            L.append(f"| ↳ ⚠️ 未完成、未計入 | | | | | | {len(a['skipped'])} | | "
                     f"`{', '.join(f'fold{f}/seed{s}' for f, s in a['skipped'])}` |")

    # ── 逐折配對 ────────────────────────────────────────────────────────────
    L += ["", "## 配對（逐折，同折相減）", "",
          "只報 **k/10 的折數**與**均值差**，**不貼三級標籤** —— "
          "DR-020 的三級規則是為 5 個 model seed 校準的，這裡的變異源是**資料切分**，"
          "沒有重新校準過，套用會是誤用（PI 裁定，Prompt 7-4）。", "",
          "「較佳折數」的方向：**ACC / Masked ACC 越大越佳、Forgetting 越小越佳** ——"
          "三軸都以「A 較佳」計數，不是一律數正號。", ""]
    for title, ka, kb in PAIRS:
        rows = paired(runs, ka, kb)
        L += [f"### {title}", ""]
        if not rows:
            L += ["⚠️ 缺資料。", ""]
            continue
        L += ["| 指標 | 逐折差值（fold 1→10） | 均值差 | A 較佳的折數 |",
              "|---|---|---|---|"]
        for r in rows:
            per = ", ".join(f"{x:+.3f}" for _f, x in r["per_fold"])
            L.append(f"| {r['label']}{' ↓' if not r['higher'] else ' ↑'} | {per} | "
                     f"**{r['mean']:+.4f}** | **{r['better']}/{r['n']}** |")
        L += [""]

    L += ["## 外部方法（基準論文 Tab. 2，reverse、10 折）", ""]
    L += [f"* {c}" for c in CAVEATS]
    L += ["* **forward 順序（其 Tab. 1）的外部數字本輪不填** —— "
          "PI 未指定，腳本不自行從論文擷取。", ""]
    L += ["| 方法 | ACC ↑ | Masked ACC ↑ | Forgetting ↓ | BWT ↑ | 出處 |",
          "|---|---|---|---|---|---|"]
    dash = lambda v: "–" if v is None else v
    for name, acc_, forg, bwt_, macc in ROWS:
        L.append(f"| {name} | {dash(acc_)} | {dash(macc)} | {dash(forg)} | "
                 f"{dash(bwt_)} | {CITATION} |")

    L += ["", "---", "",
          f"產生：`python sota/report_sota.py --tag {args.tag} --order {args.order}`。"
          f"資料源：`outputs/exp2/{args.tag}/per_slide/*.json`。", ""]

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print(f"→ {out}")
    for (arm, arch, order), rr in sorted(runs.items()):
        a = agg(rr, ORDERS[order])
        print(f"  {arm:12s} {arch:5s} {order:8s} n={a['n']}  ACC={cell(a['acc'])}  "
              f"Forgetting={cell(a['forgetting'])}")
    for title, ka, kb in PAIRS:
        for r in paired(runs, ka, kb):
            print(f"  {title:26s} {r['label']:11s} {r['mean']:+.4f}  {r['better']}/{r['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
