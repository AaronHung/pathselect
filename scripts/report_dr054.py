#!/usr/bin/env python3
"""DR-054 報表：outputs/exp3/DR054_UOLD.md（只讀 per_slide）。

第一階段：outputs/exp3/ablation_ucur（A5、B2；fold 1、seeds 0–4、累積式頭、--uold current）
  與 B5（ablation_acc：同設定但 U_old 用快照 C_old）、B6（ablation_fixed：固定頭 pod）並列；
  凍結判準：A5_ucur − A3（B5 的 A3）class-IL ≥ +1.5 pp 且 ≥ 4/5 → 進第二階段。
第二階段（若有）：outputs/exp3/sota_ucur（A5 flat 十折兩順序）與 B7 的 A3、B1/B3 的 A5 配對。
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from report_exp3 import ABL_AXES, load_abl                                # noqa: E402
from run_exp2 import ORDERS                                               # noqa: E402
from selector.text_encoder import load_config                             # noqa: E402
from sota.metrics import all_metrics                                      # noqa: E402
from sota.report_sota import load_runs, paired                            # noqa: E402

THRESH_PP, THRESH_WINS = 1.5, 4


def msd(v, nd=2):
    v = [x for x in v if x is not None]
    if not v:
        return "—"
    return f"{statistics.mean(v):.{nd}f} ± {statistics.stdev(v) if len(v) > 1 else 0:.{nd}f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", default=str(ROOT / "outputs" / "exp3"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    src = Path(a.src); out = Path(a.out) if a.out else src / "DR054_UOLD.md"
    cfg = load_config(); ls = list(cfg["tasks"]); tasks = ORDERS["reverse"]
    U = load_abl(src / "ablation_ucur" / "per_slide", "_acc_ucur", tasks, ls)
    A = load_abl(src / "ablation_acc" / "per_slide", "_acc", tasks, ls)
    Fx = load_abl(src / "ablation_fixed" / "per_slide", "", tasks, ls)
    L = ["# DR-054 — hinge 的 U_old 以當前 C_t 重算（`--uold current`）", "",
         "累積式頭、flat、pod；與 B5（U_old 用快照 C_old）、B6（固定頭 pod）同 seed 並列。**只報告。**", "",
         "## 第一階段：fold 1、seeds 0–4", "",
         "| 臂 | 設定 | class-IL | task-IL | 洩漏率 | Jaccard | ΔU(M1, all8) |", "|---|---|---|---|---|---|---|"]
    def row(arm, lab, D):
        if not D[arm]:
            L.append(f"| {arm} | {lab} | 尚未 | | | | |"); return
        S = sorted(D[arm])
        L.append(f"| {arm} | {lab} | {msd([D[arm][s]['final_class_il']*100 for s in S])} | "
                 f"{msd([D[arm][s]['final_task_il']*100 for s in S])} | {msd([D[arm][s]['mean_leak']*100 for s in S])} | "
                 f"{msd([D[arm][s]['mean_jaccard'] for s in S], 4)} | {msd([D[arm][s]['dU_M1_all8'] for s in S], 3)} |")
    for arm in ("A5", "B2"):
        row(arm, "累積式 + U_old 當前 C_t（DR-054）", U)
        row(arm, "累積式 + U_old 快照 C_old（B5）", A)
        row(arm, "固定頭 pod（B6）", Fx)
    row("A3", "累積式（B5；無 hinge，本卡對照基準）", A)
    L += ["", "## 凍結判準：A5_ucur − A3（B5）class-IL ≥ +1.5 pp 且 ≥ 4/5", ""]
    verdict = None
    if U["A5"] and A["A3"]:
        S = sorted(set(U["A5"]) & set(A["A3"]))
        d = [(U["A5"][s]["final_class_il"] - A["A3"][s]["final_class_il"]) * 100 for s in S]
        wins = sum(x > 0 for x in d)
        ok = statistics.mean(d) >= THRESH_PP and wins >= THRESH_WINS
        verdict = ok
        L += [f"* A5_ucur − A3：**{statistics.mean(d):+.2f} ± {statistics.stdev(d):.2f} pp，A5 較高 {wins}/{len(d)}**"
              f"（逐 seed {', '.join(f'{x:+.2f}' for x in d)}）→ **{'達標，進第二階段' if ok else '未達標，停止；只報告'}**"]
        for other, lab in (("A5", "A5_acc（B5）"), ):
            d2 = [(U["A5"][s]["final_class_il"] - A[other][s]["final_class_il"]) * 100 for s in S]
            L.append(f"* A5_ucur − {lab}：{statistics.mean(d2):+.2f} ± {statistics.stdev(d2):.2f} pp（{sum(x > 0 for x in d2)}/{len(d2)}）")
        d3 = [(U["A5"][s]["final_class_il"] - Fx["A5"][s]["final_class_il"]) * 100 for s in S if s in Fx["A5"]]
        if d3:
            L.append(f"* A5_ucur − A5 固定頭 pod（B6）：{statistics.mean(d3):+.2f} ± {statistics.stdev(d3):.2f} pp（{sum(x > 0 for x in d3)}/{len(d3)}）")
        if U["B2"]:
            Sb = sorted(set(U["B2"]) & set(A["B2"]))
            db = [(U["B2"][s]["final_class_il"] - A["B2"][s]["final_class_il"]) * 100 for s in Sb]
            dbf = [(U["B2"][s]["final_class_il"] - Fx["B2"][s]["final_class_il"]) * 100 for s in Sb if s in Fx["B2"]]
            L.append(f"* B2_ucur − B2_acc（B5）：{statistics.mean(db):+.2f} ± {statistics.stdev(db):.2f} pp（{sum(x > 0 for x in db)}/{len(db)}）；"
                     f"B2_ucur − B2 固定頭 pod：{statistics.mean(dbf):+.2f}（{sum(x > 0 for x in dbf)}/{len(dbf)}）")
    else:
        L.append("尚未有資料。")
    # 第二階段
    d10 = src / "sota_ucur" / "per_slide"
    L += ["", "## 第二階段：A5 flat 十折兩順序（`--uold current`）", ""]
    if d10.is_dir() and any(d10.glob("*.json")):
        ucur = load_runs(d10, list(ORDERS)); acc = load_runs(src / "sota_acc" / "per_slide", list(ORDERS))
        L += ["| 順序 | ACC | Masked ACC | Forgetting | BWT | n |", "|---|---|---|---|---|---|"]
        for order in ("reverse", "main"):
            key = ("A5", "flat", order)
            if key not in ucur:
                continue
            per = {}
            for rk, recs in ucur[key].items():
                try: per[rk] = all_metrics(recs, ORDERS[order])
                except ValueError: pass
            L.append(f"| {order} | {msd([m['acc'] for m in per.values()], 3)} | {msd([m['masked_acc'] for m in per.values()], 3)} | "
                     f"{msd([m['forgetting'] for m in per.values()], 3)} | {msd([m['bwt'] for m in per.values()], 3)} | {len(per)} |")
        L += ["", "| 配對（A − B） | 順序 | ACC | Masked ACC | Forgetting |", "|---|---|---|---|---|"]
        merged = dict(acc); merged.update({(k[0] + "_ucur", k[1], k[2]): v for k, v in ucur.items()})
        for order in ("reverse", "main"):
            for lab, ka, kb in (("A5_ucur − A3（B7）", ("A5_ucur", "flat", order), ("A3", "flat", order)),
                                ("A5_ucur − A5_acc（B1/B3）", ("A5_ucur", "flat", order), ("A5", "flat", order))):
                rows = {r["label"]: r for r in paired(merged, ka, kb)}
                cells = [f"{rows[k]['mean']:+.4f}（{rows[k]['better']}/{rows[k]['n']}）" if k in rows else "—"
                         for k in ("ACC", "Masked ACC", "Forgetting")]
                L.append(f"| {lab} | {order} | " + " | ".join(cells) + " |")
    else:
        L.append("未執行（第一階段未達標）" if verdict is False else "尚未有資料。")
    # ── 第三階段（Prompt 24）：兩層選擇器 hier ＋ 合規效用下限，十折兩順序 ──
    L += ["", "## 第三階段（Prompt 24）：A5 hier 十折兩順序（`--uold current`）", ""]
    if d10.is_dir() and any(d10.glob("*_hier_acc_ucur.json")):
        ucur = load_runs(d10, list(ORDERS)); acc = load_runs(src / "sota_acc" / "per_slide", list(ORDERS))
        L += ["| 架構 | 順序 | ACC | Masked ACC | Forgetting | BWT | n |", "|---|---|---|---|---|---|---|"]
        for arch in ("hier", "flat"):
            for order in ("reverse", "main"):
                key = ("A5", arch, order)
                if key not in ucur:
                    continue
                per = {}
                for rk, recs in ucur[key].items():
                    try: per[rk] = all_metrics(recs, ORDERS[order])
                    except ValueError: pass
                L.append(f"| {arch} | {order} | {msd([m['acc'] for m in per.values()], 3)} | {msd([m['masked_acc'] for m in per.values()], 3)} | "
                         f"{msd([m['forgetting'] for m in per.values()], 3)} | {msd([m['bwt'] for m in per.values()], 3)} | {len(per)} |")
        L += ["", "| 配對（A − B；同口徑 `--uold current` 除註明） | 順序 | ACC | Masked ACC | Forgetting |", "|---|---|---|---|---|"]
        merged = dict(acc); merged.update({(k[0] + "_ucur", k[1], k[2]): v for k, v in ucur.items()})
        for order in ("reverse", "main"):
            for lab, ka, kb in (("hier − flat（皆 ucur）", ("A5_ucur", "hier", order), ("A5_ucur", "flat", order)),
                                ("A5_hier_ucur − A3（B7，累積式 flat）", ("A5_ucur", "hier", order), ("A3", "flat", order)),
                                ("A5_hier_ucur − A5_hier_acc（B1/B3 快照口徑）", ("A5_ucur", "hier", order), ("A5", "hier", order))):
                rows = {r["label"]: r for r in paired(merged, ka, kb)}
                cells = [f"{rows[k]['mean']:+.4f}（{rows[k]['better']}/{rows[k]['n']}）" if k in rows else "—"
                         for k in ("ACC", "Masked ACC", "Forgetting")]
                L.append(f"| {lab} | {order} | " + " | ".join(cells) + " |")
    else:
        L.append("尚未有資料。")
    # ── DR-055：evidence budget sweep（B ∈ {4,16} vs B = 8）──
    L += ["", "## DR-055 — evidence budget sweep（A5 hier、`--uold current`、十折）", ""]
    b8 = load_runs(d10, list(ORDERS)) if d10.is_dir() else {}
    sw = {}
    for b in (4, 16):
        dd = src / f"sota_ucur_b{b}" / "per_slide"
        if dd.is_dir() and any(dd.glob("*.json")):
            sw[b] = load_runs(dd, list(ORDERS))
    if sw:
        L += ["| B | 順序 | ACC | Masked ACC | Forgetting | BWT | n |", "|---|---|---|---|---|---|---|"]
        for b, runs in [(4, sw.get(4)), (8, b8), (16, sw.get(16))]:
            if not runs:
                continue
            for order in ("reverse", "main"):
                key = ("A5", "hier", order)
                if key not in runs:
                    continue
                per = {}
                for rk, recs in runs[key].items():
                    try: per[rk] = all_metrics(recs, ORDERS[order])
                    except ValueError: pass
                # 自檢：per_slide 的 B 欄位必須等於本列的 B
                bs = {r.get("B") for recs in runs[key].values() for r in recs if "B" in r}
                tag = "" if bs == {b} else f" ⚠️ per_slide 的 B 欄位 = {sorted(bs)}"
                L.append(f"| {b} | {order} | {msd([m['acc'] for m in per.values()], 3)} | {msd([m['masked_acc'] for m in per.values()], 3)} | "
                         f"{msd([m['forgetting'] for m in per.values()], 3)} | {msd([m['bwt'] for m in per.values()], 3)} | {len(per)}{tag} |")
        L += ["", "| 配對（同折相減） | 順序 | ACC | Masked ACC | Forgetting |", "|---|---|---|---|---|"]
        merged = dict(b8)
        for b, runs in sw.items():
            merged.update({(f"A5b{b}", k[1], k[2]): v for k, v in runs.items()})
        for b in (4, 16):
            if b not in sw:
                continue
            for order in ("reverse", "main"):
                rows = {r["label"]: r for r in paired(merged, (f"A5b{b}", "hier", order), ("A5", "hier", order))}
                cells = [f"{rows[k]['mean']:+.4f}（{rows[k]['better']}/{rows[k]['n']}）" if k in rows else "—"
                         for k in ("ACC", "Masked ACC", "Forgetting")]
                L.append(f"| B={b} − B=8 | {order} | " + " | ".join(cells) + " |")
    else:
        L.append("尚未有資料。")
    L.append("")
    out.write_text("\n".join(L) + "\n")
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
