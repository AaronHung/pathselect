#!/usr/bin/env python3
"""exp5 的判準檢查（PI 2026-09-23 指定，先於看到數字寫好）。

判準一：Table 1 兩序的 ACC 是否都不低於 0.812（class-text top-8 零樣本參照）。
        那個 0.812 對記憶體大小免疫（零參數、與 stage 無關），所以可以跨批次比。
判準二：完整方法 − 只有 replay 的逐折配對（ACC 與 Forgetting 皆要），
        兩序是否同方向、是否至少一序達 7/10 折。
        這是本批內部的同機同批配對，可以相減。
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from report_exp5 import ORDER_LABEL, load_tag, paired          # noqa: E402

ZS = 0.812          # class-text top-8，對 |M| 免疫（outputs/exp2/sota，dossier §10.4）
WIN = 7             # 判準二的折數門檻


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--m", type=int, required=True, help="記憶體大小，例如 128")
    ap.add_argument("--src", default=str(ROOT / "outputs" / "exp5"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    src = Path(a.src)
    M = a.m

    F = {o: load_tag(src / f"m{M}_full_{s}" / "per_slide", o)
         for o, s in (("reverse", "rev"), ("main", "fwd"))}
    R = {o: load_tag(src / f"m{M}_replay_{s}" / "per_slide", o)
         for o, s in (("reverse", "rev"), ("main", "fwd"))}

    L = [f"# exp5 判準結果 —— |M| = {M}", "",
         "判準在看到數字之前就寫成腳本（`scripts/exp5_gate.py`），本檔由它產生。", ""]

    # ── 判準一 ──
    L += [f"## 判準一：Table 1 兩序的 ACC 是否都不低於 {ZS}（零樣本 class-text top-8）", "",
          "| 順序 | n | 本方法 ACC | 參照 | 差 | 結果 |", "|---|---|---|---|---|---|"]
    g1 = []
    for o in ("reverse", "main"):
        per = F[o]
        if len(per) < 10:
            L.append(f"| {ORDER_LABEL[o]} | {len(per)}/10 | 資料不完整 | {ZS} | — | ⚠️ 無法判定 |")
            g1.append(None); continue
        acc = [m["acc"] for m in per.values()]
        mean, sd = statistics.mean(acc), statistics.stdev(acc)
        ok = mean >= ZS
        g1.append(ok)
        L.append(f"| {ORDER_LABEL[o]} | {len(per)} | {mean:.4f} ± {sd:.4f} | {ZS} | "
                 f"{mean - ZS:+.4f} | {'✅ 不低於' if ok else '❌ 低於'} |")
    v1 = ("✅ **通過** —— 兩序都不低於零樣本參照。" if all(x is True for x in g1)
          else "❌ **未通過** —— 至少一序低於零樣本參照。" if all(x is not None for x in g1)
          else "⚠️ 資料不完整，無法判定。")
    L += ["", v1, ""]

    # ── 判準二 ──
    L += ["## 判準二：完整方法 − 只有 replay 的逐折配對", "",
          "| 順序 | ΔACC | 較佳折數 | ΔForgetting ↓ | 較佳折數 | n |", "|---|---|---|---|---|---|"]
    da, dfo, wa, wf = {}, {}, {}, {}
    for o in ("reverse", "main"):
        ra = paired(F[o], R[o], "acc")
        rf = paired(F[o], R[o], "forgetting")
        if not ra or not rf:
            L.append(f"| {ORDER_LABEL[o]} | 資料不完整 | | | | |"); continue
        da[o], wa[o] = ra["mean"], ra["better"]
        dfo[o], wf[o] = rf["mean"], rf["better"]
        L.append(f"| {ORDER_LABEL[o]} | {ra['mean']:+.4f} | {ra['better']}/{ra['n']} | "
                 f"{rf['mean']:+.4f} | {rf['better']}/{rf['n']} | {ra['n']} |")

    L += ["", "「較佳折數」對 ACC 是差值為正的折數，對 Forgetting 是差值為負的折數。", ""]
    if len(da) == 2:
        same_acc = (da["reverse"] > 0) == (da["main"] > 0)
        same_fo = (dfo["reverse"] < 0) == (dfo["main"] < 0)
        hit_acc = max(wa.values()) >= WIN
        hit_fo = max(wf.values()) >= WIN
        L += ["| 子判準 | 結果 |", "|---|---|",
              f"| ACC 兩序同方向 | {'✅ 是' if same_acc else '❌ 否'}"
              f"（reverse {da['reverse']:+.4f}、forward {da['main']:+.4f}） |",
              f"| ACC 至少一序達 {WIN}/10 | {'✅ 是' if hit_acc else '❌ 否'}"
              f"（最高 {max(wa.values())}/10） |",
              f"| Forgetting 兩序同方向 | {'✅ 是' if same_fo else '❌ 否'}"
              f"（reverse {dfo['reverse']:+.4f}、forward {dfo['main']:+.4f}） |",
              f"| Forgetting 至少一序達 {WIN}/10 | {'✅ 是' if hit_fo else '❌ 否'}"
              f"（最高 {max(wf.values())}/10） |", "",
              "**整體：" + ("✅ 通過" if (same_acc and hit_acc) else "❌ 未通過")
              + "**（以 ACC 為準：需兩序同方向且至少一序達 7/10）。", ""]
        if same_fo and hit_fo and not (same_acc and hit_acc):
            L += ["註：ACC 未達標，但 **Forgetting 兩序同方向且至少一序達 7/10** —— "
                  "保存機制對遺忘有一致效果，對最終準確率沒有。", ""]
    else:
        L += ["⚠️ 資料不完整，無法判定。", ""]

    txt = "\n".join(L) + "\n"
    if a.out:
        Path(a.out).write_text(txt, encoding="utf-8")
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
