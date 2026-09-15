#!/usr/bin/env python3
"""論文數字對照表（Prompt 23-A）：outputs/exp3/PAPER_NUMBERS.md。

主結果口徑：**累積式類別頭、單層選擇器（flat）、平台 pod CPU x86、十折 seed = fold**
（DR-052）。固定頭的既有數字（E2／E3／seen-class、零樣本 top-8）標為
「fixed-vocabulary setting」。所有數字由產物重算，不手抄：

  outputs/exp3/sota_acc/per_slide       累積式十折（A1、A3、A5 flat；A5 hier）
  outputs/exp3/ablation_acc/per_slide   fold-1 五 seed 六臂（累積式，flat）
  outputs/exp2/sota/per_slide           ZS-top8（固定頭；零樣本與 stage 無關）
  outputs/exp2/sota/<REPRO_SUBDIR>/*    基準論文方法的重現（sota/repro_metrics.py）
  sota/external_baselines.py            基準論文發表值（Tab. 2 反向、Tab. 1 正向）
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

from report_exp3 import abl_metrics, load_abl, ABL_ARMS                   # noqa: E402
from run_exp2 import ORDERS                                               # noqa: E402
from selector.text_encoder import load_config                             # noqa: E402
from sota.external_baselines import (MAIN_METHOD, REPRO_RUNS, REPRO_SUBDIR,  # noqa: E402
                                     ROWS, ROWS_FORWARD)
from sota.metrics import all_metrics                                      # noqa: E402
from sota.repro_metrics import find_folds, fold_metrics                   # noqa: E402
from sota.report_sota import load_runs, paired                            # noqa: E402

FIXED_NOTE = "fixed-vocabulary setting"
ORDER_LABEL = {"reverse": "reverse（ESCA→RCC→BRCA→LUNG，Tab. 2）", "main": "forward（LUNG→BRCA→RCC→ESCA，Tab. 1）"}


def per_fold(runs, key):
    if key not in runs:
        return {}
    out = {}
    for rk, recs in runs[key].items():
        try:
            out[rk] = all_metrics(recs, ORDERS[key[2]])
        except ValueError:
            pass
    return out


def ms(vals, nd=3):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "—"
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f"{statistics.mean(vals):.{nd}f} ± {sd:.{nd}f}"


def mean_only(vals, nd=3):
    vals = [v for v in vals if v is not None]
    return f"{statistics.mean(vals):.{nd}f}" if vals else "—"


def row3(m):
    return (ms([x["acc"] for x in m.values()]), mean_only([x["forgetting"] for x in m.values()]),
            mean_only([x["masked_acc"] for x in m.values()]))


def repro_numbers():
    """基準論文方法的重現：反向 mini-batch 8 與正向（原生），各十折。"""
    out = {}
    for lab, sub, _note, src in REPRO_RUNS:
        root = ROOT / "outputs" / "exp2" / "sota" / REPRO_SUBDIR / sub
        folds = find_folds(root) if root.is_dir() else {}
        per = {}
        for k, md in folds.items():
            try:
                per[k] = fold_metrics(md)
            except (ValueError, SyntaxError):
                pass
        if per:
            out[sub] = (lab, src, per)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(ROOT / "outputs" / "exp3" / "PAPER_NUMBERS.md"))
    a = ap.parse_args(argv)
    cfg = load_config(); label_space = list(cfg["tasks"]); tasks_rev = ORDERS["reverse"]
    acc = load_runs(ROOT / "outputs" / "exp3" / "sota_acc" / "per_slide", list(ORDERS))
    ref = load_runs(ROOT / "outputs" / "exp2" / "sota" / "per_slide", list(ORDERS))
    repro = repro_numbers()

    L = ["# 論文數字對照表（DR-052 之後的口徑）", "",
         "**主結果口徑**：累積式類別頭（任務 t 只在已見類別 C_t 上訓練與評估）、單層選擇器（flat）、"
         "比較協定十折（seed = fold）、平台 pod CPU x86（同平台配對見 outputs/exp3/EXP3.md）。"
         "標「" + FIXED_NOTE + "」者為固定 8 類頭的既有數字（E2／E3／seen-class、零樣本 top-8），不重跑。"
         "本檔只放數字與來源；判讀在 DR-052／DR-053 與 dossier §11。", ""]

    # ── 1. Table 1 ─────────────────────────────────────────────────────────
    L += ["## 1. Table 1 — 與發表結果並列（兩順序）", ""]
    for order, rows_pub, tab in (("reverse", ROWS, "Tab. 2"), ("main", ROWS_FORWARD, "Tab. 1")):
        L += [f"### {ORDER_LABEL[order]}", "",
              "| 方法 | ACC | Forgetting | Masked ACC | 來源 |", "|---|---|---|---|---|"]
        for name, acc_s, forg, _bwt, masked in rows_pub:
            L.append(f"| {name} | {acc_s or '—'} | {forg or '—'} | {masked or '—'} | {MAIN_METHOD} {tab}（發表值） |")
        for sub, (lab, src, per) in repro.items():
            if (src == "tab2") != (order == "reverse"):
                continue
            L.append(f"| {MAIN_METHOD}（reproduced, released code；{lab}） | "
                     f"{ms([m['acc'] for m in per.values()])} | {mean_only([m['forgetting'] for m in per.values()])} | "
                     f"{mean_only([m['masked_acc'] for m in per.values()])} | `outputs/exp2/sota/{REPRO_SUBDIR}/{sub}`（{len(per)} 折） |")
        m = per_fold(acc, ("A5", "flat", order))
        a_, f_, k_ = row3(m)
        L.append(f"| **PathSelect（ours；accumulating head，flat）** | **{a_}** | **{f_}** | **{k_}** | `outputs/exp3/sota_acc`（{len(m)} 折） |")
        L.append("")

    # ── 2. Table 2 鏈 ──────────────────────────────────────────────────────
    L += ["## 2. Table 2 — 鏈（兩順序；累積式頭，flat 除「＋兩層選擇器」）", ""]
    chain = [("無保存（sequential FT）", ("A1", "flat")), ("＋replay", ("A3", "flat")),
             ("＋蒸餾＋效用（PathSelect）", ("A5", "flat")), ("＋兩層選擇器（hier）", ("A5", "hier"))]
    for order in ("reverse", "main"):
        L += [f"### {ORDER_LABEL[order]}", "", "| 列 | ACC | Forgetting | Masked ACC | 來源 |", "|---|---|---|---|---|"]
        for lab, (arm, arch) in chain:
            m = per_fold(acc, (arm, arch, order))
            a_, f_, k_ = row3(m)
            L.append(f"| {lab} | {a_} | {f_} | {k_} | `outputs/exp3/sota_acc` {arm} {arch}（{len(m)} 折） |")
        mz = per_fold(ref, ("ZS-top8", "flat", order))
        a_, f_, k_ = row3(mz)
        L.append(f"| 零樣本 top-8（{FIXED_NOTE}；與 stage 無關） | {a_} | {f_} | {k_} | `outputs/exp2/sota` ZS-top8（{len(mz)} 折） |")
        pub = [r for r in (ROWS if order == "reverse" else ROWS_FORWARD) if r[0] == MAIN_METHOD][0]
        L.append(f"| {MAIN_METHOD}（發表） | {pub[1]} | {pub[2]} | {pub[4]} | {MAIN_METHOD} {'Tab. 2' if order == 'reverse' else 'Tab. 1'} |")
        for sub, (lab, src, per) in repro.items():
            if (src == "tab2") != (order == "reverse"):
                continue
            L.append(f"| {MAIN_METHOD}（reproduced；{lab}） | {ms([m['acc'] for m in per.values()])} | "
                     f"{mean_only([m['forgetting'] for m in per.values()])} | {mean_only([m['masked_acc'] for m in per.values()])} | `{sub}` |")
        L += ["", f"逐折配對（{order}；A − B；較佳折數依 ACC／Masked ↑、Forgetting ↓）", "",
              "| 配對 | ACC | Masked ACC | Forgetting |", "|---|---|---|---|"]
        for lab, ka, kb in (("A3 − A1（replay − 無保存）", ("A3", "flat", order), ("A1", "flat", order)),
                            ("A5 − A3（蒸餾＋效用 − replay）", ("A5", "flat", order), ("A3", "flat", order)),
                            ("hier − flat（A5）", ("A5", "hier", order), ("A5", "flat", order))):
            rows = {r["label"]: r for r in paired(acc, ka, kb)}
            cells = [f"{rows[k]['mean']:+.4f}（{rows[k]['better']}/{rows[k]['n']}）" if k in rows else "—"
                     for k in ("ACC", "Masked ACC", "Forgetting")]
            L.append(f"| {lab} | " + " | ".join(cells) + " |")
        L.append("")

    # ── 3. Table 3 ─────────────────────────────────────────────────────────
    A = load_abl(ROOT / "outputs" / "exp3" / "ablation_acc" / "per_slide", "_acc", tasks_rev, label_space)
    names = {"A2": "LoRA merge only", "A3": "+ replay", "A4": "+ replay + KD", "A5": "full (PathSelect)",
             "B1": "KD only", "B2": "utility hinge only"}
    L += ["## 3. Table 3 — fold 1、五 seed、累積式頭、flat、reverse（`outputs/exp3/ablation_acc`）", "",
          "| 臂 | class-IL | task-IL | 洩漏率 | Jaccard | ΔU(M1, C_t) | ΔU(M1, all8) |", "|---|---|---|---|---|---|---|"]
    for arm in ABL_ARMS:
        if not A[arm]:
            continue
        S = sorted(A[arm])
        L.append(f"| {arm} {names[arm]} | {ms([A[arm][s]['final_class_il'] * 100 for s in S], 2)} | "
                 f"{ms([A[arm][s]['final_task_il'] * 100 for s in S], 2)} | {ms([A[arm][s]['mean_leak'] * 100 for s in S], 2)} | "
                 f"{ms([A[arm][s]['mean_jaccard'] for s in S], 4)} | {ms([A[arm][s]['dU_M1'] for s in S], 3)} | "
                 f"{ms([A[arm][s]['dU_M1_all8'] for s in S], 3)} |")
    L += ["", "| 配對（A − B；勝 = A 較佳） | class-IL | task-IL | 洩漏率 | Jaccard | ΔU(all8) |", "|---|---|---|---|---|---|"]
    for a_, b_ in (("A5", "A3"), ("A5", "A4"), ("A5", "B1"), ("A5", "B2")):
        common = sorted(set(A[a_]) & set(A[b_]))
        if not common:
            continue
        cells = []
        for k, hib, mlt, nd in (("final_class_il", True, 100, 2), ("final_task_il", True, 100, 2),
                                ("mean_leak", False, 100, 2), ("mean_jaccard", True, 1, 4), ("dU_M1_all8", True, 1, 3)):
            d = [(A[a_][s][k] - A[b_][s][k]) * mlt for s in common]
            cells.append(f"{statistics.mean(d):+.{nd}f}（{sum((x > 0) if hib else (x < 0) for x in d)}/{len(d)}）")
        L.append(f"| {a_} − {b_} | " + " | ".join(cells) + " |")
    L.append("")

    # ── 4. 稿內數字句 ─────────────────────────────────────────────────────
    def A_(arm, arch, order, key):
        m = per_fold(acc, (arm, arch, order)); return statistics.mean(x[key] for x in m.values())
    def R_(order, key):
        m = per_fold(ref, ("ZS-top8", "flat", order)); return statistics.mean(x[key] for x in m.values())
    pub_rev = [r for r in ROWS if r[0] == MAIN_METHOD][0]; pub_fwd = [r for r in ROWS_FORWARD if r[0] == MAIN_METHOD][0]
    qr, qf = float(pub_rev[1].split("±")[0]), float(pub_fwd[1].split("±")[0])
    pr = {o: paired(acc, ("A5", "flat", o), ("A3", "flat", o)) for o in ORDERS}
    hf = {o: paired(acc, ("A5", "hier", o), ("A5", "flat", o)) for o in ORDERS}
    def pick(rows, lab): return [r for r in rows if r["label"] == lab][0]
    sents = [
        (f"Unprotected sequential selection keeps {A_('A1','flat','reverse','acc'):.2f} ACC in the reverse order; replay recovers it to {A_('A3','flat','reverse','acc'):.2f}",
         "sota_acc A1/A3 flat reverse（B7）"),
        (f"PathSelect reaches {A_('A5','flat','reverse','acc'):.3f} ACC with {A_('A5','flat','reverse','forgetting'):.3f} forgetting (reverse) and {A_('A5','flat','main','acc'):.3f} / {A_('A5','flat','main','forgetting'):.3f} (forward)",
         "sota_acc A5 flat（B1／B3）"),
        (f"It trails the strongest published result by {qr - A_('A5','flat','reverse','acc'):.3f} (reverse, {qr:.3f}) and {qf - A_('A5','flat','main','acc'):.3f} (forward, {qf:.3f})",
         f"{MAIN_METHOD} Tab. 2／Tab. 1 發表值 − sota_acc A5 flat"),
        (f"Masked ACC {A_('A5','flat','reverse','masked_acc'):.3f} (reverse) / {A_('A5','flat','main','masked_acc'):.3f} (forward)",
         "sota_acc A5 flat"),
        (f"Distillation plus utility preservation over replay alone: {pick(pr['reverse'],'ACC')['mean']:+.3f} ACC ({pick(pr['reverse'],'ACC')['better']}/10 folds, reverse), {pick(pr['main'],'ACC')['mean']:+.3f} ({pick(pr['main'],'ACC')['better']}/10, forward)",
         "配對 A5 − A3（B7／B1／B3）"),
        (f"The hierarchical selector changes ACC by {pick(hf['reverse'],'ACC')['mean']:+.3f} ({pick(hf['reverse'],'ACC')['better']}/10, reverse) and {pick(hf['main'],'ACC')['mean']:+.3f} ({pick(hf['main'],'ACC')['better']}/10, forward)",
         "配對 hier − flat（B1／B3）"),
        (f"The zero-shot top-8 reference reaches {R_('reverse','acc'):.3f} ACC ({FIXED_NOTE}) vs PathSelect {A_('A5','flat','reverse','acc'):.3f} (reverse) / {A_('A5','flat','main','acc'):.3f} (forward)",
         "outputs/exp2/sota ZS-top8（固定頭）、sota_acc A5 flat"),
    ]
    if A["A5"] and A["A3"]:
        S = sorted(set(A["A5"]) & set(A["A3"]))
        d = [(A["A5"][s]["final_class_il"] - A["A3"][s]["final_class_il"]) * 100 for s in S]
        sents.append((f"On fold 1 (five seeds) full preservation vs replay-only: {statistics.mean(d):+.2f} pp class-IL ({sum(x > 0 for x in d)}/5)",
                      "ablation_acc（B5）"))
    L += ["## 4. 稿內數字句（新口徑；每句附來源）", "", "| 句 | 來源 |", "|---|---|"]
    for s_, src in sents:
        L.append(f"| {s_} | {src} |")
    L += ["", "## 5. 固定 8 類頭的既有結果（" + FIXED_NOTE + "；不重跑）", "",
          "* E2 門控對照（A5ce vs A5，fold 1 五 seed）：`outputs/exp2/dr051/E2_GATED_CONTROL.md` —— " + FIXED_NOTE,
          "* E3 選片 vs 加權拆解：`outputs/exp2/dr051/E3_DECOMPOSITION.md` —— " + FIXED_NOTE,
          "* seen-class 檢查：`outputs/exp2/sota/SEEN_CLASS_CHECK.md` —— " + FIXED_NOTE + "（事後限制到已見類別的評估；與累積式頭的訓練口徑不同）",
          "* 零樣本 top-8：`outputs/exp2/sota/per_slide/ZS-top8_*`（無參數、與 stage 無關）—— " + FIXED_NOTE,
          "* E0 hinge 觸發率／ΔU 口徑：`outputs/exp2/dr051/E0*.md` —— " + FIXED_NOTE + "；累積式頭下的觸發率見 `outputs/exp3/DR053_HINGE.md`", ""]
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
