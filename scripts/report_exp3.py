#!/usr/bin/env python3
"""DR-052 累積式類別頭 vs 固定頭（同平台 pod CPU x86）的報表：outputs/exp3/EXP3.md。

    python scripts/report_exp3.py                              # 全部批次（有資料的節才出）
    python scripts/report_exp3.py --out outputs/exp3/B1.md     # 每批一份快照

資料來源（都在 outputs/exp3/，pod 產生）：
  sota_acc / sota_fixed        十折（reverse + main）× hier / flat，seed = fold
  ablation_acc / ablation_fixed  fold-1 五 seed 六臂（flat）
Mac 參考值：outputs/exp2/sota（十折固定頭）、outputs/exp2/{ablation,main}（fold-1 五 seed 固定頭）——
同設定但不同平台（DR-052 對數：四個 stage 後 class-IL 可差 2.8 pp），**只當參考**。

指標與 `sota/metrics.py`（ACC、Masked ACC、Forgetting、BWT）、`run_exp2.arm_metrics`
（class-IL、task-IL、洩漏率、Jaccard）同一套；累積式頭的紀錄已在各 stage 的 C_t 上取 argmax，
指標函式不需改。**本檔只放數字；「累積 − 固定」只報告差與折數，不宣稱勝負**（DR-052 判準）。
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

from report_dr051_e0 import per_slide_deltas                              # noqa: E402
from run_exp2 import ORDERS, arm_metrics                                  # noqa: E402
from selector.text_encoder import load_config                             # noqa: E402
from sota.metrics import all_metrics                                      # noqa: E402
from sota.report_sota import _provenance, cell, load_runs, paired         # noqa: E402

METRICS = [("acc", "ACC ↑"), ("masked_acc", "Masked ACC ↑"),
           ("forgetting", "Forgetting ↓"), ("bwt", "BWT ↑")]
ABL_ARMS = ["A2", "A3", "A4", "A5", "B1", "B2"]
ABL_PAIRS = [("A5", "A3"), ("A5", "A4"), ("A4", "A3"), ("A5", "B1"), ("A5", "B2")]
ABL_AXES = [("final_class_il", "class-IL", True, 100), ("final_task_il", "task-IL", True, 100),
            ("mean_leak", "洩漏率", False, 100), ("mean_jaccard", "Jaccard", True, 1),
            ("dU_M1", "ΔU(M1)", True, 1), ("dU_M1_all8", "ΔU(M1, all8)", True, 1)]


def msd(v, fmt="{:.3f}"):
    v = [x for x in v if x is not None]
    if not v:
        return "—"
    sd = statistics.stdev(v) if len(v) > 1 else 0.0
    return f"{fmt.format(statistics.mean(v))} ± {fmt.format(sd)}"


def agg(runs, tasks):
    per = {}
    for rk, recs in sorted(runs.items()):
        try:
            per[rk] = all_metrics(recs, tasks)
        except ValueError as e:
            per[rk] = {"_error": str(e)}
    good = {k: v for k, v in per.items() if "_error" not in v}
    out = {"n": len(good), "runs": sorted(good)}
    for k, _ in METRICS:
        vals = [v[k] for v in good.values() if v.get(k) is not None]
        out[k] = ((statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0)
                  if vals else None)
    return out, good


def tenfold_table(title, runs, L):
    L += [f"### {title}", "",
          "| 臂 | 架構 | 順序 | " + " | ".join(l for _k, l in METRICS) + " | n | 溯源 |",
          "|---|---|---|" + "---|" * len(METRICS) + "---|---|"]
    keys = sorted(k for k in runs if k[0] in ("A1", "A3", "A5"))
    if not keys:
        L += ["| 尚未有資料 | | | | | | | | |"]
    for key in keys:
        a, _ = agg(runs[key], ORDERS[key[2]])
        L.append(f"| {key[0]} | `{key[1]}` | {key[2]} | " + " | ".join(cell(a[k]) for k, _ in METRICS)
                 + f" | {a['n']} | `{_provenance(a['runs'])}` |")
    L.append("")


def paired_block(title, runs, ka, kb, L):
    rows = paired(runs, ka, kb)
    L += [f"#### {title}", ""]
    if not rows:
        L += ["尚未有資料。", ""]; return
    L += ["| 指標 | 逐折差值 | 均值差 | A 較佳折數 |", "|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['label']} | " + ", ".join(f"{x:+.3f}" for _f, x in r["per_fold"])
                 + f" | **{r['mean']:+.4f}** | **{r['better']}/{r['n']}** |")
    L.append("")


def diff_block(title, runs_a, runs_b, key, L, note_a="累積式", note_b="固定"):
    """同 (arm, arch, order) 下逐折差 A − B（只報告）。"""
    L += [f"#### {title}", ""]
    if key not in runs_a or key not in runs_b:
        L += ["尚未有資料。", ""]; return
    _, ga = agg(runs_a[key], ORDERS[key[2]])
    _, gb = agg(runs_b[key], ORDERS[key[2]])
    common = sorted(set(ga) & set(gb))
    if not common:
        L += ["無共同 fold。", ""]; return
    L += [f"| 指標 | {note_b} | {note_a} | 逐折差（{note_a} − {note_b}） | 均值差 | {note_a} 較高的折數 |",
          "|---|---|---|---|---|---|"]
    for k, lab in METRICS:
        d = [ga[rk][k] - gb[rk][k] for rk in common
             if ga[rk].get(k) is not None and gb[rk].get(k) is not None]
        if not d:
            continue
        L.append(f"| {lab} | {msd([gb[rk][k] for rk in common])} | {msd([ga[rk][k] for rk in common])} | "
                 + ", ".join(f"{x:+.3f}" for x in d)
                 + f" | {statistics.mean(d):+.4f} | {sum(x > 0 for x in d)}/{len(d)} |")
    L.append("")


def abl_metrics(recs, arm, seed, tasks, label_space):
    M = arm_metrics(recs, arm, tasks, seed, label_space)
    d = per_slide_deltas(recs, arm, seed, tasks)
    M["dU_M1"] = statistics.mean([statistics.mean(v) for v in d.values()]) if d else None
    r8 = [dict(r, utility_total=r.get("utility_total_all8", r["utility_total"])) for r in recs]
    d8 = per_slide_deltas(r8, arm, seed, tasks)
    M["dU_M1_all8"] = statistics.mean([statistics.mean(v) for v in d8.values()]) if d8 else None
    return M


def load_abl(d: Path, suffix: str, tasks, label_space):
    out = defaultdict(dict)
    if not d.is_dir():
        return out
    for arm in ABL_ARMS:
        for s in range(5):
            p = d / f"{arm}_reverse_seed{s}{suffix}.json"
            if p.exists():
                out[arm][s] = abl_metrics(json.loads(p.read_text()), arm, s, tasks, label_space)
    return out


def abl_table(title, A, L):
    L += [f"### {title}", "", "| 臂 | n | " + " | ".join(lab for _k, lab, _h, _m in ABL_AXES) + " |",
          "|---|---|" + "---|" * len(ABL_AXES)]
    for arm in ABL_ARMS:
        if not A[arm]:
            L.append(f"| {arm} | 0 | " + " | ".join("尚未" for _ in ABL_AXES) + " |"); continue
        cells = []
        for k, _lab, _h, m in ABL_AXES:
            vals = [A[arm][s][k] * m for s in sorted(A[arm]) if A[arm][s].get(k) is not None]
            cells.append(msd(vals, "{:.2f}" if m == 100 else "{:.4f}"))
        L.append(f"| {arm} | {len(A[arm])} | " + " | ".join(cells) + " |")
    L.append("")


def abl_pairs(title, A, L):
    L += [f"#### {title}（A − B；逐 seed；勝 = A 較佳）", "",
          "| 配對 | " + " | ".join(lab for _k, lab, _h, _m in ABL_AXES) + " |", "|---|" + "---|" * len(ABL_AXES)]
    for a, b in ABL_PAIRS:
        common = sorted(set(A[a]) & set(A[b]))
        if not common:
            L.append(f"| {a} − {b} | " + " | ".join("尚未" for _ in ABL_AXES) + " |"); continue
        cells = []
        for k, _lab, hib, m in ABL_AXES:
            d = [(A[a][s][k] - A[b][s][k]) * m for s in common
                 if A[a][s].get(k) is not None and A[b][s].get(k) is not None]
            if not d:
                cells.append("—"); continue
            wins = sum((x > 0) if hib else (x < 0) for x in d)
            sd = statistics.stdev(d) if len(d) > 1 else 0.0
            cells.append(f"{statistics.mean(d):+.2f} ± {sd:.2f}（{wins}/{len(d)}）" if m == 100
                         else f"{statistics.mean(d):+.4f}（{wins}/{len(d)}）")
        L.append(f"| {a} − {b} | " + " | ".join(cells) + " |")
    L.append("")


def abl_diff(title, A, B, L, note_a="累積式", note_b="固定"):
    L += [f"#### {title}（{note_a} − {note_b}；逐 seed；只報告）", "",
          "| 臂 | class-IL（pp） | task-IL（pp） | 洩漏率（pp） | Jaccard | ΔU(M1, all8) |",
          "|---|---|---|---|---|---|"]
    for arm in ABL_ARMS:
        common = sorted(set(A[arm]) & set(B[arm]))
        if not common:
            L.append(f"| {arm} | 尚未 | | | | |"); continue
        cells = []
        for k, m in (("final_class_il", 100), ("final_task_il", 100), ("mean_leak", 100),
                     ("mean_jaccard", 1), ("dU_M1_all8", 1)):
            d = [(A[arm][s][k] - B[arm][s][k]) * m for s in common]
            sd = statistics.stdev(d) if len(d) > 1 else 0.0
            cells.append(f"{statistics.mean(d):+.2f} ± {sd:.2f}（{note_a}較高 {sum(x > 0 for x in d)}/{len(d)}）"
                         if m == 100 else f"{statistics.mean(d):+.4f}（{sum(x > 0 for x in d)}/{len(d)}）")
        L.append(f"| {arm} | " + " | ".join(cells) + " |")
    L.append("")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", default=str(ROOT / "outputs" / "exp3"))
    ap.add_argument("--mac", default=str(ROOT / "outputs" / "exp2"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    src, mac = Path(a.src), Path(a.mac)
    out = Path(a.out) if a.out else src / "EXP3.md"
    cfg = load_config(); label_space = list(cfg["tasks"]); tasks = ORDERS["reverse"]

    def runs(d):
        return load_runs(d, list(ORDERS)) if d.is_dir() else {}
    acc = runs(src / "sota_acc" / "per_slide")
    fix = runs(src / "sota_fixed" / "per_slide")
    ref = runs(mac / "sota" / "per_slide")

    L = ["# EXP3 — 累積式類別頭（DR-052）", "",
         "**主結果平台：pod CPU x86**（RunPod，384 核，`OMP/MKL=8`、`torch.set_num_threads(8)`，10 路平行）。"
         "累積式頭與固定頭對照**都在 pod 跑**，同折同 seed 配對；Mac 的 outputs/exp2 固定頭數字**只當參考**"
         "（同設定不同平台，DR-052 對數：四個 stage 後 class-IL 可差 2.8 pp）。", "",
         "紀錄由 `scripts/run_exp2.py --head {accumulating,fixed} --out-root outputs/exp3` 產生；"
         "累積式頭在各 stage 的 C_t 上取 argmax（未見類 logit −inf）。**本檔只放數字；累積 − 固定只報告差與折數，不宣稱勝負。**", "",
         "## 1. 十折主表（seed = fold）", ""]
    tenfold_table("1a. 累積式頭（sota_acc）", acc, L)
    tenfold_table("1b. 固定頭・pod 同設定對照（sota_fixed）", fix, L)
    tenfold_table("1c. 固定頭・Mac 參考（outputs/exp2/sota）", ref, L)

    L += ["## 2. 逐折配對", ""]
    for order in ("reverse", "main"):
        L += [f"### {order}", ""]
        paired_block(f"hier − flat（累積式頭，{order}）", acc, ("A5", "hier", order), ("A5", "flat", order), L)
        paired_block(f"hier − flat（固定頭 pod，{order}）", fix, ("A5", "hier", order), ("A5", "flat", order), L)
        for arch in ("hier", "flat"):
            diff_block(f"累積 − 固定（同平台 pod；A5 `{arch}` {order}）", acc, fix, ("A5", arch, order), L)
        for arch in ("hier", "flat"):
            diff_block(f"固定頭 pod − 固定頭 Mac（平台差參考；A5 `{arch}` {order}）", fix, ref,
                       ("A5", arch, order), L, note_a="pod", note_b="Mac")
        # B7（Prompt 22）：Table 2 鏈的前兩列（A1 無保存、A3 只 replay）在累積式頭下，flat 十折
        paired_block(f"A3 − A1（累積式頭，flat，{order}）", acc, ("A3", "flat", order), ("A1", "flat", order), L)
        paired_block(f"A5 − A3（累積式頭，flat，{order}）", acc, ("A5", "flat", order), ("A3", "flat", order), L)
        for arm in ("A1", "A3"):
            diff_block(f"累積式頭 pod − 固定頭 Mac（參考；{arm} `flat` {order}）", acc, ref,
                       (arm, "flat", order), L, note_a="累積(pod)", note_b="固定(Mac)")

    L += ["## 3. fold-1 五 seed 組件消融（flat，reverse）", ""]
    A = load_abl(src / "ablation_acc" / "per_slide", "_acc", tasks, label_space)
    B = load_abl(src / "ablation_fixed" / "per_slide", "", tasks, label_space)
    R = defaultdict(dict)
    for d in (mac / "ablation" / "per_slide", mac / "main" / "per_slide"):
        for arm, v in load_abl(d, "", tasks, label_space).items():
            for s, m in v.items():
                R[arm].setdefault(s, m)
    abl_table("3a. 累積式頭（ablation_acc）", A, L)
    abl_pairs("配對・累積式頭", A, L)
    abl_table("3b. 固定頭・pod 同設定對照（ablation_fixed）", B, L)
    abl_pairs("配對・固定頭 pod", B, L)
    abl_diff("3c. 累積 − 固定（同平台 pod）", A, B, L)
    abl_diff("3d. 固定頭 pod − 固定頭 Mac（平台差參考）", B, R, L, note_a="pod", note_b="Mac")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
