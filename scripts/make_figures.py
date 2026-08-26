#!/usr/bin/env python3
"""論文圖表產生器 → `figures/`（PROMPT DOSSIER-FIGURES-20260826 §C）。

**原則**：

1. 所有數字從 `per_slide/*.json` 重算，**沿用 `run_exp2.arm_metrics`**（與 report
   共用同一個函式）—— 不另寫一套 metric。若圖與報告算出不同數字，那是最糟的結果。
2. 每張圖同時輸出 `.pdf`（向量）與 `.png`（300 dpi）。
3. 版本結果收進 `figures/figure_data.json`，含每張圖畫上去的每個數值
   （mean / std / win count / n），讓溯源檢查能驗它。
4. `--smoke` 用 fixture 跑（憲法 §3.6），fixture 涵蓋「兩個 order、seed 數不齊」（§3.6b）。

**版面**：colorblind-safe（Okabe-Ito）；臂的顏色全篇固定；std 用 error bar 或
淡色帶；win count 直接標在圖上；**不標 p 值、不標星號**；字體 ≥ 7 pt。
圖上文字不得出現被禁的敘事字（由 `tests/test_make_figures.py` 守）。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib                                                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402

from run_exp2 import ORDERS, arm_metrics                              # noqa: E402
from selector.text_encoder import load_config                         # noqa: E402

OUT_DIR = ROOT / "figures"
DATA = OUT_DIR / "figure_data.json"
EXP = ROOT / "outputs" / "exp2"
ORDER = "reverse"
TASKS = ORDERS[ORDER]
SHORT = [t.replace("tcga_", "") for t in TASKS]

#: Okabe-Ito，colorblind-safe。臂的顏色全篇固定，不隨圖變。
C = {
    "A1": "#4D4D4D",          # 深灰
    "A2": "#B0B0B0",          # 淺灰
    "A3": "#0072B2",          # 藍
    "A4": "#E69F00",          # 淺橘
    "A5": "#D55E00",          # 橘紅
    "A5nG": "#D55E00",        # 橘紅斜線填色（hatch）
    "A5g": "#CC79A7",
    "random": "#000000",      # 黑虛線
    "grid": "#009E73",
    "similarity": "#56B4E9",
    "learned-flat": "#D55E00",
}
HATCH = {"A5nG": "///"}

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "figure.dpi": 110, "savefig.bbox": "tight", "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
})


# ── 資料層 ──────────────────────────────────────────────────────────────────

def load(tag: str, arm: str, *, arch=None, order=ORDER, alloc=None, cap=None,
         root=None) -> list[dict]:
    d = (root or (EXP / tag)) / "per_slide"
    if not d.is_dir():
        return []
    out = []
    for f in sorted(d.glob("*.json")):
        for r in json.loads(f.read_text()):
            if r["arm"] != arm or r.get("order") != order:
                continue
            if arch is not None and r.get("arch") != arch:
                continue
            if alloc is not None and r.get("allocation") != alloc:
                continue
            if cap is not None and r.get("mem_capacity") != cap:
                continue
            out.append(r)
    return out


def metrics(recs, arm, label_space):
    seeds = sorted({r["seed"] for r in recs})
    return seeds, {s: arm_metrics(recs, arm, TASKS, s, label_space) for s in seeds}


def series(recs, arm, key, label_space):
    """回傳 (per-seed 值 list, mean, std, n)。"""
    seeds, M = metrics(recs, arm, label_space)
    v = [M[s][key] for s in seeds if M[s].get("per_task") and M[s].get(key) is not None]
    if not v:
        return [], float("nan"), float("nan"), 0
    return v, statistics.mean(v), (statistics.stdev(v) if len(v) > 1 else 0.0), len(v)


def paired(recs_a, arm_a, recs_b, arm_b, key, label_space, higher=True):
    sa, Ma = metrics(recs_a, arm_a, label_space)
    sb, Mb = metrics(recs_b, arm_b, label_space)
    common = sorted(set(sa) & set(sb))
    d = [Ma[s][key] - Mb[s][key] for s in common
         if Ma[s].get(key) is not None and Mb[s].get(key) is not None]
    if not d:
        return None
    w = sum((x > 0) if higher else (x < 0) for x in d)
    return {"diffs": d, "mean": statistics.mean(d),
            "std": statistics.stdev(d) if len(d) > 1 else 0.0,
            "wins": w, "n": len(d)}


def save(fig, name: str) -> list[str]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        p = OUT_DIR / f"{name}.{ext}"
        fig.savefig(p, **kw)
        try:
            paths.append(str(p.relative_to(ROOT)))
        except ValueError:          # smoke 模式輸出在暫存目錄，不在 repo 底下
            paths.append(str(p))
    plt.close(fig)
    return paths


def wl(ax, x, y, wins, n, **kw):
    """把 win count 標在圖上（不標 p 值、不標星號）。"""
    ax.annotate(f"{wins}/{n}", (x, y), fontsize=7, ha="center", **kw)


# ── Fig. 2 budget 曲線 ──────────────────────────────────────────────────────

def fig2_budget_curve(D):
    src = ROOT / "outputs" / "exp0" / "baselines_reverse_f1.json"
    if not src.exists():
        return None
    recs = json.loads(src.read_text())
    Ks = sorted({r["K"] for r in recs})
    pols = ["random", "grid", "similarity", "learned-flat"]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    data = {}
    for pol in pols:
        ms, sds = [], []
        for K in Ks:
            per_task = []
            for t in sorted({r["task_name"] for r in recs}):
                v = [r["acc"] for r in recs
                     if r["policy"] == pol and r["K"] == K and r["task_name"] == t]
                if v:
                    per_task.append(statistics.mean(v))
            ms.append(statistics.mean(per_task) if per_task else float("nan"))
            spread = [statistics.mean([r["acc"] for r in recs if r["policy"] == pol
                                       and r["K"] == K and r["task_name"] == t])
                      for t in sorted({r["task_name"] for r in recs})]
            sds.append(statistics.stdev(spread) if len(spread) > 1 else 0.0)
        style = dict(color=C[pol], marker="o", ms=3, lw=1.4)
        if pol == "random":
            style.update(ls="--")
        ax.plot(Ks, ms, label=pol, **style)
        ax.fill_between(Ks, [m - s for m, s in zip(ms, sds)],
                        [m + s for m, s in zip(ms, sds)], color=C[pol], alpha=0.10, lw=0)
        data[pol] = {"K": Ks, "mean": ms, "std": sds}
    peak_i = max(range(len(Ks)), key=lambda i: data["learned-flat"]["mean"][i])
    ax.annotate(f"peak {data['learned-flat']['mean'][peak_i]:.4f}\n@ K={Ks[peak_i]}",
                (Ks[peak_i], data["learned-flat"]["mean"][peak_i]),
                textcoords="offset points", xytext=(4, -18), fontsize=7)
    ax.set_xscale("log", base=2); ax.set_xticks(Ks)
    ax.set_xticklabels([str(k) for k in Ks])
    ax.set_xlabel("patch budget K"); ax.set_ylabel("accuracy (mean over 4 tasks)")
    ax.legend(frameon=False, loc="lower right")
    D["fig2_budget_curve"] = {"paths": save(fig, "fig2_budget_curve"), "series": data,
                              "peak_K": Ks[peak_i],
                              "peak_value": data["learned-flat"]["mean"][peak_i]}
    return True


# ── Fig. 3 遺忘三軸 ─────────────────────────────────────────────────────────

def fig3_forgetting_axes(D, ls):
    recs = load("main", "A1")
    if not recs:
        return None
    seeds, M = metrics(recs, "A1", ls)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4))
    out = {"seeds": seeds, "per_task": {}}
    for t in TASKS:
        vals = [M[s]["per_task"][t] for s in seeds if t in M[s].get("per_task", {})]
        if vals:
            out["per_task"][t] = {
                k: {"mean": statistics.mean([v[k] for v in vals]),
                    "std": statistics.stdev([v[k] for v in vals]) if len(vals) > 1 else 0.0}
                for k in ("task_il_at_learn", "task_il_at_end", "class_il_at_learn",
                          "class_il_at_end", "jaccard", "jaccard_ref",
                          "sum_u_at_learn", "sum_u_at_end")}
    P = out["per_task"]
    xs = range(len(TASKS))

    ax = axes[0]
    for key, lab, col in (("task_il", "task-IL", C["A3"]), ("class_il", "class-IL", C["A5"])):
        for suf, ls_, mk in (("at_learn", "-", "o"), ("at_end", "--", "s")):
            m = [P[t][f"{key}_{suf}"]["mean"] if t in P else float("nan") for t in TASKS]
            e = [P[t][f"{key}_{suf}"]["std"] if t in P else 0.0 for t in TASKS]
            ax.errorbar(xs, m, yerr=e, color=col, ls=ls_, marker=mk, ms=3, lw=1.2,
                        capsize=2, label=f"{lab} {'at learn' if suf=='at_learn' else '@ end'}")
    ax.set_xticks(list(xs)); ax.set_xticklabels(SHORT)
    ax.set_ylabel("accuracy"); ax.set_title("(a) accuracy", loc="left")
    ax.legend(frameon=False, fontsize=6)

    ax = axes[1]
    early = TASKS[:-1]
    xe = range(len(early))
    ax.bar([x - 0.18 for x in xe], [P[t]["jaccard"]["mean"] for t in early], 0.36,
           color=C["A1"], label="observed")
    ax.plot(list(xe), [P[t]["jaccard_ref"]["mean"] for t in early], color=C["random"],
            ls="none", marker="_", ms=12, mew=1.4, label="random overlap")
    ax.set_xticks(list(xe)); ax.set_xticklabels([t.replace("tcga_", "") for t in early])
    ax.set_ylabel("selection Jaccard"); ax.set_title("(b) selection overlap", loc="left")
    ax.legend(frameon=False, fontsize=6)

    ax = axes[2]
    for t, x in zip(early, xe):
        a, b = P[t]["sum_u_at_learn"]["mean"], P[t]["sum_u_at_end"]["mean"]
        ax.plot([x, x], [a, b], color=C["A1"], lw=1.2)
        ax.plot(x, a, "o", color=C["A3"], ms=4)
        ax.plot(x, b, "s", color=C["A5"], ms=4)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(list(xe)); ax.set_xticklabels([t.replace("tcga_", "") for t in early])
    ax.set_ylabel("Σ utility"); ax.set_title("(c) evidence utility", loc="left")

    fig.tight_layout()
    out["paths"] = save(fig, "fig3_forgetting_axes")
    D["fig3_forgetting_axes"] = out
    return True


# ── Fig. 4 主結果 ───────────────────────────────────────────────────────────

METRICS4 = [("final_task_il", "task-IL", True), ("final_class_il", "class-IL", True),
            ("mean_leak", "leakage rate", False), ("mean_jaccard", "Jaccard", True)]


def fig4_main_hier(D, ls):
    flat = {a: load("main", a) for a in ("A3", "A5")}
    hier = {a: load("hier2", a, arch="hier", alloc="per_budget")
            for a in ("A3", "A5", "A5nG")}
    if not hier["A5"]:
        return None
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.5))
    out = {"flat": {}, "hier": {}, "paired": {}}
    groups = [("flat", flat, ["A3", "A5"]), ("hier", hier, ["A3", "A5", "A5nG"])]
    for ax, (key, lab, higher) in zip(axes, METRICS4):
        pos, labels = [], []
        x = 0
        for gname, g, arms in groups:
            for a in arms:
                if not g.get(a):
                    continue
                v, m, sd, n = series(g[a], a, key, ls)
                sc = 100 if key != "mean_jaccard" else 1
                ax.bar(x, m * sc, 0.62, color=C[a], hatch=HATCH.get(a, ""),
                       edgecolor="white", linewidth=0.6)
                ax.errorbar(x, m * sc, yerr=sd * sc, color="k", capsize=2, lw=0.8)
                ax.plot([x] * len(v), [q * sc for q in v], "o", color="k", ms=1.8, alpha=0.55)
                out[gname].setdefault(a, {})[key] = {"per_seed": v, "mean": m,
                                                     "std": sd, "n": n}
                pos.append(x); labels.append(f"{gname}\n{a}")
                x += 1
            x += 0.6
        pr = paired(hier["A5"], "A5", hier["A3"], "A3", key, ls, higher)
        if pr:
            out["paired"].setdefault("A5-A3", {})[key] = pr
            sc = 100 if key != "mean_jaccard" else 1
            ax.set_title(f"{lab}\nA5−A3 {pr['mean'] * sc:+.2f} ({pr['wins']}/{pr['n']})",
                         fontsize=7.5, loc="left")
        if hier.get("A5nG"):
            pg = paired(hier["A5"], "A5", hier["A5nG"], "A5nG", key, ls, higher)
            if pg:
                out["paired"].setdefault("A5-A5nG", {})[key] = pg
        ax.set_xticks(pos); ax.set_xticklabels(labels, fontsize=6)
        ax.set_ylabel(lab + (" (pp)" if key != "mean_jaccard" else ""))
    fig.tight_layout()
    out["paths"] = save(fig, "fig4_main_hier")
    D["fig4_main_hier"] = out
    return True


# ── Fig. 5 記憶體曲線 ───────────────────────────────────────────────────────

CAPS = [64, 128, 256, 512, 1024]


def _mem(arm, cap, ls):
    if cap == 512:
        return load("hier2", arm, arch="hier", alloc="per_budget")
    return load("memory_hier", arm, arch="hier", alloc="per_budget", cap=cap)


def fig5_memory_hier(D, ls):
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.5))
    out = {"curves": {}, "paired": {}}
    for ax, (key, lab) in zip(axes, [("final_task_il", "task-IL"),
                                     ("final_class_il", "class-IL")]):
        for a in ("A3", "A5"):
            ms, sds = [], []
            for cap in CAPS:
                _v, m, sd, _n = series(_mem(a, cap, ls), a, key, ls)
                ms.append(m * 100); sds.append(sd * 100)
            ax.errorbar(CAPS, ms, yerr=sds, color=C[a], marker="o", ms=3, lw=1.4,
                        capsize=2, label=a)
            out["curves"].setdefault(a, {})[key] = {"caps": CAPS, "mean": ms, "std": sds}
        for cap in CAPS:
            pr = paired(_mem("A5", cap, ls), "A5", _mem("A3", cap, ls), "A3", key, ls)
            if pr:
                out["paired"].setdefault(str(cap), {})[key] = pr
                y = (out["curves"]["A5"][key]["mean"][CAPS.index(cap)]
                     + out["curves"]["A3"][key]["mean"][CAPS.index(cap)]) / 2
                wl(ax, cap, y, pr["wins"], pr["n"], color="k")
        ax.set_xscale("log", base=2); ax.set_xticks(CAPS)
        ax.set_xticklabels([str(c) for c in CAPS])
        ax.set_xlabel("|M|"); ax.set_ylabel(f"{lab} (pp)")
        ax.legend(frameon=False, loc="lower right")
    cross = paired(_mem("A5", 128, ls), "A5", _mem("A3", 512, ls), "A3",
                   "final_task_il", ls)
    if cross:
        out["cross_4x"] = cross
        ax = axes[0]
        y5 = out["curves"]["A5"]["final_task_il"]["mean"][CAPS.index(128)]
        y3 = out["curves"]["A3"]["final_task_il"]["mean"][CAPS.index(512)]
        ax.annotate("", xy=(512, y3), xytext=(128, y5),
                    arrowprops=dict(arrowstyle="<->", lw=0.8, color="k", ls=":"))
        ax.annotate(f"4x ({cross['mean'] * 100:+.2f}, {cross['wins']}/{cross['n']})",
                    ((128 * 512) ** 0.5, (y5 + y3) / 2), fontsize=7, ha="center",
                    va="bottom")
    fig.tight_layout()
    out["paths"] = save(fig, "fig5_memory_hier")
    D["fig5_memory_hier"] = out
    return True


# ── Fig. 6 洩漏 = 選取漂移 ──────────────────────────────────────────────────

def _confusion(recs, ls):
    last = max((r["stage"] for r in recs), default=0)
    m = [[0] * 8 for _ in range(8)]
    for r in recs:
        if r["stage"] != last:
            continue
        m[r["true"]][r["pred_class_il"]] += 1
    return m


def fig6_leakage_confusion(D, ls):
    a1, a5 = load("main", "A1"), load("hier2", "A5", arch="hier", alloc="per_budget")
    if not a1 or not a5:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.9))
    out = {}
    for ax, (name, recs, arm) in zip(axes, [("SeqFT (A1, flat)", a1, "A1"),
                                            ("Ours (A5, hier)", a5, "A5")]):
        m = _confusion(recs, ls)
        norm = [[(c / max(sum(row), 1)) for c in row] for row in m]
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for k in (2, 4, 6):
            ax.axhline(k - 0.5, color="k", lw=0.8)
            ax.axvline(k - 0.5, color="k", lw=0.8)
        _v, leak, _sd, n = series(recs, arm, "mean_leak", ls)
        ax.set_title(f"{name}\nleakage {leak * 100:.2f} pp (n={n})", fontsize=7.5)
        ax.set_xticks(range(8)); ax.set_yticks(range(8))
        lab = [f"{s}{i}" for s in SHORT for i in (0, 1)]
        ax.set_xticklabels(lab, fontsize=5.5, rotation=90)
        ax.set_yticklabels(lab, fontsize=5.5)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.grid(False)
        out[arm] = {"confusion_counts": m, "mean_leak": leak, "n": n}
    fig.tight_layout()
    out["paths"] = save(fig, "fig6_leakage_confusion")
    D["fig6_leakage_confusion"] = out
    return True


# ── Fig. S1 組織配額分佈 ────────────────────────────────────────────────────

def figS1_quota_distribution(D, ls):
    recs = load("hier2", "A5", arch="hier", alloc="per_budget")
    if not recs:
        return None
    last = max(r["stage"] for r in recs)
    fig, ax = plt.subplots(figsize=(4.6, 2.4))
    out = {}
    bottoms = [0.0] * len(TASKS)
    cmap = plt.get_cmap("tab10")
    for j in range(8):
        share = []
        for t in TASKS:
            sub = [r for r in recs if r["stage"] == last and r["task"] == t]
            tot = sum(sum(r["group_quota"]) for r in sub) or 1
            share.append(sum(r["group_quota"][j] for r in sub) / tot)
        ax.barh(range(len(TASKS)), share, left=bottoms, height=0.6,
                color=cmap(j % 10), edgecolor="white", linewidth=0.4,
                label=f"g{j}")
        out[f"group{j}"] = dict(zip(SHORT, share))
        bottoms = [b + s for b, s in zip(bottoms, share)]
    ax.set_yticks(range(len(TASKS))); ax.set_yticklabels(SHORT)
    ax.set_xlabel("share of selected patches"); ax.set_xlim(0, 1)
    ax.legend(frameon=False, fontsize=6, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.28))
    fig.tight_layout()
    out["paths"] = save(fig, "figS1_quota_distribution")
    D["figS1_quota_distribution"] = out
    return True


# ── Fig. S2 架構完整性 ──────────────────────────────────────────────────────

ARCH_EXP = [("G5", "arch", "A5", "hier_state"), ("G4", "arch", "A5", "hier_query"),
            ("G3", "arch", "A5g", "hier")]
METRICS_S2 = [("final_task_il", "task-IL", True, True),
              ("final_class_il", "class-IL", True, True),
              ("mean_leak", "leakage", False, False),
              ("mean_quota_kl", "quota KL", False, False)]


def figS2_arch_completeness(D, ls):
    base = load("hier2", "A5", arch="hier", alloc="per_budget")
    if not base:
        return None
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.4))
    out = {}
    for ax, (key, lab, higher, primary) in zip(axes, METRICS_S2):
        for i, (name, tag, arm, arch) in enumerate(ARCH_EXP):
            recs = load(tag, arm, arch=arch, alloc="per_budget")
            pr = paired(recs, arm, base, "A5", key, ls, higher) if recs else None
            if not pr:
                continue
            sc = 1 if key == "mean_quota_kl" else 100
            ax.plot([d * sc for d in pr["diffs"]], [i] * pr["n"], "o", ms=2.6,
                    color=C["A5"], alpha=0.6)
            ax.errorbar(pr["mean"] * sc, i, xerr=pr["std"] * sc, color="k",
                        marker="|", ms=9, capsize=2, lw=1.0)
            good = (pr["wins"] == pr["n"] and pr["n"] >= 5
                    and ((pr["mean"] > 0) if higher else (pr["mean"] < 0)))
            ax.annotate(f"{pr['wins']}/{pr['n']}", (pr["mean"] * sc, i + 0.28),
                        fontsize=7, ha="center",
                        fontweight="bold" if (good and not primary) else "normal")
            out.setdefault(name, {})[key] = pr
        ax.axvline(0, color="k", lw=0.7)
        ax.set_yticks(range(len(ARCH_EXP)))
        ax.set_yticklabels([e[0] for e in ARCH_EXP])
        ax.set_ylim(-0.6, len(ARCH_EXP) - 0.2)
        ax.set_xlabel(f"Δ {lab}" + ("" if key == "mean_quota_kl" else " (pp)"))
        if primary:
            ax.set_facecolor("#F2F2F2")
            ax.set_title("pre-registered", fontsize=7, loc="left", color="#555555")
    fig.tight_layout()
    out["paths"] = save(fig, "figS2_arch_completeness")
    D["figS2_arch_completeness"] = out
    return True


# ── smoke fixture ───────────────────────────────────────────────────────────

def smoke(tmp: Path) -> int:
    """用合成 fixture 跑完整條路（憲法 §3.6）。

    fixture 刻意涵蓋**兩個 order** 與**seed 數不齊**（§3.6b）—— A5 五個 seed、
    A3 三個 seed，兩個 order 都有。
    """
    import random
    rng = random.Random(0)
    for tag, arms, arch, alloc in (("main", ["A1", "A3", "A5"], "flat", "per_budget"),
                                   ("hier2", ["A3", "A5", "A5nG"], "hier", "per_budget")):
        d = tmp / tag / "per_slide"; d.mkdir(parents=True, exist_ok=True)
        for arm in arms:
            n_seed = 5 if arm in ("A5", "A5nG") else 3
            for order in ("reverse", "main"):
                for seed in range(n_seed):
                    recs = []
                    for stage in range(4):
                        for t in ORDERS[order][:stage + 1]:
                            lo = 2 * TASKS.index(t) if t in TASKS else 0
                            for k in range(6):
                                recs.append({
                                    "arm": arm, "order": order, "seed": seed,
                                    "stage": stage, "task": t, "slide_id": f"{t}_{k}",
                                    "true": lo, "pred_class_il": lo if k < 4 else (lo + 2) % 8,
                                    "pred_task_il": lo if k < 4 else lo + 1,
                                    "selected_idx": sorted(rng.sample(range(50), 8)),
                                    "group_quota": [1] * 8, "n_patch": 3000, "B": 8,
                                    "utility_total": rng.uniform(-1, 1),
                                    "arch": arch, "allocation": alloc,
                                    "mem_capacity": 512,
                                })
                    (d / f"{arm}_{order}_seed{seed}.json").write_text(json.dumps(recs))
    return 0


# ── main ────────────────────────────────────────────────────────────────────

FIGURES = [
    ("fig2_budget_curve", lambda D, ls: fig2_budget_curve(D)),
    ("fig3_forgetting_axes", fig3_forgetting_axes),
    ("fig4_main_hier", fig4_main_hier),
    ("fig5_memory_hier", fig5_memory_hier),
    ("fig6_leakage_confusion", fig6_leakage_confusion),
    ("figS1_quota_distribution", figS1_quota_distribution),
    ("figS2_arch_completeness", figS2_arch_completeness),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smoke", action="store_true", help="用合成 fixture 跑（§3.6）")
    ap.add_argument("--only", default="", help="逗號分隔的圖名，只跑這幾張")
    args = ap.parse_args(argv)

    global EXP, OUT_DIR, DATA
    tmp = None
    if args.smoke:
        import tempfile
        tmp = Path(tempfile.mkdtemp(prefix="figsmoke_"))
        smoke(tmp)
        EXP = tmp
        OUT_DIR = tmp / "figures"; DATA = OUT_DIR / "figure_data.json"

    ls = list(load_config()["tasks"])
    want = {x.strip() for x in args.only.split(",") if x.strip()}
    D, made, skipped = {}, [], []
    for name, fn in FIGURES:
        if want and name not in want:
            continue
        try:
            ok = fn(D, ls)
        except Exception as e:                                   # noqa: BLE001
            print(f"  ❌ {name}：{type(e).__name__}: {e}")
            return 1
        (made if ok else skipped).append(name)
        print(f"  {'✅' if ok else '⏭️ '} {name}" + ("" if ok else "（缺資料）"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA.write_text(json.dumps(D, indent=1, ensure_ascii=False, default=float) + "\n")
    print(f"→ {DATA}（{len(made)} 張；跳過 {len(skipped)}）")
    if args.smoke and not made:
        print("❌ smoke 模式一張圖都沒產出"); return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
