#!/usr/bin/env python3
"""G2 — L_sem 的 semantic prior 三臂消融（none / max_sim / discriminative）。

beta_s = 0.1 在每一次跑裡都開著，從未被消融過。若 L_sem 無作用，
HistoSelect 這條連結就變薄。discriminative 是 pre-registered 主線（DR-007）；
若 max_sim 或 none 勝出，照實報 —— 那是有價值的發現。
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_exp2 import ORDERS, arm_metrics                         # noqa: E402
from selector.text_encoder import load_config                            # noqa: E402

OUT_DIR = REPO_ROOT / "outputs" / "exp2" / "prior"
OUT = OUT_DIR / "PRIOR.md"
PRIORS = ["none", "max_sim", "discriminative"]
MAINLINE = "discriminative"
METRICS = [("final_task_il", "task-IL final avg", True),
           ("final_class_il", "class-IL final avg", True),
           ("mean_leak", "跨任務洩漏率", False),
           ("mean_jaccard", "selection Jaccard", True)]


def ms(vals, scale=1.0, fmt="{:.4f}"):
    vals = [v * scale for v in vals if v is not None and v == v]
    if not vals:
        return "—"
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f"{fmt.format(statistics.mean(vals))} ± {fmt.format(sd)}"


def verdict(wins, n):
    return ("**systematic**" if wins in (0, n) else "within noise" if wins <= 3
            else "directional, inconclusive")


def collect(arch: str, allocation: str = "per_budget") -> dict:
    """從所有相關 tag 蒐集 A5 的記錄，依 prior 分組。

    ⚠️ **必須同時過濾 allocation。** `hier` tag 裡是 G1 的 per_chunk 紀錄
    （階層退化成單組選取，88.6%），`hier2` 才是 per_budget 的主線。
    早期紀錄沒有 allocation 欄位，缺欄位一律視為 per_chunk。
    2026-08-24 曾因缺這道過濾，把 G1 的退化紀錄當成 discriminative 主線臂，
    產出「主線遠差於 none/max_sim」的假結論。
    """
    out = {p: [] for p in PRIORS}
    roots = [REPO_ROOT / "outputs" / "exp2" / t / "per_slide"
             for t in ("prior", "hier2", "main")]
    seen = set()
    for d in roots:
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.json")):
            for r in json.loads(f.read_text()):
                if (r["arm"] != "A5" or r["order"] != "reverse"
                        or r.get("mem_capacity", 512) != 512
                        or r.get("arch", "flat") != arch):
                    continue
                if arch == "hier" and r.get("allocation", "per_chunk") != allocation:
                    continue
                pr = r.get("prior", MAINLINE)
                key = (pr, r["seed"], r["stage"], r["task"], r["slide_id"])
                if key in seen:
                    continue
                seen.add(key)
                out[pr].append(r)
    return out


def main() -> int:
    arch = sys.argv[1] if len(sys.argv) > 1 else "hier"
    cfg = load_config()
    tasks = ORDERS["reverse"]
    groups = collect(arch)
    have = [p for p in PRIORS if groups[p]]
    if not have:
        print(f"尚無 arch={arch} 的資料"); return 1
    seeds = sorted(set.intersection(*[{r["seed"] for r in groups[p]} for p in have]))
    print(f"arch={arch}；有資料的 prior：{have}；共同 seeds {seeds}")

    M = {p: {s: arm_metrics(groups[p], "A5", tasks, s, cfg["tasks"]) for s in seeds}
         for p in have}

    L = ["# G2 — semantic prior 三臂消融（L_sem）", "",
         f"arm = A5、arch = **{arch}**（allocation = per_budget）、reverse order、"
         f"|M| = 512、seeds {seeds}、"
         "beta_s = 0.1、其餘設定與主表相同（λ 全 1.0，不調）。", "",
         f"**{MAINLINE} 是 pre-registered 主線（DR-007）。**"
         "若 max_sim 或 none 勝出，照實報 —— 那是有價值的發現"
         "（relevance 勝過 discriminability，或 semantic prior 非必要）。", "",
         "## 主表", "",
         "| prior | " + " | ".join(l for _k, l, _h in METRICS) + " |",
         "|---" * (len(METRICS) + 1) + "|"]
    for p in have:
        cells = []
        for key, _l, _h in METRICS:
            v = [M[p][s].get(key) for s in seeds if M[p][s].get("per_task")]
            cells.append(ms([x for x in v if x is not None],
                            1 if key == "mean_jaccard" else 100,
                            "{:.4f}" if key == "mean_jaccard" else "{:.2f}"))
        star = " ⭐主線" if p == MAINLINE else ""
        L.append(f"| {p}{star} | " + " | ".join(cells) + " |")

    L += ["", "## 配對比較（同 seed 相減；三級規則見 DR-020）", ""]
    pairs = [(MAINLINE, p) for p in have if p != MAINLINE]
    if "max_sim" in have and "none" in have:
        pairs.append(("max_sim", "none"))
    for key, label, higher in METRICS:
        rows = []
        for a, b in pairs:
            if a not in M or b not in M:
                continue
            d = [M[a][s][key] - M[b][s][key] for s in seeds
                 if M[a][s].get("per_task") and M[b][s].get("per_task")]
            if not d:
                continue
            sc = 1 if key == "mean_jaccard" else 100
            wins = sum((x > 0) if higher else (x < 0) for x in d)
            sd = statistics.stdev(d) if len(d) > 1 else 0.0
            rows.append(f"| {a} − {b} | " + ", ".join(f"{x * sc:+.2f}" for x in d)
                        + f" | {statistics.mean(d) * sc:+.2f} ± {sd * sc:.2f} | "
                        f"{wins}/{len(d)} | {verdict(wins, len(d))} |")
        if rows:
            L += [f"### {label}", "",
                  "| 對照 | 逐 seed 配對差值 | 配對 mean ± std | win | 判定 |",
                  "|---|---|---|---|---|"] + rows + [""]

    # 判讀骨架（只陳述數字，不下結論）
    if MAINLINE in M and "none" in M:
        d = [M[MAINLINE][s]["final_class_il"] - M["none"][s]["final_class_il"]
             for s in seeds if M[MAINLINE][s].get("per_task")
             and M["none"][s].get("per_task")]
        wins = sum(x > 0 for x in d)
        L += ["## L_sem 是否有作用（class-IL）", "",
              f"{MAINLINE} − none = **{statistics.mean(d) * 100:+.2f} pp**，"
              f"win **{wins}/{len(d)}**（{verdict(wins, len(d))}）。",
              "", "判讀由 PI 進行；此處只陳述數字。", ""]
    L += ["逐 slide 預測：`outputs/exp2/prior/per_slide/*.json`", ""]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n")
    print(f"→ {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
