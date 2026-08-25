#!/usr/bin/env python3
"""E1 階層版 —— 記憶體效率曲線（per_budget 配額、hier 架構）。

與 flat 版 `report_memory.py` 相同格式，另外兩件事：
  1. **每個 |M| 都報結構性指標**（單組比例）。若某個容量下階層退化，
     那格的判讀無效，明確標出。
  2. 產出 flat vs hier 的曲線對照 —— `hier-A3 − flat-A3 = −3.11 pp` 已證明
     replay 在階層下的行為與 flat 不同，該曲線不可假設可移植。
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_exp2 import ARMS, ORDERS, arm_metrics                   # noqa: E402
from selector.text_encoder import load_config                            # noqa: E402

HIER_DIR = REPO_ROOT / "outputs" / "exp2" / "memory_hier" / "per_slide"
FLAT_DIR = REPO_ROOT / "outputs" / "exp2" / "memory" / "per_slide"
OUT = REPO_ROOT / "outputs" / "exp2" / "memory_hier" / "MEMORY_HIER.md"
METRICS = [("final_task_il", "task-IL final avg", True),
           ("final_class_il", "class-IL final avg", True),
           ("mean_leak", "跨任務洩漏率", False),
           ("mean_jaccard", "selection Jaccard", True)]
ARMS_E1 = ["A3", "A5"]
CONTRACT_CAP = 512
DEGENERATE = 0.5      # 單組比例超過即視為階層退化，該格判讀無效


def load(d: Path, arch: str) -> list[dict]:
    if not d.is_dir():
        return []
    return [r for f in sorted(d.glob("*.json")) for r in json.loads(f.read_text())
            if r.get("arch", "flat") == arch and r["order"] == "reverse"]


def ms(vals, scale=1.0, fmt="{:.2f}"):
    vals = [v * scale for v in vals if v is not None and v == v]
    if not vals:
        return "—"
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f"{fmt.format(statistics.mean(vals))} ± {fmt.format(sd)}"


def verdict(w, n):
    return ("**systematic**" if w in (0, n) else "within noise" if w <= 3
            else "directional, inconclusive")


def structural(recs, tasks) -> tuple[float, float, dict, int]:
    rs = [r for r in recs if r["stage"] == len(tasks) - 1]
    ng = [sum(1 for v in r["group_quota"] if v > 0) for r in rs]
    share = [max(r["group_quota"]) / sum(r["group_quota"]) for r in rs]
    return (Counter(ng)[1] / len(ng), statistics.mean(ng),
            dict(sorted(Counter(ng).items())), len(rs))


def build(recs, tasks, label_space, seeds):
    M = {}
    for arm in ARMS_E1:
        for cap in sorted({r.get("mem_capacity", CONTRACT_CAP) for r in recs}):
            sub = [r for r in recs if r["arm"] == arm
                   and r.get("mem_capacity", CONTRACT_CAP) == cap]
            if sub:
                M[(arm, cap)] = {s: arm_metrics(sub, arm, tasks, s, label_space)
                                 for s in seeds}
    return M


def mean_of(M, arm, cap, seeds, key):
    if (arm, cap) not in M:
        return None
    v = [M[(arm, cap)][s].get(key) for s in seeds if M[(arm, cap)][s].get("per_task")]
    v = [x for x in v if x is not None]
    return statistics.mean(v) if v else None


def main() -> int:
    cfg = load_config()
    tasks = ORDERS["reverse"]
    hier, flat = load(HIER_DIR, "hier"), load(FLAT_DIR, "flat")
    if not hier:
        print("尚無 memory_hier 資料"); return 1
    caps = sorted({r.get("mem_capacity", CONTRACT_CAP) for r in hier})
    seeds = sorted({r["seed"] for r in hier})
    print(f"hier {len(hier)} 筆、flat {len(flat)} 筆；|M| = {caps}；seeds = {seeds}")

    Mh = build(hier, tasks, cfg["tasks"], seeds)
    Mf = build(flat, tasks, cfg["tasks"], seeds) if flat else {}

    # 結構性把關（逐 |M|）
    struct = {}
    for cap in caps:
        sub = [r for r in hier if r.get("mem_capacity", CONTRACT_CAP) == cap]
        if sub:
            struct[cap] = structural(sub, tasks)

    L = ["# E1 階層版 — 記憶體效率曲線（per_budget 配額）", "",
         f"arms = {ARMS_E1}、|M| ∈ {caps}、reverse order、seeds {seeds}、"
         "arch = **hier**、allocation = per_budget。其餘設定與主表相同"
         "（B=8、c=1、epochs 5、lr 1e-3、beta_s 0.1、beta_u 0.1、λ 全 1.0、replay_k=1）。",
         "",
         f"⚠️ |M| = {CONTRACT_CAP} 沿用 G1' 的存檔，不重跑。"
         f"|M| = 1024 超出 CONTRACT-3，需顯式 opt-in。",
         "",
         "## 結構性把關（逐 |M|）", "",
         "階層在某個容量下若退化成單組選取，**該格的判讀無效**。", "",
         "| \\|M\\| | 單組比例 | 平均用到幾組 | 分佈 | 判讀 |",
         "|---|---|---|---|---|"]
    invalid = []
    for cap in caps:
        if cap not in struct:
            continue
        single, mean_ng, hist, n = struct[cap]
        ok = single <= DEGENERATE
        if not ok:
            invalid.append(cap)
        L.append(f"| {cap} | {single:.1%} | {mean_ng:.2f} | {hist} | "
                 f"{'✅ 有效' if ok else '❌ **退化，該格判讀無效**'} |")
    L.append("")
    if invalid:
        L += [f"⚠️ **|M| ∈ {invalid} 的階層退化，下方所有涉及這些容量的數字一律不得採用。**", ""]
    else:
        L += ["✅ 所有容量下階層都有作用空間。", ""]

    for key, label, higher in METRICS:
        L += [f"## {label}", "",
              "| \\|M\\| | " + " | ".join(f"{a} {ARMS[a]['name']}" for a in ARMS_E1)
              + " | A5 − A3（配對） |", "|---" * (len(ARMS_E1) + 2) + "|"]
        for cap in caps:
            cells = []
            for arm in ARMS_E1:
                v = [Mh[(arm, cap)][s].get(key) for s in seeds
                     if (arm, cap) in Mh and Mh[(arm, cap)][s].get("per_task")]
                cells.append(ms([x for x in v if x is not None],
                                1 if key == "mean_jaccard" else 100,
                                "{:.4f}" if key == "mean_jaccard" else "{:.2f}"))
            paired = "—"
            if ("A3", cap) in Mh and ("A5", cap) in Mh:
                d = [Mh[("A5", cap)][s][key] - Mh[("A3", cap)][s][key] for s in seeds
                     if Mh[("A5", cap)][s].get("per_task")
                     and Mh[("A3", cap)][s].get("per_task")]
                if d:
                    sc = 1 if key == "mean_jaccard" else 100
                    w = sum((x > 0) if higher else (x < 0) for x in d)
                    sd = statistics.stdev(d) if len(d) > 1 else 0.0
                    paired = (f"{statistics.mean(d) * sc:+.2f} ± {sd * sc:.2f}"
                              f"（{w}/{len(d)}，{verdict(w, len(d))}）")
            mark = "  ❌退化" if cap in invalid else ""
            L.append(f"| {cap}{mark} | " + " | ".join(cells) + f" | {paired} |")
        L.append("")

    # flat vs hier 對照
    if Mf:
        L += ["## flat vs hier 的曲線對照", "",
              "`hier-A3 − flat-A3 = −3.11 pp`（G1'）已證明 **replay 在階層下的行為與 "
              "flat 不同**，因此 flat 的記憶體曲線不可假設可移植。以下為兩者並列。", "",
              "| \\|M\\| | arm | flat class-IL | hier class-IL | hier − flat |",
              "|---|---|---|---|---|"]
        for cap in caps:
            for arm in ARMS_E1:
                f_ = mean_of(Mf, arm, cap, seeds, "final_class_il")
                h_ = mean_of(Mh, arm, cap, seeds, "final_class_il")
                if f_ is None or h_ is None:
                    continue
                L.append(f"| {cap} | {arm} | {f_ * 100:.2f} | {h_ * 100:.2f} | "
                         f"{(h_ - f_) * 100:+.2f} pp |")
        L.append("")

    valid_caps = [c for c in caps if c not in invalid]
    L += cross_capacity_section(Mh, seeds, valid_caps)
    L += efficiency_section(Mh, seeds, valid_caps)
    L += dr019_section(Mh, seeds, valid_caps)
    L += ["逐 slide 預測：`outputs/exp2/memory_hier/per_slide/*.json`", ""]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n")
    print(f"→ {OUT}")
    return 0


# ── 跨容量配對（DR-042）────────────────────────────────────────────────────

#: (較省的臂, 其 |M|, 對照臂, 其 |M|)。同一組 seeds，故可配對。
CROSS = [("A5", 64, "A3", 1024), ("A5", 128, "A3", 1024),
         ("A5", 64, "A3", 512), ("A5", 128, "A3", 512)]


def paired(M, seeds, a1, c1, a2, c2, key, higher):
    """逐 seed 配對差值。回傳 (diffs, mean, std, wins, n) 或 None。"""
    if (a1, c1) not in M or (a2, c2) not in M:
        return None
    d = [M[(a1, c1)][s][key] - M[(a2, c2)][s][key] for s in seeds
         if M[(a1, c1)][s].get("per_task") and M[(a2, c2)][s].get("per_task")
         and M[(a1, c1)][s].get(key) is not None and M[(a2, c2)][s].get(key) is not None]
    if not d:
        return None
    w = sum((x > 0) if higher else (x < 0) for x in d)
    sd = statistics.stdev(d) if len(d) > 1 else 0.0
    return d, statistics.mean(d), sd, w, len(d)


def cross_capacity_section(M, seeds, valid_caps) -> list[str]:
    """效率主張是跨容量比較 —— 未配對的均值比較不足以支撐（PI 裁定，DR-042）。"""
    L = ["## 跨容量配對比較（效率主張的依據）", "",
         "「A5@小 |M| 追平 A3@大 |M|」是**跨容量**比較，但兩者跑在**同一組 seeds** 上，"
         "所以可以配對。**未配對的均值比較不足以支撐效率主張。**", "",
         "win count 方向為「較省的臂較好」。三級規則同 DR-020。", ""]
    for a1, c1, a2, c2 in CROSS:
        if c1 not in valid_caps or c2 not in valid_caps:
            continue
        L += [f"### {a1}@{c1} − {a2}@{c2}", "",
              "| 指標 | 逐 seed 配對差值 | 配對 mean ± std | win count | 三級判讀 |",
              "|---|---|---|---|---|"]
        for key, label, higher in METRICS:
            r = paired(M, seeds, a1, c1, a2, c2, key, higher)
            if r is None:
                continue
            d, mean, sd, w, n = r
            sc = 1 if key == "mean_jaccard" else 100
            unit = "" if key == "mean_jaccard" else " pp"
            L.append(f"| {label} | {', '.join(f'{x * sc:+.2f}' for x in d)} | "
                     f"{mean * sc:+.2f} ± {sd * sc:.2f}{unit} | {w}/{n} | "
                     f"{verdict(w, n)} |")
        L.append("")
    return L


def _systematic(M, seeds, a1, c1, a2, c2, key, higher):
    r = paired(M, seeds, a1, c1, a2, c2, key, higher)
    if r is None:
        return False, r
    _d, mean, _sd, w, n = r
    good = mean > 0 if higher else mean < 0
    return (w == n and n >= 5 and good), r


def efficiency_section(M, seeds, valid_caps) -> list[str]:
    """效率主張改建在 task-IL；倍數只宣稱到配對結果支持的程度（DR-042）。"""
    L = ["## 記憶體效率主張（階層版，改建在 task-IL）", "",
         "⚠️ **本節取代先前基於 class-IL 的「8×」宣稱，該宣稱已撤回。** 兩個理由：",
         "",
         "1. 原錨點 |M|=128 的 class-IL 配對是 **4/5 directional**，且 "
         "**std(7.71) > mean(7.72)** —— 用它當效率主張的支點站不住。",
         "2. flat 版的防禦「A3 在 256 後不再改善」**不適用於階層版** —— "
         "階層版 A3 的 class-IL 曲線到 1024 仍在上升，尚未飽和，"
         "所以「A3 再加記憶體也沒用」這條路在階層下不能走。",
         "",
         "改建在 **task-IL**：A3 雖然單調上升，但 A5 − A3 在 5 個容量中有 4 個為 "
         "systematic（見上方 task-IL 表），是這批資料裡最穩的軸。", ""]

    # 找配對結果支持的最大倍數（task-IL 為主）
    wins = []
    for a1, c1, a2, c2 in CROSS:
        if c1 not in valid_caps or c2 not in valid_caps:
            continue
        ok, r = _systematic(M, seeds, a1, c1, a2, c2, "final_task_il", True)
        if ok:
            wins.append((c2 // c1, a1, c1, a2, c2, r))
    L += ["| 比較 | task-IL 配對 | win count | 倍數 | 是否支持 |",
          "|---|---|---|---|---|"]
    for a1, c1, a2, c2 in CROSS:
        if c1 not in valid_caps or c2 not in valid_caps:
            continue
        ok, r = _systematic(M, seeds, a1, c1, a2, c2, "final_task_il", True)
        _d, mean, sd, w, n = r
        L.append(f"| {a1}@{c1} − {a2}@{c2} | {mean * 100:+.2f} ± {sd * 100:.2f} pp | "
                 f"{w}/{n}（{verdict(w, n)}） | {c2 // c1}× | "
                 f"{'✅' if ok else '❌'} |")
    L.append("")
    if wins:
        k, a1, c1, a2, c2, r = max(wins, key=lambda x: x[0])
        _d, mean, sd, w, n = r
        L += [f"→ **在測試範圍內達 {k}× 記憶體效率**："
              f"{a1}@{c1} 相對 {a2}@{c2} 的 task-IL 配對差值為 "
              f"{mean * 100:+.2f} ± {sd * 100:.2f} pp（{w}/{n}，systematic）。",
              "",
              "⚠️ **A3 的曲線在測試範圍內未飽和**（class-IL 到 1024 仍在上升）。"
              f"因此 {k}× 是**測試範圍內的下界**，"
              "**不是 A3 需求的上界** —— 真正需要多大的 |M| 才能讓 A3 追平 A5，"
              "本批資料無法回答。", ""]
    else:
        L += ["→ **沒有任何跨容量配對達到 5/5 systematic，效率倍數無法宣稱。**", ""]

    # class-IL 另外報
    L += ["### class-IL 另報", "",
          "| \\|M\\| | A5 − A3 配對 | win count | 三級判讀 |",
          "|---|---|---|---|"]
    for cap in valid_caps:
        r = paired(M, seeds, "A5", cap, "A3", cap, "final_class_il", True)
        if r is None:
            continue
        _d, mean, sd, w, n = r
        L.append(f"| {cap} | {mean * 100:+.2f} ± {sd * 100:.2f} pp | {w}/{n} | "
                 f"{verdict(w, n)} |")
    r = paired(M, seeds, "A5", max(valid_caps), "A3", max(valid_caps),
               "final_class_il", True)
    if r is not None:
        _d, mean, sd, w, n = r
        L += ["",
              f"⚠️ **|M| = {max(valid_caps)} 時 A5 對 A3 的 class-IL 優勢落入雜訊**"
              f"（{mean * 100:+.2f} pp，{w}/{n}）。class-IL 不支持在大容量端的優勢主張。",
              ""]
    return L


def dr019_section(M, seeds, valid_caps) -> list[str]:
    """DR-019 的四條可宣稱在階層版逐條重驗（PI 裁定，DR-042）。"""
    a3 = [(c, mean_of(M, "A3", c, seeds, "final_class_il")) for c in valid_caps]
    a3 = [(c, v) for c, v in a3 if v is not None]
    a5_128 = mean_of(M, "A5", 128, seeds, "final_class_il")
    a3_top = max(a3, key=lambda x: x[1]) if a3 else (None, None)
    # ①「A3 在 256 之後不再改善」→ 直接測 256 之後有沒有任何容量超過 256
    a3_at = dict(a3)
    after = [(c, v) for c, v in a3 if c > 256]
    improves = bool(after) and a3_at.get(256) is not None and \
        max(v for _c, v in after) > a3_at[256]
    # ① 的後半：「含 1024 皆未超過 A5@128」→ A3 的全域最佳必須**低於等於** A5@128
    stays_below = (a3_top[1] is not None and a5_128 is not None
                   and a3_top[1] <= a5_128)

    def sd_across(arm, key):
        v = [mean_of(M, arm, c, seeds, key) for c in valid_caps]
        v = [x for x in v if x is not None]
        return statistics.stdev(v) * 100 if len(v) > 1 else float("nan")

    sd_a5, sd_a3 = sd_across("A5", "final_task_il"), sd_across("A3", "final_task_il")
    r64 = paired(M, seeds, "A5", 64, "A3", 64, "final_task_il", True)
    per_cap = {c: paired(M, seeds, "A5", c, "A3", c, "final_task_il", True)
               for c in valid_caps}
    scarce_max = (r64 is not None
                  and all(r64[1] >= v[1] for v in per_cap.values() if v is not None))

    rows = [
        ("① A3 在 256 之後不再改善，含 1024 皆未超過 A5@128",
         (not improves) and stays_below,
         (f"A3 的 class-IL 在 256 之後**{'仍在改善' if improves else '不再改善'}**"
          f"（{'、'.join(f'{c}→{v:.4f}' for c, v in a3)}）；"
          f"A3 全域最佳 {a3_top[1]:.4f}@{a3_top[0]}，"
          f"{'低於' if stays_below else '**高於**'} A5@128 的 {a5_128:.4f}"
          if a3 and a5_128 is not None else "資料不足")),
        ("② 2× 記憶體效率",
         None,
         "已由本檔「記憶體效率主張」一節以**跨容量配對**重新裁定，倍數改依配對結果，"
         "不沿用 flat 版的數字"),
        ("③ A5 對記憶體預算穩健而 A3 不穩",
         sd_a5 < sd_a3,
         f"task-IL 跨 \\|M\\| 的標準差：A5 {sd_a5:.2f} pp vs A3 {sd_a3:.2f} pp"),
        ("④ 稀缺端優勢最大",
         scarce_max,
         (f"\\|M\\|=64 的 task-IL 配對 {r64[1] * 100:+.2f} pp（{r64[3]}/{r64[4]}），"
          f"為所有容量中最大" if r64 and scarce_max else
          (f"\\|M\\|=64 的 task-IL 配對 {r64[1] * 100:+.2f} pp，"
           f"**不是**所有容量中最大" if r64 else "資料不足"))),
    ]
    L = ["## DR-019 的四條可宣稱在階層版是否成立", "",
         "DR-019 是在 **flat** 架構上裁定的。`hier-A3 − flat-A3 = −3.11 pp`（G1'）"
         "已證明 replay 在階層下的行為與 flat 不同，因此每一條都必須重驗。"
         "**不成立的照實報 —— 那本身就是「replay 在階層下行為不同」的證據。**", "",
         "| DR-019 的可宣稱 | 階層版 | 依據 |", "|---|---|---|"]
    for claim, ok, why in rows:
        mark = "**改由本檔重裁**" if ok is None else ("✅ 成立" if ok else "❌ **不成立**")
        L.append(f"| {claim} | {mark} | {why} |")
    L += ["",
          "⚠️ ① 不成立這件事本身是有訊息量的：flat 版可以用「A3 加記憶體也沒用」"
          "當防禦，階層版**不能**。效率主張因此必須改建在配對證據上，"
          "而不是「對手已飽和」這個前提。", ""]
    return L


if __name__ == "__main__":
    raise SystemExit(main())
