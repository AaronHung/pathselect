#!/usr/bin/env python3
"""G1 — flat vs hierarchical 選取器的對照報告。

唯一的差異是 hierarchy（q_tau 與 state 兩邊都關著）。flat 側沿用主表存檔，
hier 側來自 `--arch hier` 的跑批；同 seed、同 |M|、同 λ、同 epochs。
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_exp2 import ARMS, ORDERS, arm_metrics                   # noqa: E402
from selector.grouping import TISSUE_GROUP_NAMES, NUM_GROUPS             # noqa: E402
from selector.text_encoder import load_config                            # noqa: E402

FLAT_DIR = REPO_ROOT / "outputs" / "exp2" / "main" / "per_slide"
HIER_DIR = REPO_ROOT / "outputs" / "exp2" / "hier" / "per_slide"
OUT = REPO_ROOT / "outputs" / "exp2" / "hier" / "HIER.md"
METRICS = [("final_task_il", "task-IL final avg", True),
           ("final_class_il", "class-IL final avg", True),
           ("mean_leak", "跨任務洩漏率", False),
           ("mean_jaccard", "selection Jaccard", True)]
ARMS_G1 = ["A3", "A5", "A5nG"]
RULE = ["### 方法學註記：win count 三級規則（DR-020）", "",
        "| win count | 名稱 |", "|---|---|",
        "| 5/5 | **systematic** |",
        "| 4/5 | **directional, inconclusive** |",
        "| ≤3/5 | **within noise** |", "",
        "**不報 p 值**（DR-016）。", ""]


def load(d: Path, arch: str) -> list[dict]:
    out = []
    for p in sorted(d.glob("*.json")):
        for r in json.loads(p.read_text()):
            if r.get("arch", "flat") != arch:
                continue
            if r.get("mem_capacity", 512) != 512 or r["order"] != "reverse":
                continue
            out.append(r)
    return out


def ms(vals, scale=1.0, fmt="{:.4f}"):
    vals = [v * scale for v in vals if v is not None and v == v]
    if not vals:
        return "—"
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f"{fmt.format(statistics.mean(vals))} ± {fmt.format(sd)}"


def verdict(wins, n):
    return ("**systematic**" if wins in (0, n) else "within noise" if wins <= 3
            else "directional, inconclusive")


def main() -> int:
    cfg = load_config()
    tasks = ORDERS["reverse"]
    flat, hier = load(FLAT_DIR, "flat"), load(HIER_DIR, "hier")
    if not hier:
        print("尚無 hier 資料"); return 1
    seeds = sorted(set(r["seed"] for r in hier) & set(r["seed"] for r in flat))
    print(f"flat {len(flat)} 筆、hier {len(hier)} 筆；共同 seeds {seeds}")

    M = {}
    for arch, recs in (("flat", flat), ("hier", hier)):
        for arm in ARMS_G1:
            sub = [r for r in recs if r["arm"] == arm]
            if sub:
                M[(arch, arm)] = {s: arm_metrics(sub, arm, tasks, s, cfg["tasks"])
                                  for s in seeds}

    L = ["# G1 — flat vs hierarchical 選取器", "",
         "**唯一的差異是 hierarchy**：q_tau 與 state 在兩側都關閉（Gate 1 的教訓 —— "
         "同時打開多件事就無法歸因）。同 seed、|M|=512、λ 全 1.0、epochs 5、"
         "B=8、c=1、reverse order。flat 側沿用主表存檔。",
         "", f"seeds {seeds}。", "",
         "## 主表", ""]
    for key, label, higher in METRICS:
        L += [f"### {label}", "",
              "| 架構 | " + " | ".join(f"{a} {ARMS[a]['name']}" for a in ARMS_G1) + " |",
              "|---" * (len(ARMS_G1) + 1) + "|"]
        for arch in ("flat", "hier"):
            cells = []
            for arm in ARMS_G1:
                if (arch, arm) not in M:
                    cells.append("—"); continue
                v = [M[(arch, arm)][s].get(key) for s in seeds
                     if M[(arch, arm)][s].get("per_task")]
                cells.append(ms([x for x in v if x is not None],
                                1 if key == "mean_jaccard" else 100,
                                "{:.4f}" if key == "mean_jaccard" else "{:.2f}"))
            L.append(f"| {arch} | " + " | ".join(cells) + " |")
        L.append("")

    L += ["## 配對比較（同 seed 相減）", ""] + RULE
    PAIRS = [(("hier", "A5"), ("flat", "A5")), (("hier", "A3"), ("flat", "A3")),
             (("hier", "A5"), ("hier", "A3")),
             (("hier", "A5"), ("hier", "A5nG"))]   # DR-022：隔離 group-level KD
    for key, label, higher in METRICS:
        rows = []
        for a, b in PAIRS:
            if a not in M or b not in M:
                continue
            d = [M[a][s][key] - M[b][s][key] for s in seeds
                 if M[a][s].get("per_task") and M[b][s].get("per_task")]
            if not d:
                continue
            sc = 1 if key == "mean_jaccard" else 100
            wins = sum((x > 0) if higher else (x < 0) for x in d)
            sd = statistics.stdev(d) if len(d) > 1 else 0.0
            rows.append(f"| {a[0]}-{a[1]} − {b[0]}-{b[1]} | "
                        + ", ".join(f"{x * sc:+.2f}" for x in d)
                        + f" | {statistics.mean(d) * sc:+.2f} ± {sd * sc:.2f} | "
                        f"{wins}/{len(d)} | {verdict(wins, len(d))} |")
        if rows:
            L += [f"### {label}", "",
                  "| 對照 | 逐 seed 配對差值 | 配對 mean ± std | win | 判定 |",
                  "|---|---|---|---|---|"] + rows + [""]

    # group 配額分佈
    L += ["## ⚠️ 跨模式比較的限制（DR-022）", "",
          "**A3 在 flat 下 F_g 完全無梯度、停在初始值**（`ste_allocation` 的注入只發生在 "
          "hierarchy 迴圈內）；在 hier 下 A3 的 F_g 會實際訓練。"
          "**因此 hier-A3 是比 flat-A3 更強的 baseline，不得直接跨模式比較 A3。**"
          "上表的 hier-A3 − flat-A3 一列只作記錄，不得單獨用來宣稱階層的效果。", "",
          "## group 配額分佈（學完 T4 後，A5）", "",
          "flat 的配額是**量測層**（選完之後統計落在哪一組）；"
          "hier 的配額是**決策層**（Group Selector 分配的名額）。", "",
          "| 架構 | task | " + " | ".join(TISSUE_GROUP_NAMES) + " |",
          "|---" * (len(TISSUE_GROUP_NAMES) + 2) + "|"]
    for arch, recs in (("flat", flat), ("hier", hier)):
        for t in tasks:
            sub = [r for r in recs if r["arm"] == "A5" and r["task"] == t
                   and r["stage"] == len(tasks) - 1 and r["seed"] in seeds]
            if not sub:
                continue
            tot = [sum(r["group_quota"][j] for r in sub) for j in range(NUM_GROUPS)]
            n = sum(tot) or 1
            L.append(f"| {arch} | {t.replace('tcga_', '')} | "
                     + " | ".join(f"{v / n:.3f}" for v in tot) + " |")

    # pre-registered 判準
    # DR-022：group-level distillation 首次驗證
    if ("hier", "A5") in M and ("hier", "A5nG") in M:
        L += ["## group-level distillation 是否有用（DR-022，首次驗證）", "",
              "`hier-A5` 與 `hier-A5nG` 的**唯一差異**是 L_KD 的 group 項係數"
              "（1.0 vs 0.0，後者完全不計算、r_new 不進計算圖）。"
              "架構圖 Panel I 畫了這一項，但在 flat 模式下 F_g 對選取零影響，"
              "所以它從未被實際測試過。", ""]
        for key, label, higher in METRICS:
            d = [M[("hier", "A5")][s_][key] - M[("hier", "A5nG")][s_][key]
                 for s_ in seeds if M[("hier", "A5")][s_].get("per_task")
                 and M[("hier", "A5nG")][s_].get("per_task")]
            if not d:
                continue
            sc = 1 if key == "mean_jaccard" else 100
            wins = sum((x > 0) if higher else (x < 0) for x in d)
            L.append(f"- **{label}**：{statistics.mean(d) * sc:+.2f}"
                     f"（{wins}/{len(d)}，{verdict(wins, len(d))}）")
        L += ["",
              "⚠️ 若 group 項無作用（≤3/5 且差值小），照實報 —— 論文會把 L_distill "
              "誠實寫成 patch-level，並在 limitation 說明 group-level 在此設定下"
              "未顯示效果。**不調 λ 搶救。**", ""]

    L += ["", "## Pre-registered 判準（DR-021，看到結果後不得修改）", ""]
    if ("hier", "A5") in M and ("flat", "A5") in M:
        d = [M[("hier", "A5")][s]["final_class_il"] - M[("flat", "A5")][s]["final_class_il"]
             for s in seeds if M[("hier", "A5")][s].get("per_task")
             and M[("flat", "A5")][s].get("per_task")]
        wins = sum(x > 0 for x in d)
        mean = statistics.mean(d) * 100
        if wins >= 4:
            call = ("**hier-A5 ≥ flat-A5（win ≥ 4/5）→ 採用階層為主線**，"
                    "架構圖與標題保持。")
        elif wins <= 1 and abs(mean) > 2.0:
            call = ("⚠️ **hier 顯著劣於 flat（win ≤ 1/5 且差值大）→ 停下來回報**，"
                    "不自行調參數搶救。由 PI 裁定改標題或改用其他方式對齊架構圖。")
        else:
            call = ("**在雜訊內 → 仍採用階層為主線**，論文誠實寫「階層在此設定下與 "
                    "flat 相當；其價值在於提供可解釋的組織層配額與 group-level 保存」"
                    "（配額分佈本身就是定性貢獻）。")
        L += [f"hier-A5 − flat-A5（class-IL）= **{mean:+.2f} pp**，"
              f"win **{wins}/{len(d)}**（{verdict(wins, len(d))}）。", "", f"→ {call}", ""]
    L += ["逐 slide 預測：`outputs/exp2/hier/per_slide/*.json`", ""]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n")
    print(f"→ {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
