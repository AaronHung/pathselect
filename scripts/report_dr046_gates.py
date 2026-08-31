#!/usr/bin/env python3
"""DR-046 gate 總表 → `docs/DR046_GATES.md`。

**只有數字，不寫解讀** —— 每節僅含逐 seed 差值、win count、以及觸發了哪一個
pre-registered 分支。判讀的文字由 PI 在 ledger 與論文裡寫。

所有數字從 `outputs/exp2/main/per_slide/*.json` 重算，沿用
`run_exp2.arm_metrics` / `filter_arch`，不另立算法。
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_exp2 import ORDERS, arm_metrics, filter_arch                 # noqa: E402
from selector.text_encoder import load_config                         # noqa: E402

SRC = ROOT / "outputs" / "exp2" / "main" / "per_slide"
OUT = ROOT / "docs" / "DR046_GATES.md"
ORDER = "reverse"
SEEDS = [0, 1, 2, 3, 4]

#: (指標 key, 顯示名, 越大越好)
METRICS = [("final_task_il", "task-IL final avg", True),
           ("final_class_il", "class-IL final avg", True),
           ("mean_leak", "跨任務洩漏率", False),
           ("mean_jaccard", "selection Jaccard", True)]

#: (gate 名稱, 說明, arch, 臂 A, 臂 B) —— 一律報 A − B
GATES = [
    ("G-W1", "warm-start（task1 全參數 FT）", "flat", "W1", "A5"),
    ("G-L2", "single continual adapter", "flat", "L2", "A5"),
    ("G-C1", "post-hoc composition（獨立 delta 相加）", "flat", "A2", "C1"),
    ("G-α", "damped merging（α = 0.5）", "flat", "A5", "A5H"),
    ("L2@hier", "single continual adapter，階層底盤（配對版）", "hier", "L2", "A5"),
]

#: 觸發的分支 —— **PI 已逐條裁定，本檔照錄，不由腳本推導**。
BRANCH = {
    "G-W1": "**分裂** → 依預註冊 **split 規則**維持現行 LoRA-from-task-1。",
    "G-L2": "A5 勝 4/5（class-IL）→ 觸發 **fresh-per-task 容量正當化條款**。",
    "G-C1": "**翻盤** → 觸發**翻盤條款**。",
    "G-α": "**分裂** → 維持 **α = 1**。",
    "L2@hier": "**未預註冊之觀察**（非 gate）。",
}

#: PI 核可的註記。逐字記錄，不加解讀。
NOTES = {
    "G-C1": "PI 核可：依**翻盤條款**誠實報告。",
    "G-α": "PI 核可：維持 **α = 1**；A5H 留存為消融證據。",
    "L2@hier": ("PI 核可：**未預註冊之觀察**。方法維持 fresh-per-task"
                "（現任者＋主指標 task-IL 5/5）；底盤交互作用於論文如實報告。"),
}


def load(arch: str) -> list[dict]:
    recs = [r for f in sorted(SRC.glob("*.json")) for r in json.loads(f.read_text())
            if r.get("order") == ORDER]
    return filter_arch(recs, arch)


def paired(recs, a: str, b: str, key: str, higher: bool, ls):
    Ma = {s: arm_metrics(recs, a, ORDERS[ORDER], s, ls) for s in SEEDS}
    Mb = {s: arm_metrics(recs, b, ORDERS[ORDER], s, ls) for s in SEEDS}
    d = [Ma[s][key] - Mb[s][key] for s in SEEDS
         if Ma[s].get(key) is not None and Mb[s].get(key) is not None]
    if not d:
        return None
    w = sum((x > 0) if higher else (x < 0) for x in d)
    return d, statistics.mean(d), (statistics.stdev(d) if len(d) > 1 else 0.0), w, len(d)


def tier(w: int, n: int, a: str, b: str) -> str:
    """三級規則，**帶方向**。

    ⚠️ 0/n 是 n/n 的鏡像 —— 代表 `b` 系統性勝出，不是 within noise。
    repo 既有的 `report_memory_hier.verdict` 就是把 `w in (0, n)` 都算 systematic；
    本檔沿用該慣例並標出是哪一邊勝，否則 G-C1 的 0/5 會被讀成「沒差異」，
    而實際上是 C1 在四個指標上全勝。
    """
    if w == n:
        return f"**systematic**（{a} 勝）"
    if w == 0:
        return f"**systematic**（{b} 勝）"
    if w == n - 1:
        return f"directional, inconclusive（{a} 向）"
    if w == 1:
        return f"directional, inconclusive（{b} 向）"
    return "within noise"


def main() -> int:
    ls = list(load_config()["tasks"])
    L = ["# DR-046 gate 總表", "",
         "五個 gate 的**逐 seed 數字與 win count**。三級規則（DR-020）："
         "5/5 = systematic、4/5 = directional inconclusive、≤3/5 = within noise。"
         "**不報 p 值**（DR-016）。", "",
         "⚠️ **本檔只放數字，不寫解讀** —— 判讀的文字在 "
         "[`docs/ledger/DR-046.md`](ledger/DR-046.md) 與論文裡。", "",
         "⚠️ 「觸發分支」一欄是 **PI 逐條裁定後照錄**，不由腳本推導；"
         "用條款名稱標註而不引節號 —— pre-registration 文件（PI 的 DR-046 v2）"
         "不在本 repo 內，無法逐字核對節次編號。", "",
         "⚠️ **win count 帶方向**：`0/5` 是 `5/5` 的鏡像（對照式右邊那個臂系統性勝出），"
         "不是 within noise。沿用 `report_memory_hier.verdict` 的既有慣例。", "",
         "資料源：`outputs/exp2/main/per_slide/*.json`，"
         "沿用 `run_exp2.arm_metrics` / `filter_arch` 重算；"
         "flat 與 hier 分開取（同名臂在兩種架構下是兩個實驗）。", ""]

    for name, desc, arch, a, b in GATES:
        recs = load(arch)
        rows = {}
        for key, lab, higher in METRICS:
            r = paired(recs, a, b, key, higher, ls)
            if r:
                rows[key] = r
        L += ["---", "", f"## {name}　{desc}", "",
              f"對照：**{a} − {b}**（arch = `{arch}`，order = `{ORDER}`，"
              f"seeds = {SEEDS}）", ""]
        if not rows:
            L += ["⚠️ 缺資料。", ""]
            continue
        L += ["| 指標 | 逐 seed 差值 | mean ± sd | win count | 三級判讀 |",
              "|---|---|---|---|---|"]
        for key, lab, _ in METRICS:
            if key not in rows:
                continue
            d, mean, sd, w, n = rows[key]
            sc = 1 if key == "mean_jaccard" else 100
            unit = "" if key == "mean_jaccard" else " pp"
            fmt = "{:+.2f}" if key != "mean_jaccard" else "{:+.3f}"
            L.append(f"| {lab} | " + ", ".join(fmt.format(x * sc) for x in d) +
                     f" | {fmt.format(mean * sc)} ± {abs(sd * sc):.2f}{unit} | "
                     f"{w}/{n} | {tier(w, n, a, b)} |")
        L += ["", f"觸發分支：{BRANCH.get(name, '—')}", ""]
        if name in NOTES:
            L += [NOTES[name], ""]

    L += ["---", "",
          "產生：`python scripts/report_dr046_gates.py`。"
          "數值由 `scripts/verify_doc_numbers.py` 溯源把關。", ""]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n")
    print(f"→ {OUT}")
    for name, _d, arch, a, b in GATES:
        recs = load(arch)
        r = paired(recs, a, b, "final_task_il", True, ls)
        if r:
            print(f"  {name:10s} {a}−{b} task-IL {r[1] * 100:+.2f} ({r[3]}/{r[4]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
