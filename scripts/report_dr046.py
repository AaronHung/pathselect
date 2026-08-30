#!/usr/bin/env python3
"""DR-046 Phase 0 — 零 GPU 離線指標報表（只讀既有紀錄，不訓練、不動任何結果檔）。

從 `outputs/exp2/main/per_slide/*.json` 重算 CL 文獻的標準指標，輸出
`docs/DR046_TABLE.md`。

⚠️ **不自創算法**：階段別遮罩、`lo` 索引、accuracy、Jaccard 全部沿用
`scripts/run_exp2.py` 的 `acc()` / `jac()` / `arm_metrics()`。本檔唯一新增的是
把它們排成 stage × task 的矩陣 a[s][j]，以及由該矩陣導出的四個彙總量。

⚠️ **自檢先行**：先用 `arm_metrics` 重算 A5 / A2 / A1 的最終 class-IL 與 task-IL，
與 `docs/RESULTS_DOSSIER.md` §4.4 既有數字比對（容差 5e-4）。**不符就整支停下**，
不寫出報表、不靜默修正。
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_exp2 import ORDERS, acc, arm_metrics, jac                   # noqa: E402
from selector.text_encoder import load_config                        # noqa: E402

SRC = ROOT / "outputs" / "exp2" / "main" / "per_slide"
OUT = ROOT / "docs" / "DR046_TABLE.md"
DOSSIER = ROOT / "docs" / "RESULTS_DOSSIER.md"
ORDER = "reverse"
TOL = 5e-4

#: 由構造不適用的欄位。R1 每 task 獨立訓練、R2 一次看完所有資料 → 各 stage 的
#: 結果相同，Forgetting 恆為 0、Jaccard 恆為 1（DR-011）。標「—」而不是印 0。
NOT_APPLICABLE = {"R1", "R2"}


# ── 讀檔 ────────────────────────────────────────────────────────────────────

def load_records() -> list[dict]:
    if not SRC.is_dir():
        raise SystemExit(f"找不到 {SRC}")
    return [r for f in sorted(SRC.glob("*.json")) for r in json.loads(f.read_text())
            if r.get("order") == ORDER]


#: 只用來決定**顯示順序**，不是白名單 —— 不在其中的臂一律照樣納入（DR-046 裁定二）。
DISPLAY_ORDER = ["A1", "A2", "A3", "A4", "A5", "A5nG", "W1", "W1B", "L2", "L2B",
                 "B1", "B2", "R1", "R2"]


def arms_present(recs, wanted: list[str] | None = None) -> list[str]:
    """per_slide 裡實際存在的所有臂（可用 --arms 縮小範圍）。"""
    have = {r["arm"] for r in recs}
    if wanted:
        missing = [a for a in wanted if a not in have]
        if missing:
            raise SystemExit(f"--arms 指定了 per_slide 裡沒有的臂：{missing}")
        have = set(wanted)
    return ([a for a in DISPLAY_ORDER if a in have]
            + sorted(have - set(DISPLAY_ORDER)))


# ── 指標矩陣 a[s][j] ────────────────────────────────────────────────────────

def matrix(recs, arm: str, seed: int, tasks: list[str], key: str):
    """a[s][j]：第 s 階段評第 j 個 task。未評到的格子為 None（下三角）。

    `acc()` 與 `pred_class_il` / `pred_task_il` 都是 run_exp2 既有的欄位與函式；
    `pred_task_il` 在寫檔時就已經套過該 task 的 `lo` 遮罩（見 run_exp2.evaluate），
    這裡不重做遮罩，避免出現第二套慣例。
    """
    sub = [r for r in recs if r["arm"] == arm and r["seed"] == seed]
    a = [[None] * len(tasks) for _ in tasks]
    for s in range(len(tasks)):
        for j, t in enumerate(tasks):
            cell = [r for r in sub if r["stage"] == s and r["task"] == t]
            if cell:
                a[s][j] = acc(cell, key)
    return a


def a_final(a) -> float:
    last = len(a) - 1
    v = [x for x in a[last] if x is not None]
    return statistics.mean(v) if v else float("nan")


def forgetting(a) -> float:
    """各舊 task 的 max_s a[s][j] − a[final][j] 的平均（只算 j < final）。

    ⚠️ `range(last)` 與下面的 `len(seen) >= 2` 是**兩道重疊的防禦**：下三角矩陣裡
    第 last 個 task 只在最後一個 stage 出現，就算把它納入迴圈也會被 `>= 2` 濾掉。
    mutation 測到「兩者只拿掉一個」不會改變輸出 —— 這是事實，不是測試漏洞，
    記在這裡免得日後有人以為其中一道可以刪。
    """
    last = len(a) - 1
    gaps = []
    for j in range(last):
        seen = [a[s][j] for s in range(last + 1) if a[s][j] is not None]
        if len(seen) >= 2 and a[last][j] is not None:
            gaps.append(max(seen) - a[last][j])
    return statistics.mean(gaps) if gaps else float("nan")


def plasticity(a) -> float:
    """對角線 a[j][j] 的平均 —— 剛學完該 task 當下的表現。"""
    v = [a[j][j] for j in range(len(a)) if a[j][j] is not None]
    return statistics.mean(v) if v else float("nan")


# ── 行為軸：Jaccard 與 ΔUtility ─────────────────────────────────────────────

def behaviour(recs, arm: str, seed: int, tasks: list[str]):
    """回傳 (selection Jaccard, ΔUtility)。

    兩者都只算前 T−1 個 task（最後一個 task 兩個時點是同一時點，
    算進去只會稀釋 —— 憲法 §3.1）。

    - **Selection Jaccard**：同一張 slide 的 `selected_idx` 自身階段 vs 最終階段，
      逐 slide 算後平均（沿用 run_exp2 的 `jac()`）。
    - **ΔUtility**：各舊 task 的 `(U_final − U_own)` 平均（DR-046 裁定二）。
      U 用 `arm_metrics` 既有的 `sum_u_at_end` / `sum_u_at_learn`，不另算一套。
      ⚠️ 原本是比值 `U_final / U_own`，已移除：`utility_total` 會變號
      （A1 的 esca 由 +22.0 掉到 −42.5），比值不是有界的保留率，會被誤讀為百分比。
    """
    last = len(tasks) - 1
    sub = [r for r in recs if r["arm"] == arm and r["seed"] == seed]
    jacs = []
    for j, t in enumerate(tasks[:last]):
        own = {r["slide_id"]: r for r in sub if r["stage"] == j and r["task"] == t}
        fin = {r["slide_id"]: r for r in sub if r["stage"] == last and r["task"] == t}
        for sid, o in own.items():
            if sid in fin:
                jacs.append(jac(o["selected_idx"], fin[sid]["selected_idx"]))
    return statistics.mean(jacs) if jacs else float("nan")


def delta_utility(M_seed, tasks: list[str]) -> float:
    """ΔUtility = 各舊 task 的 (U_final − U_own) 平均。越高越好，負值 = 退化。"""
    per = M_seed.get("per_task", {})
    d = [per[t]["sum_u_at_end"] - per[t]["sum_u_at_learn"]
         for t in tasks[:-1] if t in per]
    return statistics.mean(d) if d else float("nan")


# ── 自檢：與 RESULTS_DOSSIER §4.4 比對 ──────────────────────────────────────

def dossier_values() -> dict[str, tuple[float, float]]:
    """從 §4.4 主表抓 (task-IL, class-IL)。回傳 {arm: (task_il, class_il)}。"""
    out = {}
    for ln in DOSSIER.read_text(encoding="utf-8").splitlines():
        if not ln.startswith("|"):
            continue
        cells = [c.strip().replace("**", "") for c in ln.strip().strip("|").split("|")]
        if len(cells) < 3:
            continue
        m = re.match(r"^(A[1-5]|R[12])\b", cells[0])
        if not m:
            continue
        try:
            ti, ci = float(cells[1]), float(cells[2])
        except ValueError:
            continue
        if 0.0 <= ti <= 1.0 and 0.0 <= ci <= 1.0:
            out.setdefault(m.group(1), (ti, ci))
    return out


def self_check(recs, tasks, label_space) -> list[str]:
    """回傳比對結果的說明行；任何一項超出容差就 raise SystemExit。"""
    ref = dossier_values()
    lines = ["| 臂 | 指標 | 本檔重算 | RESULTS_DOSSIER §4.4 | 差 | 判定 |",
             "|---|---|---|---|---|---|"]
    bad = []
    for arm in ("A5", "A2", "A1"):
        if arm not in ref:
            bad.append(f"{arm}：RESULTS_DOSSIER §4.4 找不到對照值")
            continue
        seeds = sorted({r["seed"] for r in recs if r["arm"] == arm})
        M = {s: arm_metrics(recs, arm, tasks, s, label_space) for s in seeds}
        got = {"task-IL": statistics.mean([M[s]["final_task_il"] for s in seeds]),
               "class-IL": statistics.mean([M[s]["final_class_il"] for s in seeds])}
        for i, name in enumerate(("task-IL", "class-IL")):
            want = ref[arm][i]
            d = abs(got[name] - want)
            ok = d <= TOL
            lines.append(f"| {arm} | {name} | {got[name]:.6f} | {want:.4f} | "
                         f"{d:.2e} | {'✅' if ok else '❌'} |")
            if not ok:
                bad.append(f"{arm} {name}：重算 {got[name]:.6f} vs 總表 {want:.4f}"
                           f"（差 {d:.2e} > 容差 {TOL}）")
    if bad:
        print("❌ 自檢未通過，整支停下，不產出報表：")
        for b in bad:
            print(f"  - {b}")
        raise SystemExit(1)
    return lines


# ── 主流程 ──────────────────────────────────────────────────────────────────

COLS = ["A_Final", "Forgetting", "Plasticity", "Selection Jaccard", "ΔUtility"]


def fmt(v) -> str:
    return "—" if v is None else ("nan" if v != v else f"{v:.4f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arms", default="", help="逗號分隔；預設為 per_slide 內的全部臂")
    args = ap.parse_args(argv)
    wanted = [a.strip() for a in args.arms.split(",") if a.strip()]

    cfg = load_config()
    label_space = list(cfg["tasks"])
    tasks = ORDERS[ORDER]
    recs = load_records()
    arms = arms_present(recs, wanted)
    print(f"讀入 {len(recs)} 筆（order={ORDER}）；臂 = {arms}")

    check_lines = self_check(recs, tasks, label_space)
    print("✅ 自檢通過（A5 / A2 / A1 的最終 class-IL 與 task-IL 皆在容差內）")

    per_arm = {}
    for arm in arms:
        seeds = sorted({r["seed"] for r in recs if r["arm"] == arm})
        rows = {}
        for s in seeds:
            ac = matrix(recs, arm, s, tasks, "pred_class_il")
            at = matrix(recs, arm, s, tasks, "pred_task_il")
            jc = behaviour(recs, arm, s, tasks)
            M_seed = arm_metrics(recs, arm, tasks, s, label_space)
            na = arm in NOT_APPLICABLE
            rows[s] = {
                "A_Final": a_final(ac), "A_Final_task": a_final(at),
                "Forgetting": None if na else forgetting(ac),
                "Forgetting_task": None if na else forgetting(at),
                "Plasticity": plasticity(ac), "Plasticity_task": plasticity(at),
                "Selection Jaccard": None if na else jc,
                "ΔUtility": None if na else delta_utility(M_seed, tasks),
            }
        per_arm[arm] = (seeds, rows)

    L = ["# DR-046 Phase 0 — 離線 CL 指標表", "",
         f"來源：`outputs/exp2/main/per_slide/*.json`（order = **{ORDER}**，"
         f"flat 架構、B=8、c=1、\\|M\\|=512）。**純重算，未訓練、未改動任何結果檔。**", "",
         "**算法沿用 `scripts/run_exp2.py`** 的 `acc()` / `jac()` / `arm_metrics()` ——"
         "階段別遮罩與 `lo` 索引不另立一套。本檔只是把它們排成 stage × task 的"
         "矩陣 a[s][j] 並導出彙總量。", "",
         "## 指標定義", "",
         "| 欄位 | 定義 |", "|---|---|",
         "| **A_Final** | 最終階段所有 task 的平均 accuracy，`mean_j a[T−1][j]` |",
         "| **Forgetting** | 各舊 task 的 `max_s a[s][j] − a[T−1][j]` 平均（只算 j < T−1）"
         "。正值 = 退步 |",
         "| **Plasticity** | 對角線 `mean_j a[j][j]`，剛學完該 task 當下的表現 |",
         "| **Selection Jaccard** | 同一張 slide 的 `selected_idx`：自身階段 vs 最終階段，"
         "逐 slide 算後平均（沿用檔內 `jac()`），只算前 T−1 個 task |",
         "| **ΔUtility** | 各舊 task 的 `U_final − U_own` 平均（U 用 `arm_metrics` 既有的 "
         "`sum_u_at_end` / `sum_u_at_learn`）。**越高越好，負值 = 退化** |",
         "",
         "⚠️ **為什麼是差值不是比值（DR-046 裁定二）**：`utility_total` **會變號** "
         "—— A1 的 esca 由 +22.0 掉到 −42.5。比值 `U_final / U_own` 因此可能為負或"
         "絕對值極大，不是有界的保留率，放在同一張表上會被誤讀為百分比。"
         "原本的 Utility Retention 欄已移除。ΔUtility 是**加總後的差**（單位與 ΣU 相同），"
         "不是比例。",
         "",
         "⚠️ **R1 / R2 的 Forgetting / Jaccard / ΔUtility 標「—」**："
         "R1 每 task 獨立訓練、R2 一次看完所有資料，各 stage 結果相同，"
         "這三欄由構造分別恆為 0 / 1 / 1，印出來會被誤讀為「不遺忘」（DR-011）。",
         "",
         "## 自檢：與 `docs/RESULTS_DOSSIER.md` §4.4 比對（容差 5e-4）", ""]
    L += check_lines
    L += ["", "自檢未通過時腳本會直接中止、不產出本表（規格 C-8）。", "",
          "## 逐 seed 明細（class-IL）", ""]

    head = "| 臂 | seed | " + " | ".join(COLS) + " |"
    L += [head, "|---|---|" + "---|" * len(COLS)]
    for arm in arms:
        seeds, rows = per_arm[arm]
        for s in seeds:
            L.append(f"| {arm} | {s} | " +
                     " | ".join(fmt(rows[s][c]) for c in COLS) + " |")
    L += ["", "## 彙總（mean ± sd，class-IL）", "",
          "| 臂 | n seeds | " + " | ".join(COLS) + " |",
          "|---|---|" + "---|" * len(COLS)]
    for arm in arms:
        seeds, rows = per_arm[arm]
        cells = []
        for c in COLS:
            v = [rows[s][c] for s in seeds if rows[s][c] is not None
                 and rows[s][c] == rows[s][c]]
            if not v:
                cells.append("—")
            else:
                sd = statistics.stdev(v) if len(v) > 1 else 0.0
                cells.append(f"{statistics.mean(v):.4f} ± {sd:.4f}")
        L.append(f"| {arm} | {len(seeds)} | " + " | ".join(cells) + " |")

    L += ["", "## 彙總（mean ± sd，task-IL）", "",
          "| 臂 | n seeds | A_Final | Forgetting | Plasticity |",
          "|---|---|---|---|---|"]
    for arm in arms:
        seeds, rows = per_arm[arm]
        cells = []
        for c in ("A_Final_task", "Forgetting_task", "Plasticity_task"):
            v = [rows[s][c] for s in seeds if rows[s][c] is not None
                 and rows[s][c] == rows[s][c]]
            if not v:
                cells.append("—")
            else:
                sd = statistics.stdev(v) if len(v) > 1 else 0.0
                cells.append(f"{statistics.mean(v):.4f} ± {sd:.4f}")
        L.append(f"| {arm} | {len(seeds)} | " + " | ".join(cells) + " |")

    L += ["", f"產生：`python scripts/report_dr046.py`（DR-046 Phase 0）。", ""]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n")
    print(f"→ {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
