"""G3 / G4 / G5 —— 架構完整性報告。

產生 `outputs/exp2/arch/ARCH_COMPLETENESS.md`：
主表（四臂 × 指標）、三組配對比較、每個實驗的 pre-registered 判準與**自動落判**、
以及 G5 的 no-op 檢查結果。

⚠️ 判準是**字面常數**（PRE_REGISTERED），原文取自 PROMPT
G345-ARCH-COMPLETENESS-20260824，在看到任何結果之前就寫死。
落判由 `verdict()` 依 win count 三級規則自動產生，不得手改。
`tests/test_arch_report.py` 會比對常數與落判邏輯。
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_exp2 import ORDERS, Ctx, arm_metrics, ms                     # noqa: E402
from selector.text_encoder import load_config                          # noqa: E402

OUT_DIR = ROOT / "outputs" / "exp2" / "arch"
BASELINE_DIR = ROOT / "outputs" / "exp2" / "hier2" / "per_slide"
ORDER = "reverse"

#: (key, 顯示名稱, 來源目錄, arm, arch)
BASELINE = ("base", "hier-A5 基準（G1'）", BASELINE_DIR, "A5", "hier")
EXPERIMENTS = [
    ("G5", "+ E_t / B_t 狀態條件化", OUT_DIR / "per_slide", "A5", "hier_state"),
    ("G4", "+ q_tau 任務條件化", OUT_DIR / "per_slide", "A5", "hier_query"),
    ("G3", "+ group 層 L_sem (beta_g=0.1)", OUT_DIR / "per_slide", "A5g", "hier"),
]

#: 主要指標（PI pre-register）。次要指標另列。
PRIMARY = [("final_task_il", "task-IL final avg"),
           ("final_class_il", "class-IL final avg")]
SECONDARY = [("mean_leak", "跨任務洩漏率", False),
             ("mean_jaccard", "selection Jaccard", True),
             ("mean_quota_kl", "group 配額 KL", False)]

#: PI 原文判準。看到結果之後不得修改。
PRE_REGISTERED = {
    "G5": {
        "pass": "win >= 4/5 且配對為正（任一準確率軸）→ stateful 成立，圖與 "
                "\"Beyond HistoSelect\" 保留該條，論文可寫 state-conditioned "
                "sequential acquisition（仍不得寫 \"plan\"，因為輪間 detach）",
        "fail": "<= 3/5 或為負 → 從架構圖移除 Panel E 與 \"stateful\" 一詞，改寫為 "
                "budgeted top-K selection under a shared frozen head，並在 "
                "limitation 說明：在此設定下狀態條件化未帶來可測增益。不得調參搶救。",
    },
    "G4": {
        "pass": "win >= 4/5 且為正 → 圖上保留 q_tau，\"task-conditioned\" 一詞成立",
        "fail": "<= 3/5 → 圖上把 q_tau 標為 optional 或移除，論文寫：在跨器官任務序列中"
                "任務身分可由視覺特徵推得（S1，98.2/98.6%），顯式語意條件化在 patch "
                "排序與群組配額兩個層級皆未提供增益。這是有機制解釋的 null，不是失敗。"
                "同器官設定列入 future work。",
    },
    "G3": {
        "pass": "win >= 4/5 且為正 → 報告為有效變體，論文寫「完整兩層 L_sem 帶來 __ pp 增益」",
        "fail": "<= 3/5 → 論文寫有數據的發現：在 classification 設定下 group 層語意先驗"
                "不提供增益，因為 q_tau 在任務內為常數，cos(g_j, q_tau) 退化為 8 個群組的"
                "靜態排序；HistoSelect 的 q 為逐問題變動，此差異來自 query 性質而非機制本身。",
    },
}

#: G5 的判準寫明「任一準確率軸」；G4 / G3 只寫「為正」。本報告對三者採同一讀法，
#: 因為主要指標在通用設定裡就定義為 task-IL 與 class-IL 兩個 final avg。
AXIS_RULE = "任一準確率軸（task-IL 或 class-IL final avg）滿足即算通過"


def load(root: Path, arm: str, arch: str) -> list[dict]:
    if not root.exists():
        return []
    return [r for p in sorted(root.glob("*.json")) for r in json.loads(p.read_text())
            if r["arm"] == arm and r.get("arch") == arch
            and r.get("allocation") == "per_budget" and r.get("order") == ORDER]


def seeds_of(recs) -> list[int]:
    return sorted({r["seed"] for r in recs})


def tier(wins: int, n: int) -> str:
    """DR-020 三級規則。"""
    if wins == n:
        return "systematic"
    if n - wins == 1:
        return "directional, inconclusive"
    return "within noise"


def verdict(pairs: dict) -> tuple[str, list[str]]:
    """依 pre-registered 判準自動落判。

    pairs: {metric_key: (mean_diff, wins, n)}。
    通過條件 = 任一準確率軸 win >= 4/5 **且** 配對平均為正。
    """
    reasons, ok = [], False
    for key, label in PRIMARY:
        if key not in pairs:
            continue
        mean, wins, n = pairs[key]
        hit = wins >= 4 and mean > 0 and n >= 5
        ok = ok or hit
        reasons.append(f"{label}：配對 {mean * 100:+.2f} pp、win {wins}/{n}"
                       f"（{tier(wins, n)}）→ {'滿足' if hit else '不滿足'}")
    return ("PASS" if ok else "FAIL"), reasons


def collect(ctx, root: Path, arm: str, arch: str):
    recs = load(root, arm, arch)
    if not recs:
        return None
    seeds = seeds_of(recs)
    tasks = ORDERS[ORDER]
    return {"seeds": seeds,
            "M": {s: arm_metrics(recs, arm, tasks, s, ctx.label_space) for s in seeds}}


def paired(base, exp) -> dict:
    """同 seed 相減。只用兩邊共同的 seed（憲法 §1.3 / DR-034）。"""
    common = sorted(set(base["seeds"]) & set(exp["seeds"]))
    out = {}
    for key, _ in PRIMARY:
        d = [exp["M"][s][key] - base["M"][s][key] for s in common
             if base["M"][s].get(key) is not None and exp["M"][s].get(key) is not None]
        if d:
            out[key] = (statistics.mean(d), sum(x > 0 for x in d), len(d))
    for key, _, higher in SECONDARY:
        d = [exp["M"][s][key] - base["M"][s][key] for s in common
             if base["M"][s].get(key) is not None and exp["M"][s].get(key) is not None]
        if d:
            out[key] = (statistics.mean(d),
                        sum((x > 0) if higher else (x < 0) for x in d), len(d))
    out["_common"] = common
    return out


def main() -> int:
    cfg = load_config()
    ctx = Ctx(cfg)
    noop = json.loads((OUT_DIR / "noop_check.json").read_text()) \
        if (OUT_DIR / "noop_check.json").exists() else None

    base = collect(ctx, BASELINE[2], BASELINE[3], BASELINE[4])
    if base is None:
        raise SystemExit(f"找不到基準資料：{BASELINE[2]}（arm={BASELINE[3]}）")
    got = {k: collect(ctx, root, arm, arch) for k, _, root, arm, arch in EXPERIMENTS}

    L = ["# G3 / G4 / G5 —— 架構完整性",
         "",
         "圖上有三個元件在所有主結果中都是關閉或未實作的：E_t/B_t 狀態迴圈、"
         "q_tau 任務條件化、group 層 L_sem。本檔把它們逐一測出來 —— **有效就保留，"
         "無效就從圖上移除並寫成有數據的 limitation**。",
         "",
         f"通用設定：架構 hier（per_budget）、|M|=512、B=8、c=1、order={ORDER}、"
         "5 seeds、epochs 5、lr 1e-3、prior=discriminative、beta_s=0.1、beta_u=0.1、"
         "λ 全 1.0。**每個實驗只動一個變因**，對照組沿用 G1' 已有的 hier-A5 存檔"
         "（不重跑）。",
         "",
         f"**判準讀法**：{AXIS_RULE}。G5 的原文明寫「任一準確率軸」；G4 / G3 只寫"
         "「為正」，本報告對三者採同一讀法，因為通用設定已把主要指標定義為"
         "task-IL 與 class-IL 兩個 final avg。",
         ""]

    # ── G5 no-op 檢查 ──
    L += ["## G5 前置：no-op 檢查", ""]
    if noop:
        off, on = noop["state_off"], noop["state_on"]
        n = noop["config"]["n_trials"]
        L += [f"判準：{noop['criterion']}。",
              "",
              f"設定：{noop['config']['note']}，{n} 組、B={noop['config']['budget']}、"
              f"arch={noop['config']['arch']}。比較 c=1 跑八輪 vs c=8 跑一輪。",
              "",
              "| use_state | 選取集合相同 | 選取順序相同 | 判讀 |",
              "|---|---|---|---|",
              f"| False（現行主線） | {off['same_set']}/{n} | {off['same_order']}/{n} |"
              f" **no-op** —— 與 CLAIMS C-01 一致 |",
              f"| True（G5） | {on['same_set']}/{n} | {on['same_order']}/{n} |"
              f" {'**非 no-op**，state 確實進入計算' if not on['is_no_op'] else '**仍為 no-op**'} |",
              "",
              f"**判定：{noop['verdict']}** —— G5 可以進行。",
              "",
              f"⚠️ {noop['caveat']}",
              "",
              f"產物：`outputs/exp2/arch/noop_check.json`",
              ""]
    else:
        L += ["⚠️ 尚未執行（`python scripts/check_state_noop.py`）。", ""]

    # ── 主表 ──
    heads = [lab for _, lab in PRIMARY] + [lab for _, lab, _ in SECONDARY]
    L += ["## 主表", "",
          "| 臂 | 說明 | seeds | " + " | ".join(heads) + " |",
          "|---|---|---|" + "---|" * len(heads)]

    def row(key, name, data):
        if data is None:
            return f"| {key} | {name} | — | " + " | ".join("—" for _ in heads) + " |"
        cells = [ms([data["M"][s].get(k) for s in data["seeds"]])
                 for k, _ in PRIMARY]
        cells += [ms([data["M"][s].get(k) for s in data["seeds"]])
                  for k, _, _ in SECONDARY]
        return f"| {key} | {name} | {len(data['seeds'])} | " + " | ".join(cells) + " |"

    L.append(row("base", BASELINE[1], base))
    for k, name, *_ in EXPERIMENTS:
        L.append(row(k, name, got[k]))
    L += ["",
          "前兩欄是 pre-registered 的主要指標；後三欄為次要診斷。"
          "final avg 算全部 4 個 task；Jaccard 與配額 KL 只算前 3 個（CL 慣例）。",
          ""]

    # ── 逐實驗判定 ──
    L += ["## 配對比較與落判", "",
          "臂間比較一律**配對**（同 seed 相減）。win count 三級規則（DR-020）："
          "5/5 = systematic、4/5 = directional inconclusive、≤3/5 = within noise。"
          "**不報 p 值**（DR-016）。", ""]

    summary = []
    for k, name, *_ in EXPERIMENTS:
        L += [f"### {k}　{name}", ""]
        if got[k] is None:
            L += ["⚠️ 尚未有資料。", ""]
            summary.append((k, name, "PENDING"))
            continue
        pr = paired(base, got[k])
        common = pr.pop("_common")
        L += [f"共同 seeds：{common}（n={len(common)}）", "",
              "| 指標 | 逐 seed 配對差值 | 配對 mean ± std | win count | 三級判讀 |",
              "|---|---|---|---|---|"]
        for key, label in PRIMARY + [(a, b) for a, b, _ in SECONDARY]:
            if key not in pr:
                continue
            mean, wins, n = pr[key]
            d = [got[k]["M"][s][key] - base["M"][s][key] for s in common]
            scale = 100 if key in ("final_task_il", "final_class_il", "mean_leak") else 1
            unit = " pp" if scale == 100 else ""
            sd = statistics.stdev(d) if len(d) > 1 else 0.0
            L.append(f"| {label} | {', '.join(f'{x * scale:+.3f}' for x in d)} | "
                     f"{mean * scale:+.3f} ± {sd * scale:.3f}{unit} | {wins}/{n} | "
                     f"{tier(wins, n)} |")
        v, reasons = verdict(pr)
        L += ["", "**pre-registered 判準（原文，先於結果寫定）**：", "",
              f"- 通過 → {PRE_REGISTERED[k]['pass']}",
              f"- 未通過 → {PRE_REGISTERED[k]['fail']}", "",
              "**落判依據**：", ""]
        L += [f"- {r}" for r in reasons]
        L += ["", f"**判定：{v}** → "
              + (PRE_REGISTERED[k]["pass"] if v == "PASS"
                 else PRE_REGISTERED[k]["fail"]), ""]
        summary.append((k, name, v))

    L += ["## 總結", "", "| 實驗 | 變因 | 判定 | 對架構圖的處置 |", "|---|---|---|---|"]
    action = {
        ("G5", "PASS"): "保留 Panel E 與 \"stateful\"",
        ("G5", "FAIL"): "移除 Panel E 與 \"stateful\"，改寫為 budgeted top-K selection",
        ("G4", "PASS"): "保留 q_tau，\"task-conditioned\" 成立",
        ("G4", "FAIL"): "q_tau 標為 optional 或移除；寫成有機制解釋的 null",
        ("G3", "PASS"): "作為有效變體報告，主方法仍維持 patch-only",
        ("G3", "FAIL"): "維持 patch-only；寫成有數據的發現",
    }
    for k, name, v in summary:
        L.append(f"| {k} | {name} | **{v}** | {action.get((k, v), '—')} |")
    L += ["",
          "⚠️ G3 不論結果如何都**不回頭改主表** —— DR-007 pre-register 的是「用哪個 "
          "prior」，不是「用哪幾層」；本實驗是新增的消融維度。",
          "",
          "逐 slide 預測：`outputs/exp2/arch/per_slide/*.json`（基準在 "
          "`outputs/exp2/hier2/per_slide/`）。",
          ""]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "ARCH_COMPLETENESS.md").write_text("\n".join(L) + "\n")
    print(f"→ {OUT_DIR / 'ARCH_COMPLETENESS.md'}")
    for k, name, v in summary:
        print(f"  {k} {name}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
