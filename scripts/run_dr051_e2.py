#!/usr/bin/env python3
"""DR-051 E2：門控對照臂 `A5ce` —— 把 hinge 換成同一張重播切片上的**常開等權 CE**。

    python scripts/run_dr051_e2.py                # 生效性實測 → 訓練 5 seed → 配對報表
    python scripts/run_dr051_e2.py --report-only  # 只重產報表
    python scripts/run_dr051_e2.py --smoke        # 煙霧（--max-train 2 --epochs 1）

**預註冊（DR-051，凍結後才啟動）**：與 A5 flat 唯一差異是 `L_eq`：
    hinge:  max(0, U_old − U_new)，只在退步時罰
    A5ce :  CE(logits_uniform, y) = log C − U_new，**常開**，λ = λ_eq = 1.0
replay、KD、LoRA、記憶體、seed、epochs、lr、fold 全部不變。

**零改主線程式碼**：`selector/train.py::continual_terms` 以模組全域 `l_eq` 呼叫
（`from selector.continual import l_eq`），本檔只把 `selector.train.l_eq` 換成
`log C − u_new`，訓練路徑一行未動。`u_old` 被忽略 —— 這正是「常開」的定義。

**生效性實測（憲法 §2.9，啟動前）**：
1. `continual_terms.__globals__["l_eq"]` 必須是注入後的函式（不是原 hinge）；
2. 隨機 logits 上，注入版的值 == 獨立算的 `F.cross_entropy`（容差 1e-6）；
3. 同一輸入下注入版與原 hinge 的值**不同**（U_old < U_new 時 hinge = 0，CE > 0）；
4. 訓練結束後注入函式的呼叫次數 > 0，且 per_slide 的 `l_eq_fire_rate` 在 stage 1–3
   **恆為 1.0**（CE > 0 → 每步都「觸發」）—— 記錄層的第二道證據。

判準（凍結於 DR-051）：與 A5 flat（`outputs/exp2/main/`）逐 seed 配對，五軸
（class-IL final、task-IL final、洩漏率、Jaccard、ΔU：S 現行與 M1 並報），
報 A5ce − A5 的 mean ± sd 與勝場 k/5，讀法依 DR-020 三級。**結果照報，不寫「等價」。**
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_exp2 as R                                                     # noqa: E402
import selector.train as T                                               # noqa: E402
from report_dr046 import delta_utility                                   # noqa: E402
from report_dr051_e0 import per_slide_deltas                             # noqa: E402
from selector.continual import differentiable_utility, l_eq as l_eq_hinge  # noqa: E402
from selector.text_encoder import load_config                            # noqa: E402

ARM, BASE, ORDER, ARCH = "A5ce", "A5", "reverse", "flat"
TAG = "dr051_e2"
OUT_DIR = R.OUT_ROOT / TAG
REF_DIR = R.OUT_ROOT / "main" / "per_slide"
REPORT = R.OUT_ROOT / "dr051" / "E2_GATED_CONTROL.md"
SEEDS = [0, 1, 2, 3, 4]
#: 五軸（key in arm_metrics 或本檔算的 ΔU, 顯示名, 越大越好, 乘數）
AXES = [("final_class_il", "class-IL final", True, 100),
        ("final_task_il", "task-IL final", True, 100),
        ("mean_leak", "跨任務洩漏率", False, 100),
        ("mean_jaccard", "selection Jaccard", True, 1),
        ("dU_S", "ΔU（S 現行：逐任務加總再平均）", True, 1),
        ("dU_M1", "ΔU（M1：各任務先平均再跨任務平均）", True, 1)]


# ── 注入 ────────────────────────────────────────────────────────────────────

def register_arm() -> None:
    spec = dict(R.ARMS[BASE], name="Gated control: hinge → always-on equal-weight CE (λ=λ_eq)")
    assert spec["eq"] and spec["replay"] and spec["kd"] and spec["lora"]
    R.ARMS[ARM] = spec


def make_l_eq_ce(log_c: float):
    calls = {"n": 0}

    def l_eq_ce(u_new, u_old, mode="hinge"):
        calls["n"] += 1
        return log_c - u_new                 # = CE(logits_uniform, y)；常開；u_old 不用

    l_eq_ce.__calls__ = calls
    l_eq_ce.__log_c__ = log_c
    return l_eq_ce


def install(log_c: float):
    fn = make_l_eq_ce(log_c)
    T.l_eq = fn
    return fn


def effectiveness_check(fn, C: int) -> list[str]:
    out = []
    g = T.continual_terms.__globals__
    assert g["l_eq"] is fn, "continual_terms 看到的 l_eq 不是注入版 —— 注入無效"
    out.append("1. `continual_terms.__globals__['l_eq']` 是注入版 ✅")
    gen = torch.Generator().manual_seed(51)
    n_diff = 0
    for k in range(20):
        logits = torch.randn(1, C, generator=gen) * 3
        y = int(torch.randint(0, C, (1,), generator=gen))
        u_new = differentiable_utility(logits, y)
        ce_ref = float(F.cross_entropy(logits, torch.tensor([y])))
        got = float(fn(u_new, u_old=0.0))
        assert abs(got - ce_ref) <= 1e-6, f"注入版 {got} ≠ CE {ce_ref}"
        # hinge 在 U_old = U_new − 1 時為 0，CE 必 > 0
        h = float(l_eq_hinge(u_new, float(u_new) - 1.0))
        n_diff += int(abs(h - got) > 1e-6)
    out.append("2. 20 組隨機 logits：注入版 == `F.cross_entropy`（容差 1e-6）✅")
    assert n_diff == 20, f"注入版與原 hinge 只有 {n_diff}/20 組不同"
    out.append("3. 同一輸入下注入版與原 hinge 值不同 20/20（hinge 於未退步時為 0，CE > 0）✅")
    # 梯度方向：注入版的梯度 = +dCE/dθ（與 hinge 觸發時相同符號）
    logits = torch.randn(1, C, generator=gen, requires_grad=True)
    y = 0
    fn(differentiable_utility(logits, y), 0.0).backward()
    g1 = logits.grad.clone(); logits.grad = None
    F.cross_entropy(logits, torch.tensor([y])).backward()
    assert torch.allclose(g1, logits.grad, atol=1e-6)
    out.append("4. 對 logits 的梯度 == CE 的梯度 ✅")
    return out


# ── 訓練 ────────────────────────────────────────────────────────────────────

def train(seeds, extra: list[str]) -> int:
    cfg = load_config()
    C = len(cfg["tasks"]) * 2
    register_arm()
    fn = install(math.log(C))
    checks = effectiveness_check(fn, C)
    print("生效性實測：\n  " + "\n  ".join(checks), flush=True)
    sys.argv = ["run_exp2.py", "--arms", ARM, "--order", ORDER, "--arch", ARCH,
                "--fold", "1", "--seeds", ",".join(map(str, seeds)), "--tag", TAG] + extra
    rc = R.main()
    if rc != 0:
        raise SystemExit(f"❌ run_exp2.main 回傳 {rc}")
    n = fn.__calls__["n"]
    print(f"注入函式呼叫次數：{n}", flush=True)
    return n


def fire_rates(recs) -> dict[int, float | None]:
    out = {}
    for r in recs:
        out.setdefault(r["stage"], r.get("l_eq_fire_rate"))
    return out


# ── 報表 ────────────────────────────────────────────────────────────────────

def metrics(recs, arm, seed, tasks, label_space) -> dict:
    M = R.arm_metrics(recs, arm, tasks, seed, label_space)
    M["dU_S"] = delta_utility(M, tasks)
    d = per_slide_deltas(recs, arm, seed, tasks)
    M["dU_M1"] = statistics.mean([statistics.mean(v) for v in d.values()])
    return M


def _sd(v) -> float:
    return statistics.stdev(v) if len(v) > 1 else 0.0


def report(seeds, n_calls: int | None) -> None:
    cfg = load_config()
    label_space = list(cfg["tasks"])
    tasks = R.ORDERS[ORDER]
    A, B, fr = {}, {}, {}
    for s in seeds:
        pa = OUT_DIR / "per_slide" / f"{ARM}_{ORDER}_seed{s}.json"
        pb = REF_DIR / f"{BASE}_{ORDER}_seed{s}.json"
        if not pa.exists():
            raise SystemExit(f"❌ 缺 {pa}")
        ra, rb = json.loads(pa.read_text()), json.loads(pb.read_text())
        A[s] = metrics(ra, ARM, s, tasks, label_space)
        B[s] = metrics(rb, BASE, s, tasks, label_space)
        fr[s] = fire_rates(ra)
    L = ["# E2 — 門控對照：`A5ce`（常開等權 CE）vs `A5`（hinge）", "",
         "fold 1、reverse、seeds 0–4、flat。`A5ce` 與 A5 唯一差異：`L_eq` 由 "
         "`max(0, U_old − U_new)` 換成同一張重播切片上的 `CE(logits_uniform, y)`（常開，λ = λ_eq = 1）。"
         "實作為 runtime injection（`scripts/run_dr051_e2.py`），主線零改。", "",
         "判準（DR-051 凍結）：逐 seed 配對，報 **A5ce − A5** 的 mean ± sd 與勝場 k/5；"
         "「勝」= A5ce 在該軸較佳（洩漏率越低越佳，其餘越大越佳）。讀法依 DR-020 三級："
         "5/5 systematic、4/5 directional、≤3/5 within noise。**結果照報。**", "",
         "## 生效證據", ""]
    fr_ok = all(fr[s].get(st) == 1.0 for s in seeds for st in (1, 2, 3))
    L.append(f"* per_slide `l_eq_fire_rate`（stage 1/2/3）在 A5ce 恆為 1.0："
             f"{'✅' if fr_ok else '❌'} " +
             "; ".join(f"s{s}: " + "/".join(f"{fr[s].get(st)}" for st in (1, 2, 3)) for s in seeds))
    if n_calls is not None:
        L.append(f"* 注入函式在本次訓練被呼叫 {n_calls:,} 次（= 全部 replay 步數）")
    L += ["* 啟動前四項生效性實測（注入可見、值 == CE、與 hinge 不同、梯度 == CE 梯度）"
          "由本腳本在訓練前執行，任一失敗即不啟動。", "",
          "## 五軸配對（A5ce − A5；逐 seed）", "",
          "| 軸 | A5（hinge） | A5ce（常開 CE） | 差 mean ± sd | 逐 seed 差 | A5ce 較佳 |",
          "|---|---|---|---|---|---|"]
    summary = {}
    for key, name, hib, mult in AXES:
        da = [A[s][key] * mult for s in seeds]
        db = [B[s][key] * mult for s in seeds]
        d = [a - b for a, b in zip(da, db)]
        wins = sum((x > 0) if hib else (x < 0) for x in d)
        ties = sum(x == 0 for x in d)
        fmt = "{:.2f}" if mult == 100 else "{:.4f}"
        cell = lambda v: f"{fmt.format(statistics.mean(v))} ± {fmt.format(_sd(v))}"
        L.append(f"| {name} | {cell(db)} | {cell(da)} | **{('+' if statistics.mean(d) >= 0 else '')}"
                 f"{fmt.format(statistics.mean(d))} ± {fmt.format(_sd(d))}** | "
                 + ", ".join(('+' if x >= 0 else '') + fmt.format(x) for x in d)
                 + f" | **{wins}/{len(seeds)}**" + (f"（平 {ties}）" if ties else "") + " |")
        summary[key] = (statistics.mean(d), _sd(d), wins)
    L += ["", "單位：class-IL／task-IL／洩漏率為百分點（pp）；Jaccard 與 ΔU 為原始單位。", "",
          "## 逐任務（class-IL at end、Jaccard；A5ce vs A5，5 seed 平均）", "",
          "| 舊任務 | A5 class-IL@end | A5ce class-IL@end | A5 Jaccard | A5ce Jaccard |",
          "|---|---|---|---|---|"]
    for t in tasks:
        def m(D, k):
            return statistics.mean(D[s]["per_task"][t][k] for s in seeds)
        L.append(f"| {t.replace('tcga_', '')} | {m(B, 'class_il_at_end'):.4f} | "
                 f"{m(A, 'class_il_at_end'):.4f} | {m(B, 'jaccard'):.4f} | {m(A, 'jaccard'):.4f} |")
    L += ["", "## 逐軸結論（照判準的字面）", ""]
    for key, name, hib, mult in AXES:
        mean, sd, wins = summary[key]
        level = {5: "systematic", 4: "directional, inconclusive"}.get(wins, "within noise")
        better = "A5ce（常開 CE）較佳" if (mean > 0) == hib and mean != 0 else "A5（hinge）較佳"
        L.append(f"* {name}：A5ce − A5 = {mean:+.4f}，{better}方向，勝場 {wins}/{len(seeds)} → **{level}**")
    L += ["", "來源：`outputs/exp2/dr051_e2/per_slide/A5ce_reverse_seed{0..4}.json`、"
          "`outputs/exp2/main/per_slide/A5_reverse_seed{0..4}.json`。", ""]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(L) + "\n")
    print(f"→ {REPORT}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tag", default=None, help="改讀別的 tag（測試用）")
    ap.add_argument("--report", default=None, help="改寫到別的報表路徑（測試用）")
    a = ap.parse_args(argv)
    seeds = [int(x) for x in a.seeds.split(",")]
    global TAG, OUT_DIR, REPORT
    if a.tag:
        TAG = a.tag; OUT_DIR = R.OUT_ROOT / TAG
    if a.report:
        REPORT = Path(a.report)
    if a.smoke:
        TAG = "dr051_e2_smoke"; OUT_DIR = R.OUT_ROOT / TAG
        n = train(seeds, ["--max-train", "2", "--epochs", "1", "--no-resume"])
        recs = json.loads((OUT_DIR / "per_slide" / f"{ARM}_{ORDER}_seed{seeds[0]}.json").read_text())
        fr = fire_rates(recs)
        print(f"煙霧：呼叫 {n} 次；fire rate {fr}")
        return 0 if n > 0 and all(fr.get(st) == 1.0 for st in (1, 2, 3)) else 1
    n = None
    if not a.report_only:
        n = train(seeds, [])
    report(seeds, n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
