#!/usr/bin/env python3
"""DR-051 E3：選片 vs 加權拆解（只評估）—— 以 runtime injection 重跑 A5 並存各 stage 權重。

    python scripts/run_dr051_e3.py                 # 重跑（存 ckpt）+ 分析
    python scripts/run_dr051_e3.py --analyze-only  # 只分析既有 ckpt / per_slide

為什麼要重跑：`scripts/run_exp2.py` 不存 selector 權重（只落逐 slide 紀錄），
而拆解的 (c)(d) 兩格需要**最終** selector 對舊任務切片的分數 s_new。
**零改主線程式碼**：本檔只把 `run_exp2.evaluate` 換成「先存 state_dict、再呼叫原函式」
的包裝，訓練路徑一行未動。

自檢（任一失敗即停）：
1. 重跑的 per_slide 與 `outputs/exp2/main/` 既有紀錄在共同欄位上**逐筆相同**
   （selected_idx / weights_softmax / pred_* / utility_total / l_eq_fire_rate）——
   證明重跑就是當初那個模型；
2. 載入最終 ckpt 後 `run_rounds` 重現 stage-3 紀錄的 selected_idx 與權重 —— 證明 ckpt
   就是產生那份紀錄的權重；
3. (a) 由 ckpt 直接算的 CE_uniform(P_own) 與紀錄的 `log C − utility_total` 相符；
   (d) 由 ckpt 算的 CE 與紀錄 `weights_softmax` 重算的 CE 相符。

四格（舊任務 test 切片，逐任務；P_own = 學完該任務時選的 8 片，P_final = 學完全部後選的）：
  (a) P_own ＋ 等權        (b) P_final ＋ 等權
  (c) P_own ＋ s_new 加權   (d) P_final ＋ s_new 加權（= 實際推論）
(a)→(b) 只換選片、權重等權 = **選片改變**；(b)→(d) 同一組片、換成 s_new 加權 = **重加權**。
描述性，不設判準。
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_exp2 as R                                                     # noqa: E402
from selector.classifier import conch_classify, softmax_weights          # noqa: E402
from selector.rounds import run_rounds                                   # noqa: E402
from selector.text_encoder import load_config                            # noqa: E402

ARM, ORDER = "A5", "reverse"
TAG = "dr051_e3"
OUT_DIR = R.OUT_ROOT / TAG
CKPT = OUT_DIR / "ckpt"
REPORT = R.OUT_ROOT / "dr051" / "E3_DECOMPOSITION.md"
REF_DIR = R.OUT_ROOT / "main" / "per_slide"          # 既有的 A5 flat fold-1 紀錄
COMMON = ("selected_idx", "weights_softmax", "pred_class_il", "pred_task_il",
          "utility_total", "l_eq_fire_rate", "l_eq_steps", "group_quota")
TOL = 1e-9


# ── 注入：存 ckpt ─────────────────────────────────────────────────────────────

def install_ckpt_hook(arch: str) -> None:
    orig = R.evaluate
    saved = set()

    def wrapped(ctx, models, task, arm, order_name, seed, stage, args, diag=None):
        key = (arm, order_name, seed, stage)
        if key not in saved:
            CKPT.mkdir(parents=True, exist_ok=True)
            sfx = "" if arch == R.DEFAULT_ARCH else f"_{arch}"
            p = CKPT / f"{arm}_{order_name}_seed{seed}_stage{stage}{sfx}.pt"
            torch.save({"f_g": models[0].state_dict(), "f_p": models[1].state_dict(),
                        "arm": arm, "seed": seed, "stage": stage, "arch": arch},
                       p)
            saved.add(key)
            print(f"       💾 ckpt → {p.name}", flush=True)
        return orig(ctx, models, task, arm, order_name, seed, stage, args, diag)

    wrapped.__wrapped_orig__ = orig
    R.evaluate = wrapped


def train(seeds: list[int], arch: str, extra: list[str]) -> None:
    install_ckpt_hook(arch)
    assert getattr(R.evaluate, "__wrapped_orig__", None) is not None, "ckpt hook 未生效"
    sys.argv = ["run_exp2.py", "--arms", ARM, "--order", ORDER, "--arch", arch,
                "--fold", "1", "--seeds", ",".join(map(str, seeds)), "--tag", TAG] + extra
    rc = R.main()
    if rc != 0:
        raise SystemExit(f"❌ run_exp2.main 回傳 {rc}")


# ── 分析 ────────────────────────────────────────────────────────────────────

def load_recs(d: Path, seed: int, arch: str) -> list[dict]:
    sfx = "" if arch == R.DEFAULT_ARCH else f"_{arch}"
    p = d / f"{ARM}_{ORDER}_seed{seed}{sfx}.json"
    if not p.exists():
        raise SystemExit(f"❌ 缺 {p}")
    return json.loads(p.read_text())


def check_consistency(new: list[dict], ref: list[dict]) -> tuple[int, list[str]]:
    """重跑 vs 既有紀錄：共同欄位逐筆相同。回傳 (比對筆數, 差異清單)。"""
    key = lambda r: (r["stage"], r["task"], r["slide_id"])
    by = {key(r): r for r in ref}
    n, bad = 0, []
    for r in new:
        k = key(r)
        if k not in by:
            bad.append(f"{k}: 既有紀錄沒有這筆"); continue
        n += 1
        for f in COMMON:
            a, b = r.get(f), by[k].get(f)
            if isinstance(a, float) and isinstance(b, float):
                same = abs(a - b) <= TOL
            else:
                same = a == b
            if not same:
                bad.append(f"{k} {f}: 重跑 {a!r} vs 既有 {b!r}")
    return n, bad


def ce(logits: torch.Tensor, label: int) -> float:
    return float(F.cross_entropy(logits.reshape(1, -1),
                                 torch.tensor([label], dtype=torch.long)))


def analyze(seeds: list[int], arch: str, args) -> None:
    cfg = load_config()
    cfg["fold"] = 1
    ctx = R.Ctx(cfg)
    tasks = R.ORDERS[ORDER]
    last = len(tasks) - 1
    logC = math.log(ctx.f_txt.shape[0])
    ns = argparse.Namespace(budget=8, chunk=1, allocation="per_budget", arch=arch)

    L = ["# E3 — 選片 vs 加權拆解（A5 full、flat、fold 1、reverse、seeds 0–4；只評估）", "",
         "來源：`outputs/exp2/dr051_e3/`（`scripts/run_dr051_e3.py` 重跑 A5 並存各 stage "
         "權重；訓練路徑零改）。**描述性，不設判準。**", "",
         "四格 CE（舊任務 test 切片；P_own = 學完該任務時選的 B=8 片，P_final = 學完四個任務後"
         "選的；s_new = 最終 selector 的分數；等權 = 1/8）：", "",
         "| 格 | 選片 | 加權 |", "|---|---|---|",
         "| (a) | P_own | 等權 |", "| (b) | P_final | 等權 |",
         "| (c) | P_own | softmax(s_new) |", "| (d) | P_final | softmax(s_new)（= 實際推論） |", "",
         "(a)→(b) = **選片改變**（同等權）；(b)→(d) = **重加權**（同 P_final）；"
         "(a)→(c) = 重加權（同 P_own）；(c)→(d) = 選片改變（同 s_new 加權）。", ""]
    checks = []
    # 逐 seed → 逐任務 → 逐切片的四格
    cells = defaultdict(lambda: defaultdict(list))     # (seed, task) → cell → [values]
    for seed in seeds:
        new = load_recs(OUT_DIR / "per_slide", seed, arch)
        ref = load_recs(REF_DIR, seed, arch)
        n, bad = check_consistency(new, ref)
        checks.append(f"seed {seed}：重跑 vs 既有 `main/` 紀錄 {n} 筆共同欄位比對 → "
                      + ("**逐筆相同** ✅" if not bad else f"**{len(bad)} 處不同** ❌"))
        if bad:
            for b in bad[:10]:
                checks.append(f"  * {b}")
            if not args.allow_mismatch:
                raise SystemExit("❌ 重跑與既有紀錄不一致，停下（--allow-mismatch 可續）")
        sfx = "" if arch == R.DEFAULT_ARCH else f"_{arch}"
        ck = torch.load(CKPT / f"{ARM}_{ORDER}_seed{seed}_stage{last}{sfx}.pt",
                        weights_only=False)
        f_g, f_p = R.new_models(ctx, seed, True, 4)
        f_g.load_state_dict(ck["f_g"]); f_p.load_state_dict(ck["f_p"])
        f_g.eval(); f_p.eval()
        by = {(r["stage"], r["task"], r["slide_id"]): r for r in new}
        n_repro = n_slides = 0
        with torch.no_grad():
            for i_t, t in enumerate(tasks[:-1]):
                for i in range(ctx.n_slides(t, "test")):
                    rec, grp = ctx.get(t, "test", i)
                    own, fin = by[(i_t, t, rec.sid)], by[(last, t, rec.sid)]
                    P_own = torch.tensor(own["selected_idx"], dtype=torch.long)
                    P_fin = torch.tensor(fin["selected_idx"], dtype=torch.long)
                    res = run_rounds(rec.Z, grp, ctx.q0, f_g, f_p, budget=ns.budget,
                                     chunk=ns.chunk, allocation=ns.allocation,
                                     **R.ARCH[arch])
                    s_new = res.records[-1].s
                    n_slides += 1
                    # 自檢 2：ckpt 重現 stage-3 的選片與權重
                    w_fin = softmax_weights(s_new, P_fin)
                    if (res.selected.tolist() == fin["selected_idx"] and
                            all(abs(a - b) <= 1e-6 for a, b in
                                zip(w_fin.tolist(), fin["weights_softmax"]))):
                        n_repro += 1
                    Zo, Zf = rec.Z.index_select(0, P_own), rec.Z.index_select(0, P_fin)
                    y = rec.label
                    a = ce(conch_classify(Zo, None, ctx.f_txt, ctx.logit_scale), y)
                    b = ce(conch_classify(Zf, None, ctx.f_txt, ctx.logit_scale), y)
                    c = ce(conch_classify(Zo, softmax_weights(s_new, P_own),
                                          ctx.f_txt, ctx.logit_scale), y)
                    d = ce(conch_classify(Zf, w_fin, ctx.f_txt, ctx.logit_scale), y)
                    # 自檢 3：(a)(b) 與紀錄的 log C − utility_total 相符；(d) 與紀錄權重相符
                    a_rec, b_rec = logC - own["utility_total"], logC - fin["utility_total"]
                    d_rec = ce(conch_classify(Zf, torch.tensor(fin["weights_softmax"]),
                                              ctx.f_txt, ctx.logit_scale), y)
                    if abs(a - a_rec) > 1e-4 or abs(b - b_rec) > 1e-4 or abs(d - d_rec) > 1e-4:
                        raise SystemExit(f"❌ 自檢 3 失敗 seed {seed} {t} {rec.sid}: "
                                         f"a {a:.6f}/{a_rec:.6f} b {b:.6f}/{b_rec:.6f} "
                                         f"d {d:.6f}/{d_rec:.6f}")
                    for k, v in (("a", a), ("b", b), ("c", c), ("d", d)):
                        cells[(seed, t)][k].append(v)
                    cells[(seed, t)]["sel"].append(b - a)       # 選片改變（等權）
                    cells[(seed, t)]["rew"].append(d - b)       # 重加權（P_final）
                    cells[(seed, t)]["rew_own"].append(c - a)   # 重加權（P_own）
                    cells[(seed, t)]["sel_w"].append(d - c)     # 選片改變（s_new 加權）
                    cells[(seed, t)]["tot"].append(d - a)
        checks.append(f"seed {seed}：最終 ckpt 經 `run_rounds` 重現 stage-3 選片與權重 "
                      f"{n_repro}/{n_slides} 張" + ("（全數）✅" if n_repro == n_slides else " ❌"))
        if n_repro != n_slides:
            raise SystemExit("❌ 自檢 2 失敗：ckpt 無法重現既有選片")
    L += ["## 自檢", ""] + [f"* {c}" for c in checks] + [
        "* 自檢 3：每張切片 (a)(b) 與紀錄的 log C − `utility_total`、(d) 與紀錄 "
        "`weights_softmax` 重算之 CE 皆相符（容差 1e-4）✅", ""]

    # ── 表 1：平均 CE 四格（逐任務；seed 平均 ± sd）──
    def agg(t, k):
        per_seed = [statistics.mean(cells[(s, t)][k]) for s in seeds]
        return statistics.mean(per_seed), (statistics.stdev(per_seed) if len(per_seed) > 1 else 0.0)

    def fmt(t, k, sign=False):
        m, sd = agg(t, k)
        return f"{m:+.3f} ± {sd:.3f}" if sign else f"{m:.3f} ± {sd:.3f}"

    L += ["## 平均 CE（逐任務；先對切片平均，再對 5 seed 取 mean ± sd）", "",
          "| 舊任務 | n test | (a) P_own·等權 | (b) P_final·等權 | (c) P_own·s_new | (d) P_final·s_new |",
          "|---|---|---|---|---|---|"]
    for t in tasks[:-1]:
        n_t = len(cells[(seeds[0], t)]["a"])
        L.append(f"| {t.replace('tcga_', '')} | {n_t} | {fmt(t, 'a')} | {fmt(t, 'b')} | "
                 f"{fmt(t, 'c')} | {fmt(t, 'd')} |")
    L += ["", "## 逐切片 ΔCE（正 = CE 上升 = 退步；逐任務；seed mean ± sd）", "",
          "| 舊任務 | (a)→(b) 選片改變（等權） | (b)→(d) 重加權（P_final） | "
          "(a)→(c) 重加權（P_own） | (c)→(d) 選片改變（s_new） | (a)→(d) 合計 |",
          "|---|---|---|---|---|---|"]
    for t in tasks[:-1]:
        L.append(f"| {t.replace('tcga_', '')} | {fmt(t, 'sel', True)} | {fmt(t, 'rew', True)} | "
                 f"{fmt(t, 'rew_own', True)} | {fmt(t, 'sel_w', True)} | {fmt(t, 'tot', True)} |")
    # 退步切片比例
    L += ["", "### ΔCE > 0（退步）的切片比例（逐任務；5 seed 平均）", "",
          "| 舊任務 | (a)→(b) 選片改變 | (b)→(d) 重加權 | (a)→(d) 合計 |", "|---|---|---|---|"]
    for t in tasks[:-1]:
        def frac(k):
            return statistics.mean([sum(v > 0 for v in cells[(s, t)][k]) / len(cells[(s, t)][k])
                                    for s in seeds])
        L.append(f"| {t.replace('tcga_', '')} | {frac('sel'):.3f} | {frac('rew'):.3f} | {frac('tot'):.3f} |")
    # 跨任務彙總（三任務等權）
    L += ["", "### 三個舊任務等權平均", ""]
    for k, lab in (("sel", "(a)→(b) 選片改變（等權）"), ("rew", "(b)→(d) 重加權（P_final）"),
                   ("rew_own", "(a)→(c) 重加權（P_own）"), ("sel_w", "(c)→(d) 選片改變（s_new）"),
                   ("tot", "(a)→(d) 合計")):
        per_seed = [statistics.mean([statistics.mean(cells[(s, t)][k]) for t in tasks[:-1]])
                    for s in seeds]
        sd = statistics.stdev(per_seed) if len(per_seed) > 1 else 0.0
        L.append(f"* {lab}：{statistics.mean(per_seed):+.4f} ± {sd:.4f}"
                 f"（逐 seed {', '.join(f'{v:+.3f}' for v in per_seed)}）")
    L += ["", "註：(a)→(b) 與 (c)→(d) 都是「選片改變」，差別只在用哪套權重量；"
          "(b)→(d) 與 (a)→(c) 都是「重加權」，差別只在量哪一組片。四者不獨立，"
          "(a)→(d) = (a)→(b) + (b)→(d) = (a)→(c) + (c)→(d)。", ""]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(L) + "\n")
    print(f"→ {REPORT}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--arch", default="flat", choices=list(R.ARCH))
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--allow-mismatch", action="store_true",
                    help="重跑與既有紀錄不一致時仍繼續分析（結果須標明）")
    ap.add_argument("--smoke", action="store_true",
                    help="煙霧測試：--max-train 2 --epochs 1，寫到 dr051_e3_smoke，不分析")
    ap.add_argument("--tag", default=None, help="改讀別的 tag（測試用；預設 dr051_e3）")
    ap.add_argument("--report", default=None, help="改寫到別的報表路徑（測試用）")
    a = ap.parse_args(argv)
    seeds = [int(x) for x in a.seeds.split(",")]
    global TAG, OUT_DIR, CKPT, REPORT
    if a.tag:
        TAG = a.tag; OUT_DIR = R.OUT_ROOT / TAG; CKPT = OUT_DIR / "ckpt"
    if a.report:
        REPORT = Path(a.report)
    if a.smoke:
        TAG = "dr051_e3_smoke"; OUT_DIR = R.OUT_ROOT / TAG; CKPT = OUT_DIR / "ckpt"
        train(seeds, a.arch, ["--max-train", "2", "--epochs", "1", "--no-resume"])
        n = len(list(CKPT.glob("*.pt")))
        print(f"煙霧：ckpt {n} 個（應為 {4 * len(seeds)}）")
        return 0 if n == 4 * len(seeds) else 1
    if not a.analyze_only:
        train(seeds, a.arch, [])
    analyze(seeds, a.arch, a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
