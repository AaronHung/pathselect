#!/usr/bin/env python3
"""Exp 0 — Random / Grid baseline（論文 Table 1）。

第 9 版技術報告只跟「看全部 patch」比過，從來沒跟 random 選取比過。若 learned
selector 贏不了 random，整條研究線的地基就是空的。這支腳本把四條線放在完全相同
的下游路徑上比：

    selected patches → 權重 → L2 normalize → conch_classify → argmax

policy
  random        均勻隨機抽 K 個（5 seeds，報 mean±std）
  grid          依特徵原順序等距抽樣，stride = n // K（1 次）
  similarity    frozen CONCH patch-text 相似度 top-K
  learned-flat  reference/v9 skill bank 的 per-task selector top-K（純推論，不重訓）

⚠️ 權重政策不對稱，且**無法避免**：random / grid 沒有分數，只能等權；
   similarity / learned-flat 用 softmax(top-K 分數)（主線，與訓練一致）。
   這件事在 BASELINES.md 明確標註，不假裝四條線相同。

設定：4 tasks、reverse order、fold 1、K ∈ {8,16,32,64}、8-way label space。
每個 (task, K) 跑完就寫檔。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from selector.baselines import grid_indices, random_indices          # noqa: E402
from selector.evaluate import (iter_test_slides, score_based_indices,  # noqa: E402
                               select_and_classify)
from selector.flat_selector import SelectorBank, similarity_score    # noqa: E402
from selector.text_encoder import build_f_txt, load_config           # noqa: E402

BANK_PATH = REPO_ROOT / "reference" / "v9" / "skill_bank_reverse_f1.pt"
OUT_DIR = REPO_ROOT / "outputs" / "exp0"
JSON_PATH = OUT_DIR / "baselines_reverse_f1.json"
MD_PATH = OUT_DIR / "BASELINES.md"
KS = (8, 16, 32, 64)
RANDOM_SEEDS = (0, 1, 2, 3, 4)
SCORED = {"similarity": "softmax", "learned-flat": "softmax"}
UNSCORED = {"random": "uniform", "grid": "uniform"}
POLICY_ORDER = ("random", "grid", "similarity", "learned-flat")


def build_f_txt_all(cfg, device) -> torch.Tensor:
    return torch.cat([build_f_txt(t, cfg, device=device).f_txt for t in cfg["tasks"]], 0)


@torch.no_grad()
def eval_task_k(selector, f_txt, logit_scale, cfg, task, task_pos, k, max_eval):
    """單一 (task, K)：一次掃過 slide，四條線同時算。"""
    hits = {"grid": 0, "similarity": 0, "learned-flat": 0}
    hits.update({("random", s): 0 for s in RANDOM_SEEDS})
    n_slides = 0

    for rec in iter_test_slides(cfg, task, task_pos, limit=max_eval):
        Z, n = rec.Z, int(rec.Z.shape[0])
        n_slides += 1

        for seed in RANDOM_SEEDS:
            idx = random_indices(n, k, seed=seed + 1000 * n_slides)
            pred, _ = select_and_classify(Z, idx, f_txt, logit_scale, weighting="uniform")
            hits[("random", seed)] += int(pred == rec.label)

        idx = grid_indices(n, k)
        pred, _ = select_and_classify(Z, idx, f_txt, logit_scale, weighting="uniform")
        hits["grid"] += int(pred == rec.label)

        for name, scores in (("similarity", similarity_score(Z, f_txt)),
                             ("learned-flat", selector(Z, f_txt))):
            idx = score_based_indices(scores, k)
            pred, _ = select_and_classify(Z, idx, f_txt, logit_scale,
                                          scores=scores, weighting="softmax")
            hits[name] += int(pred == rec.label)

    rows = []
    for seed in RANDOM_SEEDS:
        rows.append(dict(task=task_pos, task_name=task, K=k, policy="random",
                         seed=seed, acc=hits[("random", seed)] / n_slides,
                         n_slides=n_slides))
    for name in ("grid", "similarity", "learned-flat"):
        rows.append(dict(task=task_pos, task_name=task, K=k, policy=name,
                         seed=None, acc=hits[name] / n_slides, n_slides=n_slides))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-eval", type=int, default=0, help="每 task 最多幾張 slide（0=全部）")
    ap.add_argument("--tag", default="", help="輸出檔名後綴（煙霧測試用）")
    args = ap.parse_args()

    cfg = load_config()
    device = torch.device("cpu")
    torch.manual_seed(42)

    json_path = JSON_PATH if not args.tag else JSON_PATH.with_name(
        f"{JSON_PATH.stem}_{args.tag}.json")
    md_path = MD_PATH if not args.tag else MD_PATH.with_name(
        f"{MD_PATH.stem}_{args.tag}.md")

    f_txt = build_f_txt_all(cfg, device)
    logit_scale = build_f_txt(cfg["tasks"][0], cfg, device=device).logit_scale
    bank = SelectorBank.load(str(BANK_PATH))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"f_txt {tuple(f_txt.shape)}  K={list(KS)}  seeds={list(RANDOM_SEEDS)}"
          f"  max_eval={args.max_eval or 'all'}")

    records = []
    for task_pos, task in enumerate(cfg["tasks"]):
        selector = bank.build_selector(task_pos, device)
        for k in KS:
            rows = eval_task_k(selector, f_txt, logit_scale, cfg, task, task_pos,
                               k, args.max_eval)
            records += rows
            json_path.write_text(json.dumps(records, indent=1))     # 跑完就寫檔
            summary = summarize(rows)
            print(f"  {task:10s} K={k:3d} n={rows[0]['n_slides']:4d}  "
                  + "  ".join(f"{p}={summary[p]}" for p in POLICY_ORDER), flush=True)

    write_report(records, md_path, args)
    print(f"\n→ {json_path}\n→ {md_path}")
    return 0


def summarize(rows) -> dict:
    out = {}
    rnd = [r["acc"] for r in rows if r["policy"] == "random"]
    out["random"] = (f"{statistics.mean(rnd):.4f}±"
                     f"{statistics.stdev(rnd) if len(rnd) > 1 else 0.0:.4f}")
    for p in ("grid", "similarity", "learned-flat"):
        v = [r["acc"] for r in rows if r["policy"] == p]
        out[p] = f"{v[0]:.4f}" if v else "—"
    return out


def cell(records, task, k, policy) -> str:
    v = [r["acc"] for r in records if r["task_name"] == task and r["K"] == k
         and r["policy"] == policy]
    if not v:
        return "—"
    if policy == "random":
        sd = statistics.stdev(v) if len(v) > 1 else 0.0
        return f"{statistics.mean(v):.4f} ± {sd:.4f}"
    return f"{v[0]:.4f}"


def mean_over_tasks(records, k, policy) -> float:
    tasks = sorted({r["task_name"] for r in records})
    per = []
    for t in tasks:
        v = [r["acc"] for r in records if r["task_name"] == t and r["K"] == k
             and r["policy"] == policy]
        if v:
            per.append(statistics.mean(v))
    return statistics.mean(per) if per else float("nan")


def write_report(records, md_path, args) -> None:
    tasks = [t for t in load_config()["tasks"]
             if any(r["task_name"] == t for r in records)]
    ks = sorted({r["K"] for r in records})
    L = [
        "# Exp 0 — Random / Grid baseline",
        "",
        "論文 Table 1。四條線走**完全相同**的下游路徑："
        "`selected patches → 權重 → L2 normalize → conch_classify → argmax`，"
        "唯一的差別是「選哪些 patch」與「用什麼權重」。",
        "",
        "reverse order、fold 1、8-way label space、"
        f"K ∈ {{{', '.join(map(str, ks))}}}、"
        f"random 跑 {len(RANDOM_SEEDS)} seeds ({RANDOM_SEEDS[0]}..{RANDOM_SEEDS[-1]}) 報 mean ± std、"
        "grid 跑 1 次。learned-flat 用 `reference/v9/skill_bank_reverse_f1.pt`，"
        "**純推論不重訓**。"
        + (f"\n\n⚠️ 本次為煙霧測試：每個 task 只評估前 {args.max_eval} 張 slide。"
           if args.max_eval else ""),
        "",
        "## ⚠️ 權重政策不對稱（無法避免）",
        "",
        "| policy | 選法 | 權重 |",
        "|---|---|---|",
        "| random | 均勻隨機 K 個，不重複 | **等權** |",
        "| grid | 依特徵原順序等距，stride = n // K | **等權** |",
        "| similarity | frozen CONCH patch-text 相似度 top-K | softmax(top-K 分數) |",
        "| learned-flat | per-task selector top-K | softmax(top-K 分數) |",
        "",
        "random 與 grid **沒有分數**，因此只能等權；這不是設計選擇，是這兩條線的",
        "本質限制。scored 與 unscored 之間的差距同時包含「選得比較準」與"
        "「權重政策不同」兩個因素，**不要當成純粹的選取能力差**。",
        "若要單獨看選取能力，請比較 selection-only ablation（見 DELTA_v9.md）。",
        "",
        "⚠️ **grid 不是真正的 spatial uniform**：特徵檔是 `[n, 512]` 純張量，"
        "資料集裡沒有 patch 座標，所以只能沿特徵的原始掃描序等距抽樣。",
        "",
        "## 結果",
        "",
        "| task | K | random (mean ± std) | grid | similarity | learned-flat |",
        "|---|---|---|---|---|---|",
    ]
    for t in tasks:
        for k in ks:
            L.append(f"| {t} | {k} | " + " | ".join(
                cell(records, t, k, p) for p in POLICY_ORDER) + " |")
    L += ["", "### 四 task 平均", "",
          "| K | random | grid | similarity | learned-flat | learned − random (pp) |",
          "|---|---|---|---|---|---|"]
    for k in ks:
        vals = {p: mean_over_tasks(records, k, p) for p in POLICY_ORDER}
        L.append(f"| {k} | " + " | ".join(f"{vals[p]:.4f}" for p in POLICY_ORDER)
                 + f" | {(vals['learned-flat'] - vals['random']) * 100:+.2f} |")
    L += ["", "逐筆結果：`outputs/exp0/baselines_reverse_f1.json`",
          "（欄位 task / task_name / K / policy / seed / acc / n_slides）", ""]
    md_path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
