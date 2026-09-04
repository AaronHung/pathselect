#!/usr/bin/env python3
"""DR-048 B6：正交投影式持續合併（**OPCM-Merge (adapted)**）。

出處：Bui et al., *MergeSlide: Continual Model Merging and Task-to-Class
Prompt-Aligned Inference for Lifelong Learning on Whole Slide Images*,
arXiv:2511.13099（WACV 2026），Eq. (3)–(5)。官方實作
`caodoanh2001/MergeSlide::opcm_mergeslide.py`。

**本檔是重新實作，不是移植** —— 沒有複製對方任何一行程式碼。

## 演算法（論文 Eq. 3–5）

    Δθ_t        = θ_t − θ_base                                （task vector）
    G(Δθ_t)     = U ((Uᵀ Δθ_t V) ⊙ M) Vᵀ,  UΣVᵀ = SVD(Δθ̃_{1:t−1})
    θ̃_{1:t}     = θ_base + [ λ_{t−1} Δθ̃_{1:t−1} + G(Δθ_t) ] / λ_t
    λ_t         = t · ‖λ_{t−1} Δθ̃_{1:t−1} + G(Δθ_t)‖₂ / Σ_{i≤t} ‖Δθ_i‖₂,  λ_1 = 1

`M` 是**零對角遮罩**：把 `Δθ_t` 投影到 SVD 基底後抹掉對角線，剩下的成分與
`Δθ̃_{1:t−1}` 的 Frobenius 內積為 0，因此只留下這個任務相對於舊任務的新資訊。

## 三處必須講明的落差

1. **官方程式的遮罩是 no-op。** 它寫的是 `projected_task_tv.diag().fill_(0)`，
   但 `Tensor.diag()` 對 2-D 輸入回傳**副本**（torch 2.11 實測），不是 view ——
   那一行沒有改到 `projected_task_tv`。少了遮罩，
   `U (Uᵀ Δθ_t V) Vᵀ ≡ Δθ_t`（full SVD 下 U、V 都是正交方陣），
   `G` 就退化成「直接相加」，與非線性層的 `merge_other_parameters` 完全相同。
   **本檔照論文實作**（用 `.diagonal()`，遮罩真的生效），並提供
   `mask=False` 重現官方行為以供對照。`tests/test_sota_opcm.py` 兩者都釘住。

2. **λ 的分母，論文寫 `‖θ_i‖₂`，官方程式用的是 `‖θ_i − θ_base‖₂`**
   （`get_task_vector_norm`）。本檔照**程式**，因為 `‖θ_i‖₂` 含未微調的基礎模型
   本身、量級由 θ_base 主導，會讓 λ 幾乎不隨任務變化 —— 顯然不是原意。

3. **λ 是全域純量**，對整份 state dict 攤平後取範數，不是逐矩陣算（照官方程式）。

## 本 repo 的接法（adapted）

θ_base = θ₀（`run_exp2.new_models(seed)` 的初始權重），
θ_t = θ₀ + Δ_t，Δ_t 直接取自 C1／C2 已經算好的
`outputs/exp2/main/dr046_deltas_seed{seed}.pt` —— 那是**每個任務各自從同一個 θ₀
bare 訓練**得到的四個 delta，正是 OPCM 假設的輸入。

⚠️ 因此 OPCM 只能跑在 delta 快取存在的地方：**fold 1、seed 0–4（DR-046 協定）**，
   不是 SOTA 主表的 10 折。`docs/SOTA_TABLE.md` 必須逐列標明這一點。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_exp2 import (ARCH, DEFAULT_ARCH, ORDERS, Ctx, acc, evaluate,   # noqa: E402
                      new_models)
from selector.text_encoder import load_config                           # noqa: E402

#: 與 `scripts/run_exp2.py` 同一個根，好讓 SOTA 主表只讀一個目錄
OUT_ROOT = ROOT / "outputs" / "exp2"
DELTA_CACHE = ROOT / "outputs" / "exp2" / "main" / "dr046_deltas_seed{seed}.pt"
ARM = "OPCM"


# ── 演算法 ──────────────────────────────────────────────────────────────────

def project(task_tv: torch.Tensor, prev_tv: torch.Tensor, *,
            mask: bool = True) -> torch.Tensor:
    """G(Δθ_t)：投到 `prev_tv` 的 SVD 基底、抹對角、投回來。

    `prev_tv` 全零時沒有「舊方向」可正交，直接回傳 `task_tv`（論文沒有涵蓋這個
    邊界；全零時 SVD 的 U、V 是任意正交矩陣，投影會得到 0，等於把整個任務丟掉）。
    """
    if not torch.any(prev_tv):
        return task_tv.clone()
    u, _s, vh = torch.linalg.svd(prev_tv, full_matrices=True)
    v = vh.mH
    proj = u.mH @ task_tv @ v
    if mask:
        proj = proj.clone()
        proj.diagonal().fill_(0)        # ⚠️ `.diag()` 是副本，`.diagonal()` 才是 view
    return u @ proj @ v.mH


def _norm(delta: list[dict]) -> float:
    """整份 delta（跨模型、跨鍵）攤平後的 L2 範數。"""
    flat = [v.reshape(-1) for sd in delta for v in sd.values()]
    if not flat:
        return 0.0
    return float(torch.linalg.vector_norm(torch.cat(flat)))


def _scaled(delta: list[dict], k: float) -> list[dict]:
    return [{key: v * k for key, v in sd.items()} for sd in delta]


def merge_sequence(deltas: list[list[dict]], *, mask: bool = True) -> list[list[dict]]:
    """回傳每個 stage 合併後的 Δθ̃_{1:t}（長度與 `deltas` 相同）。

    stage 0 就是 Δ_1 本身（θ̃_1 = θ_1，λ_1 = 1），與官方實作的初始化一致。
    """
    merged = [{k: v.clone() for k, v in sd.items()} for sd in deltas[0]]
    prev_lambda, norms, out = 1.0, [_norm(deltas[0])], [merged]

    for t in range(1, len(deltas)):
        norms.append(_norm(deltas[t]))
        avg = statistics.mean(norms)
        new: list[dict] = []
        for i, cur_sd in enumerate(deltas[t]):
            prev_sd, acc_sd = merged[i], {}
            for key in sorted(set(prev_sd) | set(cur_sd)):
                ref = prev_sd.get(key, cur_sd.get(key))
                prev = prev_sd.get(key, torch.zeros_like(ref))
                cur = cur_sd.get(key, torch.zeros_like(ref))
                g = project(cur, prev, mask=mask) if cur.dim() == 2 else cur
                acc_sd[key] = prev_lambda * prev + g
            new.append(acc_sd)
        n = _norm(new)
        if n == 0.0:
            raise ValueError(f"stage {t} 合併後的 task vector 全零，λ 無法定義")
        prev_lambda = n / avg                       # λ_t
        merged = _scaled(new, avg / n)              # 除以 λ_t
        out.append(merged)
    return out


def apply_delta(theta0: list[dict], delta: list[dict]) -> list[dict]:
    out = []
    for base, d in zip(theta0, delta):
        sd = {k: v.detach().clone() for k, v in base.items()}
        for k, v in d.items():
            sd[k] = sd[k] + v
        out.append(sd)
    return out


# ── 執行 ────────────────────────────────────────────────────────────────────

def run(ctx, seed: int, order_name: str, args, out_dir: Path) -> list[dict]:
    cache = Path(str(DELTA_CACHE).format(seed=seed))
    if not cache.is_file():
        raise SystemExit(f"❌ 缺 delta 快取 {cache} —— 先跑 C1（`--arms C1 --seeds {seed}`）")
    deltas = torch.load(cache, weights_only=False)
    tasks = ORDERS[order_name]
    if len(deltas) != len(tasks):
        raise SystemExit(f"❌ 快取有 {len(deltas)} 個 delta，但 order `{order_name}` "
                         f"有 {len(tasks)} 個任務")

    m0 = new_models(ctx, seed, True, args.rank)
    theta0 = [{k: v.detach().clone() for k, v in m.state_dict().items()} for m in m0]
    seq = merge_sequence(deltas, mask=not args.no_mask)

    recs = []
    for stage, merged in enumerate(seq):
        models = new_models(ctx, seed, True, args.rank)
        for m, sd in zip(models, apply_delta(theta0, merged)):
            m.load_state_dict(sd)
        print(f"    ── stage {stage}: OPCM(Δ[0..{stage}])  "
              f"‖Δθ̃‖={_norm(merged):.4f}", flush=True)
        for t in tasks[:stage + 1]:
            r = evaluate(ctx, models, t, ARM, order_name, seed, stage, args)
            recs += r
            print(f"       eval {t:10s} class-IL={acc(r, 'pred_class_il'):.4f} "
                  f"task-IL={acc(r, 'pred_task_il'):.4f}", flush=True)
    return recs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--order", default="reverse", choices=list(ORDERS))
    ap.add_argument("--arch", default=DEFAULT_ARCH, choices=list(ARCH))
    ap.add_argument("--fold", type=int, default=1,
                    help="delta 快取只在 fold 1 存在；改動這個值不會改動快取來源。")
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=1)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--allocation", default="per_budget")
    ap.add_argument("--prior", default="tissue")
    ap.add_argument("--mem-capacity", type=int, default=None)
    ap.add_argument("--tag", default="sota")
    ap.add_argument("--no-mask", action="store_true",
                    help="關掉零對角遮罩 → 重現官方程式的實際行為（G 退化成直接相加）")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config()
    cfg["fold"] = args.fold
    ctx = Ctx(cfg)
    out_dir = OUT_ROOT / args.tag / "per_slide"
    out_dir.mkdir(parents=True, exist_ok=True)

    for seed in [int(x) for x in args.seeds.split(",")]:
        suffix = "" if not args.no_mask else "_nomask"
        name = (f"{ARM}_{args.order}_seed{seed}"
                f"{'' if args.arch == DEFAULT_ARCH else '_' + args.arch}{suffix}.json")
        path = out_dir / name
        if path.exists() and not args.no_resume:
            print(f"▷ 跳過（已存在）{path.name}", flush=True)
            continue
        print(f"═══ OPCM seed={seed} order={args.order} arch={args.arch} "
              f"mask={not args.no_mask}", flush=True)
        recs = run(ctx, seed, args.order, args, out_dir)
        path.write_text(json.dumps(recs, indent=1))
        print(f"    → {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
