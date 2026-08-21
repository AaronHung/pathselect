"""訓練迴圈 —— per_task 與 joint 兩種模式。

  --mode per_task   每個 task 各訓一個 selector；q_tau 在該模式下是常數，
                    會被第一層 bias 吸收，所有 q ablation 由構造保證為 null。
  --mode joint      一個模型跑所有 task，batch 混合，每個 sample 帶自己的 q_tau。
                    後續 L4–L6 與所有 q ablation 都必須在這個模式下做。

Loss
  L_evidence = L_diag + beta_s * L_sem + beta_u * L_util
本輪只接 L_diag 與 beta_s * L_sem；L_util 與 L_CL 尚未接上（見 ENABLED_TERMS）。

CONTRACT-4：分類頭是單一 frozen head —— selected patches → score-weighted
pooling → L2 normalize → CONCH class-text logits。沒有任何 trained diagnosis head。
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from selector.grouping import assign_groups, tissue_text_features       # noqa: E402
from selector.model import GroupSelector, PatchSelector                 # noqa: E402
from selector.priors import MAINLINE_PRIOR, semantic_prior              # noqa: E402
from selector.rounds import (DEFAULT_BUDGET, DEFAULT_CHUNK,            # noqa: E402
                             DEFAULT_GROUP_GRAD, GROUP_GRAD_MODES, run_rounds)
from selector.task_query import TaskQueryBank                           # noqa: E402
from selector.text_encoder import build_f_txt, load_config              # noqa: E402

MODES = ("per_task", "joint")
#: 本輪接上的 loss 項。L_util / L_CL 之後再開。
ENABLED_TERMS = ("L_diag", "L_sem")
EPS = 1e-12


# ── CONTRACT-4：單一 frozen head ────────────────────────────────────────────

def frozen_head(Z: torch.Tensor, s: torch.Tensor, ste_mask: torch.Tensor,
                f_txt: torch.Tensor, logit_scale,
                weighting: str = "softmax") -> torch.Tensor:
    """[1, C]：selected patches → score-weighted pooling → L2 norm → class-text logits。

    weighting="softmax"（主線）權重 = softmax(s) 限制在被選中的 patch 上；乘上
    straight-through mask，梯度同時經由分數 s 與選取決策流回 F_p。
    weighting="uniform" 是 selection-only：被選中的 patch 等權，分數只影響「選誰」。
    """
    if weighting not in ("softmax", "uniform"):
        raise ValueError(f"unknown weighting: {weighting}")
    e = (torch.exp(s - s.max().detach()) if weighting == "softmax"
         else torch.ones_like(s))
    w_un = ste_mask * e
    w = w_un / w_un.sum().clamp_min(EPS)
    pooled = F.normalize((w.unsqueeze(-1) * Z).sum(0, keepdim=True), dim=-1)
    return logit_scale * (pooled @ f_txt.to(pooled.dtype).t())


# ── loss terms ──────────────────────────────────────────────────────────────

def l_diag(logits: torch.Tensor, label: int) -> torch.Tensor:
    """診斷損失：frozen head 的 cross-entropy。"""
    target = torch.tensor([int(label)], dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits.reshape(1, -1), target)


def l_sem(patch_score: torch.Tensor, prior: torch.Tensor,
          tau: float = 1.0) -> torch.Tensor:
    """語義錨 KL(q || pi)，兩者皆為 patch 上的分布（長度 n）。

    pi = softmax(s / tau)      selector 認為的 patch 重要度
    q  = softmax(prior / tau)  pre-registered semantic prior（主線 discriminative）
    沿用 v9 的形式，只把 anchor 從 max-sim 換成登記過的 prior。
    """
    log_pi = F.log_softmax(patch_score / tau, dim=0)
    q = F.softmax(prior.to(patch_score.dtype) / tau, dim=0)
    return F.kl_div(log_pi, q, reduction="sum")


def evidence_loss(logits, label, patch_score, prior, *,
                  beta_s: float = 0.1, beta_u: float = 0.1,
                  util: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
    """L_evidence = L_diag + beta_s * L_sem + beta_u * L_util。

    util 為 None（本輪的情況）時 L_util 不接上，beta_u 只是先佔位。
    """
    d = l_diag(logits, label)
    sem = l_sem(patch_score, prior)
    total = d + beta_s * sem
    parts = {"L_diag": float(d.detach()), "L_sem": float(sem.detach()),
             "L_util": None}
    if util is not None:
        u = util.mean()
        total = total + beta_u * u
        parts["L_util"] = float(u.detach())
    return total, parts


# ── 單張 slide 的一步 ────────────────────────────────────────────────────────

def train_step(Z, label, q_tau, f_txt, logit_scale, f_group, f_patch, *,
               tissue=None, grouping=None, budget=DEFAULT_BUDGET,
               chunk=DEFAULT_CHUNK, prior_kind=MAINLINE_PRIOR,
               beta_s=0.1, beta_u=0.1, n_candidate_classes=None,
               group_grad=DEFAULT_GROUP_GRAD, use_query=True, use_state=True,
               hierarchy=True, weighting="softmax"):
    """跑完一張 slide 的 chunked loop 並回傳 (loss, parts, result)。"""
    if grouping is None:
        if tissue is None:
            raise ValueError("需要 tissue text 特徵或已算好的 grouping")
        grouping = assign_groups(Z, tissue)

    result = run_rounds(Z, grouping, q_tau, f_group, f_patch,
                        budget=budget, chunk=chunk, group_grad=group_grad,
                        use_query=use_query, use_state=use_state,
                        hierarchy=hierarchy)
    if not result.records:
        raise RuntimeError("chunked loop 一輪都沒跑完，檢查候選是否為空")

    # 最後一輪的分數與累積 mask 決定 frozen head 的輸入
    s_last = result.records[-1].s
    ste = torch.zeros_like(s_last)
    for rec in result.records:
        ste = ste + rec.ste_mask
    logits = frozen_head(Z, s_last, ste, f_txt, logit_scale, weighting=weighting)

    prior = semantic_prior(Z, f_txt, kind=prior_kind,
                           n_candidate_classes=n_candidate_classes or f_txt.shape[0],
                           logit_scale=logit_scale)
    loss, parts = evidence_loss(logits, label, s_last, prior,
                                beta_s=beta_s, beta_u=beta_u)
    parts["n_selected"] = int(result.selected.numel())
    return loss, parts, result


# ── CLI ─────────────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="train the evidence selector")
    ap.add_argument("--mode", choices=MODES, default="joint")
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    ap.add_argument("--chunk", type=int, default=DEFAULT_CHUNK,
                    help="每輪選幾個（CONTRACT-1；pre-register c=8 為主線）")
    ap.add_argument("--prior", choices=("none", "max_sim", "discriminative"),
                    default=MAINLINE_PRIOR)
    ap.add_argument("--beta-s", type=float, default=0.1)
    ap.add_argument("--beta-u", type=float, default=0.1,
                    help="L_util 的權重；本輪尚未接上該項，先佔位")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--tasks", default="", help="逗號分隔；空 = config 的全部 task")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--group-grad", choices=GROUP_GRAD_MODES, default=DEFAULT_GROUP_GRAD,
                    help="F_g 的梯度路徑；主線 ste_allocation，none 僅供 ablation")
    return ap


def make_models(device="cpu") -> tuple[GroupSelector, PatchSelector]:
    return GroupSelector().to(device), PatchSelector().to(device)


def resolve_tasks(cfg, arg: str) -> list[str]:
    return [t.strip() for t in arg.split(",") if t.strip()] or list(cfg["tasks"])


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    torch.manual_seed(args.seed)
    cfg = load_config()
    device = torch.device("cpu")
    tasks = resolve_tasks(cfg, args.tasks)

    f_txt = torch.cat([build_f_txt(t, cfg, device=device).f_txt
                       for t in cfg["tasks"]], 0)                  # 8-way，兩種模式共用
    logit_scale = build_f_txt(cfg["tasks"][0], cfg, device=device).logit_scale
    tissue = tissue_text_features(cfg, device=device)
    queries = TaskQueryBank(cfg, device=device)

    print(f"mode={args.mode}  tasks={tasks}  budget={args.budget}  chunk={args.chunk}"
          f"  prior={args.prior}  beta_s={args.beta_s}  beta_u={args.beta_u}"
          f"  enabled_terms={ENABLED_TERMS}  group_grad={args.group_grad}")
    if args.group_grad == "none":
        print("  ⚠️  group_grad=none：F_g 不接收梯度（取整不可微），在整個 within-task "
              "訓練中等於固定的隨機函數。**僅供 ablation 使用** —— 用這個模式跑 "
              "+hierarchy 會得到由構造保證的 null，不是實驗結果。")

    from selector.evaluate import iter_test_slides   # 訓練資料載入沿用同一組工具

    def slides_for(task):
        pos = cfg["tasks"].index(task)
        return iter_test_slides(cfg, task, pos, limit=args.max_train)

    if args.mode == "per_task":
        for task in tasks:
            f_g, f_p = make_models(device)
            opt = torch.optim.Adam(list(f_g.parameters()) + list(f_p.parameters()),
                                   lr=args.lr, weight_decay=1e-4)
            run_epochs(args, task, [task], slides_for, queries, f_txt, logit_scale,
                       tissue, f_g, f_p, opt)
    else:
        f_g, f_p = make_models(device)
        opt = torch.optim.Adam(list(f_g.parameters()) + list(f_p.parameters()),
                               lr=args.lr, weight_decay=1e-4)
        run_epochs(args, "joint", tasks, slides_for, queries, f_txt, logit_scale,
                   tissue, f_g, f_p, opt)
    return 0


def run_epochs(args, label_str, tasks, slides_for, queries, f_txt, logit_scale,
               tissue, f_g, f_p, opt) -> None:
    for epoch in range(args.epochs):
        seen, total = 0, 0.0
        # joint：把各 task 的 slide 交錯混合，每個 sample 帶自己的 q_tau
        stream = [(t, rec) for t in tasks for rec in slides_for(t)]
        if args.mode == "joint":
            g = torch.Generator().manual_seed(args.seed + epoch)
            order = torch.randperm(len(stream), generator=g).tolist()
            stream = [stream[i] for i in order]
        for task, rec in stream:
            loss, parts, _ = train_step(
                rec.Z, rec.label, queries.get(task), f_txt, logit_scale, f_g, f_p,
                tissue=tissue, budget=args.budget, chunk=args.chunk,
                prior_kind=args.prior, beta_s=args.beta_s, beta_u=args.beta_u,
                group_grad=args.group_grad)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
            seen += 1
        print(f"  [{label_str}] epoch {epoch + 1}/{args.epochs}  "
              f"n={seen}  mean_loss={total / max(seen, 1):.4f}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
