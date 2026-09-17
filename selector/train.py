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
from selector.model import STATE_DIM, GroupSelector, PatchSelector      # noqa: E402
from selector.continual import (continual_loss, differentiable_utility,  # noqa: E402
                                l_eq, l_kd, l_util)
from selector.memory import (SelectionMemory, load_cand_features,       # noqa: E402
                             make_entry, sample_key, save_cand_features,
                             reload_features, selected_from_entry)
from selector.priors import MAINLINE_PRIOR, semantic_prior              # noqa: E402
from selector.utility import (CANDIDATE_SIZE, counterfactual_gain,      # noqa: E402
                              mask_logits,
                              top_candidates)
from selector.rounds import (DEFAULT_BUDGET, DEFAULT_CHUNK,            # noqa: E402
                             DEFAULT_GROUP_GRAD, GROUP_GRAD_MODES, run_rounds)
from selector.task_query import TaskQueryBank                           # noqa: E402
from selector.text_encoder import build_f_txt, load_config              # noqa: E402

MODES = ("per_task", "joint")
#: DR-054：hinge 的 U_old 口徑。snapshot = 快照的 u_old（現行）；current = replay 時以當前 C_t 重算。
U_OLD_MODES = ("snapshot", "current")

#: DR-056 候選級 replay 的檔案存取守門。守的是**系統層的 open 事件**，不是 code review：
#: candidate_only 的 replay 區間內，只要有人開啟路徑含 `feats-l1-s256` 的檔案就拋錯。
FEATURE_DIR_MARK = "feats-l1-s256"
_guard_depth = 0
_guard_installed = False


def _install_feature_open_guard() -> None:
    global _guard_installed
    if _guard_installed:
        return

    def _hook(event, args):
        if _guard_depth and event in ("open", "io.open") and args:
            path = str(args[0])
            if FEATURE_DIR_MARK in path:
                raise RuntimeError(
                    f"candidate_only replay 期間開啟了完整特徵檔：{path}")
    sys.addaudithook(_hook)
    _guard_installed = True


class no_feature_files:
    """with 區塊內禁止開啟原始特徵檔（DR-056 2.3 的斷言）。"""

    def __enter__(self):
        global _guard_depth
        _install_feature_open_guard()
        _guard_depth += 1
        return self

    def __exit__(self, *exc):
        global _guard_depth
        _guard_depth -= 1
        return False

#: within-task 已接上的 loss 項。CL 層的三項在 selector/continual.py。
ENABLED_TERMS = ("L_diag", "L_sem", "L_util")
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

def l_diag(logits: torch.Tensor, label: int,
           class_mask: torch.Tensor | None = None) -> torch.Tensor:
    """診斷損失：frozen head 的 cross-entropy。

    `class_mask`（DR-052 累積式頭）：未見類別的 logit 設為 −inf 後再算 CE；
    None（固定頭）時 `mask_logits` 原樣回傳，逐位元不變。
    """
    logits = mask_logits(logits, class_mask)
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
                  utility: torch.Tensor | None = None,
                  cand_idx: torch.Tensor | None = None,
                  class_mask: torch.Tensor | None = None
                  ) -> tuple[torch.Tensor, dict]:
    """L_evidence = L_diag + beta_s * L_sem + beta_u * L_util。

    utility: [len(cand_idx)] 每個候選 patch 的 counterfactual gain。
             None 或 beta_u == 0 時 L_util **完全不計算也不相加**，
             結果與未接上該項時位元相同。
    """
    d = l_diag(logits, label, class_mask)
    sem = l_sem(patch_score, prior)
    total = d + beta_s * sem
    parts = {"L_diag": float(d.detach()), "L_sem": float(sem.detach()),
             "L_util": None}
    if utility is not None and beta_u != 0.0:
        s_c = patch_score if cand_idx is None else patch_score.index_select(0, cand_idx)
        u_term = l_util(s_c, utility)
        total = total + beta_u * u_term
        parts["L_util"] = float(u_term.detach())
    return total, parts


# ── 單張 slide 的一步 ────────────────────────────────────────────────────────

def train_step(Z, label, q_tau, f_txt, logit_scale, f_group, f_patch, *,
               tissue=None, grouping=None, budget=DEFAULT_BUDGET,
               chunk=DEFAULT_CHUNK, prior_kind=MAINLINE_PRIOR,
               beta_s=0.1, beta_u=0.1, n_candidate_classes=None,
               group_grad=DEFAULT_GROUP_GRAD, use_query=True, use_state=True,
               hierarchy=True, weighting="softmax",
               candidate_size=CANDIDATE_SIZE, allocation=None,
               class_mask=None):
    """跑完一張 slide 的 chunked loop 並回傳 (loss, parts, result)。

    `class_mask`（DR-052）：累積式頭的已見類別遮罩 bool[C]；None = 固定頭（零改動）。
    影響 L_diag（遮罩 CE）、L_sem 先驗（只看 f_txt[C_t]）、反事實 teacher u_i（遮罩 CE）。
    """
    if grouping is None:
        if tissue is None:
            raise ValueError("需要 tissue text 特徵或已算好的 grouping")
        grouping = assign_groups(Z, tissue)

    result = run_rounds(Z, grouping, q_tau, f_group, f_patch,
                        budget=budget, chunk=chunk, group_grad=group_grad,
                        use_query=use_query, use_state=use_state,
                        hierarchy=hierarchy,
                        **({"allocation": allocation} if allocation else {}))
    if not result.records:
        raise RuntimeError("chunked loop 一輪都沒跑完，檢查候選是否為空")

    # 最後一輪的分數與累積 mask 決定 frozen head 的輸入
    s_last = result.records[-1].s
    ste = torch.zeros_like(s_last)
    for rec in result.records:
        ste = ste + rec.ste_mask
    logits = frozen_head(Z, s_last, ste, f_txt, logit_scale, weighting=weighting)

    if class_mask is None:
        prior = semantic_prior(Z, f_txt, kind=prior_kind,
                               n_candidate_classes=n_candidate_classes or f_txt.shape[0],
                               logit_scale=logit_scale)
    else:
        f_seen = f_txt[class_mask.to(torch.bool).to(f_txt.device)]
        prior = semantic_prior(Z, f_seen, kind=prior_kind,
                               n_candidate_classes=int(f_seen.shape[0]),
                               logit_scale=logit_scale)

    # S4-4：counterfactual gain 當監督訊號。候選集合與 evidence 都取「最後一輪的
    # 狀態」—— s_last 就是在那個狀態下產生的，兩者用同一個狀態才自洽。
    utility = cand_idx = None
    if beta_u != 0.0:
        st = result.state
        cand_idx = top_candidates(s_last.detach(), st.available_mask, candidate_size)
        if cand_idx.numel() > 0:
            utility = counterfactual_gain(
                st.evidence_sum(), st.n_selected, Z.index_select(0, cand_idx),
                f_txt, logit_scale, label, class_mask=class_mask)
        else:
            cand_idx = None

    loss, parts = evidence_loss(logits, label, s_last, prior,
                                beta_s=beta_s, beta_u=beta_u,
                                utility=utility, cand_idx=cand_idx,
                                class_mask=class_mask)
    parts["n_selected"] = int(result.selected.numel())
    parts["n_candidates"] = int(cand_idx.numel()) if cand_idx is not None else 0
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


# ── CL 層的組合（S4-2 / S4-3）───────────────────────────────────────────────
# continual.py 只放純 loss 數學（好單測）；把模型前向、記憶體重載串起來的
# orchestration 放這裡。

def fill_memory(memory: SelectionMemory, models, task: str, cfg, f_txt, logit_scale,
                tissue, *, budget=DEFAULT_BUDGET, chunk=DEFAULT_CHUNK,
                q_tau=None, spec=None, max_slides: int = 0,
                candidate_size=CANDIDATE_SIZE, class_mask=None,
                candidate_only=False, cand_store=None) -> int:
    """學完一個 task 後，把該 task 的代表性樣本寫進 Selection Memory。

    entry 不含 patch feature，只留 slide_id + cand_idx，之後用 reload_features 重載。
    汰換由 memory.policy 決定（主線 reservoir sampling）。回傳新增筆數。
    """
    from selector.evaluate import read_slide, slide_dataset
    from selector.grouping import assign_groups

    f_g, f_p = models
    spec = spec or {}
    q = q_tau if q_tau is not None else torch.zeros(512)
    ds, shift = slide_dataset(cfg, task, list(cfg["tasks"]).index(task), "train")
    n = len(ds) if max_slides <= 0 else min(max_slides, len(ds))
    added = 0
    with torch.no_grad():
        for i in range(n):
            rec = read_slide(ds, shift, i)
            grp = assign_groups(rec.Z, tissue)
            res = run_rounds(rec.Z, grp, q, f_g, f_p, budget=budget, chunk=chunk,
                             candidate_size=candidate_size, **spec)
            last = res.records[-1]
            cand = last.cand_idx
            if cand.numel() == 0:
                continue
            # DR-052：u_old 以快照當時的 C_old 算，並把 C_old 存進 entry（固定頭 None）
            u = counterfactual_gain(res.state.evidence_sum(), res.state.n_selected,
                                    rec.Z.index_select(0, cand), f_txt, logit_scale,
                                    rec.label, class_mask=class_mask)
            protos = None
            if candidate_only:             # DR-056：側倉存候選特徵、entry 存 group 原型
                if cand_store is None:
                    raise ValueError("candidate_only=True 需要 cand_store 路徑")
                save_cand_features(cand_store, task, sample_key(task, rec.sid),
                                   rec.Z.index_select(0, cand), rec.label,
                                   cand_idx=cand)
                protos = grp.prototypes
            memory.add(make_entry(task, rec.sid, res.state, last.r, cand,
                                  last.s.detach(), u, class_mask_old=class_mask,
                                  group_prototypes=protos))
            added += 1
    return added


def _candidate_terms(entry, models, f_txt, logit_scale, tissue, *, budget, chunk,
                     q_tau, spec, use_kd, use_eq, use_replay, eq_mode,
                     kd_group_weight, class_mask, u_old_mode, cand_store):
    """DR-056 候選級 replay：只吃側倉的 ≤256 個候選特徵，不開原始特徵檔。

    與全 slide replay 的定義差異（DR-056 ruling 記錄在案，不是無損壓縮）：
    * 分組原型改由**候選子集**算（`assign_groups(Z_cand, tissue)`），空 group 由
      `allocate` 的 active mask 排除；配額契約 Σ b_j = min(B, 候選數) 不變。
    * patch 評分逐列獨立，故子集上的 s 與全 slide 的 s[cand_idx] 數學上相同；
      L_KD 的 patch 項改用**位置對齊**（s_old 的順序就是 cand_idx 的順序）。
    * L_KD 的 group 項用快照存下的原型：r_new = F_g(g_stored)，與 r_old 同輸入，
      因此 KL 量的仍是「評分行為漂移」而不是「輸入漂移」。
    * hinge 的 P_old 由 `selected_from_entry` 的**位置**定位，池化向量與全 slide 相同。
    """
    from selector.grouping import assign_groups

    f_g, f_p = models
    spec = spec or {}
    if spec.get("use_state", False) or spec.get("use_query", False):
        raise NotImplementedError("候選級 replay 目前只支援 use_state=use_query=False 的主線組態")
    if entry.group_prototypes is None:
        raise ValueError("entry 沒有 group_prototypes —— 這筆快照不是 candidate_only 模式寫的")
    with no_feature_files():                     # 系統層斷言：這段不得開啟完整特徵檔
        Zc, label = load_cand_features(cand_store, entry)
        grp = assign_groups(Zc, tissue)
        res = run_rounds(Zc, grp, q_tau, f_g, f_p, budget=budget, chunk=chunk, **spec)
        last = res.records[-1]
        ste = sum(r.ste_mask for r in res.records)

        kd = eq = replay = None
        if use_kd:
            # group 項：與 r_old 同輸入（快照存的全 slide 原型）
            state0 = torch.zeros(STATE_DIM, dtype=Zc.dtype)
            r_new = f_g.score(entry.group_prototypes.to(Zc.dtype), q_tau, state0,
                              use_query=False, use_state=False)
            kd = l_kd(entry.r_old.to(r_new.dtype), r_new,
                      entry.s_old.to(last.s.dtype), last.s,     # patch 項：位置對齊
                      group_weight=kd_group_weight)
        if use_eq:
            if u_old_mode not in U_OLD_MODES:
                raise ValueError(f"unknown u_old_mode: {u_old_mode}; expected {U_OLD_MODES}")
            _idx, pos = selected_from_entry(entry, budget)
            if u_old_mode == "snapshot":
                u_old = float(entry.u_old.index_select(0, pos).sum())
                logits_uniform = mask_logits(frozen_head(Zc, last.s, ste, f_txt, logit_scale,
                                                         weighting="uniform"),
                                             entry.class_mask_old)
            else:
                ste_old = torch.zeros_like(last.s)
                ste_old[pos.to(torch.long)] = 1.0      # P_old 在子集內的位置
                with torch.no_grad():
                    logits_old = mask_logits(frozen_head(Zc, last.s.detach(), ste_old, f_txt,
                                                         logit_scale, weighting="uniform"),
                                             class_mask)
                    u_old = float(differentiable_utility(logits_old, label))
                logits_uniform = mask_logits(frozen_head(Zc, last.s, ste, f_txt, logit_scale,
                                                         weighting="uniform"),
                                             class_mask)
            eq = l_eq(differentiable_utility(logits_uniform, label), u_old, mode=eq_mode)
        if use_replay:
            replay = l_diag(frozen_head(Zc, last.s, ste, f_txt, logit_scale), label, class_mask)
    return kd, eq, replay


def continual_terms(entry, cfg, models, f_txt, logit_scale, tissue, *,
                    budget=DEFAULT_BUDGET, chunk=DEFAULT_CHUNK, q_tau=None,
                    spec=None, use_kd=True, use_eq=True, use_replay=True,
                    eq_mode="hinge", kd_group_weight=1.0, class_mask=None,
                    u_old_mode="snapshot", candidate_only=False, cand_store=None):
    """對一筆記憶體 entry 算出 (L_KD, L_eq, L_replay)；關掉的項回傳 None。

    `class_mask`（DR-052）= **當前** stage 的 C_t，只用於 replay 的 CE；
    hinge 的 U_new 遮罩到 entry 自己的 `class_mask_old`（快照時的 C_old），
    與 u_old 同口徑。固定頭兩者皆 None → 逐位元不變。

    `u_old_mode`（DR-054）：
      "snapshot"（預設）= 上述現行路徑，逐位元不變；
      "current"  = replay 時由快照的 P_old（`selected_from_entry`）等權池化、遮罩到
                   **當前** C_t 重算 U_old，U_new 也遮罩到當前 C_t —— 同口徑、不用未見
                   類別、不讀快照的 u_old／class_mask_old。

    L_replay 就是 L_diag 跑在從 M 取回的舊樣本上 —— replay 是資料機制，
    這一項沒有任何特殊之處。
    """
    from selector.grouping import assign_groups

    f_g, f_p = models
    spec = spec or {}
    q = q_tau if q_tau is not None else torch.zeros(512)
    if candidate_only:                     # DR-056：只讀側倉，不碰原始特徵檔
        return _candidate_terms(entry, models, f_txt, logit_scale, tissue, budget=budget,
                                chunk=chunk, q_tau=q, spec=spec, use_kd=use_kd,
                                use_eq=use_eq, use_replay=use_replay, eq_mode=eq_mode,
                                kd_group_weight=kd_group_weight, class_mask=class_mask,
                                u_old_mode=u_old_mode, cand_store=cand_store)
    Z, _Z_cand, label = reload_features(entry, cfg)
    grp = assign_groups(Z, tissue)
    res = run_rounds(Z, grp, q, f_g, f_p, budget=budget, chunk=chunk, **spec)
    last = res.records[-1]
    ste = sum(r.ste_mask for r in res.records)

    kd = eq = replay = None
    if use_kd:
        cand = entry.cand_idx.to(torch.long)
        kd = l_kd(entry.r_old.to(last.r.dtype), last.r,
                  entry.s_old.to(last.s.dtype), last.s.index_select(0, cand),
                  group_weight=kd_group_weight)
    if use_eq:
        if u_old_mode not in U_OLD_MODES:
            raise ValueError(f"unknown u_old_mode: {u_old_mode}; expected {U_OLD_MODES}")
        idx_old, pos = selected_from_entry(entry, budget)
        if u_old_mode == "snapshot":
            u_old = float(entry.u_old.index_select(0, pos).sum())
            logits_uniform = mask_logits(frozen_head(Z, last.s, ste, f_txt, logit_scale,
                                                     weighting="uniform"),
                                         entry.class_mask_old)
        else:                                   # DR-054 "current"
            ste_old = torch.zeros_like(last.s)
            ste_old[idx_old.to(torch.long)] = 1.0
            with torch.no_grad():
                logits_old = mask_logits(frozen_head(Z, last.s.detach(), ste_old, f_txt,
                                                     logit_scale, weighting="uniform"),
                                         class_mask)
                u_old = float(differentiable_utility(logits_old, label))
            logits_uniform = mask_logits(frozen_head(Z, last.s, ste, f_txt, logit_scale,
                                                     weighting="uniform"),
                                         class_mask)
        eq = l_eq(differentiable_utility(logits_uniform, label), u_old, mode=eq_mode)
    if use_replay:
        replay = l_diag(frozen_head(Z, last.s, ste, f_txt, logit_scale), label,
                        class_mask)
    return kd, eq, replay


def total_loss(l_evidence, kd=None, eq=None, replay=None, *,
               lambda_kd=1.0, lambda_eq=1.0, lambda_replay=1.0):
    """L_total = L_evidence + L_continual。三項全關時位元等同 L_evidence。"""
    cont, parts = continual_loss(kd, eq, replay, lambda_kd=lambda_kd,
                                 lambda_eq=lambda_eq, lambda_replay=lambda_replay,
                                 dtype=l_evidence.dtype)
    return l_evidence + cont, parts
