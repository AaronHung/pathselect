"""S4-3 / S4-4 — CL 層的保存損失，以及把 counterfactual gain 接成監督訊號。

    L_continual = λ_kd · L_KD + λ_eq · L_eq + λ_r · L_replay
    L_total     = L_evidence + L_continual

L_KD      群層 + patch 層的選取行為蒸餾：KL(r_old ‖ r_new) + KL(s_old ‖ s_new)
L_eq      utility 等價保存：新選的證據要維持舊證據的診斷效用
L_replay  **標準 experience replay**。replay 是資料機制 —— 從 Selection Memory
          取回舊樣本重新跑一次；L_replay 只是「一般任務損失（L_diag）跑在回來的
          舊資料上」，不是第三種新 loss。

三項各自可獨立開關；全部關閉時 `continual_loss` 回傳恰好為零的張量，
訓練路徑與 SeqFT 位元相同。
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

DEFAULT_LAMBDAS = {"kd": 1.0, "eq": 1.0, "replay": 1.0}
EQ_MODES = ("hinge", "l2")


# ── L_KD ────────────────────────────────────────────────────────────────────

def _kl(target_logits: torch.Tensor, student_logits: torch.Tensor,
        tau: float = 1.0) -> torch.Tensor:
    """KL(softmax(target) ‖ softmax(student))，兩者都是同一組元素上的分布。"""
    q = F.softmax(target_logits.detach().reshape(-1) / tau, dim=0)
    log_p = F.log_softmax(student_logits.reshape(-1) / tau, dim=0)
    return F.kl_div(log_p, q, reduction="sum")


def l_kd(r_old: torch.Tensor, r_new: torch.Tensor,
         s_old: torch.Tensor, s_new: torch.Tensor,
         tau: float = 1.0, group_weight: float = 1.0) -> torch.Tensor:
    """群層 + patch 層的選取行為蒸餾。

    r_*: [J] group 分數；s_*: [len(cand_idx)] 候選 patch 分數（同一組 cand_idx）。

    group_weight=0 → **完全不計算** group 項，結果與「只有 patch 蒸餾」位元相同。
    用來隔離 group-level distillation 的效果（DR-022）：架構圖 Panel I 畫了這一項，
    但在 flat 模式下 F_g 對選取零影響，所以它從未被實際測試過。
    """
    if r_old.numel() != r_new.numel():
        raise ValueError(f"r 長度不符：{r_old.numel()} vs {r_new.numel()}")
    if s_old.numel() != s_new.numel():
        raise ValueError(f"s 長度不符：{s_old.numel()} vs {s_new.numel()}")
    patch = _kl(s_old, s_new, tau)
    if group_weight == 0.0:
        return patch
    return group_weight * _kl(r_old, r_new, tau) + patch


# ── L_eq ────────────────────────────────────────────────────────────────────

def differentiable_utility(logits_uniform: torch.Tensor, label: int) -> torch.Tensor:
    """U = log C − CE(logits, y)，可微。

    與 A3 的 U 同一個定義：evidence 等權平均、frozen head、沿選取順序累加的
    counterfactual gain 會 telescope 成這個封閉形式。U > 0 代表證據推向正確類別。
    """
    C = logits_uniform.shape[-1]
    target = torch.tensor([int(label)], dtype=torch.long, device=logits_uniform.device)
    ce = F.cross_entropy(logits_uniform.reshape(1, -1), target)
    return math.log(C) - ce


def l_eq(u_new: torch.Tensor, u_old: float, mode: str = "hinge") -> torch.Tensor:
    """utility 等價保存：新選的證據要**維持**舊證據的診斷效用。

    mode="hinge"（預設，忠於「維持」的字面）：只在退步時罰，進步不罰。
    mode="l2"：雙向對齊，退步與進步都罰。
    """
    if mode not in EQ_MODES:
        raise ValueError(f"unknown eq mode: {mode}; expected {EQ_MODES}")
    gap = torch.as_tensor(float(u_old), dtype=u_new.dtype,
                          device=u_new.device) - u_new
    return F.relu(gap) if mode == "hinge" else gap.pow(2)


# ── L_util（S4-4）───────────────────────────────────────────────────────────

def l_util(patch_score: torch.Tensor, utility: torch.Tensor,
           tau: float = 1.0) -> torch.Tensor:
    """把 counterfactual gain 接成監督訊號。

    形式與 L_sem 相同（KL(target ‖ student)），只是 anchor 從 semantic prior 換成
    每個候選 patch 的 counterfactual gain u_i。兩者都是候選集合上的分布。
    ⚠️ u_i 需要 label，因此本項只在訓練期使用。
    """
    if patch_score.numel() != utility.numel():
        raise ValueError(f"長度不符：{patch_score.numel()} vs {utility.numel()}")
    return _kl(utility, patch_score, tau)


# ── 組合 ────────────────────────────────────────────────────────────────────

def continual_loss(kd: Optional[torch.Tensor] = None,
                   eq: Optional[torch.Tensor] = None,
                   replay: Optional[torch.Tensor] = None, *,
                   lambda_kd: float = 1.0, lambda_eq: float = 1.0,
                   lambda_replay: float = 1.0,
                   device=None, dtype=None) -> tuple[torch.Tensor, dict]:
    """L_continual = λ_kd·L_KD + λ_eq·L_eq + λ_r·L_replay。

    三項都是 None（全部關閉）時回傳**恰好為零**的張量 —— 加到 L_evidence 上不會
    改變任何位元，所以 baseline 與 method 走的是同一條 code path。
    """
    parts = {"L_KD": None, "L_eq": None, "L_replay": None}
    total = torch.zeros((), device=device, dtype=dtype or torch.float32)
    if kd is not None:
        total = total + lambda_kd * kd
        parts["L_KD"] = float(kd.detach())
    if eq is not None:
        total = total + lambda_eq * eq
        parts["L_eq"] = float(eq.detach())
    if replay is not None:
        total = total + lambda_replay * replay
        parts["L_replay"] = float(replay.detach())
    return total, parts


def is_disabled(kd, eq, replay) -> bool:
    """三項全關 → 等價於 SeqFT。"""
    return kd is None and eq is None and replay is None
