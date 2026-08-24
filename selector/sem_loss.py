"""L_sem 的完整兩層版本（G3）。

HistoSelect（arXiv 2603.00667v2, 式 10）的語意 IB 是**兩層**：

    L_sem = (1/M) Σ_j KL( B(r_j) ‖ B(p_j^g) )  +  KL( B(s_i) ‖ B(p_i^sem) )
            └────── group 層（本檔新增）──────┘   └──── patch 層（既有）────┘

本 repo 原本只實作 patch 項（`selector.train.l_sem`）—— 已由 mutation 實測確認：
擾動 group prototype 時 L_sem 位元不變。詳見 CLAIMS C-26。

⚠️ `beta_g = 0` 時**完全不計算** group 項，結果與 `selector.train.l_sem` 位元相同。
   這是讓 G3 的基準與既有主表可比的前提，由測試把關。

⚠️ 本檔是**新增的消融維度**，不取代主方法。DR-007 pre-register 的是「用哪個
   prior」（discriminative vs max_sim），不是「用哪幾層」；主方法維持 patch-only。
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def _kl(prior: torch.Tensor, score: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """KL(q ‖ pi)，q = softmax(prior/tau) 為 target、pi = softmax(score/tau) 為 student。

    與 `selector.train.l_sem` 的內部形式完全一致（同樣的 log_softmax / softmax /
    kl_div(reduction="sum") 順序），以確保 beta_g=0 時位元相同。
    """
    log_pi = F.log_softmax(score / tau, dim=0)
    q = F.softmax(prior.to(score.dtype) / tau, dim=0)
    return F.kl_div(log_pi, q, reduction="sum")


def group_sem_term(group_score: torch.Tensor, group_prior: torch.Tensor,
                   tau: float = 1.0) -> torch.Tensor:
    """group 層的語意錨：KL( B(r_j) ‖ B(p_j^g) )。

    group_score: [J']  非空 group 的 r_j
    group_prior: [J']  同一批 group 的 p_j^g（用與 patch 相同的 discriminative 形式）
    """
    if group_score.numel() != group_prior.numel():
        raise ValueError(f"長度不符：{group_score.numel()} vs {group_prior.numel()}")
    return _kl(group_prior, group_score, tau)


def l_sem(patch_score: torch.Tensor, patch_prior: torch.Tensor,
          group_score: Optional[torch.Tensor] = None,
          group_prior: Optional[torch.Tensor] = None,
          beta_g: float = 0.0, tau: float = 1.0) -> torch.Tensor:
    """兩層 L_sem。beta_g=0 時與 `selector.train.l_sem(patch_score, patch_prior)` 位元相同。"""
    patch_term = _kl(patch_prior, patch_score, tau)
    if beta_g == 0.0 or group_score is None or group_prior is None:
        return patch_term
    return patch_term + beta_g * group_sem_term(group_score, group_prior, tau)
