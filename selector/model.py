"""F_g / F_p —— group 與 patch 兩層評分網路，以及 top-K 的 straight-through 估計。

輸入維度（CONTRACT-2）：
    F_g:  [ g_j ; q_tau ; e_t ; B_tilde_t ]  = 512 + 512 + 512 + 1 = 1537
    F_p:  [ x_i ; q_tau ; e_t ; B_tilde_t ]  = 1537
兩者結構相同：Linear(1537 → 256) → GELU → Linear(256 → 1)。

top-K 一律走 straight-through：forward 是 hard 0/1 mask，backward 走
softmax(s / T)。禁止直接對 hard mask backprop（那條路梯度恆為零）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

FEAT_DIM = 512
QUERY_DIM = 512
STATE_DIM = FEAT_DIM + 1          # [e_t ; B_tilde_t]
SELECTOR_INPUT_DIM = FEAT_DIM + QUERY_DIM + STATE_DIM      # 1537
HIDDEN = 256


def build_input(features: torch.Tensor, q_tau: torch.Tensor,
                state_feature: torch.Tensor, *,
                use_query: bool = True, use_state: bool = True) -> torch.Tensor:
    """[N, 1537] = [features ; q_tau ; e_t ; B_tilde_t]。

    features:      [N, 512]  g_j（group 層）或 x_i（patch 層）
    q_tau:         [512]
    state_feature: [513]     EvidenceState.feature()

    use_query / use_state 為消融階梯用的開關：關掉的區塊**填零而不縮短維度**，
    網路架構與參數量在 L3–L6 之間完全相同，差異只在輸入資訊量。
    """
    if features.dim() != 2 or features.shape[1] != FEAT_DIM:
        raise ValueError(f"features must be [N, {FEAT_DIM}], got {tuple(features.shape)}")
    if q_tau.reshape(-1).shape[0] != QUERY_DIM:
        raise ValueError(f"q_tau must be [{QUERY_DIM}], got {tuple(q_tau.shape)}")
    if state_feature.reshape(-1).shape[0] != STATE_DIM:
        raise ValueError(f"state_feature must be [{STATE_DIM}], "
                         f"got {tuple(state_feature.shape)}")
    n = features.shape[0]
    q = q_tau.reshape(1, -1) if use_query else torch.zeros_like(q_tau.reshape(1, -1))
    st = (state_feature.reshape(1, -1) if use_state
          else torch.zeros_like(state_feature.reshape(1, -1)))
    tail = torch.cat([q, st], dim=-1)
    return torch.cat([features, tail.expand(n, -1).to(features.dtype)], dim=-1)


class _SelectorMLP(nn.Module):
    """Linear(1537 → 256) → GELU → Linear(256 → 1)。"""

    def __init__(self, in_dim: int = SELECTOR_INPUT_DIM, hidden: int = HIDDEN):
        super().__init__()
        self.in_dim = in_dim
        self.hidden = hidden
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        if u.shape[-1] != self.in_dim:
            raise ValueError(f"expected last dim {self.in_dim}, got {u.shape[-1]}")
        return self.mlp(u).squeeze(-1)

    def score(self, features: torch.Tensor, q_tau: torch.Tensor,
              state_feature: torch.Tensor, *, use_query: bool = True,
              use_state: bool = True) -> torch.Tensor:
        return self(build_input(features, q_tau, state_feature,
                                use_query=use_query, use_state=use_state))


class GroupSelector(_SelectorMLP):
    """F_g：吃 group prototype，輸出 r_j。"""


class PatchSelector(_SelectorMLP):
    """F_p：吃 patch embedding，輸出 s_i。"""


def straight_through_topk(scores: torch.Tensor, k: int,
                          temperature: float = 1.0,
                          mask: torch.Tensor | None = None) -> torch.Tensor:
    """[N] 0/1 mask：forward 是 hard top-K，backward 走 softmax(scores / T)。

    mask 給定時（True = 可選），不可選的位置在 forward 與 backward 都被壓成 0。
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    s = scores
    if mask is not None:
        s = s.masked_fill(~mask, float("-inf"))

    n_avail = int(mask.sum()) if mask is not None else s.shape[0]
    k_eff = max(0, min(int(k), n_avail))
    hard = torch.zeros_like(scores)
    if k_eff > 0:
        hard.scatter_(0, torch.topk(s, k_eff).indices, 1.0)

    soft = torch.softmax(s / temperature, dim=0)
    if mask is not None:
        soft = torch.nan_to_num(soft, nan=0.0)
    # forward = hard；backward 的梯度全部經由 soft。
    # 括號不可省：(hard + soft) - soft.detach() 會有浮點抵消誤差，
    # 導致 forward 值變成 0.9999999 而不是精確的 1.0。
    return hard + (soft - soft.detach())


def topk_indices(scores: torch.Tensor, k: int,
                 mask: torch.Tensor | None = None) -> torch.Tensor:
    """實際被選中的 index（與 straight_through_topk 的 forward 完全一致）。"""
    s = scores if mask is None else scores.masked_fill(~mask, float("-inf"))
    n_avail = int(mask.sum()) if mask is not None else s.shape[0]
    k_eff = max(0, min(int(k), n_avail))
    if k_eff == 0:
        return torch.empty(0, dtype=torch.long, device=scores.device)
    return torch.topk(s, k_eff).indices
