"""Counterfactual gain u_i —— 加入一個候選 patch 能讓 diagnosis loss 降多少。

必須向量化。加入一個 patch 只是對 evidence 均值做 rank-1 更新：

    S       = sum(E_t)                          # [D]
    E_cand  = (S + X) / (|E_t| + 1)             # [N, D]  一次算完全部候選
    E_cand  = normalize(E_cand)
    logits  = logit_scale * E_cand @ f_txt.T    # [N, C]  一個 matmul
    u_i     = loss(current) - loss(logits[i])

**Pre-registered 近似**：已選 evidence 用當前權重，候選 patch 以等權加入。
理由是加入之後 softmax 權重會整組重算，那是遞迴定義；固定當前權重讓 u_i 有
封閉形式、可一次矩陣算完。這個近似在此明確登記，不是實作疏漏。

⚠️ u_i 需要 label，因此**只在訓練期使用**（L_util 的監督訊號、Selection Memory
   的 u_old）。selector 的 forward 路徑永遠不呼叫本模組。
"""
from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F

#: CONTRACT-3：候選集合固定 256，與 Selection Memory 的 cand_idx 同一組。
CANDIDATE_SIZE = 256


def top_candidates(scores: torch.Tensor, available: torch.Tensor,
                   k: int = CANDIDATE_SIZE) -> torch.Tensor:
    """[<=k]：在仍可選的 patch 中依當前 s_i 取 top-k，回傳原 slide 的 index。"""
    masked = scores.masked_fill(~available, float("-inf"))
    n_avail = int(available.sum())
    if n_avail == 0:
        return torch.empty(0, dtype=torch.long, device=scores.device)
    return torch.topk(masked, min(k, n_avail)).indices


def _ce(logits: torch.Tensor, label: int) -> torch.Tensor:
    """logits [.., C] → cross-entropy against a single label。"""
    flat = logits.reshape(-1, logits.shape[-1])
    target = torch.full((flat.shape[0],), int(label), dtype=torch.long,
                        device=logits.device)
    return F.cross_entropy(flat, target, reduction="none")


def current_logits(evidence_sum: torch.Tensor, n_selected: int,
                   f_txt: torch.Tensor, logit_scale) -> torch.Tensor:
    """[1, C]：當前 evidence 的 logits；尚未選任何 patch 時回全零（= 均勻分佈）。"""
    if n_selected <= 0:
        return torch.zeros(1, f_txt.shape[0], dtype=f_txt.dtype, device=f_txt.device)
    e = F.normalize((evidence_sum / n_selected).reshape(1, -1), dim=-1)
    return logit_scale * (e @ f_txt.t())


@torch.no_grad()
def counterfactual_gain(evidence_sum: torch.Tensor, n_selected: int,
                        X_cand: torch.Tensor, f_txt: torch.Tensor,
                        logit_scale, label: int,
                        loss_fn: Optional[Callable] = None) -> torch.Tensor:
    """[N]：每個候選 patch 的 u_i = loss(current) - loss(with candidate)。

    正值代表「加進來會讓 loss 下降」，也就是這個 patch 有用。
    """
    if X_cand.dim() != 2:
        raise ValueError(f"X_cand must be [N, D], got {tuple(X_cand.shape)}")
    loss_fn = loss_fn or _ce

    E_cand = (evidence_sum.reshape(1, -1) + X_cand) / (n_selected + 1)   # [N, D]
    E_cand = F.normalize(E_cand, dim=-1)
    logits_cand = logit_scale * (E_cand @ f_txt.t())                     # [N, C]

    l_now = loss_fn(current_logits(evidence_sum, n_selected, f_txt, logit_scale), label)
    l_cand = loss_fn(logits_cand, label)                                 # [N]
    return l_now.reshape(1) - l_cand


@torch.no_grad()
def counterfactual_gain_loop(evidence_sum, n_selected, X_cand, f_txt,
                             logit_scale, label, loss_fn=None) -> torch.Tensor:
    """逐一計算的參考實作 —— **只給單元測試比對用**，正式路徑一律用向量化版。"""
    loss_fn = loss_fn or _ce
    l_now = loss_fn(current_logits(evidence_sum, n_selected, f_txt, logit_scale), label)
    out = []
    for i in range(X_cand.shape[0]):
        e = F.normalize(((evidence_sum + X_cand[i]) / (n_selected + 1)).reshape(1, -1),
                        dim=-1)
        out.append(l_now.reshape(()) - loss_fn(logit_scale * (e @ f_txt.t()), label).reshape(()))
    return torch.stack(out)


@torch.no_grad()
def sequential_utility_total(Z: torch.Tensor, selected_idx: torch.Tensor,
                            f_txt: torch.Tensor, logit_scale, label: int) -> float:
    """U(S)：沿**選取順序**累加 counterfactual gain。

    每一步的 evidence 是「到目前為止選到的 patch 的等權和」，加入下一個 patch 的
    gain 就是 loss 的下降量。整條加總會 telescope 成 loss(空證據) − loss(最終證據)，
    因此只取決於選了哪些 patch，與 selector 的參數無關。

    ⚠️ 這**不等於**「從空證據出發、各自獨立算單 patch gain 再加總」——
    後者會重複計算彼此的貢獻。A3 utility retention 用的是本函式的定義。
    """
    S = torch.zeros(Z.shape[1], dtype=Z.dtype, device=Z.device)
    total, n = 0.0, 0
    for i in selected_idx.reshape(-1).tolist():
        x = Z[i].reshape(1, -1)
        total += float(counterfactual_gain(S, n, x, f_txt, logit_scale, label)[0])
        S = S + Z[i]
        n += 1
    return total
