"""CONTRACT-1 — 每一輪把 chunk c 分配到各 group。

r_j → 非空 group 上 softmax → 乘 c → largest-remainder (Hare-Niemeyer) 取整。
保證 sum(b_j) == c 且 b_j 為非負整數。

若某 group 剩餘可選 patch 數 < 分到的 b_j，溢出依 r_j 大小回流其他仍有餘裕的
group（r_j 大者優先）。空 group 全程不參與分配。
"""
from __future__ import annotations

import torch


def _largest_remainder(quota: torch.Tensor, c: int) -> torch.Tensor:
    """Hare-Niemeyer：先取整數部分，餘數大者依序 +1，湊足 c。"""
    floor = torch.floor(quota)
    b = floor.to(torch.long)
    short = c - int(b.sum())
    if short > 0:
        remainder = quota - floor
        # 餘數大者優先；同餘數時 index 小者優先（決定性）
        order = torch.argsort(remainder, descending=True, stable=True)
        b[order[:short]] += 1
    return b


def allocate(r: torch.Tensor, c: int, group_mask: torch.Tensor,
             capacity: torch.Tensor) -> torch.Tensor:
    """回傳 b [J]，非負整數且 sum(b) == min(c, sum(capacity[非空 group]))。

    r:          [J]    group 層分數（未正規化）
    c:          int    這一輪的 chunk 大小
    group_mask: [J] bool  True = 非空 group（參與分配）
    capacity:   [J] long  每個 group 目前還剩幾個可選 patch
    """
    J = r.shape[0]
    device = r.device
    b = torch.zeros(J, dtype=torch.long, device=device)
    if c <= 0:
        return b

    active = group_mask & (capacity > 0)
    if not bool(active.any()):
        return b

    total_cap = int(capacity[active].sum())
    c_eff = min(int(c), total_cap)          # 全體都不夠時，最多只能發完剩下的

    idx = active.nonzero(as_tuple=False).reshape(-1)
    weights = torch.softmax(r.index_select(0, idx), dim=0)
    b_active = _largest_remainder(weights * c_eff, c_eff)

    # 溢出回流：超過該組可用數的部分，依 r_j 大小分給仍有餘裕者
    cap_active = capacity.index_select(0, idx)
    r_active = r.index_select(0, idx)
    overflow = int((b_active - cap_active).clamp(min=0).sum())
    b_active = torch.minimum(b_active, cap_active)
    if overflow > 0:
        order = torch.argsort(r_active, descending=True, stable=True)
        while overflow > 0:
            moved = False
            for pos in order.tolist():
                if overflow == 0:
                    break
                if b_active[pos] < cap_active[pos]:
                    b_active[pos] += 1
                    overflow -= 1
                    moved = True
            if not moved:                     # 所有 group 都滿了
                break

    b.index_copy_(0, idx, b_active)
    return b
