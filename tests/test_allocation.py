"""CONTRACT-1 — chunk 分配（largest remainder + 溢出回流）。"""
from __future__ import annotations

import torch

from selector.allocation import allocate

J = 8


def test_sum_equals_chunk_and_non_negative_integers():
    torch.manual_seed(0)
    r = torch.randn(J)
    b = allocate(r, 8, torch.ones(J, dtype=torch.bool), torch.full((J,), 100))
    assert b.dtype == torch.long
    assert int(b.sum()) == 8
    assert int(b.min()) >= 0


def test_randomised_200_cases():
    """隨機 200 組 (r, c, group sizes)：sum == c、且沒有 b_j 超過該組可用數。"""
    g = torch.Generator().manual_seed(1234)
    for case in range(200):
        j = int(torch.randint(2, 12, (1,), generator=g))
        r = torch.randn(j, generator=g) * float(torch.randint(1, 5, (1,), generator=g))
        c = int(torch.randint(1, 17, (1,), generator=g))
        cap = torch.randint(0, 20, (j,), generator=g)
        mask = torch.rand(j, generator=g) > 0.2
        b = allocate(r, c, mask, cap)

        usable = int(cap[mask & (cap > 0)].sum())
        assert int(b.sum()) == min(c, usable), (case, b.tolist(), c, usable)
        assert bool((b <= cap).all()), (case, b.tolist(), cap.tolist())
        assert bool((b[~mask] == 0).all()), (case, b.tolist(), mask.tolist())
        assert int(b.min()) >= 0


def test_empty_groups_never_receive_budget():
    r = torch.tensor([5.0, 5.0, 5.0, 5.0])
    mask = torch.tensor([True, False, True, False])
    cap = torch.tensor([10, 0, 10, 0])
    b = allocate(r, 8, mask, cap)
    assert b.tolist()[1] == 0 and b.tolist()[3] == 0
    assert int(b.sum()) == 8


def test_overflow_reflows_by_r_order():
    """某 group 容量不足時，溢出流向 r_j 較大的其他 group。"""
    r = torch.tensor([10.0, 1.0, 0.0])
    cap = torch.tensor([1, 50, 50])
    b = allocate(r, 8, torch.ones(3, dtype=torch.bool), cap)
    assert int(b.sum()) == 8
    assert int(b[0]) == 1                      # 被容量卡住
    assert int(b[1]) >= int(b[2])              # 溢出優先給 r 較大者


def test_total_capacity_shortfall_is_capped():
    r = torch.zeros(3)
    cap = torch.tensor([1, 1, 1])
    b = allocate(r, 8, torch.ones(3, dtype=torch.bool), cap)
    assert int(b.sum()) == 3 and bool((b <= cap).all())


def test_zero_chunk_allocates_nothing():
    b = allocate(torch.randn(J), 0, torch.ones(J, dtype=torch.bool), torch.full((J,), 5))
    assert int(b.sum()) == 0
