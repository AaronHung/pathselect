"""F_g / F_p 與 straight-through top-K。"""
from __future__ import annotations

import pytest
import torch

from selector.model import (HIDDEN, SELECTOR_INPUT_DIM, GroupSelector, PatchSelector,
                            build_input, straight_through_topk, topk_indices)

D = 512


def test_input_dim_is_1537():
    assert SELECTOR_INPUT_DIM == 1537 == D + D + (D + 1)


def test_build_input_layout():
    X = torch.randn(7, D)
    q = torch.randn(D)
    st = torch.randn(D + 1)
    u = build_input(X, q, st)
    assert u.shape == (7, SELECTOR_INPUT_DIM)
    assert torch.equal(u[:, :D], X)
    assert torch.equal(u[3, D:2 * D], q)
    assert torch.equal(u[3, 2 * D:], st)


def test_architecture_is_1537_256_1():
    for cls in (GroupSelector, PatchSelector):
        m = cls()
        lin = [l for l in m.mlp if isinstance(l, torch.nn.Linear)]
        assert len(lin) == 2
        assert lin[0].in_features == SELECTOR_INPUT_DIM and lin[0].out_features == HIDDEN
        assert lin[1].in_features == HIDDEN and lin[1].out_features == 1
        assert isinstance(m.mlp[1], torch.nn.GELU)


def test_ste_forward_is_hard_zero_one():
    torch.manual_seed(0)
    s = torch.randn(50, requires_grad=True)
    m = straight_through_topk(s, 8)
    vals = set(m.detach().reshape(-1).tolist())
    assert vals <= {0.0, 1.0}, vals
    assert float(m.detach().sum()) == 8.0
    assert torch.equal(m.detach().nonzero().reshape(-1).sort().values,
                       torch.topk(s.detach(), 8).indices.sort().values)


def test_ste_backward_gradient_is_non_zero():
    torch.manual_seed(0)
    model = PatchSelector()
    X, q, st = torch.randn(50, D), torch.randn(D), torch.randn(D + 1)
    s = model.score(X, q, st)
    mask = straight_through_topk(s, 8)
    # 目標必須與 mask 的分佈有關；sum(mask) 恆為 K，梯度會恆等於零
    (mask * torch.randn(50)).sum().backward()
    g = model.mlp[0].weight.grad
    assert g is not None and float(g.norm()) > 0


def test_hard_mask_alone_has_no_gradient_path():
    """對照：直接對 hard mask backprop 連計算圖都沒有 —— 這正是禁止的做法。"""
    torch.manual_seed(0)
    s = torch.randn(20, requires_grad=True)
    hard = torch.zeros_like(s)
    hard.scatter_(0, torch.topk(s, 4).indices, 1.0)
    obj = (hard * torch.randn(20)).sum()
    assert obj.grad_fn is None and not obj.requires_grad
    with pytest.raises(RuntimeError, match="does not require grad"):
        obj.backward()
    assert s.grad is None


def test_ste_respects_mask():
    torch.manual_seed(0)
    s = torch.randn(20)
    avail = torch.zeros(20, dtype=torch.bool)
    avail[5:10] = True
    m = straight_through_topk(s, 3, mask=avail)
    picked = m.detach().nonzero().reshape(-1)
    assert picked.numel() == 3 and set(picked.tolist()) <= set(range(5, 10))


def test_ste_k_larger_than_available_is_clamped():
    s = torch.randn(20)
    avail = torch.zeros(20, dtype=torch.bool)
    avail[:2] = True
    m = straight_through_topk(s, 10, mask=avail)
    assert float(m.detach().sum()) == 2.0


def test_topk_indices_matches_ste_forward():
    torch.manual_seed(0)
    s = torch.randn(40)
    avail = torch.rand(40) > 0.3
    m = straight_through_topk(s, 6, mask=avail)
    idx = topk_indices(s, 6, mask=avail)
    assert torch.equal(m.detach().nonzero().reshape(-1).sort().values,
                       idx.sort().values)


def test_temperature_must_be_positive():
    with pytest.raises(ValueError, match="temperature"):
        straight_through_topk(torch.randn(5), 2, temperature=0.0)
