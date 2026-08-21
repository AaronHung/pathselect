"""CONTRACT-2 — EvidenceState。"""
from __future__ import annotations

import pytest
import torch

from selector.state import EvidenceState

D, N, B = 512, 40, 16


def _Z():
    torch.manual_seed(0)
    return torch.nn.functional.normalize(torch.randn(N, D), dim=-1)


def test_e_t_is_all_zero_at_t0():
    st = EvidenceState(_Z(), B)
    assert st.t == 0 and st.n_selected == 0
    assert torch.equal(st.e_t, torch.zeros(D))
    assert st.B_tilde_t == 1.0


def test_feature_is_513_and_is_e_t_then_b_tilde():
    Z = _Z()
    st = EvidenceState(Z, B)
    idx = torch.tensor([1, 5, 9])
    st.update(Z.index_select(0, idx), idx)
    f = st.feature()
    assert f.shape == (D + 1,)
    assert torch.equal(f[:D], st.e_t)
    assert float(f[D]) == pytest.approx(st.B_tilde_t)


def test_e_t_equals_mean_of_selected_after_update():
    Z = _Z()
    st = EvidenceState(Z, B)
    idx = torch.tensor([3, 7, 11, 2])
    st.update(Z.index_select(0, idx), idx)
    assert torch.allclose(st.e_t, Z.index_select(0, idx).mean(0), atol=1e-6)
    idx2 = torch.tensor([20, 21])
    st.update(Z.index_select(0, idx2), idx2)
    both = torch.cat([idx, idx2])
    assert torch.allclose(st.e_t, Z.index_select(0, both).mean(0), atol=1e-6)


def test_selected_indices_leave_the_candidate_pool():
    Z = _Z()
    st = EvidenceState(Z, B)
    idx = torch.tensor([0, 4, 8])
    st.update(Z.index_select(0, idx), idx)
    cands = set(st.candidate_indices.tolist())
    assert cands.isdisjoint(set(idx.tolist()))
    assert len(cands) == N - 3
    assert not bool(st.available_mask.index_select(0, idx).any())


def test_reselecting_the_same_patch_is_rejected():
    Z = _Z()
    st = EvidenceState(Z, B)
    idx = torch.tensor([2, 3])
    st.update(Z.index_select(0, idx), idx)
    with pytest.raises(ValueError, match="重複"):
        st.update(Z.index_select(0, idx), idx)


def test_budget_counts_down():
    Z = _Z()
    st = EvidenceState(Z, B)
    for chunk in range(4):
        idx = torch.arange(chunk * 4, chunk * 4 + 4)
        st.update(Z.index_select(0, idx), idx)
    assert st.n_selected == 16 and st.B_t == 0 and st.B_tilde_t == 0.0


def test_e_t_is_detached_between_rounds():
    """CONTRACT-2：狀態不帶梯度跨輪。"""
    Z = _Z().requires_grad_(True)
    st = EvidenceState(Z, B)
    idx = torch.tensor([1, 2])
    st.update(Z.index_select(0, idx), idx)
    assert not st.e_t.requires_grad
    assert not st.feature().requires_grad


def test_reset_clears_everything():
    Z = _Z()
    st = EvidenceState(Z, B)
    idx = torch.tensor([1, 2])
    st.update(Z.index_select(0, idx), idx)
    st.reset(B)
    assert st.n_selected == 0 and torch.equal(st.e_t, torch.zeros(D))
    assert int(st.available_mask.sum()) == N
