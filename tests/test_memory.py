"""CONTRACT-3 — bounded Selection Memory 的資料結構。"""
from __future__ import annotations

import pytest
import torch

from selector.memory import (CANDIDATE_SIZE, MEMORY_CAPACITY, SelectionMemory,
                             SelectionMemoryEntry, make_entry)
from selector.state import EvidenceState

D, J = 512, 8


def _entry(tau="tcga_lung", sid="slide-0", n_cand=256):
    return SelectionMemoryEntry(
        tau=tau, slide_id=sid, e_t=torch.randn(D), B_tilde_t=0.5,
        r_old=torch.randn(J), cand_idx=torch.arange(n_cand),
        s_old=torch.randn(n_cand), u_old=torch.randn(n_cand))


def test_capacity_constant_is_512():
    assert MEMORY_CAPACITY == 512 and CANDIDATE_SIZE == 256


def test_entry_holds_no_patch_features():
    e = _entry()
    fields = set(SelectionMemoryEntry.__dataclass_fields__)
    assert fields == {"tau", "slide_id", "e_t", "B_tilde_t", "r_old",
                      "cand_idx", "s_old", "u_old"}
    # slide_id + index 就是重載的依據，entry 本身不得帶 [n, 512] 的東西
    for name in fields:
        v = getattr(e, name)
        if isinstance(v, torch.Tensor):
            assert v.dim() <= 1, name


def test_candidate_length_is_enforced():
    with pytest.raises(ValueError, match="256"):
        _entry(n_cand=257)


def test_mismatched_score_length_is_rejected():
    with pytest.raises(ValueError, match="s_old"):
        SelectionMemoryEntry(tau="t", slide_id="s", e_t=torch.randn(D), B_tilde_t=1.0,
                             r_old=torch.randn(J), cand_idx=torch.arange(10),
                             s_old=torch.randn(9), u_old=torch.randn(10))


def test_memory_is_bounded():
    m = SelectionMemory()
    for i in range(MEMORY_CAPACITY + 37):
        m.add(_entry(sid=f"slide-{i}", n_cand=4))
    assert len(m) == MEMORY_CAPACITY
    assert [e.slide_id for e in m][0] == f"slide-{37}"        # FIFO 汰換最舊的


def test_capacity_above_the_constant_is_rejected():
    with pytest.raises(ValueError):
        SelectionMemory(capacity=MEMORY_CAPACITY + 1)


def test_by_task_and_tasks():
    m = SelectionMemory()
    m.add(_entry(tau="tcga_lung", n_cand=4))
    m.add(_entry(tau="tcga_brca", n_cand=4))
    m.add(_entry(tau="tcga_lung", n_cand=4))
    assert m.tasks() == ["tcga_brca", "tcga_lung"]
    assert len(m.by_task("tcga_lung")) == 2


def test_make_entry_detaches_and_slices():
    Z = torch.nn.functional.normalize(torch.randn(100, D), dim=-1).requires_grad_(True)
    st = EvidenceState(Z, 64)
    idx = torch.tensor([1, 2, 3])
    st.update(Z.index_select(0, idx).detach(), idx)
    s_all = torch.randn(100, requires_grad=True)
    cand = torch.tensor([10, 20, 30, 40])
    e = make_entry("tcga_rcc", "sid-1", st, torch.randn(J, requires_grad=True),
                   cand, s_all)
    assert not e.e_t.requires_grad and not e.s_old.requires_grad
    assert torch.equal(e.s_old, s_all.detach().index_select(0, cand))
    assert e.u_old.shape == (4,)
