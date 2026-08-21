"""CONTRACT-3 — bounded Selection Memory 的資料結構。"""
from __future__ import annotations

import pytest
import torch

from selector.memory import (CANDIDATE_SIZE, FIFO, MEMORY_CAPACITY, ReservoirSampling,
                             SelectionMemory, SelectionMemoryEntry, make_entry,
                             reload_features, selected_from_entry)
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


def test_memory_is_bounded_under_the_default_policy():
    """|M| 永不超過 512（預設 reservoir sampling）。"""
    m = SelectionMemory()
    for i in range(MEMORY_CAPACITY * 5):
        m.add(_entry(sid=f"slide-{i}", n_cand=4))
        assert len(m) <= MEMORY_CAPACITY
    assert len(m) == MEMORY_CAPACITY and m.n_seen == MEMORY_CAPACITY * 5


def test_default_policy_is_reservoir_and_is_swappable():
    assert isinstance(SelectionMemory().policy, ReservoirSampling)
    m = SelectionMemory(capacity=8, policy=FIFO())
    for i in range(20):
        m.add(_entry(sid=f"slide-{i}", n_cand=4))
    assert len(m) == 8
    assert [e.slide_id for e in m][0] == "slide-12"           # FIFO 汰換最舊的


def test_reservoir_keeps_a_mix_of_old_and_new():
    """reservoir 不該退化成「只留最新的」。"""
    m = SelectionMemory(capacity=50, policy=ReservoirSampling(seed=0))
    for i in range(1000):
        m.add(_entry(sid=f"slide-{i}", n_cand=4))
    kept = sorted(int(e.slide_id.split("-")[1]) for e in m)
    assert len(kept) == 50
    assert min(kept) < 500 < max(kept), kept[:5]


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


def test_selected_from_entry_recovers_the_top_k_by_s_old():
    """entry 沒有直接記錄選了誰，但 s_old 足以還原。"""
    torch.manual_seed(0)
    cand = torch.arange(100, 356)
    s_old = torch.randn(256)
    e = SelectionMemoryEntry(tau="tcga_lung", slide_id="x", e_t=torch.randn(D),
                             B_tilde_t=0.5, r_old=torch.randn(J), cand_idx=cand,
                             s_old=s_old, u_old=torch.randn(256))
    idx, pos = selected_from_entry(e, 8)
    assert idx.numel() == pos.numel() == 8
    assert torch.equal(pos, torch.topk(s_old, 8).indices)
    assert torch.equal(idx, cand.index_select(0, pos))


def test_reloaded_features_are_bit_identical_to_the_original():
    """不存 patch feature，用 slide_id + index 重載；重載回來必須位元相同。"""
    import pytest
    from pathlib import Path

    from selector.evaluate import read_slide, slide_dataset
    from selector.text_encoder import load_config

    cfg = load_config()
    task = cfg["tasks"][0]
    table = Path(cfg["dataset_root_dir"] + cfg["path_table"].format(task, task.upper()))
    if not table.exists():
        pytest.skip("dataset not available")

    ds, shift = slide_dataset(cfg, task, 0, "train")
    rec = read_slide(ds, shift, 0)
    cand = torch.arange(min(CANDIDATE_SIZE, rec.Z.shape[0]))
    e = SelectionMemoryEntry(tau=task, slide_id=rec.sid, e_t=torch.randn(D),
                             B_tilde_t=1.0, r_old=torch.randn(J), cand_idx=cand,
                             s_old=torch.randn(cand.numel()),
                             u_old=torch.randn(cand.numel()))
    Z_full, Z_cand, label = reload_features(e, cfg)
    assert torch.equal(Z_full, rec.Z)
    assert torch.equal(Z_cand, rec.Z.index_select(0, cand))
    assert label == rec.label
