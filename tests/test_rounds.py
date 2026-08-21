"""CONTRACT-1 — chunked sequential loop 的行為約束。"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from selector.grouping import NUM_GROUPS, assign_groups
from selector.model import GroupSelector, PatchSelector
from selector.rounds import (DEFAULT_BUDGET, DEFAULT_CHUNK, DEFAULT_GROUP_GRAD,
                             GROUP_GRAD_MODES, run_rounds)
from selector.state import EvidenceState

D = 512


def _fixture(n=900, seed=0):
    torch.manual_seed(seed)
    Z = F.normalize(torch.randn(n, D), dim=-1)
    t = F.normalize(torch.randn(NUM_GROUPS, D), dim=-1)
    q = F.normalize(torch.randn(D), dim=-1)
    return Z, assign_groups(Z, t), q, GroupSelector(), PatchSelector()


def test_defaults_are_b64_c8():
    assert DEFAULT_BUDGET == 64 and DEFAULT_CHUNK == 8


def test_eight_rounds_and_cumulative_counts():
    Z, g, q, fg, fp = _fixture()
    res = run_rounds(Z, g, q, fg, fp)
    assert res.n_rounds == 8
    assert [r.n_selected_after for r in res.records] == [8, 16, 24, 32, 40, 48, 56, 64]


def test_each_round_sums_to_chunk_not_budget():
    Z, g, q, fg, fp = _fixture()
    res = run_rounds(Z, g, q, fg, fp)
    for rec in res.records:
        assert int(rec.b.sum()) == DEFAULT_CHUNK
        assert rec.picked.numel() == DEFAULT_CHUNK


def test_no_patch_is_selected_twice():
    Z, g, q, fg, fp = _fixture()
    res = run_rounds(Z, g, q, fg, fp)
    sel = res.selected.tolist()
    assert len(sel) == 64 == len(set(sel))


def test_a_group_may_be_picked_across_multiple_rounds():
    """同一個 group 可跨輪重複選 —— 不是每輪換一個 group。"""
    Z, g, q, fg, fp = _fixture()
    res = run_rounds(Z, g, q, fg, fp)
    hit_rounds = [(rec.b > 0).nonzero().reshape(-1).tolist() for rec in res.records]
    repeated = set(hit_rounds[0]) & set(hit_rounds[1])
    assert repeated, hit_rounds[:2]


def test_custom_chunk_changes_round_count():
    Z, g, q, fg, fp = _fixture()
    res = run_rounds(Z, g, q, fg, fp, budget=64, chunk=16)
    assert res.n_rounds == 4
    assert [r.n_selected_after for r in res.records] == [16, 32, 48, 64]
    for rec in res.records:
        assert int(rec.b.sum()) == 16


def test_candidate_set_is_capped_and_only_available():
    Z, g, q, fg, fp = _fixture(n=1000)
    res = run_rounds(Z, g, q, fg, fp)
    seen: set[int] = set()
    for rec in res.records:
        assert rec.cand_idx.numel() <= 256
        assert seen.isdisjoint(set(rec.cand_idx.tolist()))
        seen.update(rec.picked.tolist())


def test_stops_when_slide_has_fewer_patches_than_budget():
    Z, g, q, fg, fp = _fixture(n=20)
    res = run_rounds(Z, g, q, fg, fp)
    assert res.selected.numel() == 20
    assert len(set(res.selected.tolist())) == 20


def test_state_can_be_supplied_and_is_advanced():
    Z, g, q, fg, fp = _fixture()
    st = EvidenceState(Z, DEFAULT_BUDGET)
    res = run_rounds(Z, g, q, fg, fp, state=st)
    assert res.state is st and st.n_selected == 64 and st.B_t == 0


def test_group_grad_mode_does_not_change_the_forward():
    """ste_allocation 只注入梯度；forward 必須位元相同。"""
    Z, g, q, fg, fp = _fixture()
    torch.manual_seed(5)
    a = run_rounds(Z, g, q, fg, fp, group_grad="none")
    b = run_rounds(Z, g, q, fg, fp, group_grad="ste_allocation")
    assert torch.equal(a.selected, b.selected)
    for ra, rb in zip(a.records, b.records):
        assert torch.equal(ra.b, rb.b)
        assert torch.equal(ra.ste_mask.detach(), rb.ste_mask.detach())


def test_default_group_grad_is_ste_allocation():
    """PI 裁定 1：主線必須讓 F_g 收得到梯度。"""
    assert DEFAULT_GROUP_GRAD == "ste_allocation"
    assert GROUP_GRAD_MODES[0] == "ste_allocation"


def test_group_selector_receives_gradient_under_the_default():
    """預設模式下 F_g 必須有梯度 —— 否則 +hierarchy 的結論會是構造保證的 null。"""
    Z, g, q, fg, fp = _fixture()
    res = run_rounds(Z, g, q, fg, fp)
    ste = sum(rec.ste_mask for rec in res.records)
    (ste * torch.randn_like(ste)).sum().backward()
    assert fg.mlp[0].weight.grad is not None
    assert float(fg.mlp[0].weight.grad.norm()) > 0


def test_none_mode_gives_no_group_gradient_and_is_ablation_only():
    """對照 ablation：none 模式下 F_g 完全收不到梯度。"""
    Z, g, q, fg, fp = _fixture()
    res = run_rounds(Z, g, q, fg, fp, group_grad="none")
    ste = sum(rec.ste_mask for rec in res.records)
    (ste * torch.randn_like(ste)).sum().backward()
    assert fg.mlp[0].weight.grad is None
