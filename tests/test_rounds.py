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


def test_defaults_are_the_ruled_operating_point():
    """PI 裁定 A：B=8、c=1，仍是 8 rounds。"""
    assert DEFAULT_BUDGET == 8 and DEFAULT_CHUNK == 1
    assert DEFAULT_BUDGET // DEFAULT_CHUNK == 8


def test_config_matches_the_module_default_operating_point():
    from selector.text_encoder import load_config
    cfg = load_config()
    assert cfg["budget"] == DEFAULT_BUDGET and cfg["chunk"] == DEFAULT_CHUNK


def test_eight_rounds_and_cumulative_counts():
    Z, g, q, fg, fp = _fixture()
    res = run_rounds(Z, g, q, fg, fp)
    assert res.n_rounds == 8
    assert [r.n_selected_after for r in res.records] == [1, 2, 3, 4, 5, 6, 7, 8]


def test_each_round_sums_to_chunk_not_budget():
    Z, g, q, fg, fp = _fixture()
    for budget, chunk in ((8, 1), (64, 8), (8, 2)):
        res = run_rounds(Z, g, q, fg, fp, budget=budget, chunk=chunk)
        for rec in res.records:
            assert int(rec.b.sum()) == chunk
            assert rec.picked.numel() == chunk


def test_no_patch_is_selected_twice():
    Z, g, q, fg, fp = _fixture()
    res = run_rounds(Z, g, q, fg, fp, budget=64, chunk=8)
    sel = res.selected.tolist()
    assert len(sel) == 64 == len(set(sel))


class _PeakedGroupSelector(torch.nn.Module):
    """回傳固定峰值 r 的 stub —— 用來測「配額集中時的行為」。

    合成 fixture 的 group prototype 彼此太相似，真的 F_g 給出的 r 幾乎均勻，
    放大權重也只是等比例拉開、softmax 後仍接近均分。要測 CONTRACT-1 的
    「同組可跨輪重複選」必須有真正集中的配額，因此直接注入。
    """

    def __init__(self, r: torch.Tensor):
        super().__init__()
        self.r = torch.nn.Parameter(r.clone())

    def score(self, features, q_tau, state_feature, **_kw):
        return self.r


def test_a_group_may_be_picked_across_multiple_rounds():
    """CONTRACT-1：同一個 group 可跨輪重複選 —— 不是每輪換一個 group。

    ⚠️ 在 per_budget 配額下這要在 **r 有峰值**時才觀察得到：r 近乎均勻時配額是
    每組 1 個，自然每輪換組（那不是違反契約，是配額本來就該平均）。
    """
    Z, g, q, _fg, fp = _fixture()
    peaked = torch.tensor([6.0, 3.0, 1.0, 0.0, 0.0, -1.0, -3.0, -6.0])
    res = run_rounds(Z, g, q, _PeakedGroupSelector(peaked), fp, budget=8, chunk=1)
    per_round = [int(rec.b.argmax()) for rec in res.records]
    assert per_round[0] == per_round[1], f"配額集中時前兩輪應同組：{per_round}"
    assert len(set(per_round)) < len(per_round), f"不該每輪都換組：{per_round}"


def test_uniform_group_scores_spread_one_per_group():
    """對照：r 均勻時配額每組 1 個，八輪剛好走遍八組。"""
    Z, g, q, _fg, fp = _fixture()
    res = run_rounds(Z, g, q, _PeakedGroupSelector(torch.zeros(8)), fp,
                     budget=8, chunk=1)
    per_round = [int(rec.b.argmax()) for rec in res.records]
    assert len(set(per_round)) == 8, per_round


def test_custom_chunk_changes_round_count():
    Z, g, q, fg, fp = _fixture()
    res = run_rounds(Z, g, q, fg, fp, budget=64, chunk=16)
    assert res.n_rounds == 4
    assert [r.n_selected_after for r in res.records] == [16, 32, 48, 64]
    for rec in res.records:
        assert int(rec.b.sum()) == 16


def test_candidate_set_is_capped_and_only_available():
    Z, g, q, fg, fp = _fixture(n=1000)
    res = run_rounds(Z, g, q, fg, fp, budget=64, chunk=8)
    seen: set[int] = set()
    for rec in res.records:
        assert rec.cand_idx.numel() <= 256
        assert seen.isdisjoint(set(rec.cand_idx.tolist()))
        seen.update(rec.picked.tolist())


def test_stops_when_slide_has_fewer_patches_than_budget():
    Z, g, q, fg, fp = _fixture(n=20)
    res = run_rounds(Z, g, q, fg, fp, budget=64, chunk=8)
    assert res.selected.numel() == 20
    assert len(set(res.selected.tolist())) == 20


def test_state_can_be_supplied_and_is_advanced():
    Z, g, q, fg, fp = _fixture()
    st = EvidenceState(Z, DEFAULT_BUDGET)
    res = run_rounds(Z, g, q, fg, fp, state=st)
    assert res.state is st and st.n_selected == DEFAULT_BUDGET and st.B_t == 0


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


def test_flat_mode_ignores_group_quota_but_still_records_it():
    """L3 / L4 的 flat 模式：直接在全部候選上取 top-c。"""
    Z, g, q, fg, fp = _fixture()
    res = run_rounds(Z, g, q, fg, fp, hierarchy=False)
    assert res.selected.numel() == DEFAULT_BUDGET
    for rec in res.records:
        assert int(rec.b.sum()) == DEFAULT_CHUNK          # b 仍記錄落在哪個 group
    flat = run_rounds(Z, g, q, fg, fp, hierarchy=False, budget=8, chunk=8)
    top8 = torch.topk(flat.records[0].s.detach(), 8).indices
    assert torch.equal(flat.selected.sort().values, top8.sort().values)


def test_ablation_switches_zero_the_input_blocks_not_the_architecture():
    Z, g, q, fg, fp = _fixture()
    n_params = sum(p.numel() for p in fp.parameters())
    for uq in (True, False):
        for us in (True, False):
            res = run_rounds(Z, g, q, fg, fp, use_query=uq, use_state=us)
            assert res.selected.numel() == DEFAULT_BUDGET
    assert sum(p.numel() for p in fp.parameters()) == n_params


def test_score_reuse_is_numerically_identical_to_recomputation():
    """use_state=False 的分數重用必須與逐輪重算位元相同。"""
    Z, g, q, fg, fp = _fixture()
    res = run_rounds(Z, g, q, fg, fp, use_state=False)
    first = res.records[0].s
    for rec in res.records[1:]:
        assert torch.equal(first, rec.s)


def test_stateful_scores_change_between_rounds():
    """use_state=True 時分數必須逐輪改變 —— 否則 E_t 條件化沒有作用。"""
    Z, g, q, fg, fp = _fixture()
    res = run_rounds(Z, g, q, fg, fp, use_state=True)
    assert not torch.equal(res.records[0].s, res.records[-1].s)
