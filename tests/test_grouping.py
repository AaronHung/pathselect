"""tissue grouping — argmax 指派、group prototype、空 group mask。"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from selector.grouping import (NUM_GROUPS, TISSUE_GROUP_NAMES, TISSUE_PROMPTS,
                               Grouping, assign_groups, group_capacity)

D = 512


def _fixture(n=60, j=NUM_GROUPS, seed=0):
    torch.manual_seed(seed)
    Z = F.normalize(torch.randn(n, D), dim=-1)
    t = F.normalize(torch.randn(j, D), dim=-1)
    return Z, t


def test_prompts_are_constant_strings():
    assert NUM_GROUPS == 8 == len(TISSUE_GROUP_NAMES)
    for group in TISSUE_PROMPTS:
        assert isinstance(group, tuple) and group
        for s in group:
            assert isinstance(s, str) and s.strip()


def test_assignment_is_argmax_cosine():
    Z, t = _fixture()
    g = assign_groups(Z, t)
    expected = (F.normalize(Z, dim=-1) @ F.normalize(t, dim=-1).t()).argmax(-1)
    assert torch.equal(g.assignment, expected)
    assert g.assignment.shape == (Z.shape[0],)


def test_prototype_is_the_group_mean():
    Z, t = _fixture()
    g = assign_groups(Z, t)
    for j in range(g.num_groups):
        members = g.members(j)
        if members.numel() == 0:
            continue
        assert torch.allclose(g.prototypes[j], Z.index_select(0, members).mean(0),
                              atol=1e-5), j


def test_empty_groups_are_kept_but_masked():
    """空 group 保留（index 穩定）但 mask=False，prototype 為零向量。"""
    Z, t = _fixture(n=40)
    # 讓所有 patch 都最靠近 group 0：把 group 0 的方向放大到必勝
    Z = F.normalize(t[0].reshape(1, -1).repeat(Z.shape[0], 1)
                    + 0.01 * torch.randn_like(Z), dim=-1)
    g = assign_groups(Z, t)
    assert int(g.mask.sum()) == 1 and bool(g.mask[0])
    assert g.num_groups == NUM_GROUPS                      # 空 group 沒有被刪掉
    for j in range(1, NUM_GROUPS):
        assert int(g.sizes[j]) == 0
        assert torch.equal(g.prototypes[j], torch.zeros(D))


def test_sizes_sum_to_n():
    Z, t = _fixture(n=137)
    g = assign_groups(Z, t)
    assert int(g.sizes.sum()) == 137


def test_group_capacity_tracks_availability():
    Z, t = _fixture(n=60)
    g = assign_groups(Z, t)
    avail = torch.ones(60, dtype=torch.bool)
    assert torch.equal(group_capacity(g, avail), g.sizes)
    avail[g.members(int(g.mask.nonzero()[0]))] = False
    cap = group_capacity(g, avail)
    assert int(cap.sum()) == int(avail.sum())


def test_grouping_dataclass_shape_contract():
    Z, t = _fixture()
    g = assign_groups(Z, t)
    assert isinstance(g, Grouping)
    assert g.prototypes.shape == (NUM_GROUPS, D)
    assert g.mask.shape == (NUM_GROUPS,) and g.mask.dtype == torch.bool
