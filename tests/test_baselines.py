"""Exp 0 baseline 選取政策的基本性質。"""
from __future__ import annotations

import torch

from selector.baselines import grid_indices, random_indices


def test_random_is_reproducible_and_without_replacement():
    a = random_indices(1000, 64, seed=3)
    b = random_indices(1000, 64, seed=3)
    assert torch.equal(a, b)
    assert a.numel() == 64 and len(set(a.tolist())) == 64
    assert int(a.min()) >= 0 and int(a.max()) < 1000


def test_random_seeds_differ():
    assert not torch.equal(random_indices(1000, 64, 0), random_indices(1000, 64, 1))


def test_grid_is_evenly_spaced():
    idx = grid_indices(1000, 64)
    assert idx.numel() == 64
    gaps = (idx[1:] - idx[:-1]).unique()
    assert gaps.numel() == 1 and int(gaps[0]) == 1000 // 64


def test_both_fall_back_to_all_patches_when_k_exceeds_n():
    for fn in (lambda n, k: random_indices(n, k, 0), grid_indices):
        assert torch.equal(fn(10, 64), torch.arange(10))


def test_grid_indices_stay_in_range():
    for n in (17, 100, 743, 1263):
        for k in (8, 16, 32, 64):
            idx = grid_indices(n, k)
            assert int(idx.max()) < n, (n, k)
