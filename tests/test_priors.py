"""semantic prior —— 三種都測，主線是 discriminative。"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from selector.priors import (MAINLINE_PRIOR, PRIOR_KINDS, assert_full_class_space,
                             semantic_prior)

D, C = 512, 8


def test_mainline_is_discriminative():
    assert MAINLINE_PRIOR == "discriminative"
    assert set(PRIOR_KINDS) == {"none", "max_sim", "discriminative"}


def test_none_prior_is_all_zero():
    X = torch.randn(10, D)
    f = F.normalize(torch.randn(C, D), dim=-1)
    p = semantic_prior(X, f, kind="none", n_candidate_classes=C)
    assert torch.equal(p, torch.zeros(10))


def test_discriminative_is_zero_for_uniform_distribution():
    """所有 class 的 cos 完全相同 → softmax 均勻 → H = log C → p ≈ 0。"""
    f = F.normalize(torch.randn(C, D), dim=-1)
    # 讓 x 與每個 class text 等距：取所有 class 的平均方向
    x = F.normalize(f.mean(0), dim=-1).reshape(1, -1)
    # 直接構造完全均勻的情形：class text 互為旋轉時仍有殘差，這裡用退化 f
    f_uniform = f[0].reshape(1, -1).repeat(C, 1)
    p = semantic_prior(x, f_uniform, kind="discriminative",
                       n_candidate_classes=C, temperature=0.07)
    assert float(p[0]) == pytest.approx(0.0, abs=1e-5)


def test_discriminative_is_one_for_one_hot_distribution():
    """某一類的 cos 遠高於其他 → softmax 近 one-hot → H ≈ 0 → p ≈ 1。"""
    f = F.normalize(torch.eye(C, D), dim=-1)
    x = f[3].reshape(1, -1)                       # 與第 3 類完全對齊、與其餘正交
    p = semantic_prior(x, f, kind="discriminative",
                       n_candidate_classes=C, temperature=0.01)
    assert float(p[0]) == pytest.approx(1.0, abs=1e-5)


def test_discriminative_is_bounded_and_monotone_in_peakiness():
    f = F.normalize(torch.eye(C, D), dim=-1)
    x_peaky = f[0].reshape(1, -1)
    x_flat = F.normalize(f.sum(0), dim=-1).reshape(1, -1)
    X = torch.cat([x_peaky, x_flat], 0)
    p = semantic_prior(X, f, n_candidate_classes=C, temperature=0.07)
    assert bool(((p >= 0) & (p <= 1)).all())
    assert float(p[0]) > float(p[1])


def test_max_sim_is_min_max_normalised():
    torch.manual_seed(0)
    X = F.normalize(torch.randn(50, D), dim=-1)
    f = F.normalize(torch.randn(C, D), dim=-1)
    p = semantic_prior(X, f, kind="max_sim", n_candidate_classes=C)
    assert float(p.min()) == pytest.approx(0.0, abs=1e-6)
    assert float(p.max()) == pytest.approx(1.0, abs=1e-6)
    raw = (X @ f.t()).amax(-1)
    assert torch.equal(raw.argsort(), p.argsort())          # 單調，排序不變


def test_max_sim_all_equal_returns_half():
    f = F.normalize(torch.randn(C, D), dim=-1)
    X = f[0].reshape(1, -1).repeat(4, 1)
    p = semantic_prior(X, f, kind="max_sim", n_candidate_classes=C)
    assert torch.allclose(p, torch.full((4,), 0.5))


def test_partial_class_space_is_rejected():
    """只餵 true class 是 label leakage，必須被 assert 擋下（不是靠註解）。"""
    X = torch.randn(5, D)
    f_full = F.normalize(torch.randn(C, D), dim=-1)
    with pytest.raises(AssertionError, match="leakage"):
        semantic_prior(X, f_full[:1], kind="discriminative", n_candidate_classes=C)
    with pytest.raises(AssertionError, match="leakage"):
        semantic_prior(X, f_full[:2], kind="max_sim", n_candidate_classes=C)


def test_assert_helper_rejects_single_class():
    with pytest.raises(AssertionError):
        assert_full_class_space(torch.randn(1, D), 1)


def test_group_level_uses_the_same_function():
    """group 層用 g_j 同法計算 —— 同一個函式、同一組 class prompt。"""
    torch.manual_seed(0)
    G = F.normalize(torch.randn(8, D), dim=-1)
    f = F.normalize(torch.randn(C, D), dim=-1)
    p = semantic_prior(G, f, n_candidate_classes=C, temperature=0.07)
    assert p.shape == (8,) and bool(((p >= 0) & (p <= 1)).all())


def test_unknown_kind_raises():
    with pytest.raises(ValueError, match="unknown prior"):
        semantic_prior(torch.randn(3, D), torch.randn(C, D), kind="oracle",
                       n_candidate_classes=C)
