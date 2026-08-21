"""counterfactual gain —— 向量化結果必須等於迴圈參考實作。"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from selector.utility import (CANDIDATE_SIZE, counterfactual_gain,
                              counterfactual_gain_loop, current_logits,
                              top_candidates)

D, C = 512, 8


def test_vectorised_matches_loop_in_float64():
    """小 N 下逐一比對；用 float64 排除 BLAS 分塊造成的尾數差。"""
    torch.manual_seed(0)
    X = F.normalize(torch.randn(12, D, dtype=torch.float64), dim=-1)
    f = F.normalize(torch.randn(C, D, dtype=torch.float64), dim=-1)
    S = torch.randn(D, dtype=torch.float64)
    ls = torch.tensor(56.3477, dtype=torch.float64)
    a = counterfactual_gain(S, 5, X, f, ls, label=3)
    b = counterfactual_gain_loop(S, 5, X, f, ls, label=3)
    assert torch.allclose(a, b, rtol=0, atol=1e-12), (a - b).abs().max()


def test_vectorised_matches_loop_in_float32_within_tolerance():
    torch.manual_seed(1)
    X = F.normalize(torch.randn(16, D), dim=-1)
    f = F.normalize(torch.randn(C, D), dim=-1)
    S = torch.randn(D)
    a = counterfactual_gain(S, 4, X, f, 56.3477, label=6)
    b = counterfactual_gain_loop(S, 4, X, f, 56.3477, label=6)
    assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max()


def test_works_from_empty_evidence():
    """t=0：current evidence 為空，current logits 取均勻分佈。"""
    torch.manual_seed(2)
    X = F.normalize(torch.randn(8, D, dtype=torch.float64), dim=-1)
    f = F.normalize(torch.randn(C, D, dtype=torch.float64), dim=-1)
    S = torch.zeros(D, dtype=torch.float64)
    assert torch.equal(current_logits(S, 0, f, 56.3477), torch.zeros(1, C, dtype=f.dtype))
    a = counterfactual_gain(S, 0, X, f, 56.3477, label=1)
    b = counterfactual_gain_loop(S, 0, X, f, 56.3477, label=1)
    assert torch.allclose(a, b, atol=1e-12)


def test_gain_is_positive_for_a_patch_pointing_at_the_true_class():
    f = F.normalize(torch.eye(C, D, dtype=torch.float64), dim=-1)
    S = torch.zeros(D, dtype=torch.float64)
    X = torch.stack([f[2], f[5]])
    u = counterfactual_gain(S, 0, X, f, 56.3477, label=2)
    assert float(u[0]) > 0 > float(u[1])


def test_shape_guard():
    f = F.normalize(torch.randn(C, D), dim=-1)
    with pytest.raises(ValueError, match=r"\[N, D\]"):
        counterfactual_gain(torch.zeros(D), 0, torch.randn(D), f, 1.0, 0)


def test_candidate_set_is_256_and_respects_availability():
    assert CANDIDATE_SIZE == 256
    torch.manual_seed(3)
    scores = torch.randn(1000)
    avail = torch.ones(1000, dtype=torch.bool)
    avail[:900] = False
    idx = top_candidates(scores, avail)
    assert idx.numel() == 100                       # 只剩 100 個可選
    assert set(idx.tolist()) <= set(range(900, 1000))

    avail = torch.ones(1000, dtype=torch.bool)
    idx = top_candidates(scores, avail)
    assert idx.numel() == CANDIDATE_SIZE
    assert torch.equal(idx, torch.topk(scores, CANDIDATE_SIZE).indices)


def test_no_candidates_returns_empty():
    idx = top_candidates(torch.randn(10), torch.zeros(10, dtype=torch.bool))
    assert idx.numel() == 0
