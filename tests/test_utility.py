"""counterfactual gain —— 向量化結果必須等於迴圈參考實作。"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from selector.utility import (CANDIDATE_SIZE, counterfactual_gain, _ce,
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


def test_sequential_total_is_not_the_sum_of_independent_gains():
    """兩種「utility 總和」不是同一件事 —— 這個混淆真的發生過。

    sequential：evidence 隨選取順序累積，加總會 telescope 成
                loss(空證據) − loss(最終證據)。
    獨立加總：  每個 patch 都從空證據算起，彼此的貢獻被重複計算。
    """
    from selector.utility import sequential_utility_total

    torch.manual_seed(0)
    Z = F.normalize(torch.randn(50, D, dtype=torch.float64), dim=-1)
    f = F.normalize(torch.randn(C, D, dtype=torch.float64), dim=-1)
    idx = torch.arange(8)
    ls, label = 56.3477, 3

    seq = sequential_utility_total(Z, idx, f, ls, label)
    indep = float(counterfactual_gain(torch.zeros(D, dtype=Z.dtype), 0,
                                      Z.index_select(0, idx), f, ls, label).sum())
    assert abs(seq - indep) > 1e-6, (seq, indep)


def test_sequential_total_telescopes_to_the_loss_reduction():
    """U(S) = loss(空證據) − loss(最終等權證據)。"""
    import math

    from selector.utility import sequential_utility_total

    torch.manual_seed(1)
    Z = F.normalize(torch.randn(40, D, dtype=torch.float64), dim=-1)
    f = F.normalize(torch.randn(C, D, dtype=torch.float64), dim=-1)
    idx = torch.arange(8)
    ls, label = 56.3477, 5

    got = sequential_utility_total(Z, idx, f, ls, label)
    e = F.normalize(Z.index_select(0, idx).mean(0).reshape(1, -1), dim=-1)
    final = _ce(ls * (e @ f.t()), label).reshape(())
    want = math.log(C) - float(final)
    assert got == pytest.approx(want, abs=1e-9)


def test_sequential_total_is_order_independent():
    """telescoping 的直接推論：換順序不改總和（浮點誤差內）。"""
    from selector.utility import sequential_utility_total

    torch.manual_seed(2)
    Z = F.normalize(torch.randn(40, D, dtype=torch.float64), dim=-1)
    f = F.normalize(torch.randn(C, D, dtype=torch.float64), dim=-1)
    ls, label = 56.3477, 2
    a = sequential_utility_total(Z, torch.tensor([3, 1, 7, 9]), f, ls, label)
    b = sequential_utility_total(Z, torch.tensor([9, 7, 1, 3]), f, ls, label)
    assert a == pytest.approx(b, abs=1e-9)
