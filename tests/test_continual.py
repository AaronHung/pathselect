"""S4-3 — CL 層的保存損失。

最關鍵的一條是「全部關閉時等價於 SeqFT（位元相同）」：它保證 baseline 與
method 走的是同一條 code path。
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from selector.continual import (DEFAULT_LAMBDAS, EQ_MODES, continual_loss,
                                differentiable_utility, is_disabled, l_eq, l_kd,
                                l_util)

J, C, N = 8, 8, 256


def test_default_lambdas_are_one():
    assert DEFAULT_LAMBDAS == {"kd": 1.0, "eq": 1.0, "replay": 1.0}


# ── 全關 == SeqFT ───────────────────────────────────────────────────────────

def test_all_terms_off_is_exactly_zero():
    total, parts = continual_loss(None, None, None)
    assert torch.equal(total, torch.zeros(()))
    assert parts == {"L_KD": None, "L_eq": None, "L_replay": None}
    assert is_disabled(None, None, None)


def test_all_terms_off_leaves_l_evidence_bit_identical():
    """L_total = L_evidence + L_continual，全關時位元不變 —— 同一條 code path。"""
    torch.manual_seed(0)
    l_evidence = torch.randn(()).abs() + 1.0
    total, _ = continual_loss(None, None, None, dtype=l_evidence.dtype)
    assert torch.equal(l_evidence + total, l_evidence)


@pytest.mark.parametrize("which", ["kd", "eq", "replay"])
def test_each_term_is_independently_switchable(which):
    torch.manual_seed(0)
    vals = {"kd": torch.tensor(2.0), "eq": torch.tensor(3.0),
            "replay": torch.tensor(5.0)}
    kwargs = {k: (v if k == which else None) for k, v in vals.items()}
    total, parts = continual_loss(kwargs["kd"], kwargs["eq"], kwargs["replay"])
    assert float(total) == float(vals[which])
    key = {"kd": "L_KD", "eq": "L_eq", "replay": "L_replay"}[which]
    assert parts[key] == float(vals[which])
    assert all(parts[k] is None for k in parts if k != key)


def test_lambdas_scale_each_term():
    total, _ = continual_loss(torch.tensor(2.0), torch.tensor(3.0), torch.tensor(5.0),
                              lambda_kd=0.5, lambda_eq=2.0, lambda_replay=0.1)
    assert float(total) == pytest.approx(0.5 * 2 + 2.0 * 3 + 0.1 * 5)


# ── L_KD ────────────────────────────────────────────────────────────────────

def test_l_kd_is_zero_when_behaviour_is_unchanged():
    torch.manual_seed(0)
    r, s = torch.randn(J), torch.randn(N)
    assert float(l_kd(r, r.clone(), s, s.clone())) == pytest.approx(0.0, abs=1e-6)


def test_l_kd_grows_as_behaviour_drifts():
    torch.manual_seed(0)
    r_old, s_old = torch.randn(J), torch.randn(N)
    near = l_kd(r_old, r_old + 0.01 * torch.randn(J), s_old, s_old + 0.01 * torch.randn(N))
    far = l_kd(r_old, torch.randn(J), s_old, torch.randn(N))
    assert float(far) > float(near)


def test_l_kd_only_backprops_into_the_student():
    r_old = torch.randn(J, requires_grad=True)
    r_new = torch.randn(J, requires_grad=True)
    s_old = torch.randn(N, requires_grad=True)
    s_new = torch.randn(N, requires_grad=True)
    l_kd(r_old, r_new, s_old, s_new).backward()
    assert r_old.grad is None and s_old.grad is None
    assert r_new.grad is not None and s_new.grad is not None


def test_l_kd_length_guards():
    with pytest.raises(ValueError, match="r 長度"):
        l_kd(torch.randn(J), torch.randn(J + 1), torch.randn(N), torch.randn(N))
    with pytest.raises(ValueError, match="s 長度"):
        l_kd(torch.randn(J), torch.randn(J), torch.randn(N), torch.randn(N - 1))


# ── L_eq ────────────────────────────────────────────────────────────────────

def test_differentiable_utility_matches_log_c_minus_ce():
    torch.manual_seed(0)
    logits = torch.randn(1, C)
    u = differentiable_utility(logits, 3)
    ce = F.cross_entropy(logits, torch.tensor([3]))
    assert float(u) == pytest.approx(math.log(C) - float(ce), rel=1e-6)


def test_utility_is_positive_when_evidence_points_at_the_truth():
    logits = torch.zeros(1, C)
    logits[0, 2] = 10.0
    assert float(differentiable_utility(logits, 2)) > 0
    assert float(differentiable_utility(logits, 5)) < 0


def test_l_eq_hinge_only_penalises_regression():
    u_new = torch.tensor(1.0, requires_grad=True)
    assert float(l_eq(u_new, u_old=2.0)) == pytest.approx(1.0)     # 退步 → 罰
    assert float(l_eq(u_new, u_old=0.5)) == 0.0                    # 進步 → 不罰


def test_l_eq_l2_penalises_both_directions():
    u_new = torch.tensor(1.0)
    assert float(l_eq(u_new, 2.0, mode="l2")) == pytest.approx(1.0)
    assert float(l_eq(u_new, 0.0, mode="l2")) == pytest.approx(1.0)


def test_l_eq_modes_and_guard():
    assert EQ_MODES == ("hinge", "l2")
    with pytest.raises(ValueError, match="eq mode"):
        l_eq(torch.tensor(1.0), 0.0, mode="oracle")


def test_l_eq_is_differentiable():
    u_new = torch.tensor(0.5, requires_grad=True)
    l_eq(u_new, 2.0).backward()
    assert u_new.grad is not None and float(u_new.grad) < 0    # 提高 u_new 可降 loss


# ── L_util ──────────────────────────────────────────────────────────────────

def test_l_util_is_zero_when_scores_match_utility():
    torch.manual_seed(0)
    u = torch.randn(N)
    assert float(l_util(u, u)) == pytest.approx(0.0, abs=1e-6)


def test_l_util_penalises_disagreement_with_utility():
    torch.manual_seed(0)
    u = torch.randn(N)
    assert float(l_util(-u, u)) > float(l_util(u * 0.9, u))


def test_l_util_length_guard():
    with pytest.raises(ValueError, match="長度不符"):
        l_util(torch.randn(N), torch.randn(N - 1))


def test_total_loss_is_bit_identical_when_all_cl_terms_are_off():
    """SeqFT 與 method 走同一條 code path：三項全關時 L_total == L_evidence。"""
    from selector.train import total_loss
    torch.manual_seed(0)
    l_ev = torch.randn(()).abs() + 1.0
    out, parts = total_loss(l_ev)
    assert torch.equal(out, l_ev)
    assert parts == {"L_KD": None, "L_eq": None, "L_replay": None}


def test_total_loss_adds_terms_when_enabled():
    from selector.train import total_loss
    l_ev = torch.tensor(1.0)
    out, parts = total_loss(l_ev, kd=torch.tensor(2.0), eq=torch.tensor(3.0),
                            replay=torch.tensor(4.0), lambda_kd=1.0,
                            lambda_eq=0.5, lambda_replay=0.25)
    assert float(out) == pytest.approx(1.0 + 2.0 + 1.5 + 1.0)
    assert parts["L_KD"] == 2.0 and parts["L_eq"] == 3.0 and parts["L_replay"] == 4.0
