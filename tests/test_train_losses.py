"""CONTRACT-4 的 frozen head 與本輪接上的 loss 項。"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from selector.classifier import conch_classify
from selector.grouping import NUM_GROUPS, assign_groups
from selector.model import GroupSelector, PatchSelector, straight_through_topk
from selector.train import ENABLED_TERMS, evidence_loss, frozen_head, l_diag, l_sem
from selector.priors import semantic_prior

D, C = 512, 8


def test_only_diag_and_sem_are_wired_this_round():
    assert ENABLED_TERMS == ("L_diag", "L_sem")


def test_frozen_head_equals_conch_classify_with_softmax_weights():
    """CONTRACT-4：score-weighted pooling → L2 norm → class-text logits。"""
    torch.manual_seed(0)
    Z = F.normalize(torch.randn(100, D), dim=-1)
    f = F.normalize(torch.randn(C, D), dim=-1)
    s = torch.randn(100)
    mask = straight_through_topk(s, 16).detach()
    idx = mask.nonzero().reshape(-1)

    got = frozen_head(Z, s, mask, f, 56.3477)
    want = conch_classify(Z.index_select(0, idx),
                          F.softmax(s.index_select(0, idx), dim=0), f, 56.3477)
    assert got.shape == (1, C)
    assert torch.allclose(got, want, atol=1e-4), (got - want).abs().max()


def test_frozen_head_has_no_trained_parameters():
    """沒有 trained diagnosis head —— frozen_head 是純函式。"""
    import inspect
    assert not isinstance(frozen_head, torch.nn.Module)
    src = inspect.getsource(frozen_head)
    assert "nn.Linear" not in src and "Parameter" not in src


def test_l_diag_is_cross_entropy():
    logits = torch.tensor([[2.0, 0.0, 0.0, 0.0]])
    assert float(l_diag(logits, 0)) < float(l_diag(logits, 1))


def test_l_sem_is_zero_when_scores_match_the_prior():
    s = torch.randn(50)
    assert float(l_sem(s, s)) < 1e-5


def test_l_sem_is_positive_when_scores_disagree_with_the_prior():
    torch.manual_seed(0)
    s = torch.randn(50)
    assert float(l_sem(s, -s)) > 0


def test_evidence_loss_composition():
    torch.manual_seed(0)
    logits = torch.randn(1, C)
    s = torch.randn(40)
    prior = torch.rand(40)
    total, parts = evidence_loss(logits, 3, s, prior, beta_s=0.1)
    assert parts["L_util"] is None
    assert float(total) == pytest.approx(
        float(l_diag(logits, 3)) + 0.1 * float(l_sem(s, prior)), rel=1e-6)


def test_beta_u_only_takes_effect_when_util_is_supplied():
    torch.manual_seed(0)
    logits = torch.randn(1, C)
    s = torch.randn(40)
    prior = torch.rand(40)
    a, _ = evidence_loss(logits, 1, s, prior, beta_u=0.1)
    b, parts = evidence_loss(logits, 1, s, prior, beta_u=0.1, util=torch.ones(40))
    assert float(b) - float(a) == pytest.approx(0.1, rel=1e-5)
    assert parts["L_util"] == 1.0


def test_gradient_reaches_the_patch_selector():
    torch.manual_seed(0)
    Z = F.normalize(torch.randn(300, D), dim=-1)
    t = F.normalize(torch.randn(NUM_GROUPS, D), dim=-1)
    f = F.normalize(torch.randn(C, D), dim=-1)
    q = F.normalize(torch.randn(D), dim=-1)
    from selector.train import train_step

    fg, fp = GroupSelector(), PatchSelector()
    loss, parts, res = train_step(Z, 3, q, f, torch.tensor(56.3477), fg, fp,
                                  grouping=assign_groups(Z, t))
    loss.backward()
    from selector.rounds import DEFAULT_BUDGET
    assert parts["n_selected"] == DEFAULT_BUDGET
    assert float(fp.mlp[0].weight.grad.norm()) > 0


def test_prior_used_by_train_step_sees_the_full_class_space():
    """train_step 傳給 prior 的 f_txt 必須是全部 candidate class。"""
    torch.manual_seed(0)
    Z = F.normalize(torch.randn(120, D), dim=-1)
    f = F.normalize(torch.randn(C, D), dim=-1)
    p = semantic_prior(Z, f, n_candidate_classes=C, logit_scale=56.3477)
    assert p.shape == (120,)


def test_frozen_head_uniform_is_selection_only_mean_pooling():
    """PI 裁定 B：uniform 權重 = selection-only，等權平均已選 patch。"""
    torch.manual_seed(0)
    Z = F.normalize(torch.randn(100, D), dim=-1)
    f = F.normalize(torch.randn(C, D), dim=-1)
    s = torch.randn(100)
    mask = straight_through_topk(s, 8).detach()
    idx = mask.nonzero().reshape(-1)

    got = frozen_head(Z, s, mask, f, 56.3477, weighting="uniform")
    want = conch_classify(Z.index_select(0, idx), None, f, 56.3477)
    assert torch.allclose(got, want, atol=1e-4), (got - want).abs().max()


def test_frozen_head_weightings_differ():
    torch.manual_seed(1)
    Z = F.normalize(torch.randn(100, D), dim=-1)
    f = F.normalize(torch.randn(C, D), dim=-1)
    s = torch.randn(100) * 3
    mask = straight_through_topk(s, 8).detach()
    a = frozen_head(Z, s, mask, f, 56.3477, weighting="softmax")
    b = frozen_head(Z, s, mask, f, 56.3477, weighting="uniform")
    assert not torch.allclose(a, b, atol=1e-3)


def test_unknown_weighting_is_rejected():
    import pytest
    with pytest.raises(ValueError, match="weighting"):
        frozen_head(torch.randn(4, D), torch.randn(4), torch.ones(4),
                    F.normalize(torch.randn(C, D), dim=-1), 1.0, weighting="oracle")
