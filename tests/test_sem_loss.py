"""G3 — 兩層 L_sem。最關鍵的是 beta_g=0 時與既有版本位元相同。"""
from __future__ import annotations

import pytest
import torch

from selector.sem_loss import group_sem_term, l_sem as l_sem2
from selector.train import l_sem as l_sem_patch_only

N, J = 256, 8


def _fx(seed=0):
    torch.manual_seed(seed)
    return (torch.randn(N), torch.rand(N), torch.randn(J), torch.rand(J))


def test_beta_g_zero_is_bit_identical_to_the_existing_patch_only_version():
    """G3 的基準必須與既有主表可比 —— 位元相同，不是 allclose。"""
    for seed in range(5):
        s, p, r, pg = _fx(seed)
        assert torch.equal(l_sem2(s, p), l_sem_patch_only(s, p))
        assert torch.equal(l_sem2(s, p, r, pg, beta_g=0.0), l_sem_patch_only(s, p))


def test_group_term_is_added_when_beta_g_non_zero():
    s, p, r, pg = _fx(1)
    got = l_sem2(s, p, r, pg, beta_g=0.1)
    want = l_sem_patch_only(s, p) + 0.1 * group_sem_term(r, pg)
    assert torch.allclose(got, want, rtol=0, atol=0)
    assert not torch.equal(got, l_sem_patch_only(s, p))


def test_group_term_zero_when_score_matches_prior():
    torch.manual_seed(2)
    r = torch.randn(J)
    assert float(group_sem_term(r, r)) == pytest.approx(0.0, abs=1e-6)


def test_group_term_backprops_into_the_score():
    """group 分數要收到梯度。

    ⚠️ prior **沒有** detach —— 這是刻意與既有 `train.l_sem` 保持一致（它也沒有），
    否則 beta_g=0 的「位元相同」就不是走同一條路徑。實務上 prior 一律由
    `semantic_prior`（@torch.no_grad）產生，不帶梯度，所以沒有差別。
    這裡斷言的是**實際行為**，不是我期望的行為。
    """
    r = torch.randn(J, requires_grad=True)
    pg = torch.rand(J, requires_grad=True)
    group_sem_term(r, pg).backward()
    assert r.grad is not None and float(r.grad.abs().sum()) > 0
    assert pg.grad is not None, "與 train.l_sem 一致：prior 未 detach"


def test_prior_from_semantic_prior_never_requires_grad():
    """實務上 prior 來自 @torch.no_grad 的 semantic_prior，故上述差異不影響訓練。"""
    import torch.nn.functional as F

    from selector.priors import semantic_prior
    G = F.normalize(torch.randn(J, 512), dim=-1)
    f = F.normalize(torch.randn(8, 512), dim=-1)
    assert not semantic_prior(G, f, n_candidate_classes=8).requires_grad


def test_group_score_and_prior_length_guard():
    with pytest.raises(ValueError, match="長度不符"):
        group_sem_term(torch.randn(J), torch.randn(J - 1))


def test_missing_group_inputs_falls_back_to_patch_only():
    s, p, r, pg = _fx(3)
    assert torch.equal(l_sem2(s, p, None, pg, beta_g=0.1), l_sem_patch_only(s, p))
    assert torch.equal(l_sem2(s, p, r, None, beta_g=0.1), l_sem_patch_only(s, p))


def test_beta_g_zero_keeps_group_score_out_of_the_graph():
    """beta_g=0 時 group_score 不得進入計算圖。

    數值上 `patch + 0.0 * group` 位元相同，所以純數值測試抓不到「忘了提早 return」
    這個 mutation —— 但它會把 r 拉進圖裡。與 `continual.l_kd` 在 group_weight=0
    時提早 return 的契約一致。
    """
    s, p, _, pg = _fx(4)
    s = s.requires_grad_(True)
    r = torch.randn(J, requires_grad=True)
    l_sem2(s, p, r, pg, beta_g=0.0).backward()
    assert s.grad is not None and float(s.grad.abs().sum()) > 0
    assert r.grad is None, "beta_g=0 時 group_score 不該出現在圖中"
