"""DR-052 累積式類別頭（--head accumulating）。

四組守門（預註冊於 docs/ledger/DR-052.md「測試」節）：
1. 固定頭逐位元不變：class_mask=None 與全 1 遮罩在同一輸入下 logits／loss／選片相同；
   None 路徑不做任何運算（mask_logits 回傳同一個物件）。
2. 一致性：遮罩後的 argmax == `report_seen_class_check.seen_columns` 的事後限制；
   `class_mask_for` 的 C_t 在反向序為 2/4/6/8 類且落在正確的列。
3. hinge 遮罩：U_new = log|C_old| − CE_masked；未見類 logit 任意改動不影響 U_new；
   `continual_terms` 用的是 entry 的 C_old 而非當前 C_t。
4. 累積式先驗：|C_t| = 2 時 = 1 − H/log 2，且等於直接把 f_txt 截成兩列。

⚠️ 每條斷言都先用「反例」證明會 FAIL（tests/README.md §1）：
   例如把 mask_logits 的 −inf 換成 −1e4，第 3 組的「未見類不影響」與第 1 組的
   bit-identity 都會抓到。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from selector.continual import differentiable_utility                     # noqa: E402
from selector.grouping import NUM_GROUPS, Grouping                          # noqa: E402
from selector.memory import make_entry                                      # noqa: E402
from selector.model import GroupSelector, PatchSelector                     # noqa: E402
from selector.priors import semantic_prior                                  # noqa: E402
from selector.train import l_diag, train_step                               # noqa: E402
from selector.utility import (counterfactual_gain, mask_logits,             # noqa: E402
                              sequential_utility_total)

LABEL_SPACE = ["tcga_esca", "tcga_rcc", "tcga_brca", "tcga_lung"]
REVERSE = ["tcga_esca", "tcga_rcc", "tcga_brca", "tcga_lung"]
C = 8


def _synthetic(seed=0, n=64, d=512):
    g = torch.Generator().manual_seed(seed)
    Z = torch.randn(n, d, generator=g)
    f_txt = F.normalize(torch.randn(C, d, generator=g), dim=-1)
    assignment = torch.randint(0, NUM_GROUPS, (n,), generator=g)
    protos = torch.stack([Z[assignment == j].mean(0) if (assignment == j).any()
                          else torch.zeros(d) for j in range(NUM_GROUPS)])
    mask = torch.tensor([(assignment == j).any().item() for j in range(NUM_GROUPS)])
    sizes = torch.tensor([(assignment == j).sum().item() for j in range(NUM_GROUPS)])
    grp = Grouping(assignment=assignment, prototypes=protos, mask=mask, sizes=sizes)
    return Z, f_txt, grp


def _models(seed=0):
    torch.manual_seed(seed)
    return GroupSelector(), PatchSelector()


# ── 1. 固定頭逐位元不變 ─────────────────────────────────────────────────────

def test_mask_none_returns_same_object():
    x = torch.randn(1, C)
    assert mask_logits(x, None) is x


def test_all_true_mask_is_bitwise_identical_to_none():
    Z, f_txt, grp = _synthetic()
    full = torch.ones(C, dtype=torch.bool)
    outs = []
    for cm in (None, full):
        f_g, f_p = _models()
        loss, parts, res = train_step(Z, 3, torch.zeros(512), f_txt, torch.tensor(56.0),
                                      f_g, f_p, grouping=grp, budget=8, chunk=1,
                                      use_query=False, use_state=False, hierarchy=False,
                                      class_mask=cm)
        outs.append((float(loss.detach()), res.selected.tolist(), parts["L_diag"]))
    assert outs[0] == outs[1]
    # 反例：部分遮罩必須改變 loss（否則遮罩根本沒接上）
    part = torch.zeros(C, dtype=torch.bool); part[2:4] = True
    f_g, f_p = _models()
    loss_p, _, _ = train_step(Z, 3, torch.zeros(512), f_txt, torch.tensor(56.0),
                              f_g, f_p, grouping=grp, budget=8, chunk=1,
                              use_query=False, use_state=False, hierarchy=False,
                              class_mask=part)
    assert float(loss_p) != outs[0][0]


def test_utility_paths_bitwise_identical_under_full_mask():
    Z, f_txt, _ = _synthetic(1)
    full = torch.ones(C, dtype=torch.bool)
    idx = torch.tensor([0, 5, 9, 13, 21, 30, 41, 60])
    assert (sequential_utility_total(Z, idx, f_txt, 56.0, 4)
            == sequential_utility_total(Z, idx, f_txt, 56.0, 4, class_mask=full))
    S, n = Z[:3].sum(0), 3
    a = counterfactual_gain(S, n, Z[10:20], f_txt, 56.0, 4)
    b = counterfactual_gain(S, n, Z[10:20], f_txt, 56.0, 4, class_mask=full)
    assert torch.equal(a, b)
    lg = torch.randn(1, C)
    assert torch.equal(l_diag(lg, 2), l_diag(lg, 2, full))


# ── 2. 一致性：遮罩 argmax == seen_columns 事後限制；C_t 大小與位置 ────────────

def test_class_mask_matches_seen_columns_per_stage():
    from report_seen_class_check import seen_columns
    from run_exp2 import class_mask_for

    g = torch.Generator().manual_seed(7)
    for stage in range(4):
        m = class_mask_for(LABEL_SPACE, REVERSE[:stage + 1])
        assert int(m.sum()) == 2 * (stage + 1)
        cols = seen_columns("reverse", stage, LABEL_SPACE)
        assert m.nonzero().reshape(-1).tolist() == cols
        for _ in range(50):
            lg = torch.randn(C, generator=g)
            masked_arg = int(mask_logits(lg, m).argmax())
            posthoc_arg = cols[int(lg[cols].argmax())]
            assert masked_arg == posthoc_arg
    # 正向序：C_0 = lung 的兩列（6, 7）
    m0 = class_mask_for(LABEL_SPACE, ["tcga_lung"])
    assert m0.nonzero().reshape(-1).tolist() == [6, 7]


def test_mask_rejects_fewer_than_two_classes_and_wrong_length():
    lg = torch.randn(1, C)
    with pytest.raises(ValueError):
        mask_logits(lg, torch.tensor([True] + [False] * 7))
    with pytest.raises(ValueError):
        mask_logits(lg, torch.ones(7, dtype=torch.bool))


# ── 3. hinge 遮罩 ─────────────────────────────────────────────────────────────

def test_differentiable_utility_uses_masked_class_count():
    g = torch.Generator().manual_seed(3)
    lg = torch.randn(1, C, generator=g)
    m = torch.zeros(C, dtype=torch.bool); m[:4] = True          # C_old = 4 類
    y = 1
    u = differentiable_utility(mask_logits(lg, m), y)
    ce_manual = float(F.cross_entropy(lg[:, :4], torch.tensor([y])))
    assert float(u) == pytest.approx(math.log(4) - ce_manual, abs=1e-6)
    # 未見類的 logit 任意改動不影響 U_new
    lg2 = lg.clone(); lg2[:, 4:] += 100.0
    assert torch.equal(differentiable_utility(mask_logits(lg2, m), y), u)
    # 固定頭（mask=None）行為不變：C = 8
    u8 = differentiable_utility(lg, y)
    assert float(u8) == pytest.approx(math.log(8) - float(F.cross_entropy(lg, torch.tensor([y]))),
                                      abs=1e-6)


def test_continual_terms_hinge_uses_entry_mask_not_current_mask():
    """U_new 遮罩到快照的 C_old；當前 C_t 只影響 replay 的 CE。"""
    from selector.train import continual_terms, fill_memory  # noqa: F401
    import selector.train as T

    Z, f_txt, grp = _synthetic(5)
    f_g, f_p = _models(5)
    # 假 entry：C_old = 前 4 類；cand 與分數隨便給
    cand = torch.arange(16)
    m_old = torch.zeros(C, dtype=torch.bool); m_old[:4] = True
    entry = make_entry("tcga_rcc", "sid-1", None, torch.randn(NUM_GROUPS), cand,
                       torch.randn(Z.shape[0]), torch.randn(16), class_mask_old=m_old)
    assert entry.class_mask_old is not None and int(entry.class_mask_old.sum()) == 4

    # 用 monkeypatch 把 reload_features 換成回傳合成 Z，避免讀檔
    orig = T.reload_features
    T.reload_features = lambda e, cfg: (Z, Z[cand], 2)
    try:
        cur6 = torch.zeros(C, dtype=torch.bool); cur6[:6] = True
        cur8 = torch.ones(C, dtype=torch.bool)
        spec = dict(use_query=False, use_state=False, hierarchy=False, allocation="per_budget")
        tissue = F.normalize(torch.randn(NUM_GROUPS, 512, generator=torch.Generator().manual_seed(1)), dim=-1)
        _, eq6, rep6 = continual_terms(entry, {}, (f_g, f_p), f_txt, 56.0, tissue,
                                       spec=spec, use_kd=False, class_mask=cur6)
        _, eq8, rep8 = continual_terms(entry, {}, (f_g, f_p), f_txt, 56.0, tissue,
                                       spec=spec, use_kd=False, class_mask=cur8)
        assert torch.equal(eq6.detach(), eq8.detach())         # hinge 只看 C_old
        assert not torch.equal(rep6.detach(), rep8.detach())   # replay 看當前 C_t
    finally:
        T.reload_features = orig


# ── 4. 累積式先驗 ────────────────────────────────────────────────────────────

def test_prior_on_two_seen_classes():
    Z, f_txt, _ = _synthetic(9)
    m = torch.zeros(C, dtype=torch.bool); m[2:4] = True
    p = semantic_prior(Z, f_txt[m], kind="discriminative", n_candidate_classes=2,
                       logit_scale=56.0)
    cos = F.normalize(Z, dim=-1) @ F.normalize(f_txt[2:4], dim=-1).t()
    logp = F.log_softmax(cos * 56.0, dim=-1)
    H = -(logp.exp() * logp).sum(-1)
    assert torch.allclose(p, (1 - H / math.log(2)).clamp(0, 1), atol=1e-6)
    p_full = semantic_prior(Z, f_txt, kind="discriminative", n_candidate_classes=8,
                            logit_scale=56.0)
    assert not torch.allclose(p, p_full)                        # 反例：確實不同


def test_stage_mask_none_for_fixed_head():
    from run_exp2 import stage_mask
    import argparse
    ctx = argparse.Namespace(label_space=LABEL_SPACE)
    assert stage_mask(argparse.Namespace(head="fixed"), ctx, REVERSE, 2) is None
    m = stage_mask(argparse.Namespace(head="accumulating"), ctx, REVERSE, 1)
    assert m.tolist() == [True, True, True, True, False, False, False, False]
