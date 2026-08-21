"""V1 — 訓練與評估必須走同一個分類器，且在相同輸入下位元相同。

背景：先前回報的訓練路徑 [8.3458, 10.4836] 與評估路徑 [8.2943, 10.4545] 不相等。
差異來源不是分類器，而是「兩條路選到不同的 patch、且用了不同的聚合權重」：
  - 訓練：top-64 by selector score，權重 = softmax(top-64 分數)
  - 評估：multiround（redundancy_weight=0.5）選到另一組 patch，權重 = 等權
本檔把 indices 與權重都固定，證明 conch_classify 本身兩邊完全一致。

注意：不可用「在子集上重算 selector 分數」來取得評估權重 —— 那不是位元穩定的
（同一列在 batch=n 與 batch=k 下走不同的 BLAS 分塊，差 ~1e-9）。權重必須由訓練
路徑算好後原封不動帶到評估路徑，因此 make_predict_fn 收固定 weights。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from selector.classifier import conch_classify, make_predict_fn
from selector.flat_selector import EvidenceSelector
from selector.multiround import (ContinualSequentialNavigationAgent, ObserveConfig,
                                 SelectorBank, top_k_select)

SEED = 0
N_PATCH, N_CLASS, DIM, BUDGET = 300, 8, 512, 64
TASK_ID = 0


def _fixture():
    """固定 seed 的合成場景：selector / f_txt / Z 都是決定性的。"""
    torch.manual_seed(SEED)
    selector = EvidenceSelector(feat_dim=DIM).eval()
    f_txt = F.normalize(torch.randn(N_CLASS, DIM), dim=-1)
    Z = F.normalize(torch.randn(N_PATCH, DIM), dim=-1)
    logit_scale = torch.tensor(56.3477)
    bank = SelectorBank(feat_dim=DIM)
    bank.add_skill(TASK_ID, selector)
    return selector, f_txt, Z, logit_scale, bank


@torch.no_grad()
def _train_path(selector, Z, f_txt, logit_scale):
    """訓練時的分類路徑：top-K → softmax 權重 → conch_classify。"""
    score = selector(Z, f_txt)                       # [n]
    top, idx = torch.topk(score, BUDGET)
    w = F.softmax(top, dim=0)                        # [k]
    return conch_classify(Z.index_select(0, idx), w, f_txt, logit_scale), idx, w


def test_same_indices_and_weights_give_bit_identical_logits():
    """固定 indices 與權重 → 訓練路徑與評估路徑位元相同。"""
    selector, f_txt, Z, logit_scale, _ = _fixture()
    logits_train, idx, w = _train_path(selector, Z, f_txt, logit_scale)

    predict_fn = make_predict_fn(f_txt, logit_scale, weights=w)   # 評估路徑的分類器
    logits_eval = predict_fn(Z.index_select(0, idx))

    assert logits_train.shape == logits_eval.shape == (1, N_CLASS)
    assert torch.equal(logits_train, logits_eval), (
        f"train={logits_train.tolist()} eval={logits_eval.tolist()}")


def test_one_shot_agent_selects_exactly_the_training_top_k():
    """證明先前的數字差異來自「選到不同 patch」，不是分類器。

    one-shot（step_size≥budget、redundancy_weight=0）必須選到與訓練完全相同的
    index 序列；normalize_base 的 z-score 是單調變換，不改變 top-K 與其順序。
    """
    selector, f_txt, Z, logit_scale, bank = _fixture()
    _, idx_train, w = _train_path(selector, Z, f_txt, logit_scale)

    agent = ContinualSequentialNavigationAgent(
        bank, f_txt, make_predict_fn(f_txt, logit_scale, weights=w),
        ObserveConfig(budget=BUDGET, step_size=10 ** 9, redundancy_weight=0.0))
    _, selected = agent.predict(Z, task_id=TASK_ID)

    assert torch.equal(selected, idx_train), (
        f"selected={selected[:8].tolist()} train={idx_train[:8].tolist()}")


def test_one_shot_agent_logits_are_bit_identical_to_training():
    """end-to-end：one-shot agent 的 logits 與訓練路徑位元相同。"""
    selector, f_txt, Z, logit_scale, bank = _fixture()
    logits_train, _, w = _train_path(selector, Z, f_txt, logit_scale)

    agent = ContinualSequentialNavigationAgent(
        bank, f_txt, make_predict_fn(f_txt, logit_scale, weights=w),
        ObserveConfig(budget=BUDGET, step_size=10 ** 9, redundancy_weight=0.0))
    logits_eval, _ = agent.predict(Z, task_id=TASK_ID)

    assert torch.equal(logits_train, logits_eval), (
        f"train={logits_train.tolist()} eval={logits_eval.tolist()}")


def test_multiround_reorders_the_same_subset():
    """對照組一：redundancy_weight>0 的 multiround 改變的是「觀察順序」。

    在這個 fixture 上它選到的 patch 集合與訓練 top-K 完全相同，只是順序不同；
    順序不同會讓加權和的浮點累加順序改變 -> 尾數差 ~1e-6，與先前看到的 0.05
    差距不同量級。
    """
    selector, f_txt, Z, logit_scale, bank = _fixture()
    _, idx_train, w = _train_path(selector, Z, f_txt, logit_scale)

    with torch.no_grad():
        scores = selector(Z, f_txt)
    agent = ContinualSequentialNavigationAgent(
        bank, f_txt, make_predict_fn(f_txt, logit_scale, scores=scores),
        ObserveConfig(budget=BUDGET, step_size=16, redundancy_weight=0.5))
    _, selected = agent.predict(Z, task_id=TASK_ID)

    assert torch.equal(selected.sort().values, idx_train.sort().values)   # 同集合
    assert not torch.equal(selected, idx_train)                           # 不同順序


def test_weight_policy_is_what_moved_the_logits():
    """對照組二：先前 0.05 的差距來自「權重政策」，不是分類器。

    同一組 indices、同一個 conch_classify，只換權重（softmax vs 等權）就足以
    重現該量級的差異；把權重也固定住，差異歸零（見上面的 bit-identical 測試）。
    """
    selector, f_txt, Z, logit_scale, _ = _fixture()
    logits_softmax, idx, _w = _train_path(selector, Z, f_txt, logit_scale)
    logits_uniform = conch_classify(Z.index_select(0, idx), None, f_txt, logit_scale)

    assert not torch.equal(logits_softmax, logits_uniform)
    gap = (logits_softmax - logits_uniform).abs().max()
    assert gap > 1e-3, f"權重政策造成的差距應該是可見量級，實得 {float(gap)}"


def test_softmax_is_the_default_weighting():
    """拍板 2：主線權重是 softmax(top-K 分數)，等權要用參數明確叫出來。"""
    selector, f_txt, Z, logit_scale, _ = _fixture()
    with torch.no_grad():
        scores = selector(Z, f_txt)
    idx = top_k_select(scores, BUDGET)
    Z_sel = Z.index_select(0, idx)

    default_fn = make_predict_fn(f_txt, logit_scale, scores=scores)
    ablation_fn = make_predict_fn(f_txt, logit_scale, weighting="uniform")
    expected_softmax = conch_classify(
        Z_sel, F.softmax(scores.index_select(0, idx), dim=0), f_txt, logit_scale)

    assert torch.equal(default_fn(Z_sel, idx), expected_softmax)
    assert torch.equal(ablation_fn(Z_sel, idx),
                       conch_classify(Z_sel, None, f_txt, logit_scale))


def test_softmax_weighting_requires_scores_and_idx():
    """主線權重拿不到分數時必須明確報錯，不准無聲退回等權。"""
    _, f_txt, Z, logit_scale, _ = _fixture()
    import pytest
    with pytest.raises(ValueError, match="softmax"):
        make_predict_fn(f_txt, logit_scale)(Z[:BUDGET])


def test_uniform_weights_equal_plain_mean():
    """weights=None 的等權聚合 == 直接平均，確保預設沒有隱藏縮放。"""
    _, f_txt, Z, logit_scale, _ = _fixture()
    sub = Z[:BUDGET]
    a = conch_classify(sub, None, f_txt, logit_scale)
    b = conch_classify(sub, torch.full((BUDGET,), 1.0 / BUDGET), f_txt, logit_scale)
    assert torch.equal(a, b)


def test_top_k_select_matches_torch_topk():
    """one-shot 的選取原語就是 torch.topk，順序一致。"""
    selector, f_txt, Z, _, _ = _fixture()
    with torch.no_grad():
        score = selector(Z, f_txt)
    assert torch.equal(top_k_select(score, BUDGET), torch.topk(score, BUDGET).indices)
