"""G1 — flat 與 hierarchical 之間**只有一個開關**不同。

Gate 1 的教訓：同時打開多件事就無法歸因。這一組測試把「唯一差異是 hierarchy」
釘死，避免之後有人順手把 q_tau 或 state 一起打開。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_exp2 import ARCH, DEFAULT_ARCH                        # noqa: E402
from selector.grouping import NUM_GROUPS, assign_groups                # noqa: E402
from selector.model import GroupSelector, PatchSelector                # noqa: E402
from selector.rounds import run_rounds                                 # noqa: E402

D = 512


def _fixture(seed=0, n=600):
    torch.manual_seed(seed)
    Z = F.normalize(torch.randn(n, D), dim=-1)
    t = F.normalize(torch.randn(NUM_GROUPS, D), dim=-1)
    return Z, assign_groups(Z, t), torch.zeros(D), GroupSelector(), PatchSelector()


def test_arch_table_differs_only_in_hierarchy():
    """兩個組態的差異只准有 hierarchy 一個 key。"""
    flat, hier = ARCH["flat"], ARCH["hier"]
    assert set(flat) == set(hier)
    diff = {k for k in flat if flat[k] != hier[k]}
    assert diff == {"hierarchy"}, f"不只 hierarchy 不同：{diff}"
    assert flat["hierarchy"] is False and hier["hierarchy"] is True


def test_query_and_state_stay_off_in_both():
    """G1 只驗證階層；q_tau 與 state 兩邊都必須關著。"""
    for name, spec in ARCH.items():
        assert spec["use_query"] is False, f"{name} 打開了 q_tau"
        assert spec["use_state"] is False, f"{name} 打開了 state"


def test_default_arch_is_flat_so_existing_results_reproduce():
    assert DEFAULT_ARCH == "flat"


def test_group_selector_has_no_effect_on_selection_under_flat():
    """flat 下 F_g 的輸出不影響選取 —— 這正是 group KD 從未被測到的原因。"""
    Z, g, q, f_g, f_p = _fixture()
    torch.manual_seed(99)
    f_g2 = GroupSelector()
    with torch.no_grad():
        for p in f_g2.parameters():
            p.mul_(5.0).add_(torch.randn_like(p))
    a = run_rounds(Z, g, q, f_g, f_p, budget=8, chunk=1, **ARCH["flat"]).selected
    b = run_rounds(Z, g, q, f_g2, f_p, budget=8, chunk=1, **ARCH["flat"]).selected
    assert torch.equal(a, b), "flat 下換掉 F_g 竟然改變了選取"


def test_group_selector_does_affect_selection_under_hier():
    """階層下 F_g 必須真的參與決策，否則 G1 沒有意義。"""
    Z, g, q, f_g, f_p = _fixture()
    torch.manual_seed(99)
    f_g2 = GroupSelector()
    with torch.no_grad():
        for p in f_g2.parameters():
            p.mul_(5.0).add_(torch.randn_like(p))
    a = run_rounds(Z, g, q, f_g, f_p, budget=8, chunk=1, **ARCH["hier"]).selected
    b = run_rounds(Z, g, q, f_g2, f_p, budget=8, chunk=1, **ARCH["hier"]).selected
    assert not torch.equal(a, b), "階層下換掉 F_g 竟然不改變選取"


def test_group_selector_receives_l_diag_gradient_only_under_hier():
    """flat 下 F_g 收不到 L_diag 的梯度；階層下必須收得到。"""
    from selector.train import frozen_head

    f_txt = F.normalize(torch.randn(8, D), dim=-1)
    grads = {}
    for name in ("flat", "hier"):
        Z, g, q, f_g, f_p = _fixture()
        res = run_rounds(Z, g, q, f_g, f_p, budget=8, chunk=1, **ARCH[name])
        ste = sum(r.ste_mask for r in res.records)
        frozen_head(Z, res.records[-1].s, ste, f_txt, 56.3477).sum().backward()
        grads[name] = f_g.mlp[0].weight.grad
    assert grads["flat"] is None, "flat 下 F_g 不該收到 L_diag 梯度"
    assert grads["hier"] is not None and float(grads["hier"].norm()) > 0


@pytest.mark.parametrize("name", list(ARCH))
def test_both_arches_respect_the_budget_contract(name):
    """兩種架構都必須遵守 B=8、c=1、8 rounds、不重複選取。"""
    Z, g, q, f_g, f_p = _fixture()
    res = run_rounds(Z, g, q, f_g, f_p, budget=8, chunk=1, **ARCH[name])
    sel = res.selected.tolist()
    assert res.n_rounds == 8
    assert len(sel) == 8 == len(set(sel))
    for rec in res.records:
        assert int(rec.b.sum()) == 1
