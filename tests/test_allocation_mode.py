"""DR-025 — 配額口徑：對整個 budget（per_budget）而非對 chunk（per_chunk）。

G1 的教訓：per_chunk 在 c=1 時 largest-remainder 只有一個名額可發、必然給
argmax(r)；r 逐輪不變 ⇒ 每輪同一組 ⇒ 階層退化成「先挑一組再取該組 top-c」。
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from selector.grouping import NUM_GROUPS, assign_groups                  # noqa: E402
from selector.model import GroupSelector, PatchSelector                  # noqa: E402
from selector.rounds import (ALLOCATION_MODES, DEFAULT_ALLOCATION,       # noqa: E402
                             run_rounds)

HIER = dict(use_query=False, use_state=False, hierarchy=True)


def _groups_used(seed, allocation, budget=8, chunk=1):
    torch.manual_seed(seed)
    Z = F.normalize(torch.randn(1200, 512), dim=-1)
    t = F.normalize(torch.randn(NUM_GROUPS, 512), dim=-1)
    g = assign_groups(Z, t)
    f_g, f_p = GroupSelector(), PatchSelector()
    res = run_rounds(Z, g, torch.zeros(512), f_g, f_p, budget=budget, chunk=chunk,
                     allocation=allocation, **HIER)
    tot = [0] * NUM_GROUPS
    for rec in res.records:
        for j, v in enumerate(rec.b.tolist()):
            tot[j] += v
    return tot, res


def test_default_allocation_is_per_budget():
    assert DEFAULT_ALLOCATION == "per_budget"
    assert ALLOCATION_MODES[0] == "per_budget"


def test_per_chunk_degenerates_to_a_single_group_at_c1():
    """記錄 G1 的失敗模式，避免有人把預設改回去而沒人發現。"""
    counts = [sum(1 for v in _groups_used(s, "per_chunk")[0] if v > 0) for s in range(20)]
    hist = Counter(counts)
    assert hist[1] >= 18, f"per_chunk 在 c=1 下應該幾乎全部退化為單組，實得 {dict(hist)}"


def test_per_budget_spreads_across_groups_at_c1():
    """新口徑必須讓預算攤到多個 group。"""
    counts = [sum(1 for v in _groups_used(s, "per_budget")[0] if v > 0) for s in range(20)]
    single = sum(1 for c in counts if c == 1)
    assert single / len(counts) <= 0.5, f"仍有 {single}/{len(counts)} 張退化為單組"
    assert min(counts) >= 2


@pytest.mark.parametrize("allocation", list(ALLOCATION_MODES))
def test_budget_is_always_fully_spent(allocation):
    """配額口徑不得讓預算花不完（第一版的漂移就是這樣被抓到的）。"""
    for seed in range(10):
        tot, res = _groups_used(seed, allocation)
        assert sum(tot) == 8, f"{allocation} seed{seed} 只花了 {sum(tot)}"
        assert res.selected.numel() == 8
        assert len(set(res.selected.tolist())) == 8


def test_per_budget_quota_respects_group_scores():
    """r 有峰值時配額必須集中到高分組，不是永遠均分。"""
    from selector.allocation import allocate
    cap = torch.full((NUM_GROUPS,), 500)
    mask = torch.ones(NUM_GROUPS, dtype=torch.bool)
    flat_q = allocate(torch.zeros(NUM_GROUPS), 8, mask, cap)
    peaked = allocate(torch.tensor([6., 3., 1., 0., 0., -1., -3., -6.]), 8, mask, cap)
    assert int(flat_q.max()) <= 2, f"均勻 r 不該集中：{flat_q.tolist()}"
    assert int(peaked.max()) >= 4, f"集中的 r 應該集中配額：{peaked.tolist()}"
    assert int(flat_q.sum()) == int(peaked.sum()) == 8


def test_flat_is_unaffected_by_allocation_mode():
    """flat 不走配額，兩種口徑必須位元相同（既有結果不受影響）。"""
    outs = []
    for allocation in ALLOCATION_MODES:
        torch.manual_seed(3)
        Z = F.normalize(torch.randn(900, 512), dim=-1)
        t = F.normalize(torch.randn(NUM_GROUPS, 512), dim=-1)
        g = assign_groups(Z, t)
        torch.manual_seed(7)
        f_g, f_p = GroupSelector(), PatchSelector()
        outs.append(run_rounds(Z, g, torch.zeros(512), f_g, f_p, budget=8, chunk=1,
                               allocation=allocation, use_query=False,
                               use_state=False, hierarchy=False).selected)
    assert torch.equal(outs[0], outs[1])
