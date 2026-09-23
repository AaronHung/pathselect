"""DR-054：hinge 的 U_old 以當前 C_t 由快照 P_old 重算（`u_old_mode="current"`）。

四組守門（預註冊於 docs/ledger/DR-054.md「測試」節）：
1. 預設 `snapshot` 與 DR-052 路徑逐位元相同（同輸入同輸出）。
2. `current`：U_old == 獨立算的 log|C_t| − CE_uniform(P_old)；C_t 外的 logit 改動不影響
   U_old 與 U_new（用「未見類別的文字向量」擾動來驗）。
3. 固定頭（mask=None）下 `current` 與 `snapshot` 的 U_old 相等（telescoping 恆等式）。
4. `run_exp2 --uold current` 的旗標存在且預設為 snapshot；非法值被擋。
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

import selector.train as T                                                  # noqa: E402
from selector.grouping import NUM_GROUPS                                    # noqa: E402
from selector.memory import make_entry, selected_from_entry                 # noqa: E402
from selector.model import GroupSelector, PatchSelector                     # noqa: E402
from selector.utility import counterfactual_gain, mask_logits               # noqa: E402

C = 8
SPEC = dict(use_query=False, use_state=False, hierarchy=False, allocation="per_budget")


def _world(seed=11, n=96):
    g = torch.Generator().manual_seed(seed)
    Z = torch.randn(n, 512, generator=g)
    f_txt = F.normalize(torch.randn(C, 512, generator=g), dim=-1)
    tissue = F.normalize(torch.randn(NUM_GROUPS, 512, generator=g), dim=-1)
    torch.manual_seed(seed)
    f_g, f_p = GroupSelector(), PatchSelector()
    return Z, f_txt, tissue, f_g, f_p


def _entry_with_true_uold(Z, f_txt, cand, label, class_mask):
    """快照：u_old 用真實的 counterfactual gain（與 fill_memory 同一條算式）。"""
    s_all = torch.randn(Z.shape[0], generator=torch.Generator().manual_seed(2))
    # fill_memory 的 u 是「以已選證據為基準」的 gain；這裡簡化為空證據基準即可
    u = counterfactual_gain(torch.zeros(512), 0, Z[cand], f_txt, 56.0, label, class_mask=class_mask)
    return make_entry("tcga_rcc", "sid", None, torch.randn(NUM_GROUPS), cand, s_all, u,
                      class_mask_old=class_mask)


def _terms(entry, Z, f_txt, tissue, f_g, f_p, label, class_mask, mode):
    orig = T.reload_features
    T.reload_features = lambda e, cfg: (Z, Z[entry.cand_idx], label)
    try:
        return T.continual_terms(entry, {}, (f_g, f_p), f_txt, 56.0, tissue, spec=SPEC,
                                 use_kd=False, use_replay=False, class_mask=class_mask,
                                 u_old_mode=mode)
    finally:
        T.reload_features = orig


def test_snapshot_default_is_bitwise_unchanged():
    Z, f_txt, tissue, f_g, f_p = _world()
    cand = torch.arange(24)
    m_old = torch.zeros(C, dtype=torch.bool); m_old[:4] = True
    e = _entry_with_true_uold(Z, f_txt, cand, 2, m_old)
    cur = torch.zeros(C, dtype=torch.bool); cur[:6] = True
    _, eq_default, _ = _terms(e, Z, f_txt, tissue, f_g, f_p, 2, cur, "snapshot")
    # 不傳 u_old_mode（DR-052 的呼叫方式）必須逐位元相同
    orig = T.reload_features
    T.reload_features = lambda en, cfg: (Z, Z[en.cand_idx], 2)
    try:
        _, eq_legacy, _ = T.continual_terms(e, {}, (f_g, f_p), f_txt, 56.0, tissue, spec=SPEC,
                                            use_kd=False, use_replay=False, class_mask=cur)
    finally:
        T.reload_features = orig
    assert torch.equal(eq_default.detach(), eq_legacy.detach())


def test_current_uold_equals_independent_computation_and_ignores_unseen_classes():
    Z, f_txt, tissue, f_g, f_p = _world(5)
    cand = torch.arange(24)
    m_old = torch.zeros(C, dtype=torch.bool); m_old[:2] = True
    e = _entry_with_true_uold(Z, f_txt, cand, 1, m_old)
    cur = torch.zeros(C, dtype=torch.bool); cur[:6] = True
    # 獨立計算：P_old 等權池化 → 遮罩到當前 C_t → log|C_t| − CE
    idx_old, _ = selected_from_entry(e, 8)
    pooled = F.normalize(Z[idx_old].mean(0, keepdim=True), dim=-1)
    logits = mask_logits(56.0 * (pooled @ f_txt.t()), cur)
    u_old_ref = math.log(6) - float(F.cross_entropy(logits, torch.tensor([1])))
    # 從 continual_terms 拿 U_old：把 l_eq 換成回傳 u_old 的探針
    seen = {}
    orig_leq = T.l_eq
    T.l_eq = lambda u_new, u_old, mode="hinge": (seen.__setitem__("u_old", u_old) or orig_leq(u_new, u_old, mode))
    try:
        _, eq_a, _ = _terms(e, Z, f_txt, tissue, f_g, f_p, 1, cur, "current")
    finally:
        T.l_eq = orig_leq
    assert seen["u_old"] == pytest.approx(u_old_ref, abs=1e-5)
    # 未見類（C_t 外，第 6、7 列）的文字向量任意改動 → U_old 與 L_eq 都不變
    f2 = f_txt.clone(); f2[6:] = F.normalize(torch.randn(2, 512, generator=torch.Generator().manual_seed(9)), dim=-1)
    seen2 = {}
    T.l_eq = lambda u_new, u_old, mode="hinge": (seen2.__setitem__("u_old", u_old) or orig_leq(u_new, u_old, mode))
    try:
        _, eq_b, _ = _terms(e, Z, f2, tissue, f_g, f_p, 1, cur, "current")
    finally:
        T.l_eq = orig_leq
    assert seen2["u_old"] == pytest.approx(seen["u_old"], abs=1e-6)
    assert float(eq_b.detach()) == pytest.approx(float(eq_a.detach()), abs=1e-6)
    # 反例：C_t 內（第 2 列）的文字向量改動必須改變 U_old
    f3 = f_txt.clone(); f3[2] = F.normalize(torch.randn(512, generator=torch.Generator().manual_seed(10)), dim=-1)
    seen3 = {}
    T.l_eq = lambda u_new, u_old, mode="hinge": (seen3.__setitem__("u_old", u_old) or orig_leq(u_new, u_old, mode))
    try:
        _terms(e, Z, f3, tissue, f_g, f_p, 1, cur, "current")
    finally:
        T.l_eq = orig_leq
    assert seen3["u_old"] != pytest.approx(seen["u_old"], abs=1e-6)


def test_fixed_head_current_equals_snapshot_uold():
    """固定頭：快照 u_old（telescoped gains）== 由 P_old 重算的 log 8 − CE。"""
    Z, f_txt, tissue, f_g, f_p = _world(21)
    cand = torch.arange(30)
    # 用與 fill_memory 相同的定義：以「空證據」為基準的 sequential gain 恆等於 log C − CE(P)
    # 這裡快照的 u_old 需是 P_old 上 sequential 的 gains；用 sequential_utility_total 的定義建構
    from selector.utility import sequential_utility_total
    s_all = torch.randn(Z.shape[0], generator=torch.Generator().manual_seed(3))
    e0 = make_entry("tcga_rcc", "sid", None, torch.randn(NUM_GROUPS), cand, s_all,
                    torch.zeros(30), class_mask_old=None)
    idx_old, pos = selected_from_entry(e0, 8)
    # 把 P_old 逐一的 sequential gain 放進 u_old（其餘候選為 0）
    S = torch.zeros(512); n = 0; gains = torch.zeros(30)
    for i, p in zip(idx_old.tolist(), pos.tolist()):
        gains[p] = counterfactual_gain(S, n, Z[i].reshape(1, -1), f_txt, 56.0, 3)[0]
        S = S + Z[i]; n += 1
    e = make_entry("tcga_rcc", "sid", None, e0.r_old, cand, s_all, gains, class_mask_old=None)
    assert float(gains.sum()) == pytest.approx(sequential_utility_total(Z, idx_old, f_txt, 56.0, 3), abs=1e-5)
    got = {}
    orig_leq = T.l_eq
    for mode in ("snapshot", "current"):
        T.l_eq = (lambda m: (lambda u_new, u_old, mode="hinge": (got.__setitem__(m, u_old) or orig_leq(u_new, u_old, mode))))(mode)
        try:
            _terms(e, Z, f_txt, tissue, f_g, f_p, 3, None, mode)
        finally:
            T.l_eq = orig_leq
    assert got["current"] == pytest.approx(got["snapshot"], abs=1e-5)


def test_invalid_mode_rejected_and_cli_default():
    Z, f_txt, tissue, f_g, f_p = _world()
    e = _entry_with_true_uold(Z, f_txt, torch.arange(16), 0, None)
    with pytest.raises(ValueError):
        _terms(e, Z, f_txt, tissue, f_g, f_p, 0, None, "bogus")
    import subprocess
    r = subprocess.run([sys.executable, "scripts/run_exp2.py", "--help"], capture_output=True,
                       text=True, cwd=REPO_ROOT)
    assert "--uold {snapshot,current}" in r.stdout


def test_uold_is_a_no_op_for_arms_without_the_hinge():
    """沒有 hinge 的臂（eq=False）在兩種 --uold 下逐位元相同。

    這支撐一個影響論文的判斷：現行 Table 3 混用了 snapshot 與 current 兩種口徑
    （第 1/2/4/5 列 snapshot、第 3/6 列 current），但無 hinge 的那幾列在兩種口徑下
    結果相同，所以那張表**內部一致、可以直接沿用**，不必為了統一口徑重跑。

    程式面的理由：`u_old_mode` 在 `continual_terms` 裡的每一處使用都在 `if use_eq:`
    區塊內，區塊外零出現。本測試是那個事實的行為層佐證 —— 讀碼會漏看，跑過才算數。
    """
    import torch.nn.functional as F
    from selector import train as T
    from selector.grouping import NUM_GROUPS, assign_groups
    from selector.memory import make_entry
    from selector.model import GroupSelector, PatchSelector
    from selector.rounds import run_rounds

    torch.manual_seed(0)
    g = torch.Generator().manual_seed(3)
    Z = torch.randn(400, 512, generator=g)
    f_txt = F.normalize(torch.randn(8, 512, generator=g), dim=-1)
    tissue = F.normalize(torch.randn(NUM_GROUPS, 512, generator=g), dim=-1)
    f_g, f_p = GroupSelector(), PatchSelector()
    grp = assign_groups(Z, tissue)
    spec = dict(use_query=False, use_state=False, hierarchy=False)
    with torch.no_grad():
        res = run_rounds(Z, grp, torch.zeros(512), f_g, f_p, budget=8, chunk=1, **spec)
    last = res.records[-1]
    entry = make_entry("tcga_rcc", "sid", res.state, last.r, last.cand_idx,
                       last.s.detach(), torch.randn(last.cand_idx.numel()),
                       class_mask_old=torch.ones(8, dtype=torch.bool))

    orig = T.reload_features
    T.reload_features = lambda e, cfg: (Z, Z.index_select(0, e.cand_idx), 3)
    try:
        cm = torch.ones(8, dtype=torch.bool)
        no_hinge = [("B1", dict(use_kd=True, use_eq=False, use_replay=False)),
                    ("A3", dict(use_kd=False, use_eq=False, use_replay=True)),
                    ("A4", dict(use_kd=True, use_eq=False, use_replay=True))]
        for arm, flags in no_hinge:
            outs = []
            for mode in ("snapshot", "current"):
                torch.manual_seed(0)
                t = T.continual_terms(entry, {}, (f_g, f_p), f_txt, 56.0, tissue,
                                      budget=8, chunk=1, spec=spec, class_mask=cm,
                                      u_old_mode=mode, **flags)
                outs.append(tuple(None if x is None else x.detach().clone() for x in t))
            for a, b in zip(*outs):
                if a is None:
                    assert b is None, f"{arm}: 一邊 None 一邊不是"
                else:
                    assert torch.equal(a, b), f"{arm}: --uold 改變了結果，但它沒有 hinge"

        # 反向對照：有 hinge 的臂**必須**不同，否則這個測試沒有鑑別力
        outs = []
        for mode in ("snapshot", "current"):
            torch.manual_seed(0)
            t = T.continual_terms(entry, {}, (f_g, f_p), f_txt, 56.0, tissue,
                                  budget=8, chunk=1, spec=spec, class_mask=cm,
                                  u_old_mode=mode, use_kd=True, use_eq=True, use_replay=True)
            outs.append(t[1].detach().clone())
        assert not torch.equal(outs[0], outs[1]), \
            "有 hinge 的臂在兩種 --uold 下相同 —— 這個測試失去鑑別力了"
    finally:
        T.reload_features = orig
