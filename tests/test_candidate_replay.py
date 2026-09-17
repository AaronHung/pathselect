"""DR-056（exp/candidate-replay 分支）：候選級 replay 與磁碟側倉。

守門六組：
1. 預設 `candidate_only=False` 不需要側倉 —— 簽名的預設值就是舊路徑。
2. 側倉往返：寫入後可**獨立量測 bytes**；entry 帶 [J, D] 原型、不帶 [n, D] 特徵。
3. `no_feature_files` 真的攔得住：區塊內開啟含 `feats-l1-s256` 的路徑即拋錯。
4. 候選級 replay 全程**不開任何原始特徵檔**，只開側倉那一個檔（系統層 audit 觀測）。
5. L_KD 的 group 項吃的是快照原型（r_new = F_g(g_stored)）：換掉 entry 的原型會改變
   L_KD，`group_weight=0` 時不再改變 —— 證明基準是 g_stored 而非候選子集的原型。
6. hinge 的 U_old（`--uold current`）在子集上與全 slide 逐值相同：P_old 的等權池化
   向量不因「只保留候選」而改變。**這一項是候選級 replay 唯一的無損之處**，
   L_KD 的 group 項與 run_rounds 的配額都不是（見 DR-056 的定義變更說明）。
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import selector.train as T                                                   # noqa: E402
from selector.continual import differentiable_utility                        # noqa: E402
from selector.grouping import NUM_GROUPS, assign_groups                      # noqa: E402
from selector.memory import (cand_store_path, load_cand_features, make_entry,  # noqa: E402
                             sample_key, save_cand_features, selected_from_entry)
from selector.model import GroupSelector, PatchSelector                      # noqa: E402
from selector.rounds import run_rounds                                       # noqa: E402
from selector.train import _candidate_terms, frozen_head, no_feature_files   # noqa: E402
from selector.utility import mask_logits                                     # noqa: E402

C = 8
SPEC = dict(use_query=False, use_state=False, hierarchy=True)
BUDGET, SCALE = 8, 56.0


def _world(seed: int = 5, n: int = 900):
    g = torch.Generator().manual_seed(seed)
    Z = torch.randn(n, 512, generator=g)
    f_txt = F.normalize(torch.randn(C, 512, generator=g), dim=-1)
    tissue = F.normalize(torch.randn(NUM_GROUPS, 512, generator=g), dim=-1)
    torch.manual_seed(seed)
    return Z, f_txt, tissue, GroupSelector(), PatchSelector()


def _snapshot(Z, tissue, f_g, f_p, *, label=2, store=None, tau="tcga_rcc", sid="sid-1"):
    """模擬 fill_memory 的 candidate_only 分支：跑一次 run_rounds → 側倉 ＋ 帶原型的 entry。"""
    grp = assign_groups(Z, tissue)
    with torch.no_grad():
        res = run_rounds(Z, grp, torch.zeros(512), f_g, f_p,
                         budget=BUDGET, chunk=1, **SPEC)
    last = res.records[-1]
    n_bytes = None
    if store is not None:
        n_bytes = save_cand_features(store, tau, sample_key(tau, sid),
                                     Z.index_select(0, last.cand_idx), label)
    entry = make_entry(tau, sid, res.state, last.r, last.cand_idx, last.s.detach(),
                       torch.randn(last.cand_idx.numel()),
                       group_prototypes=grp.prototypes)
    return entry, n_bytes, grp


def _terms(entry, f_g, f_p, f_txt, tissue, store, **over):
    kw = dict(budget=BUDGET, chunk=1, q_tau=torch.zeros(512), spec=SPEC,
              use_kd=True, use_eq=True, use_replay=True, eq_mode="hinge",
              kd_group_weight=1.0, class_mask=None, u_old_mode="current",
              cand_store=store)
    kw.update(over)
    return _candidate_terms(entry, (f_g, f_p), f_txt, SCALE, tissue, **kw)


def test_default_path_does_not_need_the_side_store():
    import inspect
    sig = inspect.signature(T.continual_terms)
    assert sig.parameters["candidate_only"].default is False
    assert sig.parameters["cand_store"].default is None
    assert inspect.signature(T.fill_memory).parameters["candidate_only"].default is False


def test_side_store_roundtrip_and_measured_bytes(tmp_path):
    Z, _f_txt, tissue, f_g, f_p = _world()
    entry, n_bytes, grp = _snapshot(Z, tissue, f_g, f_p, label=3, store=tmp_path)
    Zc, label = load_cand_features(tmp_path, entry)
    assert label == 3
    assert Zc.dtype == torch.float32 and Zc.shape == (entry.cand_idx.numel(), 512)
    assert torch.equal(Zc, Z.index_select(0, entry.cand_idx))
    # 可獨立量測：回傳的 bytes 就是檔案大小，且 ≈ k × 512 × 4（＋ torch 標頭）
    assert n_bytes == cand_store_path(tmp_path, entry).stat().st_size
    payload = entry.cand_idx.numel() * 512 * 4
    assert payload <= n_bytes < payload + 4096
    # entry 裡多出來的是 [J, D] 原型（8 個向量），不是 [n, D] 的特徵
    assert entry.group_prototypes.shape == (NUM_GROUPS, 512)
    assert torch.equal(entry.group_prototypes, grp.prototypes)


def test_missing_side_store_file_is_a_hard_error(tmp_path):
    Z, _f_txt, tissue, f_g, f_p = _world()
    entry, _b, _g = _snapshot(Z, tissue, f_g, f_p)          # 沒寫側倉
    with pytest.raises(FileNotFoundError, match="候選側倉缺檔"):
        load_cand_features(tmp_path, entry)


def test_entry_without_prototypes_is_rejected(tmp_path):
    Z, f_txt, tissue, f_g, f_p = _world()
    entry, _b, _g = _snapshot(Z, tissue, f_g, f_p, store=tmp_path)
    bare = dataclasses.replace(entry, group_prototypes=None)
    with pytest.raises(ValueError, match="group_prototypes"):
        _terms(bare, f_g, f_p, f_txt, tissue, tmp_path)


def test_guard_blocks_feature_file_open(tmp_path):
    d = tmp_path / f"{T.FEATURE_DIR_MARK}_CONCH"
    d.mkdir()
    f = d / "slide.pt"
    f.write_bytes(b"0")
    open(f, "rb").close()                                   # 區塊外：開得了
    with pytest.raises(RuntimeError, match="完整特徵檔"):
        with no_feature_files():
            open(f, "rb").close()
    open(f, "rb").close()                                   # 離開區塊後恢復


def test_candidate_replay_opens_only_the_side_store(tmp_path):
    Z, f_txt, tissue, f_g, f_p = _world(7)
    entry, _b, _g = _snapshot(Z, tissue, f_g, f_p, store=tmp_path)
    seen, armed = [], []

    def _spy(event, args):                                  # 系統層觀測：記錄 open
        if armed and event in ("open", "io.open") and args:
            seen.append(str(args[0]))

    sys.addaudithook(_spy)                                  # audit hook 無法移除 → 用旗標控制
    armed.append(True)
    try:
        kd, eq, replay = _terms(entry, f_g, f_p, f_txt, tissue, tmp_path)
    finally:
        armed.clear()
    assert kd is not None and eq is not None and replay is not None
    assert [p for p in seen if T.FEATURE_DIR_MARK in p] == []
    assert str(cand_store_path(tmp_path, entry)) in seen


def test_kd_group_term_uses_the_stored_prototypes(tmp_path):
    Z, f_txt, tissue, f_g, f_p = _world(11)
    entry, _b, _g = _snapshot(Z, tissue, f_g, f_p, store=tmp_path)
    moved = dataclasses.replace(entry, group_prototypes=entry.group_prototypes + 0.5)
    kw = dict(use_eq=False, use_replay=False)
    base_kd = _terms(entry, f_g, f_p, f_txt, tissue, tmp_path, **kw)[0]
    moved_kd = _terms(moved, f_g, f_p, f_txt, tissue, tmp_path, **kw)[0]
    assert not torch.equal(base_kd.detach(), moved_kd.detach())   # 原型是 group 項的輸入
    # group_weight=0 → group 項完全不計算（l_kd 的既有語義）→ 換原型不再有影響
    off = dict(kw, kd_group_weight=0.0)
    a = _terms(entry, f_g, f_p, f_txt, tissue, tmp_path, **off)[0]
    b = _terms(moved, f_g, f_p, f_txt, tissue, tmp_path, **off)[0]
    assert torch.equal(a.detach(), b.detach())


def test_uold_current_matches_the_full_slide_value(tmp_path):
    """P_old 的等權池化向量與全 slide 相同 → U_old 不因候選級 replay 而改變。"""
    Z, f_txt, tissue, f_g, f_p = _world(13)
    entry, _b, _g = _snapshot(Z, tissue, f_g, f_p, label=1, store=tmp_path)
    Zc, label = load_cand_features(tmp_path, entry)
    idx_old, pos = selected_from_entry(entry, BUDGET)
    cur = torch.zeros(C, dtype=torch.bool)
    cur[:6] = True                                          # 當前 C_t
    ste_full = torch.zeros(Z.shape[0]); ste_full[idx_old] = 1.0
    ste_sub = torch.zeros(Zc.shape[0]); ste_sub[pos] = 1.0
    u_full = differentiable_utility(
        mask_logits(frozen_head(Z, torch.zeros(Z.shape[0]), ste_full, f_txt, SCALE,
                                weighting="uniform"), cur), label)
    u_sub = differentiable_utility(
        mask_logits(frozen_head(Zc, torch.zeros(Zc.shape[0]), ste_sub, f_txt, SCALE,
                                weighting="uniform"), cur), label)
    assert float(u_sub) == pytest.approx(float(u_full), abs=1e-6)


def test_reservoir_replay_matches_a_real_memory():
    """DR-056 資料量實測用的無模型重放，必須與真正的 SelectionMemory 逐筆相同。

    這是「真正需要保留的是哪 |M| 筆」那個數字的依據：ReservoirSampling 只吃
    random.Random(0) 與 (len, capacity, n_seen)，entry 的身分只由加入順序決定。
    """
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from measure_dr056_store import replay_reservoir
    from selector.memory import SelectionMemory

    order = ["tcga_esca", "tcga_rcc", "tcga_brca", "tcga_lung"]
    sids = {t: [f"{t}-{i:04d}" for i in range(n)]
            for t, n in zip(order, (120, 616, 763, 95))}
    cap = 128
    mem = SelectionMemory(capacity=cap)
    protos = torch.zeros(NUM_GROUPS, 512)
    for task in order:
        for sid in sids[task]:
            mem.add(make_entry(task, sid, None, torch.zeros(NUM_GROUPS),
                               torch.zeros(4, dtype=torch.long), torch.zeros(8),
                               group_prototypes=protos))
    real = [(e.tau, e.sample_key) for e in mem]
    replayed = [(t, sample_key(t, s)) for t, s in replay_reservoir(order, sids, cap)]
    assert len(real) == cap
    assert replayed == real                      # 順序與內容都要一致
