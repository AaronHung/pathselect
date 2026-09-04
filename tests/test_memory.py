"""CONTRACT-3 — bounded Selection Memory 的資料結構。"""
from __future__ import annotations

import pytest
import torch

from selector.memory import (CANDIDATE_SIZE, FIFO, MEMORY_CAPACITY, SAMPLE_KEY_BITS,
                             ReservoirSampling, SampleKeyIndex, SelectionMemory,
                             SelectionMemoryEntry, make_entry, reload_features,
                             sample_key, selected_from_entry)
from selector.state import EvidenceState

D, J = 512, 8


def _entry(tau="tcga_lung", sid="slide-0", n_cand=256):
    return SelectionMemoryEntry(
        tau=tau, sample_key=sample_key(tau, sid),
        r_old=torch.randn(J), cand_idx=torch.arange(n_cand),
        s_old=torch.randn(n_cand), u_old=torch.randn(n_cand))


def test_capacity_constant_is_512():
    assert MEMORY_CAPACITY == 512 and CANDIDATE_SIZE == 256


def test_entry_holds_no_patch_features():
    e = _entry()
    fields = set(SelectionMemoryEntry.__dataclass_fields__)
    assert fields == {"tau", "sample_key", "r_old", "cand_idx", "s_old", "u_old"}
    # sample_key + index 就是重載的依據，entry 本身不得帶 [n, 512] 的東西
    for name in fields:
        v = getattr(e, name)
        if isinstance(v, torch.Tensor):
            assert v.dim() <= 1, name


def test_candidate_length_is_enforced():
    with pytest.raises(ValueError, match="256"):
        _entry(n_cand=257)


def test_mismatched_score_length_is_rejected():
    with pytest.raises(ValueError, match="s_old"):
        SelectionMemoryEntry(tau="t", sample_key=sample_key("t", "s"),
                             r_old=torch.randn(J), cand_idx=torch.arange(10),
                             s_old=torch.randn(9), u_old=torch.randn(10))


def test_memory_is_bounded_under_the_default_policy():
    """|M| 永不超過 512（預設 reservoir sampling）。"""
    m = SelectionMemory()
    for i in range(MEMORY_CAPACITY * 5):
        m.add(_entry(sid=f"slide-{i}", n_cand=4))
        assert len(m) <= MEMORY_CAPACITY
    assert len(m) == MEMORY_CAPACITY and m.n_seen == MEMORY_CAPACITY * 5


def test_default_policy_is_reservoir_and_is_swappable():
    assert isinstance(SelectionMemory().policy, ReservoirSampling)
    m = SelectionMemory(capacity=8, policy=FIFO())
    for i in range(20):
        m.add(_entry(sid=f"slide-{i}", n_cand=4))
    assert len(m) == 8
    index = SampleKeyIndex.from_slide_ids("tcga_lung", [f"slide-{i}" for i in range(20)])
    assert index.resolve(next(iter(m)).sample_key) == "slide-12"   # FIFO 汰換最舊的


def test_reservoir_keeps_a_mix_of_old_and_new():
    """reservoir 不該退化成「只留最新的」。"""
    m = SelectionMemory(capacity=50, policy=ReservoirSampling(seed=0))
    for i in range(1000):
        m.add(_entry(sid=f"slide-{i}", n_cand=4))
    index = SampleKeyIndex.from_slide_ids("tcga_lung", [f"slide-{i}" for i in range(1000)])
    kept = sorted(int(index.resolve(e.sample_key).split("-")[1]) for e in m)
    assert len(kept) == 50
    assert min(kept) < 500 < max(kept), kept[:5]


def test_capacity_above_the_contract_is_rejected_by_default():
    """CONTRACT-3：|M| <= 512。超過必須顯式 opt-in，不能誤用。"""
    with pytest.raises(ValueError, match="CONTRACT-3"):
        SelectionMemory(capacity=MEMORY_CAPACITY + 1)
    with pytest.raises(ValueError, match="為正"):
        SelectionMemory(capacity=0)


def test_capacity_above_the_contract_needs_explicit_opt_in():
    m = SelectionMemory(capacity=1024, allow_over_contract=True)
    assert m.capacity == 1024
    for i in range(1500):
        m.add(_entry(sid=f"s{i}", n_cand=4))
        assert len(m) <= 1024
    assert len(m) == 1024


def test_by_task_and_tasks():
    m = SelectionMemory()
    m.add(_entry(tau="tcga_lung", n_cand=4))
    m.add(_entry(tau="tcga_brca", n_cand=4))
    m.add(_entry(tau="tcga_lung", n_cand=4))
    assert m.tasks() == ["tcga_brca", "tcga_lung"]
    assert len(m.by_task("tcga_lung")) == 2


def test_make_entry_detaches_and_slices():
    Z = torch.nn.functional.normalize(torch.randn(100, D), dim=-1).requires_grad_(True)
    st = EvidenceState(Z, 64)
    idx = torch.tensor([1, 2, 3])
    st.update(Z.index_select(0, idx).detach(), idx)
    s_all = torch.randn(100, requires_grad=True)
    cand = torch.tensor([10, 20, 30, 40])
    e = make_entry("tcga_rcc", "sid-1", st, torch.randn(J, requires_grad=True),
                   cand, s_all)
    assert not e.s_old.requires_grad
    assert torch.equal(e.s_old, s_all.detach().index_select(0, cand))
    assert e.u_old.shape == (4,)


def test_selected_from_entry_recovers_the_top_k_by_s_old():
    """entry 沒有直接記錄選了誰，但 s_old 足以還原。"""
    torch.manual_seed(0)
    cand = torch.arange(100, 356)
    s_old = torch.randn(256)
    e = SelectionMemoryEntry(tau="tcga_lung", sample_key=sample_key("tcga_lung", "x"),
                             r_old=torch.randn(J), cand_idx=cand,
                             s_old=s_old, u_old=torch.randn(256))
    idx, pos = selected_from_entry(e, 8)
    assert idx.numel() == pos.numel() == 8
    assert torch.equal(pos, torch.topk(s_old, 8).indices)
    assert torch.equal(idx, cand.index_select(0, pos))


def test_reloaded_features_are_bit_identical_to_the_original():
    """不存 patch feature，用 sample_key + index 重載；重載回來必須位元相同。"""
    import pytest
    from pathlib import Path

    from selector.evaluate import read_slide, slide_dataset
    from selector.text_encoder import load_config

    cfg = load_config()
    task = cfg["tasks"][0]
    table = Path(cfg["dataset_root_dir"] + cfg["path_table"].format(task, task.upper()))
    if not table.exists():
        pytest.skip("dataset not available")

    ds, shift = slide_dataset(cfg, task, 0, "train")
    rec = read_slide(ds, shift, 0)
    cand = torch.arange(min(CANDIDATE_SIZE, rec.Z.shape[0]))
    e = SelectionMemoryEntry(tau=task, sample_key=sample_key(task, rec.sid),
                             r_old=torch.randn(J), cand_idx=cand,
                             s_old=torch.randn(cand.numel()),
                             u_old=torch.randn(cand.numel()))
    Z_full, Z_cand, label = reload_features(e, cfg)
    assert torch.equal(Z_full, rec.Z)
    assert torch.equal(Z_cand, rec.Z.index_select(0, cand))
    assert label == rec.label


# ── schema v2：sample_key 與對照表（DR-048 B5）────────────────────────────────

def test_sample_key_is_stable_across_processes():
    """不能用內建 `hash()` —— 它對 str 有 per-process 隨機化。

    這裡把值寫死：換了實作而值變了，舊的記憶庫就對不回原本的 slide。
    """
    assert sample_key("tcga_rcc", "SID-1") == 215102019384988136
    assert 0 <= sample_key("tcga_lung", "x") < (1 << SAMPLE_KEY_BITS)


def test_sample_key_depends_on_both_tau_and_slide_id():
    assert sample_key("tcga_rcc", "s") != sample_key("tcga_lung", "s")
    assert sample_key("tcga_rcc", "s1") != sample_key("tcga_rcc", "s2")


def test_key_fits_in_int64():
    import numpy as np
    for tau in ("tcga_esca", "tcga_rcc", "tcga_brca", "tcga_lung"):
        for i in range(200):
            k = sample_key(tau, f"slide-{i}")
            assert int(np.int64(k)) == k, "超出 int64 正數範圍"


def test_index_round_trips_every_slide_id():
    sids = [f"TCGA-{i:04d}-01Z" for i in range(500)]
    index = SampleKeyIndex.from_slide_ids("tcga_brca", sids)
    assert len(index) == 500
    for sid in sids:
        assert index.resolve(sample_key("tcga_brca", sid)) == sid


def test_index_rejects_a_collision_instead_of_overwriting():
    """碰撞機率極低，但不能靜默覆蓋 —— 覆蓋會讓 replay 讀到別張 slide。"""
    index = SampleKeyIndex()
    index.register("t", "a")
    index._to_sid[sample_key("t", "b")] = "OTHER"       # 手工造出碰撞
    with pytest.raises(RuntimeError, match="碰撞"):
        index.register("t", "b")


def test_unknown_key_raises_rather_than_returning_none():
    with pytest.raises(KeyError, match="不在對照表"):
        SampleKeyIndex.from_slide_ids("t", ["a"]).resolve(12345)


def test_entry_rejects_non_int_sample_key():
    """字串 key 必須當場擋下 —— 否則 v1 的呼叫端會靜默寫進一個壞 entry。"""
    with pytest.raises(TypeError, match="sample_key"):
        SelectionMemoryEntry(tau="t", sample_key="slide-0", r_old=torch.randn(J),
                             cand_idx=torch.arange(4), s_old=torch.randn(4),
                             u_old=torch.randn(4))
    with pytest.raises(ValueError, match="int64"):
        SelectionMemoryEntry(tau="t", sample_key=-1, r_old=torch.randn(J),
                             cand_idx=torch.arange(4), s_old=torch.randn(4),
                             u_old=torch.randn(4))


def test_entry_carries_no_state_fields_any_more():
    """v2 拿掉 e_t / B_tilde_t：它們是只寫不讀的死負載（見模組 docstring）。"""
    e = _entry()
    assert not hasattr(e, "e_t") and not hasattr(e, "B_tilde_t")


def test_make_entry_ignores_state_argument():
    """`state` 留在簽名裡但不再被讀 —— 傳 None 也要能組出 entry。"""
    e = make_entry("tcga_rcc", "sid-1", None, torch.randn(J),
                   torch.tensor([10, 20, 30, 40]), torch.randn(100))
    assert e.sample_key == sample_key("tcga_rcc", "sid-1")
