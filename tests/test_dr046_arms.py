"""DR-046 Phase A 的四個新臂：規格與兩個旗標的行為。

⚠️ 這些臂會驅動一次 5 seeds × 4 stage 的正式跑。臂設錯要幾小時後才看得出來，
   所以先在這裡把規格與旗標效果釘死。
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import run_exp2 as R                                                  # noqa: E402

NEW = ("W1", "W1B", "L2", "L2B")


# ── 臂規格 ──────────────────────────────────────────────────────────────────

def test_four_new_arms_exist_with_the_specified_flags():
    spec = {
        "W1": dict(mode="sequential", lora=True, kd=True, eq=True, replay=True,
                   warmstart=True),
        "W1B": dict(mode="sequential", lora=True, kd=False, eq=False, replay=False,
                    warmstart=True),
        "L2": dict(mode="sequential", lora=True, kd=True, eq=True, replay=True,
                   merge_each=False),
        "L2B": dict(mode="sequential", lora=True, kd=False, eq=False, replay=False,
                    merge_each=False),
    }
    for arm, want in spec.items():
        assert arm in R.ARMS, f"缺少 {arm}"
        got = R.ARMS[arm]
        for k, v in want.items():
            assert got.get(k) == v, f"{arm} 的 {k}：期望 {v}，實得 {got.get(k)}"


def test_b_variants_differ_from_full_only_in_preservation():
    """W1B 與 W1（L2B 與 L2）只差在 kd/eq/replay，其餘必須相同。"""
    for full, b in (("W1", "W1B"), ("L2", "L2B")):
        a, c = dict(R.ARMS[full]), dict(R.ARMS[b])
        for d in (a, c):
            d.pop("name")
            for k in ("kd", "eq", "replay"):
                d.pop(k, None)
        assert a == c, f"{full} 與 {b} 除了保存機制之外還有差異：{a} vs {c}"


def test_existing_arms_keep_default_flags():
    """既有臂不得被新旗標影響（預設 warmstart=False、merge_each=True）。"""
    for arm in ("A1", "A2", "A3", "A4", "A5", "A5nG", "B1", "B2", "R1", "R2"):
        s = R.ARMS[arm]
        assert s.get("warmstart", False) is False, f"{arm} 意外帶了 warmstart"
        assert s.get("merge_each", True) is True, f"{arm} 意外帶了 merge_each"


def test_paired_comparisons_include_the_four_new_pairs():
    for pair in (("W1", "A5"), ("L2", "A5"), ("W1B", "A2"), ("L2B", "A2")):
        assert pair in R.PAIRED_COMPARISONS, f"缺少配對 {pair}"


# ── wrap_with_lora 的自檢 ───────────────────────────────────────────────────

def test_wrap_with_lora_is_functionally_a_no_op():
    """B 初始為零 ⇒ ΔW = 0 ⇒ 掛上去不改變輸出（逐位元）。"""
    from selector.model import SELECTOR_INPUT_DIM, GroupSelector, PatchSelector
    torch.manual_seed(0)
    models = (GroupSelector(), PatchSelector())
    x = torch.randn(8, SELECTOR_INPUT_DIM)
    before = [m(x).detach().clone() for m in models]
    R.wrap_with_lora(models, rank=4)
    for m, ref in zip(models, before):
        assert torch.equal(m(x).detach(), ref), "掛 LoRA 改變了輸出"
    from selector.lora import lora_layers
    assert all(lora_layers(m) for m in models), "沒有真的掛上 LoRA"


def test_wrap_with_lora_aborts_when_output_changes(monkeypatch):
    """自檢要有牙齒：前向若真的變了必須 SystemExit，不得只印警告。"""
    from selector.lora import LoRALinear
    from selector.model import GroupSelector, PatchSelector
    torch.manual_seed(0)
    models = (GroupSelector(), PatchSelector())

    real_init = LoRALinear.__init__

    def sabotage(self, base, r=4, alpha=None):
        real_init(self, base, r=r, alpha=alpha)
        with torch.no_grad():
            self.lora_B.fill_(0.1)          # ΔW ≠ 0 → 前向改變
    monkeypatch.setattr(LoRALinear, "__init__", sabotage)
    with pytest.raises(SystemExit, match="位元相同"):
        R.wrap_with_lora(models, rank=4)


# ── 兩個旗標對流程的實際影響 ────────────────────────────────────────────────

def _trace(arm, monkeypatch):
    """跑 run_arm 但把訓練/評估全部換成 spy，只記錄呼叫序列。"""
    calls = []

    def fake_train_stage(ctx, a, models, tasks, seed, args, memory, rng, *,
                         use_lora=None):
        calls.append(("train", tasks[0], use_lora))
        return {}

    def fake_merge(*models):
        calls.append(("merge", None, None))

    def fake_evaluate(ctx, models, task, *a, **k):
        return []

    def fake_fill(*a, **k):
        return 0

    def fake_wrap(models, rank):
        calls.append(("wrap", None, rank))

    monkeypatch.setattr(R, "train_stage", fake_train_stage)
    monkeypatch.setattr(R, "merge_lora", fake_merge)
    monkeypatch.setattr(R, "evaluate", fake_evaluate)
    monkeypatch.setattr(R, "fill_memory", fake_fill)
    monkeypatch.setattr(R, "wrap_with_lora", fake_wrap)
    monkeypatch.setattr(R, "new_models", lambda ctx, seed, use_lora, rank:
                        (calls.append(("new", None, use_lora)), (None, None))[1])

    ctx = types.SimpleNamespace(device=torch.device("cpu"), cfg={}, f_txt=None,
                                logit_scale=None, tissue=None)
    args = types.SimpleNamespace(rank=4, mem_capacity=None, mem_slides=0,
                                 budget=8, chunk=1, arch="flat",
                                 allocation="per_budget", replay_k=1)
    R.run_arm(ctx, arm, "reverse", 0, args, Path("."))
    return calls


def test_warmstart_trains_full_params_at_stage0_then_wraps(monkeypatch):
    calls = _trace("W1", monkeypatch)
    assert calls[0] == ("new", None, False), "warm-start 建模時不該先掛 LoRA"
    trains = [c for c in calls if c[0] == "train"]
    assert trains[0][2] is False, "stage 0 必須訓練全參數"
    assert all(c[2] is True for c in trains[1:]), "stage 1+ 必須只訓 LoRA"
    kinds = [c[0] for c in calls]
    assert kinds.count("wrap") == 1, "應該只掛一次 LoRA"
    assert kinds.index("wrap") < kinds.index("train", kinds.index("wrap") - 10 + 1) \
        or True                                   # 順序由下一條更精確地檢查
    # wrap 必須發生在 stage 0 之後、stage 1 訓練之前
    i_wrap = kinds.index("wrap")
    i_train1 = [i for i, c in enumerate(calls) if c[0] == "train"][1]
    assert i_wrap < i_train1, "LoRA 必須在 stage 1 開始訓練前掛好"


def test_warmstart_does_not_merge_at_stage0(monkeypatch):
    calls = _trace("W1", monkeypatch)
    i_wrap = [i for i, c in enumerate(calls) if c[0] == "wrap"][0]
    assert not any(c[0] == "merge" for c in calls[:i_wrap]), \
        "stage 0 沒有 LoRA，不該 merge"
    assert sum(c[0] == "merge" for c in calls) == 3, "stage 1–3 各 merge 一次"


def test_merge_each_false_merges_only_once_at_the_end(monkeypatch):
    calls = _trace("L2", monkeypatch)
    assert sum(c[0] == "merge" for c in calls) == 1, "L2 只該在最後合併一次"
    assert calls[-1][0] == "merge" or calls[-2][0] == "merge", "合併應在最後一個 stage"
    assert all(c[2] is True for c in calls if c[0] == "train"), "L2 全程只訓 LoRA"


def test_baseline_arm_merges_every_stage(monkeypatch):
    """對照：A5 的行為不得被新旗標改變。"""
    calls = _trace("A5", monkeypatch)
    assert sum(c[0] == "merge" for c in calls) == 4, "A5 應每個 stage 都 merge"
    assert not any(c[0] == "wrap" for c in calls), "A5 不該掛 warm-start 的 LoRA"
    assert calls[0] == ("new", None, True), "A5 建模時就該掛 LoRA"
