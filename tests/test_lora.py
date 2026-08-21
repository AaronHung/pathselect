"""S4-1 — LoRA + sequential merge。"""
from __future__ import annotations

import pytest
import torch

from selector.lora import (DEFAULT_RANK, LoRALinear, PerTaskLoRABank, apply_lora,
                           base_norm, delta_norm, lora_layers, lora_parameters,
                           merge_lora, n_parameters)
from selector.model import GroupSelector, PatchSelector, SELECTOR_INPUT_DIM

D = SELECTOR_INPUT_DIM


def _models(seed=0, r=DEFAULT_RANK):
    torch.manual_seed(seed)
    f_g, f_p = GroupSelector(), PatchSelector()
    apply_lora(f_g, r=r), apply_lora(f_p, r=r)
    return f_g, f_p


def _train_a_bit(f_p, steps=3, seed=0):
    """隨便走幾步，讓 ΔW 不為零。"""
    torch.manual_seed(seed + 100)
    opt = torch.optim.Adam(lora_parameters(f_p), lr=1e-2)
    for _ in range(steps):
        x = torch.randn(16, D)
        loss = (f_p(x) - torch.randn(16)).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()


def test_wrapping_does_not_change_forward_at_init():
    """B 初始化為零 → ΔW 恰好為零 → 包上 LoRA 不改變任何輸出。"""
    torch.manual_seed(0)
    f_p = PatchSelector()
    x = torch.randn(32, D)
    before = f_p(x).detach().clone()
    apply_lora(f_p)
    assert torch.equal(before, f_p(x))
    assert delta_norm(f_p) == 0.0


def test_forward_is_bit_identical_across_merge():
    """核心約束：merge 前後 forward 位元相同。"""
    f_g, f_p = _models()
    _train_a_bit(f_p)
    x = torch.randn(64, D)
    with torch.no_grad():
        before = f_p(x).clone()
    assert delta_norm(f_p) > 0            # 確認真的有東西可以 merge
    merge_lora(f_g, f_p)
    with torch.no_grad():
        after = f_p(x)
    assert torch.equal(before, after), (before - after).abs().max()


def test_lora_is_zero_after_merge():
    f_g, f_p = _models()
    _train_a_bit(f_p)
    merge_lora(f_g, f_p)
    assert delta_norm(f_g, f_p) == 0.0
    for layer in lora_layers(f_p):
        assert torch.equal(layer.lora_B, torch.zeros_like(layer.lora_B))
        assert torch.equal(layer.effective_weight(), layer.weight)


def test_merge_updates_base_weight():
    """W_t = W_{t-1} + ΔW_t：base weight 真的被改了。"""
    f_g, f_p = _models()
    w0 = base_norm(f_p)
    _train_a_bit(f_p)
    merge_lora(f_g, f_p)
    assert base_norm(f_p) != w0


def test_parameter_count_does_not_grow_with_tasks():
    """參數總量不隨 task 數成長。"""
    f_g, f_p = _models()
    counts = [n_parameters(f_g, f_p)]
    for t in range(4):
        _train_a_bit(f_p, seed=t)
        merge_lora(f_g, f_p)
        counts.append(n_parameters(f_g, f_p))
    assert len(set(counts)) == 1, counts


def test_only_lora_parameters_require_grad():
    f_g, f_p = _models()
    trainable = [p for m in (f_g, f_p) for p in m.parameters() if p.requires_grad]
    assert len(trainable) == len(lora_parameters(f_g, f_p))
    for layer in lora_layers(f_p):
        assert not layer.weight.requires_grad
        if layer.bias is not None:
            assert not layer.bias.requires_grad


def test_lora_parameter_budget_is_small():
    f_g, f_p = _models(r=4)
    n_lora = sum(p.numel() for p in lora_parameters(f_g, f_p))
    n_all = n_parameters(f_g, f_p)
    assert n_lora < 0.1 * n_all, (n_lora, n_all)


def test_rank_is_configurable():
    for r in (1, 2, 8):
        f_g, f_p = _models(r=r)
        for layer in lora_layers(f_p):
            assert layer.lora_A.shape[0] == r and layer.lora_B.shape[1] == r


def test_invalid_rank_rejected():
    torch.manual_seed(0)
    with pytest.raises(ValueError, match="rank"):
        LoRALinear(torch.nn.Linear(4, 4), r=0)


def test_gradient_flows_into_lora_only():
    f_g, f_p = _models()
    x = torch.randn(8, D)
    f_p(x).sum().backward()
    for layer in lora_layers(f_p):
        assert layer.lora_A.grad is not None or layer.lora_B.grad is not None
        assert layer.weight.grad is None


def test_per_task_bank_is_oracle_only_and_restores_exactly():
    """per-task bank 是 oracle upper bound 的實作，不是主方法。"""
    f_g, f_p = _models()
    bank = PerTaskLoRABank()
    _train_a_bit(f_p, seed=1)
    bank.snapshot("tcga_esca", f_g, f_p)
    snap = [l.delta_w().clone() for l in lora_layers(f_p)]
    _train_a_bit(f_p, seed=2)
    assert not torch.equal(snap[0], lora_layers(f_p)[0].delta_w())
    bank.restore("tcga_esca", f_g, f_p)
    for a, layer in zip(snap, lora_layers(f_p)):
        assert torch.equal(a, layer.delta_w())
    assert bank.tasks() == ["tcga_esca"] and len(bank) == 1
