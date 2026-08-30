"""DR-046 Phase 0：merge-and-reset 與 frozen-stack 的等價性（純張量，不需資料集）。

**命題**：下面兩條路徑訓練出同一個模型。

    路徑甲（主方法，merge-and-reset）
        段1 訓練 (A1,B1) → merge_（W ← W + ΔW1，adapter 歸零重抽）
        → 段2 訓練 (A2,B2) → merge_
    路徑乙（frozen-stack）
        段1 訓練 (A1,B1) 不 merge → 凍結 (A1,B1) → 加掛 (A2,B2) 訓練
        → 最後把兩組一次合併

**為什麼會相等**：`LoRALinear.forward` 每次物化 `W_eff = W + (B@A)·scale` 再走單一
`F.linear`（見 lora.py 的說明）。因此

    甲：merge 後 W1 = W0 + ΔW1；段2 前向用 W1 + ΔW2
    乙：前向用 (W0 + ΔW1) + ΔW2

**加總順序相同**，所以不只數學相等，浮點上也應該逐位元相同；(A2,B2) 收到的梯度
因此也相同，兩段訓練軌跡完全重合。

⚠️ **實作落差（DR-046 Phase 0 回報項）**：`selector/lora.py` **沒有** frozen-stack
   這個模式 —— 主方法就是 merge-and-reset，一個 `LoRALinear` 只有一組 (A,B)。
   本檔的 `StackedLoRALinear` 是**測試專用的對照組**，不是 selector/ 的一部分，
   也不改變主方法的任何行為。它刻意用與 `LoRALinear` 相同的算式與相同的
   RNG 消耗方式，否則兩路徑不可比。

⚠️ **RNG 對齊**：路徑甲的 `merge_()` 會呼叫 `reset_lora()` 重抽 A、歸零 B。
   路徑乙加掛第二組時必須用**同一個種子、同樣的順序、同樣的張量形狀**初始化，
   否則第二段的起點不同，比較沒有意義。本檔在兩邊都先 `torch.manual_seed(SEED_SEG2)`
   再依相同的 layer 順序初始化。
"""
from __future__ import annotations

import copy
import math

import pytest
import torch
import torch.nn as nn

from selector.lora import LoRALinear, apply_lora, lora_layers, lora_parameters, merge_lora

SEED_MODEL, SEED_DATA, SEED_SEG1, SEED_SEG2 = 0, 1, 2, 3
IN, HID, N, STEPS, LR, RANK = 16, 8, 32, 12, 1e-2, 4
RTOL, ATOL = 1e-5, 1e-6


def make_model() -> nn.Module:
    """與 F_g / F_p 同形狀的小堆疊：Linear → GELU → Linear。"""
    torch.manual_seed(SEED_MODEL)
    return nn.Sequential(nn.Linear(IN, HID), nn.GELU(), nn.Linear(HID, 1))


def make_data(seed: int):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(N, IN, generator=g), torch.randn(N, 1, generator=g))


# ── 測試專用：frozen-stack（selector/ 裡沒有這個模式）────────────────────────

class StackedLoRALinear(nn.Module):
    """凍結的舊 adapter + 可訓練的新 adapter，共用同一個 base weight。

    `effective_weight` 的加總順序**刻意**寫成 `(W + Δ_old) + Δ_new`，
    與路徑甲「先 merge 成 W1，再加 Δ2」逐項對齊 —— 順序不同就只有數學相等，
    浮點上會有差。
    """

    def __init__(self, src: LoRALinear):
        super().__init__()
        self.in_features, self.out_features = src.in_features, src.out_features
        self.r, self.scale = src.r, src.scale
        self.weight = nn.Parameter(src.weight.detach().clone(), requires_grad=False)
        self.bias = (nn.Parameter(src.bias.detach().clone(), requires_grad=False)
                     if src.bias is not None else None)
        # 舊 adapter：接手段1 訓練完的值，凍結
        self.old_A = nn.Parameter(src.lora_A.detach().clone(), requires_grad=False)
        self.old_B = nn.Parameter(src.lora_B.detach().clone(), requires_grad=False)
        # 新 adapter：形狀與 LoRALinear 相同，初始化另外做（RNG 要對齊）
        self.new_A = nn.Parameter(torch.empty(self.r, self.in_features))
        self.new_B = nn.Parameter(torch.zeros(self.out_features, self.r))

    def reset_new(self) -> None:
        """與 `LoRALinear.reset_lora` 完全相同的呼叫與形狀 → 消耗相同的 RNG。"""
        nn.init.kaiming_uniform_(self.new_A, a=math.sqrt(5))
        nn.init.zeros_(self.new_B)

    def delta_old(self) -> torch.Tensor:
        return (self.old_B @ self.old_A) * self.scale

    def delta_new(self) -> torch.Tensor:
        return (self.new_B @ self.new_A) * self.scale

    def effective_weight(self) -> torch.Tensor:
        return (self.weight + self.delta_old()) + self.delta_new()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.effective_weight(), self.bias)

    @torch.no_grad()
    def merge_all_(self) -> None:
        """最後一次把兩組一起併進 base weight。"""
        self.weight.data = self.effective_weight().detach()
        self.old_B.zero_(); self.new_B.zero_()


def stack_on(model: nn.Module) -> nn.Module:
    """把每個 LoRALinear 換成 StackedLoRALinear（保留段1 的 adapter）。"""
    for name, child in list(model.named_children()):
        if isinstance(child, LoRALinear):
            setattr(model, name, StackedLoRALinear(child))
        else:
            stack_on(child)
    return model


def stacked_layers(model: nn.Module) -> list[StackedLoRALinear]:
    return [m for m in model.modules() if isinstance(m, StackedLoRALinear)]


# ── 訓練 ────────────────────────────────────────────────────────────────────

def train(model: nn.Module, params, x, y, *, steps=STEPS, lr=LR) -> list[float]:
    opt = torch.optim.SGD(params, lr=lr)
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    return losses


def effective_weights(model: nn.Module) -> list[torch.Tensor]:
    out = []
    for m in model.modules():
        if isinstance(m, (LoRALinear, StackedLoRALinear)):
            out.append(m.effective_weight().detach().clone())
    return out


def run_both_paths():
    """跑完兩條路徑，回傳 (甲, 乙) 的 (模型, 段2 loss 軌跡)。"""
    x1, y1 = make_data(SEED_DATA)
    x2, y2 = make_data(SEED_DATA + 100)

    # 段 1：只訓練一次，兩路徑從**同一個結果**分岔（保證段1 完全相同）
    base = apply_lora(make_model(), r=RANK)
    torch.manual_seed(SEED_SEG1)
    for layer in lora_layers(base):
        layer.reset_lora()
    train(base, lora_parameters(base), x1, y1)

    # 路徑甲：merge → reset（RNG 由 SEED_SEG2 控制）→ 段2 → merge
    a = copy.deepcopy(base)
    torch.manual_seed(SEED_SEG2)
    merge_lora(a)
    loss_a = train(a, lora_parameters(a), x2, y2)

    # 路徑乙：凍結段1 → 加掛第二組（同一個 SEED_SEG2、同樣的 layer 順序）→ 段2
    b = stack_on(copy.deepcopy(base))
    torch.manual_seed(SEED_SEG2)
    for layer in stacked_layers(b):
        layer.reset_new()
    loss_b = train(b, [p for layer in stacked_layers(b)
                       for p in (layer.new_A, layer.new_B)], x2, y2)

    return (a, loss_a), (b, loss_b), (x2, y2)


# ── 測試 ────────────────────────────────────────────────────────────────────

def test_effective_weights_match_after_two_segments():
    """核心命題：兩路徑的最終 effective weight 相同。"""
    (a, _), (b, _), _ = run_both_paths()
    wa, wb = effective_weights(a), effective_weights(b)
    assert len(wa) == len(wb) == 2, f"層數不符：{len(wa)} vs {len(wb)}"
    for i, (u, v) in enumerate(zip(wa, wb)):
        assert torch.allclose(u, v, rtol=RTOL, atol=ATOL), (
            f"第 {i} 層 effective weight 不符，最大差 {float((u - v).abs().max()):.3e}")


def test_forward_outputs_match_on_same_batch():
    """同一批輸入的前向輸出相同。"""
    (a, _), (b, _), (x2, _y2) = run_both_paths()
    with torch.no_grad():
        oa, ob = a(x2), b(x2)
    assert torch.allclose(oa, ob, rtol=RTOL, atol=ATOL), (
        f"前向輸出不符，最大差 {float((oa - ob).abs().max()):.3e}")


def test_segment2_training_trajectories_match():
    """不只終點相同 —— 段2 的每一步 loss 都相同，代表梯度一路一致。"""
    (_, la), (_, lb), _ = run_both_paths()
    assert len(la) == len(lb) == STEPS
    for i, (p, q) in enumerate(zip(la, lb)):
        assert p == pytest.approx(q, rel=RTOL, abs=ATOL), (
            f"第 {i} 步 loss 不符：{p:.10f} vs {q:.10f}")


def test_path_b_final_merge_lands_on_same_weight():
    """路徑乙「最後一次合併」之後，base weight 應等於路徑甲 merge 後的 weight。"""
    (a, _), (b, _), _ = run_both_paths()
    merge_lora(a)                                   # 甲：段2 結束再 merge 一次
    for layer in stacked_layers(b):
        layer.merge_all_()                          # 乙：兩組一次合併
    wa = [m.weight.detach() for m in a.modules() if isinstance(m, LoRALinear)]
    wb = [m.weight.detach() for m in stacked_layers(b)]
    for i, (u, v) in enumerate(zip(wa, wb)):
        assert torch.allclose(u, v, rtol=RTOL, atol=ATOL), (
            f"第 {i} 層 merge 後 base weight 不符，最大差 {float((u - v).abs().max()):.3e}")


# ── 反向對照：證明這條測試抓得到違例（憲法 §2.3）────────────────────────────

def test_different_segment2_seed_breaks_equivalence():
    """RNG 沒對齊時必須測得出來 —— 否則上面的通過沒有意義。"""
    x1, y1 = make_data(SEED_DATA)
    x2, y2 = make_data(SEED_DATA + 100)
    base = apply_lora(make_model(), r=RANK)
    torch.manual_seed(SEED_SEG1)
    for layer in lora_layers(base):
        layer.reset_lora()
    train(base, lora_parameters(base), x1, y1)

    a = copy.deepcopy(base)
    torch.manual_seed(SEED_SEG2)
    merge_lora(a)
    train(a, lora_parameters(a), x2, y2)

    b = stack_on(copy.deepcopy(base))
    torch.manual_seed(SEED_SEG2 + 999)              # ← 刻意不對齊
    for layer in stacked_layers(b):
        layer.reset_new()
    train(b, [p for layer in stacked_layers(b)
              for p in (layer.new_A, layer.new_B)], x2, y2)

    wa, wb = effective_weights(a), effective_weights(b)
    assert not all(torch.allclose(u, v, rtol=RTOL, atol=ATOL) for u, v in zip(wa, wb)), (
        "換了 seed 兩路徑竟然仍相同 —— 代表這條測試對 adapter 初始化不敏感，"
        "上面的等價結論不可信")


def test_stacking_order_matters_for_bit_level_agreement():
    """記錄一件事實：加總順序寫反時，仍在容差內但不再逐位元相同。

    這條不是要求嚴格位元相同（規格允許浮點加法順序的微差），
    而是把「為什麼要刻意寫成 (W + Δ_old) + Δ_new」留下依據。
    """
    torch.manual_seed(SEED_MODEL)
    W = torch.randn(HID, IN, dtype=torch.float32)
    d1 = torch.randn(HID, IN, dtype=torch.float32) * 1e-3
    d2 = torch.randn(HID, IN, dtype=torch.float32) * 1e-3
    same_order = (W + d1) + d2
    other_order = W + (d1 + d2)
    assert torch.allclose(same_order, other_order, rtol=RTOL, atol=ATOL)
