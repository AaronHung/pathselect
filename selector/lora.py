"""S4-1 — LoRA + sequential merge。

CONTRACT：W_t = W_{t-1} + ΔW_t。每個 task 只訓練 LoRA，task 結束後把 ΔW merge
進 base weight，LoRA 歸零重新開始。最終**永遠只有一個 shared selector**，
推論不需要 task id，參數總量不隨 task 數成長。

⚠️ 為了讓 merge 前後 forward **位元相同**，forward 每次都物化
    W_eff = W + (B @ A) * scale
再走單一 `F.linear`，而不是把 LoRA 當旁路另外加總。理由：

    x @ (W + ΔW).T        ≠(位元)  x @ W.T + x @ ΔW.T

兩者數學相等但浮點捨入不同。物化之後，merge 做的是 `W ← W_eff`（同一段算式、
同樣的位元）再把 B 歸零，於是 merge 後的 `W_eff = W_merged + 0` 與 merge 前
逐位元相同。

per-task LoRA bank 另存一份，那是 **oracle upper bound 的實作**，不是主方法。
"""
from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_RANK = 4


class LoRALinear(nn.Module):
    """包住一個 nn.Linear：W_eff = W + (B @ A) * (alpha / r)。

    A 隨機初始化、B 初始化為零 → 初始 ΔW 恰好為零，包上去不改變任何輸出。
    """

    def __init__(self, base: nn.Linear, r: int = DEFAULT_RANK,
                 alpha: float | None = None):
        super().__init__()
        if r <= 0:
            raise ValueError(f"rank must be positive, got {r}")
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.r = int(r)
        self.alpha = float(alpha if alpha is not None else r)
        self.scale = self.alpha / self.r

        self.weight = nn.Parameter(base.weight.detach().clone(), requires_grad=False)
        self.bias = (nn.Parameter(base.bias.detach().clone(), requires_grad=False)
                     if base.bias is not None else None)
        self.lora_A = nn.Parameter(torch.empty(self.r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.r))
        self.reset_lora()

    # ── LoRA state ──────────────────────────────────────────────────────────

    def reset_lora(self) -> None:
        """A 重新隨機初始化、B 歸零 → ΔW 恰好為零。"""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def delta_w(self) -> torch.Tensor:
        return (self.lora_B @ self.lora_A) * self.scale

    def effective_weight(self) -> torch.Tensor:
        return self.weight + self.delta_w()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.effective_weight(), self.bias)

    # ── merge ───────────────────────────────────────────────────────────────

    @torch.no_grad()
    def merge_(self) -> None:
        """W ← W + ΔW，然後 LoRA 歸零。forward 在 merge 前後位元相同。"""
        self.weight.data = self.effective_weight().detach()
        self.reset_lora()

    def lora_state(self) -> dict:
        return {"lora_A": self.lora_A.detach().clone(),
                "lora_B": self.lora_B.detach().clone()}

    def load_lora_state(self, state: dict) -> None:
        with torch.no_grad():
            self.lora_A.copy_(state["lora_A"])
            self.lora_B.copy_(state["lora_B"])


# ── module-level helpers ────────────────────────────────────────────────────

def apply_lora(module: nn.Module, r: int = DEFAULT_RANK,
               alpha: float | None = None) -> nn.Module:
    """就地把 module 底下所有 nn.Linear 換成 LoRALinear。base weight 一併凍結。"""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, r=r, alpha=alpha))
        else:
            apply_lora(child, r=r, alpha=alpha)
    return module


def lora_layers(module: nn.Module) -> list[LoRALinear]:
    return [m for m in module.modules() if isinstance(m, LoRALinear)]


def lora_parameters(*modules: nn.Module) -> list[nn.Parameter]:
    """只回傳 LoRA 參數（optimizer 只該看到這些）。"""
    out = []
    for m in modules:
        for layer in lora_layers(m):
            out += [layer.lora_A, layer.lora_B]
    return out


def merge_lora(*modules: nn.Module) -> None:
    """把所有 LoRA 併進 base weight 並歸零。task 結束時呼叫一次。"""
    for m in modules:
        for layer in lora_layers(m):
            layer.merge_()


def delta_norm(*modules: nn.Module) -> float:
    """所有 ΔW 的 Frobenius 範數（診斷用；merge 後應為 0）。"""
    total = 0.0
    for m in modules:
        for layer in lora_layers(m):
            total += float(layer.delta_w().detach().pow(2).sum())
    return math.sqrt(total)


def base_norm(*modules: nn.Module) -> float:
    """所有 base weight 的 Frobenius 範數（診斷用）。"""
    total = 0.0
    for m in modules:
        for layer in lora_layers(m):
            total += float(layer.weight.pow(2).sum())
    return math.sqrt(total)


def n_parameters(*modules: nn.Module) -> int:
    return sum(p.numel() for m in modules for p in m.parameters())


class PerTaskLoRABank:
    """per-task LoRA 快照。

    ⚠️ 這是 **oracle upper bound 的實作**，不是主方法：它需要推論時已知 task id，
    而且容量隨 task 數成長。主方法只有一個 merge 後的 shared selector。
    """

    def __init__(self):
        self._states: dict[str, list[dict]] = {}

    def snapshot(self, task: str, *modules: nn.Module) -> None:
        self._states[task] = [copy.deepcopy(layer.lora_state())
                              for m in modules for layer in lora_layers(m)]

    def restore(self, task: str, *modules: nn.Module) -> None:
        layers = [layer for m in modules for layer in lora_layers(m)]
        states = self._states[task]
        if len(layers) != len(states):
            raise ValueError(f"layer 數不符：{len(layers)} vs {len(states)}")
        for layer, st in zip(layers, states):
            layer.load_lora_state(st)

    def tasks(self) -> list[str]:
        return sorted(self._states)

    def __len__(self) -> int:
        return len(self._states)
