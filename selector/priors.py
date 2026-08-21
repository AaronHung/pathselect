"""Semantic prior — 只看 frozen CONCH 的 patch/group 對 class text 的相似度。

三種都實作，主線由 PI 裁定為 `discriminative`：

  none            不用 prior（全零）
  max_sim         p_i = max_c cos(x_i, t_c)，min-max 正規化到 [0, 1]
  discriminative  z_ic = cos(x_i, t_c) / T ; p_i = 1 - H(softmax(z_i)) / log(C)

⚠️ 一律使用「全部」candidate class prompts。只餵 true class 會把 label 洩進
   selector，這由 `assert_full_class_space()` 在每次呼叫時強制檢查，不是靠註解。
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

PRIOR_KINDS = ("none", "max_sim", "discriminative")
MAINLINE_PRIOR = "discriminative"


def assert_full_class_space(f_txt: torch.Tensor, n_candidate_classes: int) -> None:
    """f_txt 必須涵蓋**全部** candidate class，不得只是 true class 的子集。"""
    assert f_txt.dim() == 2, f"f_txt must be [C, D], got {tuple(f_txt.shape)}"
    assert n_candidate_classes >= 2, (
        f"candidate class 至少要 2 類才談得上 prior，got {n_candidate_classes}")
    assert f_txt.shape[0] == n_candidate_classes, (
        f"prior 必須看全部 {n_candidate_classes} 個 candidate class，"
        f"實得 f_txt 只有 {f_txt.shape[0]} 類 —— 這通常代表只餵了 true class，"
        f"是 label leakage。")


def _min_max(v: torch.Tensor) -> torch.Tensor:
    """正規化到 [0, 1]；全等時回 0.5（沒有任何 patch 比較突出）。"""
    lo, hi = v.min(), v.max()
    span = hi - lo
    if float(span) <= 1e-12:
        return torch.full_like(v, 0.5)
    return (v - lo) / span


@torch.no_grad()
def semantic_prior(X: torch.Tensor, f_txt: torch.Tensor, *,
                   kind: str = MAINLINE_PRIOR,
                   n_candidate_classes: Optional[int] = None,
                   temperature: Optional[float] = None,
                   logit_scale: Optional[torch.Tensor | float] = None
                   ) -> torch.Tensor:
    """[N]：每個 patch（或 group prototype）的 semantic prior。

    X:      [N, D]  patch embedding 或 group prototype
    f_txt:  [C, D]  **全部** candidate class 的文字特徵
    kind:   PRIOR_KINDS 之一
    temperature: discriminative 的 T。None 時取 1 / logit_scale（與 frozen head
                 同一個溫度）；logit_scale 也沒給時退回 0.07。
    """
    if kind not in PRIOR_KINDS:
        raise ValueError(f"unknown prior kind: {kind}; expected one of {PRIOR_KINDS}")
    if kind == "none":
        return torch.zeros(X.shape[0], dtype=X.dtype, device=X.device)

    assert_full_class_space(f_txt, n_candidate_classes
                            if n_candidate_classes is not None else f_txt.shape[0])

    cos = F.normalize(X, dim=-1) @ F.normalize(f_txt, dim=-1).t()      # [N, C]
    if kind == "max_sim":
        return _min_max(cos.amax(dim=-1))

    # discriminative（主線）
    if temperature is None:
        temperature = (1.0 / float(logit_scale)) if logit_scale is not None else 0.07
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    C = cos.shape[1]
    logp = F.log_softmax(cos / temperature, dim=-1)                    # [N, C]
    entropy = -(logp.exp() * logp).sum(-1)                             # [N]
    import math
    return (1.0 - entropy / math.log(C)).clamp(0.0, 1.0)
