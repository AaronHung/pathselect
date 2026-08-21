"""Exp 0 的無訓練 baseline 選取政策。

這些 policy **沒有分數**，只決定「選哪些 patch」；下游一律走
`selector.evaluate.select_and_classify`，與 learned policy 完全相同。

⚠️ 特徵檔是 [n, 512] 純張量，不含 patch 座標（資料集裡沒有 .h5、沒有 coords 檔），
所以 `grid_indices` 只能沿**特徵的原始順序**等距抽樣。那個順序來自切 patch 時的
掃描序，是 spatial uniform 的代理，不是真正的空間均勻取樣。
"""
from __future__ import annotations

import numpy as np
import torch


def random_indices(n: int, k: int, seed: int, device=None) -> torch.Tensor:
    """從 n 個 patch 均勻抽 k 個不重複（numpy RandomState，可重現）。"""
    if k <= 0 or k >= n:
        return torch.arange(n, device=device)
    rs = np.random.RandomState(seed)
    idx = np.sort(rs.choice(n, size=k, replace=False))
    return torch.as_tensor(idx, dtype=torch.long, device=device)


def grid_indices(n: int, k: int, device=None) -> torch.Tensor:
    """依原順序等距抽樣，stride = n // k。"""
    if k <= 0 or k >= n:
        return torch.arange(n, device=device)
    stride = n // k
    idx = np.arange(k) * stride
    return torch.as_tensor(idx, dtype=torch.long, device=device)
