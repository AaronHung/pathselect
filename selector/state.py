"""CONTRACT-2 — evidence state 是顯式張量。

單張 slide 在 chunked sequential loop 中的狀態：

    e_t = 0                        (t = 0)
    e_t = mean(z for z in E_t)     (t > 0)
    B_tilde_t = B_t / B_0

selector 的輸入尾巴一律是 `feature() = [e_t ; B_tilde_t]`，共 513 維。

⚠️ e_t 在輪與輪之間 detach（CONTRACT-2）。不做 BPTT。
   代價：這是 state-conditioned greedy acquisition，每一輪只依當前狀態決策，
   不對未來輪次做任何預估。
"""
from __future__ import annotations

import torch


class EvidenceState:
    """E_t、e_t、B_t、已選 indices、候選 mask。

    Z: [n, D] 整張 slide 的 patch embedding（不複製，只持有參考）。
    """

    def __init__(self, Z: torch.Tensor, budget: int):
        self.Z = Z
        self.n = int(Z.shape[0])
        self.dim = int(Z.shape[1])
        self.reset(budget)

    # ── lifecycle ───────────────────────────────────────────────────────────

    def reset(self, B0: int) -> None:
        """回到 t=0：沒有任何已選 patch，全部 patch 都是候選。"""
        if B0 <= 0:
            raise ValueError(f"budget must be positive, got {B0}")
        self.B0 = int(B0)
        self.B_t = int(B0)
        self.t = 0
        self.selected: list[int] = []
        self._sum = torch.zeros(self.dim, dtype=self.Z.dtype, device=self.Z.device)
        self._available = torch.ones(self.n, dtype=torch.bool, device=self.Z.device)

    def update(self, z_new: torch.Tensor, idx_new: torch.Tensor) -> None:
        """把這一輪選到的 patch 併入 evidence，並從候選移除。

        z_new:   [k, D]
        idx_new: [k]  這些 patch 在原 slide 中的 index
        """
        idx_new = idx_new.reshape(-1).to(torch.long)
        if z_new.shape[0] != idx_new.shape[0]:
            raise ValueError(f"z_new {tuple(z_new.shape)} vs idx_new {tuple(idx_new.shape)}")
        if idx_new.numel() == 0:
            self.t += 1
            return
        if not bool(self._available.index_select(0, idx_new).all()):
            raise ValueError("同一個 patch 不得重複選取（已選的必須先從候選移除）")

        # e_t 在輪與輪之間 detach —— 狀態不帶梯度回上一輪。
        self._sum = self._sum + z_new.detach().sum(0)
        self._available.index_fill_(0, idx_new, False)
        self.selected.extend(int(i) for i in idx_new.tolist())
        self.B_t = max(self.B0 - len(self.selected), 0)
        self.t += 1

    # ── state ───────────────────────────────────────────────────────────────

    @property
    def n_selected(self) -> int:
        return len(self.selected)

    @property
    def available_mask(self) -> torch.Tensor:
        """[n] bool：True = 仍可被選。"""
        return self._available

    @property
    def candidate_indices(self) -> torch.Tensor:
        return self._available.nonzero(as_tuple=False).reshape(-1)

    @property
    def e_t(self) -> torch.Tensor:
        """[D]：t=0 時全零，t>0 時為已選 patch 的平均。"""
        if not self.selected:
            return torch.zeros(self.dim, dtype=self.Z.dtype, device=self.Z.device)
        return self._sum / len(self.selected)

    @property
    def B_tilde_t(self) -> float:
        return self.B_t / self.B0

    def evidence_sum(self) -> torch.Tensor:
        """[D]：sum(E_t)。counterfactual gain 的 rank-1 更新要用。"""
        return self._sum

    def feature(self) -> torch.Tensor:
        """[D+1] = [e_t ; B_tilde_t]，接在 selector 輸入的尾端。"""
        b = torch.tensor([self.B_tilde_t], dtype=self.Z.dtype, device=self.Z.device)
        return torch.cat([self.e_t, b], dim=0)

    def summary(self) -> dict:
        return {"t": self.t, "n_selected": self.n_selected, "B_t": self.B_t,
                "B_tilde_t": round(self.B_tilde_t, 4),
                "n_candidates": int(self._available.sum())}
