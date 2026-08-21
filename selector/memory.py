"""CONTRACT-3 — bounded Selection Memory 的資料結構（本輪只定義，不實作 CL）。

entry = (tau, slide_id, e_t, B_tilde_t, r_old[J], cand_idx[256],
         s_old[cand_idx], u_old[cand_idx])

不存 patch feature：只留 slide_id + index，需要時從特徵檔重載。
容量上限 |M| <= 512，寫成常數 MEMORY_CAPACITY。

符號約定（PI 裁定）：**J = group 數（=8）**，**|M| = 記憶體容量（<=512）**。
r_old 是 group 層分數向量，長度為 J。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

#: |M| <= 512
MEMORY_CAPACITY = 512
#: 與 utility.CANDIDATE_SIZE 同一組候選
CANDIDATE_SIZE = 256


@dataclass(frozen=True)
class SelectionMemoryEntry:
    """一次「在某個狀態下做過的選擇」的壓縮紀錄。不含任何 patch feature。"""
    tau: str                     # task identity
    slide_id: str                # 用來從特徵檔重載
    e_t: torch.Tensor            # [D]   當時的 evidence 均值
    B_tilde_t: float             # 當時的剩餘預算比例
    r_old: torch.Tensor          # [J]   當時的 group 分數
    cand_idx: torch.Tensor       # [<=256] 候選 patch 在該 slide 的 index
    s_old: torch.Tensor          # [<=256] 候選當時的 patch 分數
    u_old: torch.Tensor          # [<=256] 候選當時的 counterfactual gain

    def __post_init__(self):
        n = self.cand_idx.reshape(-1).shape[0]
        if n > CANDIDATE_SIZE:
            raise ValueError(f"cand_idx 最多 {CANDIDATE_SIZE} 個，got {n}")
        for name in ("s_old", "u_old"):
            v = getattr(self, name).reshape(-1)
            if v.shape[0] != n:
                raise ValueError(f"{name} 長度 {v.shape[0]} 與 cand_idx {n} 不符")
        if self.e_t.reshape(-1).shape[0] == 0:
            raise ValueError("e_t 不得為空")


class SelectionMemory:
    """有界的 entry 儲存（FIFO）。本輪只提供結構與容量保證，不做任何 CL 邏輯。"""

    def __init__(self, capacity: int = MEMORY_CAPACITY):
        if capacity <= 0 or capacity > MEMORY_CAPACITY:
            raise ValueError(f"capacity 必須在 1..{MEMORY_CAPACITY}，got {capacity}")
        self.capacity = capacity
        self._entries: list[SelectionMemoryEntry] = []

    def add(self, entry: SelectionMemoryEntry) -> None:
        self._entries.append(entry)
        if len(self._entries) > self.capacity:
            self._entries.pop(0)

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    def tasks(self) -> list[str]:
        return sorted({e.tau for e in self._entries})

    def by_task(self, tau: str) -> list[SelectionMemoryEntry]:
        return [e for e in self._entries if e.tau == tau]

    def clear(self) -> None:
        self._entries.clear()


def make_entry(tau: str, slide_id: str, state, r_old: torch.Tensor,
               cand_idx: torch.Tensor, s_all: torch.Tensor,
               u_cand: Optional[torch.Tensor] = None) -> SelectionMemoryEntry:
    """從當前狀態組一筆 entry（全部 detach 到 CPU，避免拖住計算圖）。"""
    cand_idx = cand_idx.reshape(-1).to(torch.long)
    u = (u_cand if u_cand is not None
         else torch.zeros(cand_idx.shape[0], dtype=torch.float32))
    return SelectionMemoryEntry(
        tau=tau, slide_id=slide_id,
        e_t=state.e_t.detach().cpu(), B_tilde_t=float(state.B_tilde_t),
        r_old=r_old.detach().cpu(),
        cand_idx=cand_idx.detach().cpu(),
        s_old=s_all.detach().cpu().index_select(0, cand_idx.detach().cpu()),
        u_old=u.detach().cpu().reshape(-1))
