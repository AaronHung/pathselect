"""CONTRACT-3 — bounded Selection Memory 的資料結構（本輪只定義，不實作 CL）。

entry = (tau, slide_id, e_t, B_tilde_t, r_old[J], cand_idx[256],
         s_old[cand_idx], u_old[cand_idx])

不存 patch feature：只留 slide_id + index，需要時從特徵檔重載。
容量上限 |M| <= 512，寫成常數 MEMORY_CAPACITY。

符號約定（PI 裁定）：**J = group 數（=8）**，**|M| = 記憶體容量（<=512）**。
r_old 是 group 層分數向量，長度為 J。
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Protocol

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


class ReplacementPolicy(Protocol):
    """汰換策略介面 —— 可替換，主線用 reservoir sampling。"""

    def offer(self, entries: list, entry, capacity: int, n_seen: int) -> None:
        ...


class ReservoirSampling:
    """標準 reservoir sampling：任何時刻 M 都是「至今所有 entry」的均勻樣本。

    未滿就直接放；滿了之後以機率 capacity / n_seen 取代一個隨機位置。
    """

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def offer(self, entries, entry, capacity, n_seen) -> None:
        if len(entries) < capacity:
            entries.append(entry)
            return
        j = self.rng.randrange(n_seen)
        if j < capacity:
            entries[j] = entry


class FIFO:
    """對照用的汰換策略：永遠留最新的。"""

    def offer(self, entries, entry, capacity, n_seen) -> None:
        entries.append(entry)
        if len(entries) > capacity:
            entries.pop(0)


class SelectionMemory:
    """有界的 entry 儲存。汰換策略可替換，預設 reservoir sampling。"""

    def __init__(self, capacity: int = MEMORY_CAPACITY,
                 policy: Optional[ReplacementPolicy] = None,
                 allow_over_contract: bool = False):
        """allow_over_contract：CONTRACT-3 把 |M| 凍結在 512，預設超過就拒絕。

        只有記憶體效率曲線（E1）需要跑到 1024 —— 那是**刻意探測契約之外**的
        診斷點，必須顯式開啟，不會被誤用。
        """
        if capacity <= 0:
            raise ValueError(f"capacity 必須為正，got {capacity}")
        if capacity > MEMORY_CAPACITY and not allow_over_contract:
            raise ValueError(
                f"capacity {capacity} 超過 CONTRACT-3 的 |M| <= {MEMORY_CAPACITY}；"
                f"要刻意探測契約之外請顯式傳 allow_over_contract=True")
        self.capacity = capacity
        self.policy = policy or ReservoirSampling()
        self.n_seen = 0
        self._entries: list[SelectionMemoryEntry] = []

    def add(self, entry: SelectionMemoryEntry) -> None:
        self.n_seen += 1
        self.policy.offer(self._entries, entry, self.capacity, self.n_seen)
        if len(self._entries) > self.capacity:      # 任何策略都不得越界
            raise RuntimeError(f"|M| 超過上限 {self.capacity}")

    def sample(self, k: int, rng: Optional[random.Random] = None) -> list:
        """取 k 筆做 replay；k 大於現有數量時全給。"""
        if k >= len(self._entries):
            return list(self._entries)
        return (rng or random.Random(0)).sample(self._entries, k)

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
        self.n_seen = 0


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


def reload_features(entry: SelectionMemoryEntry, cfg: dict) -> tuple:
    """依 slide_id + index 從特徵檔重載候選 patch。entry 本身不存 feature。

    回傳 (Z_full [n, D], Z_cand [len(cand_idx), D], label)。
    label 也是重載來的（來自表格），不佔 entry 欄位 —— CONTRACT-3 的欄位是凍結的。
    """
    from .evaluate import read_slide, slide_dataset

    task_pos = list(cfg["tasks"]).index(entry.tau)
    ds, shift = slide_dataset(cfg, entry.tau, task_pos, "train")
    by_sid = {str(s): i for i, s in enumerate(ds.sids)}
    if entry.slide_id not in by_sid:
        raise KeyError(f"slide_id {entry.slide_id} 不在 {entry.tau} 的 train split")
    rec = read_slide(ds, shift, by_sid[entry.slide_id])
    return rec.Z, rec.Z.index_select(0, entry.cand_idx.to(torch.long)), rec.label


def selected_from_entry(entry: SelectionMemoryEntry, k: int) -> tuple:
    """entry 沒有直接記錄「選了誰」，但 s_old 足以還原：候選中分數最高的 k 個。

    回傳 (原 slide 的 index [k], 在 cand_idx 中的位置 [k])。
    """
    k = min(k, entry.s_old.numel())
    pos = torch.topk(entry.s_old, k).indices
    return entry.cand_idx.index_select(0, pos), pos
