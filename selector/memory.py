"""CONTRACT-3 — bounded Selection Memory 的資料結構。

entry（**schema v2**，DR-048 B5）＝

    (tau, sample_key, r_old[J], cand_idx[<=256], s_old[cand_idx], u_old[cand_idx])

不存 patch feature：只留 key + index，需要時從特徵檔重載。
容量上限 |M| <= 512，寫成常數 MEMORY_CAPACITY。

符號約定（PI 裁定）：**J = group 數（=8）**，**|M| = 記憶體容量（<=512）**。
r_old 是 group 層分數向量，長度為 J。

## v1 → v2 的兩處改動（DR-048 B5）

1. **拿掉 `e_t`（[512] float32）與 `B_tilde_t`（float）**。這兩欄由 `make_entry`
   寫入，但**整個 repo 沒有任何地方讀它們** —— replay 的損失（`selector/train.py`
   的 `continual_terms`）只用 `cand_idx / r_old / s_old / u_old`，重載只用身分欄。
   它們佔掉每筆 entry 約 2 KB，是最大的一塊死負載。
   ⚠️ 因為沒有讀取端，移除**不會改變任何訓練行為**；DR-048 B5 以
   「舊/新 schema 各跑一次 A5 seed0 1 epoch，逐位元比對 per_slide」釘住這件事。

2. **`slide_id: str` → `sample_key: int`（int64）**。key ↔ slide_id 的對照表
   **不放在記憶庫裡**：key 是 `(tau, slide_id)` 的穩定雜湊（`sample_key()`），
   任何時候都能從 split 的 sid 清單重建（`SampleKeyIndex.from_slide_ids`）。
   記憶庫因此不必攜帶字串，也不必額外持久化一份對照檔。
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Optional, Protocol

import torch

#: |M| <= 512
MEMORY_CAPACITY = 512
#: 與 utility.CANDIDATE_SIZE 同一組候選
CANDIDATE_SIZE = 256


#: sample_key 的位寬。取 63 bit（不含符號位）→ 落在 int64 的正數範圍。
SAMPLE_KEY_BITS = 63


def sample_key(tau: str, slide_id: str) -> int:
    """`(tau, slide_id)` → 穩定的 int64 key。

    用 blake2b 而不是內建 `hash()`：後者對 str 有 per-process 隨機化，
    換一次行程就換一組 key，記憶庫就不可重現了。
    """
    h = hashlib.blake2b(f"{tau}/{slide_id}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") & ((1 << SAMPLE_KEY_BITS) - 1)


@dataclass(frozen=True)
class SelectionMemoryEntry:
    """一次「在某個狀態下做過的選擇」的壓縮紀錄。不含任何 patch feature。"""
    tau: str                     # task identity
    sample_key: int              # int64；對照表在記憶庫之外（SampleKeyIndex）
    r_old: torch.Tensor          # [J]   當時的 group 分數
    cand_idx: torch.Tensor       # [<=256] 候選 patch 在該 slide 的 index
    s_old: torch.Tensor          # [<=256] 候選當時的 patch 分數
    u_old: torch.Tensor          # [<=256] 候選當時的 counterfactual gain

    def __post_init__(self):
        if not isinstance(self.sample_key, int) or isinstance(self.sample_key, bool):
            raise TypeError(f"sample_key 必須是 int，got {type(self.sample_key).__name__}")
        if not 0 <= self.sample_key < (1 << SAMPLE_KEY_BITS):
            raise ValueError(f"sample_key 超出 int64 正數範圍：{self.sample_key}")
        n = self.cand_idx.reshape(-1).shape[0]
        if n > CANDIDATE_SIZE:
            raise ValueError(f"cand_idx 最多 {CANDIDATE_SIZE} 個，got {n}")
        for name in ("s_old", "u_old"):
            v = getattr(self, name).reshape(-1)
            if v.shape[0] != n:
                raise ValueError(f"{name} 長度 {v.shape[0]} 與 cand_idx {n} 不符")


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


class SampleKeyIndex:
    """`sample_key` → `slide_id` 的對照表。**放在記憶庫之外**。

    key 是 `(tau, slide_id)` 的純函數，所以這張表隨時可以從 split 的 sid 清單
    重建 —— 記憶庫不必攜帶它，也不必額外存一份檔案。

    碰撞在 63 bit / 千餘張 slide 的規模下機率約 1e-13，但**還是檢查**：
    同一個 key 對上兩個不同的 slide_id 會直接報錯，不靜默覆蓋。
    """

    def __init__(self) -> None:
        self._to_sid: dict[int, str] = {}

    @classmethod
    def from_slide_ids(cls, tau: str, slide_ids) -> "SampleKeyIndex":
        idx = cls()
        for sid in slide_ids:
            idx.register(tau, str(sid))
        return idx

    def register(self, tau: str, slide_id: str) -> int:
        k = sample_key(tau, slide_id)
        prev = self._to_sid.get(k)
        if prev is not None and prev != slide_id:
            raise RuntimeError(f"sample_key 碰撞：{k} 同時對應 {prev!r} 與 {slide_id!r}")
        self._to_sid[k] = slide_id
        return k

    def resolve(self, key: int) -> str:
        if key not in self._to_sid:
            raise KeyError(f"sample_key {key} 不在對照表裡（表內 {len(self._to_sid)} 筆）")
        return self._to_sid[key]

    def __len__(self) -> int:
        return len(self._to_sid)


def make_entry(tau: str, slide_id: str, state, r_old: torch.Tensor,
               cand_idx: torch.Tensor, s_all: torch.Tensor,
               u_cand: Optional[torch.Tensor] = None) -> SelectionMemoryEntry:
    """從當前狀態組一筆 entry（全部 detach 到 CPU，避免拖住計算圖）。

    ⚠️ `state` 保留在簽名裡但**不再被讀取** —— schema v2 拿掉了 `e_t` /
    `B_tilde_t`（見模組 docstring）。留著參數是為了不動 `selector/train.py`
    的呼叫端；哪天真的要用狀態，接線點還在原處。
    """
    cand_idx = cand_idx.reshape(-1).to(torch.long)
    u = (u_cand if u_cand is not None
         else torch.zeros(cand_idx.shape[0], dtype=torch.float32))
    return SelectionMemoryEntry(
        tau=tau, sample_key=sample_key(tau, slide_id),
        r_old=r_old.detach().cpu(),
        cand_idx=cand_idx.detach().cpu(),
        s_old=s_all.detach().cpu().index_select(0, cand_idx.detach().cpu()),
        u_old=u.detach().cpu().reshape(-1))


def reload_features(entry: SelectionMemoryEntry, cfg: dict) -> tuple:
    """依 sample_key + index 從特徵檔重載候選 patch。entry 本身不存 feature。

    回傳 (Z_full [n, D], Z_cand [len(cand_idx), D], label)。
    label 也是重載來的（來自表格），不佔 entry 欄位 —— CONTRACT-3 的欄位是凍結的。

    對照表當場從該 split 的 sid 清單重建（`SampleKeyIndex.from_slide_ids`），
    因此 v2 不需要在記憶庫或磁碟上另存 key → path 的映射。
    """
    from .evaluate import read_slide, slide_dataset

    task_pos = list(cfg["tasks"]).index(entry.tau)
    ds, shift = slide_dataset(cfg, entry.tau, task_pos, "train")
    index = SampleKeyIndex.from_slide_ids(entry.tau, ds.sids)
    try:
        slide_id = index.resolve(entry.sample_key)
    except KeyError as exc:
        raise KeyError(f"sample_key {entry.sample_key} 不在 {entry.tau} 的 "
                       f"train split（{len(index)} 張）") from exc
    by_sid = {str(s): i for i, s in enumerate(ds.sids)}
    rec = read_slide(ds, shift, by_sid[slide_id])
    return rec.Z, rec.Z.index_select(0, entry.cand_idx.to(torch.long)), rec.label


def selected_from_entry(entry: SelectionMemoryEntry, k: int) -> tuple:
    """entry 沒有直接記錄「選了誰」，但 s_old 足以還原：候選中分數最高的 k 個。

    回傳 (原 slide 的 index [k], 在 cand_idx 中的位置 [k])。
    """
    k = min(k, entry.s_old.numel())
    pos = torch.topk(entry.s_old, k).indices
    return entry.cand_idx.index_select(0, pos), pos
