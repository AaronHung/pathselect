#!/usr/bin/env python3
"""DR-048 B5：Selection Memory 的實際佔用 → `docs/MEMORY_FOOTPRINT.md`。

報四個數字（PI 指定）：**payload bytes**、**serialized bytes**、
**逐 entry 的 mean / max**、**|M| = 512 滿載總量**。schema v1 與 v2 並列。

兩種量法都報，因為它們回答的是不同的問題：

* **payload** = 純資料位元組（tensor 的 `element_size × nelement`，加上純量與
  字串本身）。這是「理論上需要搬多少資料」。
* **serialized** = `torch.save` 到 BytesIO 的實際長度。這是「存到磁碟／送過網路
  要多少」，含 pickle 與 zip 容器的額外開銷；小張量時開銷佔比很高，所以
  payload 單獨看會低估。

⚠️ **v1 的資料結構在本檔就地重建**（`_V1Entry`），只為了對照，不是可用的記憶庫；
   `selector/memory.py` 已經是 v2。這與 `tests/test_merge_equivalence.py` 建立
   測試專用對照組的作法一致（DR-046 裁定一）。

⚠️ 候選長度取自**真實資料**：從 `outputs/exp2/main/per_slide/*.json` 讀每張 slide
   的 `n_patch`，候選長度 = min(CANDIDATE_SIZE, n_patch)。同時另報「全 256」的
   最壞情況 —— |M| = 512 的預算必須撐得住最壞情況，不是平均值。
"""
from __future__ import annotations

import io
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from selector.memory import (CANDIDATE_SIZE, MEMORY_CAPACITY,          # noqa: E402
                             SelectionMemoryEntry, sample_key)

PER_SLIDE = ROOT / "outputs" / "exp2" / "main" / "per_slide"
OUT = ROOT / "docs" / "MEMORY_FOOTPRINT.md"
J = 8            # group 數
D = 512          # CONCH 特徵維度（v1 的 e_t 長度）
#: 真實 slide_id 的長度（TCGA barcode + UUID），量字串成本用
SID = "TCGA-R6-A6Y2-01Z-00-DX1.9B1F1957-2C59-4D5A-BCE7-9FCBF1417B42"


def _v1(n: int) -> dict:
    """schema v1 的欄位。**僅供對照**，不參與任何訓練路徑。"""
    return {"tau": "tcga_brca", "slide_id": SID, "e_t": torch.zeros(D),
            "B_tilde_t": 0.5, "r_old": torch.zeros(J),
            "cand_idx": torch.arange(n), "s_old": torch.zeros(n),
            "u_old": torch.zeros(n)}


def _v2(n: int) -> dict:
    return {"tau": "tcga_brca", "sample_key": sample_key("tcga_brca", SID),
            "r_old": torch.zeros(J), "cand_idx": torch.arange(n),
            "s_old": torch.zeros(n), "u_old": torch.zeros(n)}


def _v2_entry(n: int) -> SelectionMemoryEntry:
    """真正的 v2 entry 物件 —— 只拿來對照 dataclass 包裝的固定開銷。"""
    return SelectionMemoryEntry(**_v2(n))


def payload_bytes(entry: dict) -> int:
    """純資料位元組。tensor 用 element_size × nelement；str 用 UTF-8 長度。

    刻意**不用** `sys.getsizeof`：它含 CPython 物件表頭，會把「資料本身多大」
    和「這個直譯器的物件開銷多大」混在一起。int / float 一律算 8（int64）。
    """
    total = 0
    for v in entry.values():
        if isinstance(v, torch.Tensor):
            total += v.element_size() * v.nelement()
        elif isinstance(v, str):
            total += len(v.encode())
        else:                                    # int / float → 8 bytes
            total += 8
    return total


def serialized_bytes(entry) -> int:
    """`torch.save` 的實際長度。

    兩個 schema 都以**欄位字典**的形式量，兩邊的容器完全相同 ——
    否則 v2 會多背一層 dataclass 包裝，比較就不是同一把尺。
    dataclass 的固定開銷另外單獨報一行。
    """
    buf = io.BytesIO()
    torch.save(entry, buf)
    return buf.getbuffer().nbytes


def observed_candidate_lengths() -> list[int]:
    """從真實 per_slide 產物讀 n_patch → 候選長度 = min(256, n_patch)。"""
    ns = []
    for f in sorted(PER_SLIDE.glob("*.json")):
        for r in json.loads(f.read_text()):
            n = r.get("n_patch")
            if n:
                ns.append(min(CANDIDATE_SIZE, int(n)))
    return ns


def stats(make, lengths: list[int]) -> dict:
    pay = [payload_bytes(make(n)) for n in lengths]
    ser = [serialized_bytes(make(n)) for n in lengths]
    return {"payload_mean": statistics.mean(pay), "payload_max": max(pay),
            "ser_mean": statistics.mean(ser), "ser_max": max(ser)}


def kb(x: float) -> str:
    """位元組原值 + KB —— 只印 KB 會把「實測分布」與「最壞情況」的差異捨掉。"""
    return f"{x:,.0f} B（{x / 1024:.2f} KB）"


def mb(x: float) -> str:
    return f"{x / 1024 / 1024:.2f} MB"


def main() -> int:
    lengths = observed_candidate_lengths()
    if not lengths:
        print(f"⚠️ {PER_SLIDE} 沒有 per_slide 產物，無法取真實候選長度")
        return 1
    uniq = sorted(set(lengths))
    L = ["# Selection Memory 佔用量（schema v1 vs v2）", "",
         "DR-048 B5。產生：`python scripts/report_memory_footprint.py`。", "",
         "v2 相對 v1 的改動：拿掉 `e_t`（[512] float32）與 `B_tilde_t`，"
         "`slide_id: str` 改成 `sample_key: int64`。"
         "兩欄在整個 repo 都**只寫不讀**，移除不改變任何訓練行為 "
         "（見 `selector/memory.py` 模組說明與 DR-048 卡的位元等價驗證）。", "",
         "## 量法", "",
         "* **payload** = tensor 的 `element_size × nelement` + 字串的 UTF-8 位元組 "
         "+ 每個純量 8 bytes。不含 CPython 物件表頭。",
         "* **serialized** = `torch.save` 到 BytesIO 的長度（含 pickle／zip 容器開銷）。", "",
         f"候選長度取自真實產物 `outputs/exp2/main/per_slide/*.json` 的 `n_patch`："
         f"共 {len(lengths):,} 筆，長度介於 {min(uniq)}–{max(uniq)}，"
         f"其中 **{sum(x == CANDIDATE_SIZE for x in lengths) / len(lengths) * 100:.1f}% "
         f"撞到 {CANDIDATE_SIZE} 的上限**。", "",
         "⚠️ 因此下面兩張表幾乎相同 —— 絕大多數 slide 的候選都是滿的。"
         "保留兩張是為了讓「實測」與「預算上限」分開可讀，不是筆誤。", ""]

    for title, lens, note in [
            ("實測分布（真實 n_patch）", lengths, ""),
            (f"最壞情況（候選滿 {CANDIDATE_SIZE}）", [CANDIDATE_SIZE],
             f"|M| = {MEMORY_CAPACITY} 的預算必須撐得住這一欄，不是上面那欄。")]:
        s1, s2 = stats(_v1, lens), stats(_v2, lens)
        L += [f"## {title}", ""]
        if note:
            L += [note, ""]
        L += ["| 量法 | v1 mean | v1 max | v2 mean | v2 max | max 降幅 |",
              "|---|---|---|---|---|---|",
              f"| payload | {kb(s1['payload_mean'])} | {kb(s1['payload_max'])} | "
              f"{kb(s2['payload_mean'])} | {kb(s2['payload_max'])} | "
              f"−{(1 - s2['payload_max'] / s1['payload_max']) * 100:.1f}% |",
              f"| serialized | {kb(s1['ser_mean'])} | {kb(s1['ser_max'])} | "
              f"{kb(s2['ser_mean'])} | {kb(s2['ser_max'])} | "
              f"−{(1 - s2['ser_max'] / s1['ser_max']) * 100:.1f}% |", "",
              f"|M| = {MEMORY_CAPACITY} 滿載（用 max，即最壞情況）：",
              "",
              f"* payload：v1 {mb(s1['payload_max'] * MEMORY_CAPACITY)} → "
              f"v2 {mb(s2['payload_max'] * MEMORY_CAPACITY)}",
              f"* serialized：v1 {mb(s1['ser_max'] * MEMORY_CAPACITY)} → "
              f"v2 {mb(s2['ser_max'] * MEMORY_CAPACITY)}", ""]

    e2 = _v2(CANDIDATE_SIZE)
    L += ["## v2 的逐欄拆解（候選滿 256）", "",
          "| 欄位 | dtype × 長度 | payload |", "|---|---|---|"]
    for name, v in e2.items():
        if isinstance(v, torch.Tensor):
            desc, b = f"{v.dtype} × {v.nelement()}", v.element_size() * v.nelement()
        elif isinstance(v, str):
            desc, b = f"str（{len(v)} 字）", len(v.encode())
        else:
            desc, b = type(v).__name__, 8
        L.append(f"| `{name}` | {desc} | {b:,} B |")
    wrap = serialized_bytes(_v2_entry(CANDIDATE_SIZE)) - serialized_bytes(e2)
    L += ["", f"合計 {payload_bytes(e2):,} B。"
          f"（真正的 `SelectionMemoryEntry` 物件序列化後比欄位字典多 {wrap:,} B，"
          "那是 dataclass 包裝的固定開銷，v1 也有，故上表兩邊都以欄位字典量。）"
          "`cand_idx` 是 int64，現在是最大的一欄 —— 降到 int32 可以再省一半，"
          "但那會動到 `index_select` 的呼叫端，本輪不做。", ""]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n")
    print(f"→ {OUT}")
    for tag, make in (("v1", _v1), ("v2", _v2)):
        e = make(CANDIDATE_SIZE)
        print(f"  {tag}  payload {payload_bytes(e):,} B   "
              f"serialized {serialized_bytes(e):,} B   "
              f"×{MEMORY_CAPACITY} = {mb(serialized_bytes(e) * MEMORY_CAPACITY)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
