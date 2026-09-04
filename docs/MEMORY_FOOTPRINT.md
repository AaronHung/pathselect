# Selection Memory 佔用量（schema v1 vs v2）

DR-048 B5。產生：`python scripts/report_memory_footprint.py`。

v2 相對 v1 的改動：拿掉 `e_t`（[512] float32）與 `B_tilde_t`，`slide_id: str` 改成 `sample_key: int64`。兩欄在整個 repo 都**只寫不讀**，移除不改變任何訓練行為 （見 `selector/memory.py` 模組說明與 DR-048 卡的位元等價驗證）。

## 量法

* **payload** = tensor 的 `element_size × nelement` + 字串的 UTF-8 位元組 + 每個純量 8 bytes。不含 CPython 物件表頭。
* **serialized** = `torch.save` 到 BytesIO 的長度（含 pickle／zip 容器開銷）。

候選長度取自真實產物 `outputs/exp2/main/per_slide/*.json` 的 `n_patch`：共 45,520 筆，長度介於 224–256，其中 **99.5% 撞到 256 的上限**。

⚠️ 因此下面兩張表幾乎相同 —— 絕大多數 slide 的候選都是滿的。保留兩張是為了讓「實測」與「預算上限」分開可讀，不是筆誤。

## 實測分布（真實 n_patch）

| 量法 | v1 mean | v1 max | v2 mean | v2 max | max 降幅 |
|---|---|---|---|---|---|
| payload | 6,250 B（6.10 KB） | 6,253 B（6.11 KB） | 4,142 B（4.05 KB） | 4,145 B（4.05 KB） | −33.7% |
| serialized | 8,726 B（8.52 KB） | 8,729 B（8.52 KB） | 6,362 B（6.21 KB） | 6,365 B（6.22 KB） | −27.1% |

|M| = 512 滿載（用 max，即最壞情況）：

* payload：v1 3.05 MB → v2 2.02 MB
* serialized：v1 4.26 MB → v2 3.11 MB

## 最壞情況（候選滿 256）

|M| = 512 的預算必須撐得住這一欄，不是上面那欄。

| 量法 | v1 mean | v1 max | v2 mean | v2 max | max 降幅 |
|---|---|---|---|---|---|
| payload | 6,253 B（6.11 KB） | 6,253 B（6.11 KB） | 4,145 B（4.05 KB） | 4,145 B（4.05 KB） | −33.7% |
| serialized | 8,729 B（8.52 KB） | 8,729 B（8.52 KB） | 6,365 B（6.22 KB） | 6,365 B（6.22 KB） | −27.1% |

|M| = 512 滿載（用 max，即最壞情況）：

* payload：v1 3.05 MB → v2 2.02 MB
* serialized：v1 4.26 MB → v2 3.11 MB

## v2 的逐欄拆解（候選滿 256）

| 欄位 | dtype × 長度 | payload |
|---|---|---|
| `tau` | str（9 字） | 9 B |
| `sample_key` | int | 8 B |
| `r_old` | torch.float32 × 8 | 32 B |
| `cand_idx` | torch.int64 × 256 | 2,048 B |
| `s_old` | torch.float32 × 256 | 1,024 B |
| `u_old` | torch.float32 × 256 | 1,024 B |

合計 4,145 B。（真正的 `SelectionMemoryEntry` 物件序列化後比欄位字典多 64 B，那是 dataclass 包裝的固定開銷，v1 也有，故上表兩邊都以欄位字典量。）`cand_idx` 是 int64，現在是最大的一欄 —— 降到 int32 可以再省一半，但那會動到 `index_select` 的呼叫端，本輪不做。

