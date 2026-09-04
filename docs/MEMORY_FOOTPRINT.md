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

## 特徵記錄大小（原始 patch 特徵，供對照）

記憶庫存的是 **key + index**，不是特徵；下面這一節是「如果改成存特徵」的量體，用來說明為什麼不存。

N = 每張 slide 的 patch 數，特徵記錄 = **N × 512 × 4 bytes**（CONCH float32）。N 由特徵檔大小反推：`(size − 747) / (512×4)`，整除性逐檔驗證。

| task | slide 數 | N 平均 | N 中位數 | 記錄平均 | 記錄中位數 |
|---|---|---|---|---|---|
| `tcga_esca` | 158 | 3,261 | 3,144 | 6.37 MB | 6.14 MB |
| `tcga_rcc` | 937 | 3,540 | 3,606 | 6.91 MB | 7.04 MB |
| `tcga_brca` | 1133 | 2,784 | 2,683 | 5.44 MB | 5.24 MB |
| `tcga_lung` | 1054 | 3,094 | 2,900 | 6.04 MB | 5.66 MB |
| **四 task 合計** | **3282** | **3,123** | **3,060** | **6.10 MB** | **5.98 MB** |

N 的範圍 35–16,848。

### 與記憶庫並列

* **一張 slide 的特徵記錄**（平均）：6.10 MB　／　中位數 5.98 MB
* **整個記憶庫 |M| = 512**（schema v2，最壞情況）：**3.11 MB**

換句話說，**整個記憶庫（512 筆）約等於 51% 張 slide 的特徵**（3.11 MB vs 一張 6.10 MB）—— 連一張都不到。這是「不存 feature，只存 key + index」這個設計的量化理由。

對照另一個尺度：四個 task 全部 3,282 張 slide 的特徵合計 **19.5 GB**；記憶庫是其中的 0.02%。

