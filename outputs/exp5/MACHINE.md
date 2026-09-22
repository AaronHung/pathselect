# outputs/exp5 的執行環境（M = 64 備份實驗）

| 項目 | 值 |
|---|---|
| 機器 | RunPod，GPU **RTX 5090**（32,607 MiB；**本實驗不使用 GPU**），CPU **AMD EPYC 9J14 96-Core** |
| 主機可見核心 | 384 |
| 容器 CPU 配額 | **45.9 核**（cgroup v2：4,590,000 / 100,000） |
| RAM | 755 GB |
| 儲存 | `/workspace` 為網路檔案系統（`mfs#eur-is-1.runpod.net`）；特徵庫 20 GB / 3,282 檔 |
| Python / torch | **3.11.13 / 2.8.0+cu128**（venv `/workspace/venvs/ps5`），numpy 2.4.6 |
| 每 run 執行緒 | **OMP/MKL/OpenBLAS/torch = 8** |
| 平行數 | 10 路（工作佇列，一個 run 結束就補下一個） |
| branch / commit | `exp/m64-5090-backup` / `081bdcd165bdc02180c7544962e09ef80d80b98f` |
| repo 路徑（pod） | `/workspace/pathselect-exp5` |
| 日期 | 2026-09-23 |

## 為什麼執行緒數固定在 8

torch 的規約順序會隨執行緒數改變，換了就無法與 exp3 逐位元比對。exp3 用的是 8，
本批沿用。平行數是唯一可調的旋鈕。

## 與 exp3 的環境差異（影響協定核對的解讀）

| 項目 | exp3 | 本批 |
|---|---|---|
| Python | 3.11.10 | 3.11.13 |
| torch | 2.8.0+cu128 | 2.8.0+cu128（相同） |
| CPU | AMD EPYC 7702（4090 pod）／5090 pod | AMD EPYC 9J14 |

⚠️ **跨機器的數字不得相減。** outputs/exp3 與 outputs/exp4 是在別台機器產生的；
本批與它們的比較只能當**參考**，不得作為配對統計。本批內部的所有配對
（不同 M、不同 arm、不同順序）都在這一台、這一批完成。

## H200 那套

主要數據由同學在 H200 上跑。**兩套數字絕不混用**，選用規則見 `docs/ledger/DR-057.md`。
