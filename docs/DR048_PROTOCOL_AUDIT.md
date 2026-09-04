# DR-048 A1 —— 協定稽核：本機 vs QPMIL-VL 官方

官方事實取自 **github.com/can-can-ya/QPMIL-VL @ main**（2026-09-04 取得）：`configs/main.yaml`、`scripts/main.sh`、`dataset/data_utils.py`、`dataset/WSI.py`。

⚠️ 本稽核**不引入任何 QPMIL 程式碼**，只記錄其設定值供比對；`tests/test_no_banned_deps.py` 仍然有效。

## 差異表

| 面向 | QPMIL-VL 官方 | 本 repo | 判定 | 備註 |
|---|---|---|---|---|
| 切分檔路徑 | `/{}/datasplit/fold_{}.npz` | `/{}/datasplit/fold_{}.npz` | ✅ 一致 |  |
| 特徵路徑 | `/{}/feats-l1-s256_{}/pt_files` | `/{}/feats-l1-s256_{}/pt_files` | ✅ 一致 |  |
| 標籤表路徑 | `/{}/table/{}_path_subtype_x10_processed.csv` | `/{}/table/{}_path_subtype_x10_processed.csv` | ✅ 一致 |  |
| 特徵格式 | pt | pt | ✅ 一致 |  |
| 骨幹 | CONCH | CONCH | ✅ 一致 |  |
| 資料集 fold 數 | `total_fold: 10` | esca=10、rcc=10、brca=10、lung=10 | ✅ 一致 |  |
| 切分鍵 | train/val/test（病人層級） | test_patients、train_patients、val_patients | ✅ 一致 | `read_datasplit_npz` 與官方**逐字相同** |
| 表格→標籤 | `retrieve_from_table_clf` | 同名同實作 | ✅ 一致 | `data/table_utils.py` 與官方**逐字相同** → 子型別對映由構造一致 |
| **任務順序** | forward：lung → brca → rcc → esca | 主線 `reverse`：esca → rcc → brca → lung | ❌ **不同** | 本 repo 的 `main` order 才等於官方 forward |
| 8-way 標籤索引 | lung=0/1、brca=2/3、rcc=4/5、esca=6/7 | esca=0/1、rcc=2/3、brca=4/5、lung=6/7 | ❌ **不同** | 純置換：8-way ACC 不受影響，但**逐類別／混淆矩陣的比較必須先對映** |
| 跑幾折 / 平均對象 | 10 折（`main.sh` SEED 1..10），對 fold 平均 | `configs` 寫死 `fold: 1`；主線對 **model seed** 平均 | ❌ **不同** | 本 repo 目前只跑 fold 1；`run_exp2.py` 尚無 `--fold` |
| epochs | [12, 12, 12, 12] | 5 | ❌ **不同** | 屬**我們方法的訓練設定**，非資料協定；不影響切分可比性 |

## 切分檔的內部一致性（本機自檢）

| task | 病人總數 | 十折 test 聯集 | 折內重疊 |
|---|---|---|---|
| tcga_esca | 148 | 148 | ✅ 無 |
| tcga_rcc | 738 | 738 | ✅ 無 |
| tcga_brca | 891 | 891 | ✅ 無 |
| tcga_lung | 868 | 868 | ✅ 無 |

## fold 1 的成員數（本機）

| task | train | val | test |
|---|---|---|---|
| tcga_esca | 118 | 15 | 15 |
| tcga_rcc | 590 | 74 | 74 |
| tcga_brca | 712 | 89 | 90 |
| tcga_lung | 694 | 87 | 87 |

## 結論

**路徑、切分檔格式、切分鍵、表格→標籤對映全部一致**（後兩者的載入程式碼與官方逐字相同，故子型別對映由構造相同）。**每折成員切片**無法與官方 repo 對照——他們的 repo 不含切分檔，切分檔就在本機這份作者公開的 benchmark 裡；本檔改以內部一致性（折內無重疊、十折 test 聯集 = 病人總數）驗證。

**兩處實質不同，且都不是換切分檔的 adapter 能對齊的：**

1. **任務順序相反。** 官方發表協定是 forward（lung→brca→rcc→esca），repo 內唯一的評測範本也是 `forward-order.xlsx`。本 repo 的主線是 `reverse`。要引用其發表數字當 baseline，我們必須跑 **`--order main`**（= 官方 forward）。
2. **平均對象不同。** 官方對 **10 個 fold** 平均；本 repo 主線對 **5 個 model seed @ fold 1** 平均。兩者不是同一種變異來源，不可直接並列。

8-way 標籤索引雖然相反，但那是純置換，8-way ACC 不受影響 —— 只有逐類別或混淆矩陣的比較需要先對映。

產生：`python scripts/audit_qpmil_protocol.py`（官方事實 2026-09-04 取得）。

