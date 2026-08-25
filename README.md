# pathselect

WSI（whole-slide image）**patch 選取**的持續學習方法。在固定的 patch 預算下，
逐步選出最能支持診斷的少數 patch，並在任務序列中保住既有任務的選取行為。

分類頭是**凍結**的 —— 選中的 patch 經分數加權池化後直接與 CONCH 的類別文字嵌入比對，
沒有任何可訓練的診斷頭。因此「遺忘」可以 100% 歸因於**選取漂移**，而不是分類器漂移。

---

## 給第一次看這個 repo 的人

1. **先讀 [`docs/ledger/INDEX.md`](docs/ledger/INDEX.md)** —— 決策現況，一頁看完。
   每張 `DR-*.md` 記錄一次裁定：脈絡、考慮過的選項、最後的判斷、支撐的證據。
2. **再讀 [`docs/CONSTITUTION.md`](docs/CONSTITUTION.md)** —— 研究紀律。
   三級 win count 規則、不報 p 值、負面結果照實報、機制生效性實測是實驗啟動的門檻等。
   搭配 [`docs/CLAIMS.md`](docs/CLAIMS.md)（**不可宣稱**的清單）一起看。
3. **實驗結果在 [`outputs/`](outputs/)** —— 各子目錄的對應見
   [`docs/CODEBASE_MAP.md`](docs/CODEBASE_MAP.md)。
   每個實驗都有逐 slide 存檔，報告由腳本重算產生，不手抄數字。
4. **[`outputs/exp2/seqft/ckpt/`](outputs/exp2/seqft/ckpt/)** 保留了 SeqFT 各階段的
   模型狀態（24 個檔、72 MB），供結果重算之用 —— 不是最終模型權重。
5. **[`reference/`](reference/) 為唯讀存檔**，附 [`SHA256SUMS.txt`](reference/SHA256SUMS.txt)。
   前一版的產物，只用於回歸比對，不得修改。

如果只想看一個檔案感受這個 repo 在做什麼，建議
[`docs/CLAIMS.md`](docs/CLAIMS.md) —— 那裡記錄的是**已經被證明不能說的話**。

---

## 跑起來

```bash
pip install torch torchvision h5py pyyaml numpy pandas
python -m pytest tests/ -q           # 全部應為綠（目前 1049 條）
```

CONCH 的推論程式碼 vendored 在 `third_party/`（來源與版本見該目錄）。

### ⚠️ 路徑需自行修改

`configs/pathselect.yaml` 內的路徑是**作者機器上的絕對路徑**，clone 之後必須改：

```yaml
conch_ckpt_path: /Users/aaron/research/01_navipath/checkpoints/conch/pytorch_model.bin
dataset_root_dir: /Users/aaron/research/can_dataset
```

`scripts/pipeline_*.sh` 開頭的 `cd /Users/aaron/research/02_pathselect` 同理。

CONCH 權重需自行向 [Mahmood Lab](https://github.com/mahmoodlab/CONCH) 申請，本 repo 不含。
特徵檔（TCGA patch embeddings）亦不含。

---

## 這個 repo 的一個特點

方法本身之外，這裡有一套**決策治理系統**：每次裁定、每個負面結果、每次方法學上的
失誤都寫成紀錄。包括：

- **已撤回的主張**（例如記憶體效率從 8× 撤回、依配對證據改為 4×）
- **「規格寫了、架構圖畫了，但從未生效」的元件**共四例，分屬架構 / 環境 / 實作 /
  接線四層失效模式（`CLAIMS.md` C-26）
- **mutation check 紀律**：每條新斷言都必須用一個人造的違規證明它抓得到；
  這條規則也適用於科學主張（用擾動或替換證明關掉某機制會改變輸出）

負面結果與撤回都留在紀錄裡，不刪除。
