# 重現檢查的相容性修補紀錄

對方程式針對 torch 1.11 / transformers 4.46.1 撰寫；本機是 Apple Silicon，
torch 1.11 沒有對應 wheel，只能用新版。以下修補**僅處理 API 變更，
不改變任何數值行為**；凡是會影響數值的差異一律停下回報，不自行修改。

| # | 檔案:行 | 原因 | 改動 |
|---|---|---|---|
| 1 | `manager/manager.py` L116 與 L122（兩處，寫法不同：`self.cfg['lrs_verbose']` 與 `cfg_lr_scheduler.verbose`）| `torch.optim.lr_scheduler.ReduceLROnPlateau` 在新版 PyTorch 移除了 `verbose` 參數（對方針對 torch 1.11，本機 2.14） | 移除 `verbose=self.cfg['lrs_verbose']` 這個 kwarg。它只控制是否列印學習率調整訊息，**對數值零影響**。原檔備份於 `manager/manager.py.orig` |
| 2 | `manager/manager.py` L532（訓練迴圈內） | `torch.cuda.set_device()` 被**無條件**呼叫，本機無 CUDA 直接拋 `AttributeError`。（他們的 `utils/tools.py:92` 其實有 CPU fallback，但這一行繞過了它） | 加上 `if torch.cuda.is_available():` 守衛。在有 CUDA 的機器上行為完全不變；無 CUDA 時原本就跑不了，**對數值零影響** |
| 3 | `utils/evaluator_clf.py` 等 | `np.long` / `np.float` / `np.int` / `np.bool` 這些別名在 NumPy 1.24 移除（對方釘 numpy 1.21.2；本機 venv 已降到 1.26.4 但仍高於 1.24） | 換成 `np.int64` / `np.float64` / `np.bool_`。**別名與本尊語意相同**，對數值零影響 |
| 4 | `class_ensemble/class_ensemble_reverse.json`（**新增副本，原檔未動**） | `utils/tools.py::get_current_ensemble_classes` 依 JSON 的**鍵序**累積類別、遇到當前 dataset 才停，**完全不看 config 的 `dataset_names`** —— 任務順序實際上寫死在這個資料檔裡。光改 config 跑 reverse，第一個任務就會累積到全部 8 類（`tunable_v` 只有 2 類 → shape mismatch） | 複製一份並把鍵序改成 reverse（esca→rcc→brca→lung）；**每個 dataset 的類別內容逐字不變**（已驗證）。這是設定層而非模型層的改動 |

## 兩個平台實際需要的修補不同

| # | Mac（CPU，torch 2.14、venv） | Pod（RTX 4090，torch 2.4.1） |
|---|---|---|
| 1 相依缺失 | 需要（wandb / timm / torchvision） | 需要（+ scikit-learn / pandas / matplotlib） |
| 2 transformers 版本 | 需要（base 是 5.5.3） | 需要（釘 4.46.1） |
| 3 `ReduceLROnPlateau(verbose=)` | **需要**（torch 2.14 已移除） | **不需要**（torch 2.4.1 仍支援） |
| 4 `np.long` 等別名 | 需要 | 需要 |
| 5 `torch.cuda.set_device` 守衛 | **需要**（無 CUDA） | **不需要**（有 CUDA） |
| — reverse 鍵序的 class_ensemble | 需要 | 需要 |

**GPU 路徑只動了 #4 與 class_ensemble 鍵序**，比 Mac 少兩道，更貼近其原始碼 ——
`docs/SOTA_TABLE.md` 的重現數字用的是 GPU 路徑。

⚠️ **`numpy<2` 必須最後安裝**：`pip install scikit-learn` 會把 numpy 2.x 裝回來，
導致 `np.Inf` 再度失敗。這個順序陷阱在 pod 上實際踩過一次。
