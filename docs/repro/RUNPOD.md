# RunPod 重現環境（DR-048 / QPMIL-VL 重現檢查）

## 現況

| | |
|---|---|
| pod id | `jnd934wacl9pk6`（名稱 `lexical_tan_frog`） |
| GPU | 1× RTX 4090（24 GB） |
| 規格 | 64 vCPU、62 GB RAM、容器磁碟 **20 GB** |
| 費用 | **$0.740 / 小時** |
| 映像 | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` |
| SSH | `ssh -p 30556 root@157.157.221.29` |

⚠️ **容器磁碟只有 20 GB，不符「≥60 GB」的規格**，但 `/workspace` 是 RunPod
**網路磁碟區**（MooseFS，624 TB 可用且跨 pod 持久），資料與結果全部放在那裡，
實質需求已滿足。重建新 pod 時只要掛同一個網路磁碟區，資料不必重傳。

## 工作目錄

```
/workspace/datasets/can_dataset          資料集（20 GB，**早已存在**，本輪未重傳）
/workspace/src/navipath/checkpoints/conch/pytorch_model.bin   CONCH 權重
/workspace/repro_qpmil/QPMIL-VL          對方 repo（clone，commit 3a7a769）
/workspace/repro_qpmil/cfg/              改寫後的設定（reverse_full / reverse_smoke）
/workspace/repro_qpmil/results/          執行結果
/workspace/repro_qpmil/logs/             逐折 log 與 .done 標記
/workspace/repro_qpmil/run_folds.sh      十折 runner（同卡並行 3 折、可續跑）
```

⚠️ **pod 上沒有、也不得有 pathselect 的任何程式碼**（`selector/`、`sota/`）。
只有對方 repo、shim、資料、結果。

## 從零重建（約 30 分鐘，資料已在網路磁碟區則約 10 分鐘）

```bash
# 1. 開 pod：4090、掛同一個網路磁碟區、映像 runpod/pytorch:2.4.x-cuda12.4
runpodctl get pod                     # 取得新的 SSH host/port

# 2. 連進去
ssh -p <PORT> root@<HOST>

# 3. 系統套件
apt-get update -qq && apt-get install -y -qq tmux rsync

# 4. 對方 repo
mkdir -p /workspace/repro_qpmil && cd /workspace/repro_qpmil
git clone --depth 1 https://github.com/can-can-ya/QPMIL-VL.git
cd QPMIL-VL

# 5. Python 相依（版本照其 requirements.txt 釘住；numpy 必須 <2）
pip install 'transformers==4.46.1' 'tokenizers==0.20.1' 'timm==1.0.11' \
            wandb openpyxl h5py seaborn scikit-learn pandas matplotlib
pip install 'numpy<2'                 # ⚠️ 必須放最後：sklearn 會把 numpy 2.x 裝回來

# 6. 套用仍需要的 shim（見 SHIMS.md；CUDA 路徑下只需要 #4）
#    np.long / np.float / np.int / np.bool → np.int64 / np.float64 / np.bool_
python - <<'PY'
import pathlib, re
R={r"np\.long\b":"np.int64", r"np\.float\b(?!\d|_)":"np.float64",
   r"np\.int\b(?!\d|p|e|8|_)":"np.int64", r"np\.bool\b(?!_|8)":"np.bool_"}
for f in pathlib.Path(".").rglob("*.py"):
    t=f.read_text(errors="ignore"); o=t
    for a,b in R.items(): t=re.sub(a,b,t)
    if t!=o: f.write_text(t); print("patched",f)
PY

# 7. reverse 鍵序的 class_ensemble（**跑 reverse 必要**，見 SHIMS.md #4 的說明）
python - <<'PY'
import json, pathlib
p=pathlib.Path("class_ensemble/class_ensemble.json"); d=json.load(open(p))
cn=d['0']['classnames']; rev=["tcga_esca","tcga_rcc","tcga_brca","tcga_lung"]
d['0']['classnames']={k:cn[k] for k in rev}
pathlib.Path("class_ensemble/class_ensemble_reverse.json").write_text(json.dumps(d,indent=1))
PY

# 8. 設定檔：改四個路徑 + reverse 三個欄位（見 cfg/reverse_full.yaml）
#    dataset_root_dir / conch_ckpt_path / result_dir / class_ensemble_path
#    dataset_names / dataset_label_shift / dataset_subtype_num

# 9. 在 tmux 裡跑（關蓋、斷線都不影響）
tmux new-session -d -s repro 'bash /workspace/repro_qpmil/run_folds.sh 2>&1 | tee /workspace/repro_qpmil/run_folds.log'
```

## 實測數字

| | Mac（CPU，M 系列） | Pod（RTX 4090） |
|---|---|---|
| smoke：1 epoch × 4 任務 | **46 分** | **3 分 14 秒** |
| 單折（其預設 12 epoch）推估 | ~9.2 小時 | **~39 分** |
| 加速比 | 1× | **約 15×** |

同卡並行 3 折時 GPU 使用率約 43%、顯存 4.8 GB / 24 GB —— 並行 3 折仍有餘裕。

## 本輪實際執行（2026-09-07）

十折全部完成，逐折耗時（3 折並行）：

| fold | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| 秒 | 1125 | 1248 | 1077 | 1392 | 1352 | 1048 | 894 | 1189 | 1125 | — |

（fold 10 的計時因其跨折彙總步驟先行崩潰而未寫出，但**它自己的指標已完整產出**。）
八折並行批次牆鐘 **61 分鐘**；fold 7 事後單獨重跑 894 秒。

### 兩個中途故障

1. **fold 7**：pod 既有資料集的 `tcga_brca/datasplit/fold_7.npz` 是 **0 bytes**
   （Mac 端 19,601 bytes）。補傳該檔後重跑成功。已全面檢查 40 個切分檔，
   其餘 39 個正常。
2. **fold 10**：其程式在最後一折會**跨折彙總**，預期所有折共用同一個 `--time`
   目錄；本輪每折給了不同的 `--time`（`fold1`…`fold10`），因此它找不到前面幾折而
   崩潰。**崩潰發生在該折自身訓練與評估完成之後**，指標檔完整，不需重跑。
   ⚠️ 日後要用它自身的跨折彙總，必須讓十折共用同一個 `--time`。
