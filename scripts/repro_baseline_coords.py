"""DR-048：外部基準方法**重現檢查**的座標。唯讀資料模組，不寫任何檔案。

PI 指定的重現檢查（Prompt 10 步驟 5）：把對方公開的實作 clone 到 repo 之外，
指向本機的同一份資料集，先 smoke 再決定要不要跑十折。

⚠️ 本檔含被禁的識別字 —— **必須指名才能寫出 clone URL 與產物目錄名**。
   於 `tests/test_no_banned_deps.py::EXEMPT` 有窄例外，相對地它
   **不得寫檔**（`test_only_read_only_scripts_may_be_exempt` 會靜態檢查）。
   實際的 clone／執行／寫檔由 `scripts/repro_external.py` 負責，那一支不含禁用字。

⚠️ **clone 目標在 repo 之外**（`~/research/ext/`），不進 `selector/`、`sota/`，
   也不進版控 —— 我們只讀它的執行結果，不把它的程式碼帶進本專案。
"""
from __future__ import annotations

#: 對方公開實作的 git 位置與 clone 後的目錄名（放在 repo 之外）
CLONE_URL = "https://github.com/can-can-ya/QPMIL-VL.git"
CLONE_DIRNAME = "QPMIL-VL"

#: 進入點與設定檔（相對 clone 目錄）
ENTRY = "main.py"
CONFIG_REL = "configs/main.yaml"

#: 產物子目錄名（呼叫端自行接在實驗根目錄底下）與不可行紀錄的檔名
OUT_SUBDIR = "repro_qpmil"
INFEASIBLE_NAME = "QPMIL_REPRO_INFEASIBLE.txt"

#: 需要覆寫成本機值的設定鍵。值由呼叫端提供（本檔不知道本機路徑）。
PATCH_KEYS = {
    "dataset_root_dir": "資料集根目錄",
    "conch_ckpt_path": "CONCH 權重",
}

#: ⚠️ 對方的 `utils/tools.py::get_current_ensemble_classes` 是依
#: `class_ensemble.json` 的**鍵序**累積類別、遇到當前 dataset 才停 ——
#: 它完全不看 config 的 `dataset_names`。因此任務順序其實**寫死在那個資料檔裡**，
#: 光改 config 的 `dataset_names` 會讓第一個任務就累積到全部 8 類
#: （其可訓練的任務向量只有 2 類 → shape mismatch）。
#: 跑 reverse 必須另備一份鍵序重排的副本。
CLASS_ENSEMBLE_REVERSE = "class_ensemble/class_ensemble_reverse.json"

#: reverse 順序（其 Tab. 2）的任務序與對應的標籤位移／子型別數。
REVERSE_DATASET_NAMES = ["tcga_esca", "tcga_rcc", "tcga_brca", "tcga_lung"]
REVERSE_LABEL_SHIFT = [0, 2, 4, 6]
REVERSE_SUBTYPE_NUM = [2, 2, 2, 2]

#: smoke 用的 epoch 數（四個任務各 1）。正式跑用對方預設，不覆寫。
SMOKE_EPOCHS = [1, 1, 1, 1]

#: 單折上限（分鐘）。smoke 推估超過就不跑十折 —— PI 指定 60。
FOLD_BUDGET_MIN = 60
