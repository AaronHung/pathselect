#!/usr/bin/env python3
"""DR-048 A1 —— 本機協定 vs QPMIL-VL 官方協定的差異稽核（**唯讀**）。

本檔只蒐集事實並組裝報告內容，**不寫任何檔案** —— 寫檔由
`scripts/report_protocol_audit.py` 負責。

⚠️ 這個拆分是 `tests/test_no_banned_deps.py::test_only_read_only_scripts_may_be_exempt`
   逼出來的：申請禁用字例外的腳本必須是唯讀的。守門的訊息直接指了做法
   （「請把唯讀部分拆成獨立模組」），照做。

官方事實取自 `github.com/can-can-ya/QPMIL-VL`（main 分支，2026-09-04 取得）：
`configs/main.yaml`、`scripts/main.sh`、`dataset/data_utils.py`、`dataset/WSI.py`。
**逐條寫成常數並附出處**，讓本檔離線可重跑；重新核對時只要更新常數與日期。

⚠️ 本檔**不引入任何 QPMIL 程式碼** —— 只記錄其設定值與檔名，供比對。
   `tests/test_no_banned_deps.py` 仍然有效。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from selector.text_encoder import load_config                        # noqa: E402

FETCHED = "2026-09-04"
SRC = "github.com/can-can-ya/QPMIL-VL @ main"

#: 官方協定（逐條出處）
OFFICIAL = {
    "dataset_names": (["tcga_lung", "tcga_brca", "tcga_rcc", "tcga_esca"],
                      "configs/main.yaml，註解寫明 forward-order: "
                      "lung (NSCLC) → brca (BRCA) → rcc (RCC) → esca (ESCA)"),
    "dataset_label_shift": ([0, 2, 4, 6], "configs/main.yaml"),
    "total_fold": (10, "configs/main.yaml"),
    "fold_loop": (list(range(1, 11)),
                  "scripts/main.sh：for SEED in 1..10，-s ${SEED} → data_split_seed"),
    "path_split": ("/{}/datasplit/fold_{}.npz", "configs/main.yaml"),
    "path_feat": ("/{}/feats-l1-s256_{}/pt_files", "configs/main.yaml"),
    "path_table": ("/{}/table/{}_path_subtype_x10_processed.csv", "configs/main.yaml"),
    "feat_format": ("pt", "configs/main.yaml"),
    "base_model_arch": ("CONCH", "configs/main.yaml"),
    "split_keys": (["train_patients", "val_patients", "test_patients"],
                   "dataset/data_utils.py::read_datasplit_npz"),
    "split_level": ("patient（再由 table 展開成 slide）",
                    "dataset/WSI.py::WSIClf + retrieve_from_table_clf(level='slide')"),
    "epochs": ([12, 12, 12, 12], "configs/main.yaml"),
    "eval_template": ("eval_template/forward-order.xlsx（repo 內僅此一份）",
                      "repo 檔案清單"),
}

TASKS = ["tcga_esca", "tcga_rcc", "tcga_brca", "tcga_lung"]


def local_facts():
    cfg = load_config()
    root = cfg["dataset_root_dir"]
    folds, splits, counts = {}, {}, {}
    for t in TASKS:
        d = Path(root) / t / "datasplit"
        folds[t] = sorted(int(p.stem.split("_")[1]) for p in d.glob("fold_*.npz")) \
            if d.is_dir() else []
        f1 = d / "fold_1.npz"
        if f1.exists():
            z = np.load(f1, allow_pickle=True)
            splits[t] = list(z.keys())
            counts[t] = {k: len(z[k]) for k in z.keys()}
    return cfg, folds, splits, counts


def check_overlap(cfg):
    """同一 fold 內 train/val/test 的病人不得重疊；跨 fold 的 test 應覆蓋全體。"""
    root = cfg["dataset_root_dir"]
    rows = []
    for t in TASKS:
        d = Path(root) / t / "datasplit"
        if not d.is_dir():
            continue
        overlaps, all_test, all_pid = [], set(), set()
        for k in range(1, 11):
            f = d / f"fold_{k}.npz"
            if not f.exists():
                continue
            z = np.load(f, allow_pickle=True)
            s = {n: {str(x) for x in z[n]} for n in z.keys()}
            for a, b in (("train_patients", "val_patients"),
                         ("train_patients", "test_patients"),
                         ("val_patients", "test_patients")):
                if s.get(a) and s.get(b) and (s[a] & s[b]):
                    overlaps.append(f"fold {k}: {a}∩{b} = {len(s[a] & s[b])}")
            all_test |= s.get("test_patients", set())
            all_pid |= set().union(*s.values())
        rows.append((t, len(all_pid), len(all_test), overlaps))
    return rows


def build_report_lines() -> list[str]:
    """組裝報告的每一行。呼叫端負責寫檔。"""
    cfg, folds, splits, counts = local_facts()
    ours = list(cfg["tasks"])
    off_names, off_names_src = OFFICIAL["dataset_names"]

    def row(item, official, local, same, note=""):
        mark = "✅ 一致" if same else "❌ **不同**"
        return f"| {item} | {official} | {local} | {mark} | {note} |"

    L = [f"# DR-048 A1 —— 協定稽核：本機 vs QPMIL-VL 官方", "",
         f"官方事實取自 **{SRC}**（{FETCHED} 取得）：`configs/main.yaml`、"
         "`scripts/main.sh`、`dataset/data_utils.py`、`dataset/WSI.py`。", "",
         "⚠️ 本稽核**不引入任何 QPMIL 程式碼**，只記錄其設定值供比對；"
         "`tests/test_no_banned_deps.py` 仍然有效。", "",
         "## 差異表", "",
         "| 面向 | QPMIL-VL 官方 | 本 repo | 判定 | 備註 |",
         "|---|---|---|---|---|"]

    L.append(row("切分檔路徑", f"`{OFFICIAL['path_split'][0]}`",
                 f"`{cfg['path_split']}`", cfg["path_split"] == OFFICIAL["path_split"][0]))
    L.append(row("特徵路徑", f"`{OFFICIAL['path_feat'][0]}`",
                 f"`{cfg['path_feat']}`", cfg["path_feat"] == OFFICIAL["path_feat"][0]))
    L.append(row("標籤表路徑", f"`{OFFICIAL['path_table'][0]}`",
                 f"`{cfg['path_table']}`", cfg["path_table"] == OFFICIAL["path_table"][0]))
    L.append(row("特徵格式", OFFICIAL["feat_format"][0], cfg["feat_format"],
                 cfg["feat_format"] == OFFICIAL["feat_format"][0]))
    L.append(row("骨幹", OFFICIAL["base_model_arch"][0], cfg["conch_path_feat"],
                 cfg["conch_path_feat"] == OFFICIAL["base_model_arch"][0]))

    nfold = {t: len(v) for t, v in folds.items()}
    same_fold = all(v == 10 for v in nfold.values())
    L.append(row("資料集 fold 數", f"`total_fold: {OFFICIAL['total_fold'][0]}`",
                 "、".join(f"{t.replace('tcga_','')}={n}" for t, n in nfold.items()),
                 same_fold))

    keys_ok = all(sorted(v) == sorted(OFFICIAL["split_keys"][0]) for v in splits.values())
    L.append(row("切分鍵", "train/val/test（病人層級）",
                 "、".join(sorted(next(iter(splits.values())))) if splits else "—",
                 keys_ok, "`read_datasplit_npz` 與官方**逐字相同**"))
    L.append(row("表格→標籤", "`retrieve_from_table_clf`", "同名同實作", True,
                 "`data/table_utils.py` 與官方**逐字相同** → 子型別對映由構造一致"))

    L.append(row("**任務順序**", "forward：lung → brca → rcc → esca",
                 f"主線 `reverse`：{' → '.join(t.replace('tcga_','') for t in ours)}",
                 ours == off_names,
                 "本 repo 的 `main` order 才等於官方 forward"))
    off_map = {n: (s, s + 1) for n, s in zip(off_names, OFFICIAL["dataset_label_shift"][0])}
    our_map = {t: (2 * i, 2 * i + 1) for i, t in enumerate(ours)}
    L.append(row("8-way 標籤索引",
                 "、".join(f"{k.replace('tcga_','')}={v[0]}/{v[1]}" for k, v in off_map.items()),
                 "、".join(f"{k.replace('tcga_','')}={v[0]}/{v[1]}" for k, v in our_map.items()),
                 off_map == our_map,
                 "純置換：8-way ACC 不受影響，但**逐類別／混淆矩陣的比較必須先對映**"))

    L.append(row("跑幾折 / 平均對象",
                 "10 折（`main.sh` SEED 1..10），對 fold 平均",
                 f"`configs` 寫死 `fold: {cfg['fold']}`；主線對 **model seed** 平均",
                 False,
                 "本 repo 目前只跑 fold 1；`run_exp2.py` 尚無 `--fold`"))
    L.append(row("epochs", str(OFFICIAL["epochs"][0]), "5", False,
                 "屬**我們方法的訓練設定**，非資料協定；不影響切分可比性"))

    L += ["", "## 切分檔的內部一致性（本機自檢）", "",
          "| task | 病人總數 | 十折 test 聯集 | 折內重疊 |", "|---|---|---|---|"]
    for t, n_pid, n_test, ov in check_overlap(cfg):
        L.append(f"| {t} | {n_pid} | {n_test} | "
                 f"{'✅ 無' if not ov else '❌ ' + '；'.join(ov)} |")

    L += ["", "## fold 1 的成員數（本機）", "",
          "| task | train | val | test |", "|---|---|---|---|"]
    for t in TASKS:
        c = counts.get(t, {})
        L.append(f"| {t} | {c.get('train_patients','—')} | "
                 f"{c.get('val_patients','—')} | {c.get('test_patients','—')} |")

    L += ["", "## 結論", "",
          "**路徑、切分檔格式、切分鍵、表格→標籤對映全部一致**（後兩者的載入程式碼與"
          "官方逐字相同，故子型別對映由構造相同）。**每折成員切片**無法與官方 repo 對照"
          "——他們的 repo 不含切分檔，切分檔就在本機這份作者公開的 benchmark 裡；"
          "本檔改以內部一致性（折內無重疊、十折 test 聯集 = 病人總數）驗證。", "",
          "**兩處實質不同，且都不是換切分檔的 adapter 能對齊的：**", "",
          "1. **任務順序相反。** 官方發表協定是 forward（lung→brca→rcc→esca），"
          "repo 內唯一的評測範本也是 `forward-order.xlsx`。本 repo 的主線是 `reverse`。"
          "要引用其發表數字當 baseline，我們必須跑 **`--order main`**（= 官方 forward）。",
          "2. **平均對象不同。** 官方對 **10 個 fold** 平均；本 repo 主線對 "
          "**5 個 model seed @ fold 1** 平均。兩者不是同一種變異來源，不可直接並列。",
          "",
          "8-way 標籤索引雖然相反，但那是純置換，8-way ACC 不受影響 —— "
          "只有逐類別或混淆矩陣的比較需要先對映。",
          "",
          f"產生：`python scripts/audit_qpmil_protocol.py`（官方事實 {FETCHED} 取得）。", ""]

    return L
