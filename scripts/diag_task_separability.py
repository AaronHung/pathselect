#!/usr/bin/env python3
"""S1 — 任務可分離性 probe（診斷，**不是方法的一部分**）。

問題：四個 task 是四個不同器官（esca / rcc / brca / lung）。食道與腎臟切片在
視覺上差很多，task identity 可能從影像特徵本身就免費可得。若是如此，q_tau 提供
的是模型早已擁有的資訊。

做法：
  A. 每張 slide 取 CONCH patch feature 的平均 → [n_slides, 512]
  B. 每張 slide 取 8 個 tissue group prototype 串接 → [n_slides, 8*512]
  兩種輸入各訓一個 multinomial logistic regression 預測 4-way task id，
  train split 訓練、test split 評估，報 accuracy 與 confusion matrix。

⚠️ 本檔不被 selector 匯入，也不寫入任何模型權重。純診斷。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from selector.evaluate import read_slide, slide_dataset                  # noqa: E402
from selector.grouping import (NUM_GROUPS, TISSUE_GROUP_NAMES,           # noqa: E402
                               assign_groups, tissue_text_features)
from selector.text_encoder import load_config                            # noqa: E402

OUT_DIR = REPO_ROOT / "outputs" / "exp1" / "diag"
SEED = 0


def build_features(cfg, tissue, split: str):
    """回傳 (X_mean [N,512], X_group [N,8*512], y [N], slide_ids)。"""
    X_mean, X_group, y, sids = [], [], [], []
    for pos, task in enumerate(cfg["tasks"]):
        ds, shift = slide_dataset(cfg, task, pos, split)
        for i in range(len(ds)):
            rec = read_slide(ds, shift, i)
            X_mean.append(rec.Z.mean(0))
            g = assign_groups(rec.Z, tissue)
            X_group.append(g.prototypes.reshape(-1))       # 空 group 為零向量
            y.append(pos)
            sids.append(rec.sid)
        print(f"  {split:5s} {task:10s} {len(ds):4d} slides", flush=True)
    return (torch.stack(X_mean).numpy(), torch.stack(X_group).numpy(),
            np.array(y), sids)


def fit_probe(Xtr, ytr, Xte, yte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    clf = make_pipeline(
        StandardScaler(),
        # sklearn >= 1.7 移除了 multi_class 參數；多類別 + lbfgs 預設就是 multinomial
        LogisticRegression(max_iter=5000, random_state=SEED))
    clf.fit(Xtr, ytr)
    pred_te = clf.predict(Xte)
    return {
        "train_acc": float(clf.score(Xtr, ytr)),
        "test_acc": float((pred_te == yte).mean()),
        "confusion": confusion_matrix(yte, pred_te, labels=list(range(4))).tolist(),
        "pred_test": pred_te.tolist(),
    }


def md_confusion(cm, tasks) -> list[str]:
    L = ["| true \\ pred | " + " | ".join(tasks) + " | n |", "|---" * (len(tasks) + 2) + "|"]
    for i, t in enumerate(tasks):
        row = cm[i]
        L.append(f"| **{t}** | " + " | ".join(str(v) for v in row) + f" | {sum(row)} |")
    return L


def main() -> int:
    cfg = load_config()
    tasks = list(cfg["tasks"])
    tissue = tissue_text_features(cfg, device="cpu")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = OUT_DIR / "_probe_features.npz"
    if cache.exists():
        print(f"reusing cached features: {cache}", flush=True)
        z = np.load(cache, allow_pickle=True)
        Xm_tr, Xg_tr, y_tr = z["Xm_tr"], z["Xg_tr"], z["y_tr"]
        Xm_te, Xg_te, y_te, sid_te = z["Xm_te"], z["Xg_te"], z["y_te"], list(z["sid_te"])
    else:
        print("building features…", flush=True)
        Xm_tr, Xg_tr, y_tr, _ = build_features(cfg, tissue, "train")
        Xm_te, Xg_te, y_te, sid_te = build_features(cfg, tissue, "test")
        np.savez_compressed(cache, Xm_tr=Xm_tr, Xg_tr=Xg_tr, y_tr=y_tr,
                            Xm_te=Xm_te, Xg_te=Xg_te, y_te=y_te,
                            sid_te=np.array(sid_te, dtype=object))

    res = {
        "mean_patch_512": fit_probe(Xm_tr, y_tr, Xm_te, y_te),
        "group_prototypes_4096": fit_probe(Xg_tr, y_tr, Xg_te, y_te),
    }
    for k, v in res.items():
        print(f"  {k:24s} train={v['train_acc']:.4f}  test={v['test_acc']:.4f}")

    (OUT_DIR / "per_slide_task_probe.json").write_text(json.dumps([
        {"slide_id": s, "task": tasks[int(t)], "true": int(t),
         "pred_mean_patch": int(res["mean_patch_512"]["pred_test"][i]),
         "pred_group_proto": int(res["group_prototypes_4096"]["pred_test"][i])}
        for i, (s, t) in enumerate(zip(sid_te, y_te))], indent=1))

    short = [t.replace("tcga_", "") for t in tasks]
    n_tr = [int((y_tr == i).sum()) for i in range(4)]
    n_te = [int((y_te == i).sum()) for i in range(4)]
    L = [
        "# S1 — 任務可分離性 probe",
        "",
        "⚠️ **這是診斷，不是方法的一部分。** 本 probe 不被 selector 匯入，"
        "不影響任何 Exp 0 / Exp 1 的數字。",
        "",
        "問題：四個 task 是四個不同器官，task identity 可能從影像特徵本身就免費可得。",
        "",
        "做法：每張 slide 取一個固定長度的表徵，訓一個 multinomial logistic "
        "regression 預測 4-way task id（train split 訓練、test split 評估）。"
        f"特徵先做 StandardScaler，max_iter=5000，random_state={SEED}。",
        "",
        f"- train：{sum(n_tr)} 張（" + "、".join(
            f"{s} {n}" for s, n in zip(short, n_tr)) + "）",
        f"- test：{sum(n_te)} 張（" + "、".join(
            f"{s} {n}" for s, n in zip(short, n_te)) + "）",
        f"- 多數類基準（test）：{max(n_te) / sum(n_te):.4f}；隨機猜：0.2500",
        "",
        "## 結果",
        "",
        "| 輸入表徵 | 維度 | train accuracy | test accuracy |",
        "|---|---|---|---|",
        f"| slide 平均 CONCH patch feature | 512 | "
        f"{res['mean_patch_512']['train_acc']:.4f} | "
        f"{res['mean_patch_512']['test_acc']:.4f} |",
        f"| 8 個 tissue group prototype 串接 | {NUM_GROUPS * 512} | "
        f"{res['group_prototypes_4096']['train_acc']:.4f} | "
        f"{res['group_prototypes_4096']['test_acc']:.4f} |",
        "",
        "group prototype 的順序為 " + "、".join(TISSUE_GROUP_NAMES) + "；空 group 補零向量。",
        "",
        "## Confusion matrix（test split）",
        "",
        "### A. slide 平均 patch feature（512-d）",
        "",
    ]
    L += md_confusion(res["mean_patch_512"]["confusion"], short)
    L += ["", "### B. group prototype 串接（4096-d）", ""]
    L += md_confusion(res["group_prototypes_4096"]["confusion"], short)
    L += ["", "逐 slide 預測：`outputs/exp1/diag/per_slide_task_probe.json`", ""]
    (OUT_DIR / "TASK_SEPARABILITY.md").write_text("\n".join(L) + "\n")
    print(f"\n→ {OUT_DIR / 'TASK_SEPARABILITY.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
