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
#: 附錄 A：patch 層 probe 的 train 端每張 slide 取樣數。
#: train 全集是 2273 × ~3400 ≈ 7.7M 個 patch（float32 約 15.8 GB），機器只有 16 GB，
#: 放不下也不必放；train 端固定子取樣，**test 端跑全部 patch**（串流，不materialize）。
PATCH_PER_TRAIN_SLIDE = 64


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


def build_patch_train(cfg, n_per_slide: int = PATCH_PER_TRAIN_SLIDE):
    """[N, 512] 單一 patch 特徵 + task id。train 端每張 slide 固定取樣 n_per_slide 個。"""
    rs = np.random.RandomState(SEED)
    X, y = [], []
    for pos, task in enumerate(cfg["tasks"]):
        ds, shift = slide_dataset(cfg, task, pos, "train")
        for i in range(len(ds)):
            Z = read_slide(ds, shift, i).Z
            n = int(Z.shape[0])
            k = min(n_per_slide, n)
            idx = rs.choice(n, size=k, replace=False)
            X.append(Z[torch.as_tensor(idx, dtype=torch.long)].numpy())
            y.append(np.full(k, pos))
        print(f"  patch-train {task:10s} {len(ds):4d} slides", flush=True)
    return np.concatenate(X), np.concatenate(y)


def eval_patch_probe(clf, cfg):
    """test 端逐 slide 串流預測**全部** patch，累積 confusion matrix。"""
    cm = np.zeros((4, 4), dtype=np.int64)
    per_slide = []
    for pos, task in enumerate(cfg["tasks"]):
        ds, shift = slide_dataset(cfg, task, pos, "test")
        for i in range(len(ds)):
            rec = read_slide(ds, shift, i)
            pred = clf.predict(rec.Z.numpy())
            for c in range(4):
                cm[pos, c] += int((pred == c).sum())
            per_slide.append({"slide_id": rec.sid, "task": task, "true": pos,
                              "n_patch": int(rec.Z.shape[0]),
                              "pred_patch_correct_frac": float((pred == pos).mean()),
                              "pred_patch_majority": int(np.bincount(pred, minlength=4).argmax())})
        print(f"  patch-test  {task:10s} {len(ds):4d} slides", flush=True)
    return cm, per_slide


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

    # ── 附錄 A：patch 層 probe ─────────────────────────────────────────────
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    print("patch-level probe…", flush=True)
    Xp, yp = build_patch_train(cfg)
    print(f"  train patches: {Xp.shape}", flush=True)
    patch_clf = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=1000, random_state=SEED))
    patch_clf.fit(Xp, yp)
    patch_train_acc = float(patch_clf.score(Xp, yp))
    cm_patch, per_slide_patch = eval_patch_probe(patch_clf, cfg)
    patch_test_acc = float(np.trace(cm_patch) / cm_patch.sum())
    slide_vote_acc = float(np.mean([r["pred_patch_majority"] == r["true"]
                                    for r in per_slide_patch]))
    print(f"  patch_level_512          train={patch_train_acc:.4f}  "
          f"test={patch_test_acc:.4f}  (slide 多數決 {slide_vote_acc:.4f})")
    res["patch_level_512"] = {"train_acc": patch_train_acc, "test_acc": patch_test_acc,
                              "confusion": cm_patch.tolist(),
                              "slide_majority_vote_acc": slide_vote_acc,
                              "n_train_patches": int(Xp.shape[0]),
                              "n_test_patches": int(cm_patch.sum())}
    (OUT_DIR / "per_slide_patch_probe.json").write_text(
        json.dumps(per_slide_patch, indent=1))

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
        f"| **單一 patch feature（附錄 A）** | 512 | "
        f"{res['patch_level_512']['train_acc']:.4f} | "
        f"**{res['patch_level_512']['test_acc']:.4f}** |",
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

    pl = res["patch_level_512"]
    L += ["", "## 附錄 A — 單一 patch 的 task 可分離性", "",
          "S1 的主 probe 用全片平均，但 F_p 的輸入是**單一 512-d patch**。這一節把"
          "輸入換成單一 patch feature，label 是該 slide 的 task id，split 沿用同一組。",
          "",
          f"- **train 端子取樣**：全集是 2273 張 × 平均約 3400 個 patch ≈ 7.7M 個，"
          f"float32 約 15.8 GB，這台機器放不下。改為每張 train slide 固定隨機取 "
          f"{PATCH_PER_TRAIN_SLIDE} 個（`np.random.RandomState({SEED})`），"
          f"共 {pl['n_train_patches']:,} 個 patch。",
          f"- **test 端不取樣**：279 張 test slide 的**全部** {pl['n_test_patches']:,} "
          f"個 patch 都評估（逐 slide 串流預測，不一次載入）。",
          f"- max_iter=1000（patch 數量大，比主 probe 的 5000 低）。",
          "",
          "| 指標 | 值 |", "|---|---|",
          f"| patch 層 train accuracy | {pl['train_acc']:.4f} |",
          f"| **patch 層 test accuracy** | **{pl['test_acc']:.4f}** |",
          f"| 同一個 patch probe 做 slide 多數決 | {pl['slide_majority_vote_acc']:.4f} |",
          f"| 多數類基準（patch 層） | "
          f"{max(sum(r) for r in pl['confusion']) / pl['n_test_patches']:.4f} |",
          "", "### Confusion matrix（test split，patch 層，單位：patch 數）", ""]
    L += md_confusion(pl["confusion"], short)
    L += ["", "逐 slide 預測：`outputs/exp1/diag/per_slide_task_probe.json`"
          "（slide 層）、`outputs/exp1/diag/per_slide_patch_probe.json`"
          "（patch 層，含每張 slide 的 patch 正確率與多數決）", ""]
    (OUT_DIR / "TASK_SEPARABILITY.md").write_text("\n".join(L) + "\n")
    print(f"\n→ {OUT_DIR / 'TASK_SEPARABILITY.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
