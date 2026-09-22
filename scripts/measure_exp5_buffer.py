#!/usr/bin/env python3
"""exp5：replay 實際需要的特徵量（實測，在 pod 上跑）。

replay 不是讀記憶庫本身 —— entry 不存 feature，它是拿 sample_key 去**重新載入那張切片
的完整特徵檔**（`selector/memory.py::reload_features`）。所以「buffer 有多大」這個問題的
誠實答案是：**留在記憶庫裡的那 |M| 筆所對應的完整特徵檔有多大**。

留下的是哪 |M| 筆可以無模型精確重放：`ReservoirSampling` 只吃 `random.Random(0)` 與
(len, capacity, n_seen)，entry 的身分只由加入順序決定，而 `fill_memory` 依 dataset 順序
逐張加入。本倉已有一組測試把這個重放與真正的 SelectionMemory 逐筆比對過。

只讀，不寫任何實驗產物。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_exp2 import ORDERS                                    # noqa: E402
from selector.evaluate import slide_dataset                    # noqa: E402
from selector.memory import ReservoirSampling                  # noqa: E402
from selector.text_encoder import load_config                  # noqa: E402


def retained(order_tasks, sids_by_task, capacity):
    pol = ReservoirSampling()
    entries, n_seen = [], 0
    for task in order_tasks:
        for sid in sids_by_task[task]:
            n_seen += 1
            pol.offer(entries, (task, str(sid)), capacity, n_seen)
    return entries


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--caps", default="16,32,64,128,256,512")
    ap.add_argument("--orders", default="reverse,main")
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    cfg = load_config()
    cfg["fold"] = a.fold
    tasks = list(cfg["tasks"])

    sids_by_task, size_by = {}, {}
    train_total = 0
    for pos, task in enumerate(tasks):
        ds, _ = slide_dataset(cfg, task, pos, "train")
        sids_by_task[task] = [str(s) for s in ds.sids]
        feat_dir = Path(cfg["dataset_root_dir"] +
                        cfg["path_feat"].format(task, cfg["conch_path_feat"]))
        for sid in sids_by_task[task]:
            hits = sorted(feat_dir.glob(f"{sid}*"))
            n = hits[0].stat().st_size if hits else 0
            size_by[(task, sid)] = n
            train_total += n

    entries = []
    for cap in [int(x) for x in a.caps.split(",")]:
        for order in a.orders.split(","):
            keep = retained(ORDERS[order], sids_by_task, cap)
            got = [size_by.get(k, 0) for k in keep]
            found = sum(1 for x in got if x)
            entries.append({
                "capacity": cap, "order": order, "fold": a.fold,
                "retained": len(keep), "found": found,
                "total_bytes": sum(got),
                "total_MiB": sum(got) / 2**20,
                "pct_of_train": 100.0 * sum(got) / train_total if train_total else 0.0,
                "by_task": {t: sum(1 for x, _ in keep if x == t)
                            for t in sorted({x for x, _ in keep})},
            })

    out = {"train_files": sum(len(v) for v in sids_by_task.values()),
           "train_total_bytes": train_total,
           "train_total_GiB": train_total / 2**30,
           "note": "留在記憶庫的 |M| 筆所對應的完整特徵檔大小總和。"
                   "replay 重新載入整張切片的特徵，不是只讀候選。",
           "entries": entries}
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"訓練特徵庫：{out['train_files']} 檔、{out['train_total_GiB']:.2f} GiB")
    for e in entries:
        print(f"  |M|={e['capacity']:<4} {e['order']:<8} 保留 {e['found']:>3}/{e['retained']:<3} "
              f"= {e['total_MiB']:8.1f} MiB（{e['pct_of_train']:5.2f}%）  {e['by_task']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
