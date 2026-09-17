#!/usr/bin/env python3
"""DR-056 第 7 步的資料量實測（在 pod 上跑；只讀，不動任何產物）。

輸出 JSON：
  side_store    候選側倉每檔 bytes 的分布 ＋ 總量（= 候選級 replay 每步實際讀入的量）
  full_slide    同一批 slide 的完整特徵檔 bytes 分布 ＋ 總量（= 現行 replay 每步讀入的量）
  paired        逐筆配對（同一張 slide 的側倉 vs 完整檔），給出實測倍率
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from selector.evaluate import slide_dataset                       # noqa: E402
from selector.memory import SampleKeyIndex, sample_key            # noqa: E402
from selector.text_encoder import load_config                     # noqa: E402


def replay_reservoir(order_tasks, sids_by_task, capacity):
    """無模型重放 reservoir 汰換，回傳最後留在記憶庫的 (tau, sid)。

    `ReservoirSampling` 只吃 `random.Random(0)` 與 (len, capacity, n_seen)，
    entry 的身分只由「加入順序」決定 —— fill_memory 依 dataset 順序逐張加入，
    且本批四個 run 的 log 顯示每個 task 的 `記憶體 +N` 都等於該 task 的 train
    張數（無 cand 為空的跳過），所以這個重放與訓練時逐筆相同。
    """
    from selector.memory import ReservoirSampling
    pol = ReservoirSampling()
    entries, n_seen = [], 0
    for task in order_tasks:
        for sid in sids_by_task[task]:
            n_seen += 1
            pol.offer(entries, (task, str(sid)), capacity, n_seen)
    return entries


def dist(v):
    if not v:
        return None
    return {"n": len(v), "min": min(v), "mean": statistics.mean(v), "max": max(v),
            "total_bytes": sum(v), "total_GiB": sum(v) / 2**30}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True, help="cand_store 目錄")
    ap.add_argument("--out", required=True)
    ap.add_argument("--order", default="reverse", choices=["reverse", "main"],
                    help="任務順序；決定 reservoir 的加入序（reverse / main = forward）")
    ap.add_argument("--mem-capacity", type=int, default=128,
                    help="|M| 上限；用來算「真正需要保留」的量（見輸出的 bounded 欄位）")
    a = ap.parse_args(argv)
    store = Path(a.store)
    cfg = load_config()
    tasks = list(cfg["tasks"])

    # key → (sid, 完整特徵檔路徑)。key 是 (tau, sid) 的純函數，可從 split 重建。
    lookup: dict[tuple[str, int], tuple[str, Path]] = {}
    sids_by_task: dict[str, list] = {}
    for pos, task in enumerate(tasks):
        ds, _shift = slide_dataset(cfg, task, pos, "train")
        sids_by_task[task] = [str(s) for s in ds.sids]
        idx = SampleKeyIndex.from_slide_ids(task, ds.sids)
        feat_dir = Path(cfg["dataset_root_dir"] +
                        cfg["path_feat"].format(task, cfg["conch_path_feat"]))
        for key, sid in idx._to_sid.items():
            hits = sorted(feat_dir.glob(f"{sid}*"))
            lookup[(task, key)] = (sid, hits[0] if hits else None)

    side, full, paired, missing = [], [], [], []
    for f in sorted(store.glob("*.pt")):
        tau, _, keys = f.stem.rpartition("_")
        rec = lookup.get((tau, int(keys)))
        s = f.stat().st_size
        side.append(s)
        if rec is None or rec[1] is None:
            missing.append(f.name); continue
        sid, fp = rec
        b = fp.stat().st_size
        full.append(b)
        paired.append({"tau": tau, "sid": sid, "side_bytes": s, "full_bytes": b,
                       "ratio": b / s})

    # ⚠️ 現行 fill_memory 對**每一張快照過的 slide** 都寫側倉，寫入發生在
    # memory.add()（reservoir 汰換）之前，所以側倉的檔數 = 看過的 slide 數，
    # **不受 |M| 上限約束**。真正需要保留的只有留在記憶庫裡的 |M| 筆。
    # 兩個數字都報：as_written = 現行實作實際佔用；bounded = 方法本身的需求。
    # 真正需要保留的是最後留在記憶庫的那 |M| 筆 —— 用無模型重放算出**確切**是哪些，
    # 再把它們的側倉檔案大小加總（不是用平均估）。
    from run_exp2 import ORDERS
    retained = replay_reservoir(ORDERS[a.order], sids_by_task, a.mem_capacity)
    ret_bytes, ret_missing = [], []
    for tau, sid in retained:
        f = store / f"{tau}_{sample_key(tau, sid)}.pt"
        (ret_bytes.append(f.stat().st_size) if f.is_file() else ret_missing.append(f.name))
    bounded = {"order": a.order, "capacity": a.mem_capacity,
               "entries": len(retained), "found": len(ret_bytes),
               "missing": ret_missing[:5],
               "total_bytes": sum(ret_bytes), "total_GiB": sum(ret_bytes) / 2**30,
               "by_task": {t: sum(1 for x, _ in retained if x == t)
                           for t in sorted({x for x, _ in retained})},
               "note": "無模型重放 ReservoirSampling(seed=0) 得到確切的保留集合，"
                       "再加總其側倉檔案大小。現行實作未在汰換時刪檔，故側倉"
                       "實際佔用（as_written）大於這個數。"}

    out = {"store": str(store), "mem_capacity": a.mem_capacity,
           "bounded_requirement": bounded,
           "side_store": dist(side), "full_slide": dist(full),
           "missing_lookup": missing,
           "ratio": {"min": min((p["ratio"] for p in paired), default=None),
                     "mean": statistics.mean([p["ratio"] for p in paired]) if paired else None,
                     "max": max((p["ratio"] for p in paired), default=None),
                     "total": (sum(p["full_bytes"] for p in paired) /
                               sum(p["side_bytes"] for p in paired)) if paired else None},
           "paired": paired}
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    sd, fl = out["side_store"], out["full_slide"]
    print(f"側倉   n={sd['n']}  mean {sd['mean']:,.0f} B  總計 {sd['total_GiB']:.4f} GiB")
    if fl:
        print(f"完整檔 n={fl['n']}  mean {fl['mean']:,.0f} B  總計 {fl['total_GiB']:.4f} GiB")
        print(f"倍率   mean {out['ratio']['mean']:.1f}×   總量 {out['ratio']['total']:.1f}×")
    b = out["bounded_requirement"]
    print(f"真正需要保留（|M|={a.mem_capacity}, order={a.order}）："
          f"{b['found']}/{b['entries']} 筆 = {b['total_GiB']:.4f} GiB  分佈 {b['by_task']}")
    print("⚠️ 側倉檔數 = 看過的 slide 數，不受 |M| 約束（寫入早於 reservoir 汰換）。")
    if missing:
        print(f"⚠️ {len(missing)} 檔對不到 slide：{missing[:3]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
