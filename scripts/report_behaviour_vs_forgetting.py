#!/usr/bin/env python3
"""SEEDS S-03：行為保留度是否預測準確率保留度？（DR-046 的依據）

S-03 寫「brca 6/6 方向一致」，但 repo 裡**沒有對應產物**（憲法 §2.8）。本檔把它
重算出來：每個 (order, seed) 取三個「非最後學」的 task，看
**Jaccard 最高的那個 task 是不是 forgetting 最小的那個**。

⚠️ 不得為了湊 6/6 改定義。定義先寫死在這裡：
- Jaccard = 該 task 學完當下 vs 學完 T4 的選取集合重疊（逐 slide 算後平均）
- forgetting = class-IL accuracy 在同兩個時點的差（pp，正 = 退步）
- 「一致」= argmax(Jaccard) 與 argmin(forgetting) 是同一個 task
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SRC = ROOT / "outputs" / "exp2" / "seqft" / "per_slide"
OUT = ROOT / "outputs" / "exp2" / "seqft" / "BEHAVIOUR_VS_FORGETTING.md"
ORDERS = {"reverse": ["tcga_esca", "tcga_rcc", "tcga_brca", "tcga_lung"],
          "main": ["tcga_lung", "tcga_brca", "tcga_rcc", "tcga_esca"]}


def jac(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 1.0


def acc(recs):
    return sum(r["pred_softmax"] == r["true"] for r in recs) / len(recs) if recs else float("nan")


def main() -> int:
    recs = [r for f in sorted(SRC.glob("*.json")) for r in json.loads(f.read_text())]
    if not recs:
        print("尚無 seqft 資料"); return 1
    last = 3
    rows, hits = [], []
    for order, tasks in ORDERS.items():
        for seed in sorted({r["seed"] for r in recs if r["order"] == order}):
            per = {}
            for i, t in enumerate(tasks[:last]):          # 只看非最後學的三個
                at_i = [r for r in recs if r["order"] == order and r["seed"] == seed
                        and r["stage"] == i and r["task"] == t]
                at_e = [r for r in recs if r["order"] == order and r["seed"] == seed
                        and r["stage"] == last and r["task"] == t]
                if not at_i or not at_e:
                    continue
                by = {r["slide_id"]: r for r in at_e}
                pairs = [(r, by[r["slide_id"]]) for r in at_i if r["slide_id"] in by]
                per[t] = {
                    "jaccard": statistics.mean([jac(a["selected_idx"], b["selected_idx"])
                                                for a, b in pairs]),
                    "forget": (acc(at_i) - acc(at_e)) * 100,
                }
            if len(per) < 3:
                continue
            best_j = max(per, key=lambda t: per[t]["jaccard"])
            best_f = min(per, key=lambda t: per[t]["forget"])
            ok = best_j == best_f
            hits.append(ok)
            rows.append((order, seed, per, best_j, best_f, ok))

    n_ok = sum(hits)
    L = ["# 行為保留度是否預測準確率保留度？（SEEDS S-03 的重算）", "",
         "S-03 寫「brca 6/6 方向一致」，但 repo 裡沒有對應產物。本檔重算。", "",
         "**定義（先寫死，不因結果調整）**：每個 (order, seed) 取三個非最後學的 task；",
         "「一致」= Jaccard 最高的 task 與 forgetting 最小的 task 是同一個。",
         "Jaccard 為逐 slide 算後平均；forgetting 為 class-IL accuracy 的差（pp，正 = 退步）。", "",
         f"## 結果：**{n_ok}/{len(hits)}** 一致", "",
         "| order | seed | Jaccard 最高 | forgetting 最小 | 一致？ | 逐 task（Jaccard｜forgetting pp） |",
         "|---|---|---|---|---|---|"]
    for order, seed, per, bj, bf, ok in rows:
        detail = "；".join(f"{t.replace('tcga_', '')} {v['jaccard']:.4f}｜{v['forget']:+.2f}"
                           for t, v in per.items())
        L.append(f"| {order} | {seed} | {bj.replace('tcga_', '')} | "
                 f"{bf.replace('tcga_', '')} | {'✅' if ok else '❌'} | {detail} |")

    brca = sum(1 for _o, _s, per, bj, _bf, _k in rows if bj == "tcga_brca")
    L += ["",
          f"其中 Jaccard 最高者為 **brca** 的批次：{brca}/{len(rows)}。", "",
          ("✅ **S-03 的「6/6 方向一致」成立**，本檔即其產物。"
           if n_ok == len(hits) and len(hits) == 6 else
           f"⚠️ **S-03 的「6/6 方向一致」不成立** —— 實際為 **{n_ok}/{len(hits)}**。"
           "S-03 該句應改為實際數字（憲法 §2.4）。"),
          "",
          "⚠️ 這是 **3-seed × 2 order 的觀察**，n 太小，不足以支撐「行為保留度預測"
          "準確率保留度」的一般性宣稱（憲法 §1.2）。可安全寫的是逐批次的事實。",
          ""]
    OUT.write_text("\n".join(L) + "\n")
    print(f"→ {OUT}")
    print(f"  {n_ok}/{len(hits)} 一致；Jaccard 最高為 brca 的批次 {brca}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
