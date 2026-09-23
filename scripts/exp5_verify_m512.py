#!/usr/bin/env python3
"""把本批的 M=512 reverse 十折與 exp3 的同設定逐折比對。

論文即將退回 M=512，整張 Table 1/2/3 都建立在 exp3 的那批上。P0 只對過 fold 1
逐位元相同，其餘九折未驗證 —— 本檔補上那九折。

比對到逐筆：每一折的 selected_idx 與預測是否完全相同，而不只是均值接近。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from report_exp5 import load_tag                               # noqa: E402

NEW = ROOT / "outputs/exp5/m512_full_rev/per_slide"
REF = ROOT / "outputs/exp3/sota_ucur/per_slide"


def ref_file(fold: int) -> Path:
    """exp3 的命名：fold 1 不加後綴，其餘加 _f{fold}。"""
    s = "" if fold == 1 else f"_f{fold}"
    return REF / f"A5_reverse_seed{fold}{s}_hier_acc_ucur.json"


def new_file(fold: int) -> Path:
    s = "" if fold == 1 else f"_f{fold}"
    return NEW / f"A5_reverse_seed{fold}_M512{s}_hier_acc_ucur.json"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    L = ["# exp5：M = 512 reverse 十折與 exp3 的逐折比對", "",
         "論文即將退回 M = 512，整張 Table 1／2／3 都建立在 `outputs/exp3/sota_ucur` 上。",
         "P0 的協定核對只對過 fold 1，本檔補上其餘九折。**比對到逐筆，不只看均值。**", "",
         "| fold | n | 本批 class-IL | exp3 class-IL | selected_idx 不同 | 預測不同 | 判定 |",
         "|---|---|---|---|---|---|---|"]
    identical, differing, missing = [], [], []
    for fold in range(1, 11):
        nf, rf = new_file(fold), ref_file(fold)
        if not nf.is_file() or not rf.is_file():
            missing.append(fold)
            L.append(f"| {fold} | — | {'缺本批' if not nf.is_file() else ''}"
                     f"{'缺 exp3' if not rf.is_file() else ''} | | | | ⚠️ |")
            continue
        N = {(r["stage"], r["task"], r["slide_id"]): r for r in json.loads(nf.read_text())}
        R = {(r["stage"], r["task"], r["slide_id"]): r for r in json.loads(rf.read_text())}
        ks = sorted(set(N) & set(R))
        d_idx = sum(N[k]["selected_idx"] != R[k]["selected_idx"] for k in ks)
        d_prd = sum(N[k]["pred_class_il"] != R[k]["pred_class_il"] for k in ks)
        an = sum(N[k]["pred_class_il"] == N[k]["true"] for k in ks) / len(ks)
        ar = sum(R[k]["pred_class_il"] == R[k]["true"] for k in ks) / len(ks)
        same = d_idx == 0 and d_prd == 0
        (identical if same else differing).append(fold)
        L.append(f"| {fold} | {len(ks)} | {an:.8f} | {ar:.8f} | {d_idx} | {d_prd} | "
                 f"{'✅ 逐位元相同' if same else '❌ 有差異'} |")

    # 十折彙總
    per = load_tag(NEW, "reverse")
    L += [""]
    if per:
        acc = [per[k]["acc"] for k in sorted(per)]
        fo = [per[k]["forgetting"] for k in sorted(per)]
        mk = [per[k]["masked_acc"] for k in sorted(per)]
        L += ["## 十折彙總", "",
              "| 來源 | n | ACC | Forgetting | Masked ACC |", "|---|---|---|---|---|",
              f"| 本批（5090 pod） | {len(per)} | {statistics.mean(acc):.4f} ± "
              f"{statistics.stdev(acc):.4f} | {statistics.mean(fo):.4f} | {statistics.mean(mk):.4f} |",
              "| 稿件現用（exp3） | 10 | 0.834 ± 0.031 | 0.101 | 0.914 |", ""]

    L += ["## 判定", ""]
    if missing:
        L += [f"⚠️ 有 {len(missing)} 折缺檔（{missing}），無法完整判定。", ""]
    if differing:
        L += [f"❌ **{len(differing)} 折有差異**（fold {differing}）。"
              "論文的地基**不是**在本機可逐位元重現的；引用 exp3 的數字時必須標明平台。", ""]
    elif identical and not missing:
        L += [f"✅ **十折全部逐位元相同。** 本批與 exp3 在這個設定下完全等價 —— "
              "論文即將退回的 M = 512 那套數字，在本機得到獨立重現。", "",
              "這比 P0 的單折核對強得多：整張 Table 1 的 PathSelect 列、Table 2 的第 4 列，"
              "以及所有由它們導出的敘述，其計算基礎都已驗證。", ""]
    txt = "\n".join(L) + "\n"
    if a.out:
        Path(a.out).write_text(txt, encoding="utf-8")
    print(txt)
    return 0 if not differing and not missing else 1


if __name__ == "__main__":
    sys.exit(main())
