#!/usr/bin/env python3
"""exp5 第 4 步：協定核對。

把本批的 chk_m512_f1 與 outputs/exp3/sota_ucur 的 fold 1 逐筆比對。

判讀（DR-057 凍結，不在看到數字後調整）：
  stage 0 相同          → 協定正確，繼續。stage 0 沒有 replay，所以它與 |M| 無關，
                          一致就代表資料、切分、標籤空間、選擇器、評估口徑全部對上。
  全部逐位元相同        → 記錄「與 exp3 同平台」。
  stage 0 同、之後分歧  → 平台浮點差異（機制見 DR-052 的 Mac/pod 分歧），繼續。
  stage 0 就不同        → 設定有誤，停下。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "outputs/exp3/sota_ucur/per_slide/A5_reverse_seed1_hier_acc_ucur.json"


def key(r: dict) -> tuple:
    return (r["stage"], r["task"], r["slide_id"])


def acc(recs: list[dict], field: str) -> float:
    return sum(r[field] == r["true"] for r in recs) / len(recs) if recs else float("nan")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--new", required=True, help="本批的 per_slide JSON")
    ap.add_argument("--ref", default=str(REF))
    ap.add_argument("--out", default=None, help="把判讀寫成 markdown")
    a = ap.parse_args(argv)

    new = json.loads(Path(a.new).read_text())
    ref = json.loads(Path(a.ref).read_text())
    N = {key(r): r for r in new}
    R = {key(r): r for r in ref}
    shared = sorted(set(N) & set(R))

    L = [f"# exp5 協定核對", "",
         f"* 本批：`{a.new}`（{len(new)} 筆）",
         f"* 基準：`{a.ref}`（{len(ref)} 筆）",
         f"* 共同的 (stage, task, slide)：{len(shared)}", ""]
    if set(N) != set(R):
        L += [f"⚠️ 記錄集合不同：本批多 {len(set(N)-set(R))}、基準多 {len(set(R)-set(N))}", ""]

    stages = sorted({k[0] for k in shared})
    L += ["每個 stage 的 n 是**該時點評估的所有任務**加總（學完 t 個任務就評估 t 個），",
          "class-IL 也是同一個口徑。", "",
          "| stage | 評估的任務 | n | 本批 class-IL | 基準 class-IL | selected_idx 不同 | pred 不同 |",
          "|---|---|---|---|---|---|---|"]
    per_stage = {}
    for s in stages:
        ks = [k for k in shared if k[0] == s]
        nn = [N[k] for k in ks]; rr = [R[k] for k in ks]
        d_idx = sum(N[k]["selected_idx"] != R[k]["selected_idx"] for k in ks)
        d_prd = sum(N[k]["pred_class_il"] != R[k]["pred_class_il"] for k in ks)
        per_stage[s] = (len(ks), d_idx, d_prd)
        tl = ", ".join(t.replace("tcga_", "") for t in
                       sorted({k[1] for k in ks}, key=lambda x: [k[1] for k in ks].index(x)))
        L.append(f"| {s} | {tl} | {len(ks)} | {acc(nn,'pred_class_il'):.8f} | "
                 f"{acc(rr,'pred_class_il'):.8f} | {d_idx} | {d_prd} |")

    s0 = per_stage.get(0)
    total_idx = sum(v[1] for v in per_stage.values())
    total_prd = sum(v[2] for v in per_stage.values())
    first_div = next((s for s in stages if per_stage[s][1] or per_stage[s][2]), None)

    if s0 is None:
        verdict, ok = "⚠️ 沒有共同的 stage 0 記錄，無法核對。", False
    elif s0[1] or s0[2]:
        verdict = (f"❌ **stage 0 就不同**（selected_idx {s0[1]}/{s0[0]}、"
                   f"pred {s0[2]}/{s0[0]}）。stage 0 不跑 replay，與 |M| 無關，"
                   "所以這代表設定有誤，不是浮點差異。**停下佇列。**")
        ok = False
    elif total_idx == 0 and total_prd == 0:
        verdict = "✅ **全部逐位元相同** → 本批與 exp3 同平台，數字可直接對讀。"
        ok = True
    else:
        verdict = (f"✅ **stage 0 逐筆相同**，協定正確。分歧自 stage {first_div} 起出現"
                   f"（全程 selected_idx {total_idx}、pred {total_prd} 筆不同）→ "
                   "記錄為平台浮點差異經 top-k 離散決策放大（機制同 DR-052），繼續執行。"
                   "⚠️ 本批與 exp3 的數字**不得相減**。")
        ok = True

    L += ["", "## 判讀", "", verdict, ""]
    txt = "\n".join(L) + "\n"
    if a.out:
        Path(a.out).write_text(txt, encoding="utf-8")
    print(txt)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
