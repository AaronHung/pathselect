#!/usr/bin/env python3
"""驗證 Exp 2 的 A1（SeqFT，CL 三項全關）與 S2 的結果位元相同。

這是 S4 那條「三項全關等價 SeqFT」單元測試的實跑驗證。

⚠️ 只在 beta_u = 0 下成立：S2 是在 L_util 接上之前跑的（evidence_loss 當時
   永遠 util=None）。beta_u = 0.1 會讓 L_util 進入 L_evidence、改變梯度，
   A1 就不可能重現 S2 —— 這是設定衝突，不是實作錯誤。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
S2_DIR = REPO_ROOT / "outputs" / "exp2" / "seqft" / "per_slide"
A1_DIR = REPO_ROOT / "outputs" / "exp2" / "verify_a1" / "per_slide"
KEY = ("order", "seed", "stage", "task", "slide_id")
COMPARE = ("true", "selected_idx", "weights_softmax", "group_quota", "utility_total")


def index(records, pred_key):
    out = {}
    for r in records:
        out[tuple(r[k] for k in KEY)] = {**{c: r[c] for c in COMPARE},
                                         "pred": r[pred_key]}
    return out


def main() -> int:
    s2 = [r for p in sorted(S2_DIR.glob("*.json")) for r in json.loads(p.read_text())]
    a1 = [r for p in sorted(A1_DIR.glob("*.json")) for r in json.loads(p.read_text())]
    seeds = sorted({r["seed"] for r in a1})
    s2 = [r for r in s2 if r["order"] == "reverse" and r["seed"] in seeds]
    print(f"S2 記錄 {len(s2)} 筆、A1 記錄 {len(a1)} 筆（reverse, seeds {seeds}）")

    S, A = index(s2, "pred_softmax"), index(a1, "pred_class_il")
    only_s2, only_a1 = set(S) - set(A), set(A) - set(S)
    if only_s2 or only_a1:
        print(f"⚠️ key 不對齊：只在 S2 {len(only_s2)} 筆、只在 A1 {len(only_a1)} 筆")

    shared = sorted(set(S) & set(A))
    diffs = {c: 0 for c in COMPARE + ("pred",)}
    for k in shared:
        for c in diffs:
            if S[k][c] != A[k][c]:
                diffs[c] += 1
    print(f"比對 {len(shared)} 筆共同記錄：")
    for c, n in diffs.items():
        print(f"  {c:18s} 不同 {n} 筆")
    ok = all(n == 0 for n in diffs.values()) and not only_s2 and not only_a1
    print("\n結論：" + ("✅ 位元相同 —— CL 三項全關確實等價於 SeqFT"
                        if ok else "❌ 不相同"))
    if not ok:
        for k in shared:
            if any(S[k][c] != A[k][c] for c in diffs):
                print(f"\n第一筆差異 {k}")
                for c in diffs:
                    if S[k][c] != A[k][c]:
                        print(f"  {c}: S2={str(S[k][c])[:90]}")
                        print(f"  {'':{len(c)}}  A1={str(A[k][c])[:90]}")
                break
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
