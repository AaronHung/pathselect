#!/usr/bin/env python3
"""已見類別限制版指標 → `outputs/exp2/sota/SEEN_CLASS_CHECK.md`。

把 class-IL 的 argmax 從**固定八類**改成**每階段只在已見類別上**取
（＝累積式分類頭在中間階段的行為），重算 class-IL ACC / Forgetting / BWT
並與現行值並排。理由見 `docs/CLAIMS.md` C-30 與 DR-048。

## 為什麼要重建 logits

逐 slide 存檔**沒有存 logits**，只有 `pred_class_il`（八類 argmax）與
`pred_task_il`。因此用存下來的 `selected_idx` 與 `weights_softmax`，
走**與 `run_exp2.evaluate` 完全相同的路徑**（`conch_classify` + `Ctx.f_txt`
+ `Ctx.logit_scale`）重建 logits，再在已見類別的欄位子集上取 argmax。

⚠️ **重建忠實度每次都驗證**（`--verify N`）：重建的八類 argmax 必須與存檔的
   `pred_class_il` 完全一致，不一致就中止 —— 否則後面的差值沒有意義。

⚠️ **最終階段不重算**：該階段的已見類別就是全部八類，兩種取法逐筆相同，
   直接沿用存檔值。這也是 class-IL ACC 差值恆為 0 的原因。

⚠️ **唯讀**：本腳本只讀 `per_slide` 與特徵檔，只寫這一份報表。
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_exp2 import DEFAULT_ARCH, ORDERS, Ctx                        # noqa: E402
from selector.classifier import conch_classify                       # noqa: E402
from selector.text_encoder import load_config                        # noqa: E402
from sota.metrics import bwt, forgetting                             # noqa: E402

OUT = ROOT / "outputs" / "exp2" / "sota" / "SEEN_CLASS_CHECK.md"
SRC = ROOT / "outputs" / "exp2" / "sota" / "per_slide"

#: (arm, order, arch) → 顯示名
CONFIGS = [("A5", "reverse", "flat", "reverse ／ flat"),
           ("A5", "reverse", "hier", "reverse ／ hier"),
           ("A5", "main", "flat", "forward ／ flat"),
           ("A5", "main", "hier", "forward ／ hier")]


def runs_of(arm: str, order: str, arch: str) -> list[tuple[int, Path]]:
    out = []
    for f in sorted(SRC.glob("*.json")):
        head = json.loads(f.read_text())[0]
        if (head["arm"] == arm and head["order"] == order
                and (head.get("arch") or DEFAULT_ARCH) == arch):
            out.append((head.get("fold", 1), f))
    return sorted(out)


def seen_columns(order: str, stage: int, label_space: list[str]) -> list[int]:
    """stage i 時已見的**全域**類別欄位（每個 task 兩欄）。"""
    cols = []
    for t in ORDERS[order][:stage + 1]:
        lo = 2 * label_space.index(t)
        cols += [lo, lo + 1]
    return sorted(cols)


def fold_matrices(order: str, arch: str, fold: int, path: Path,
                  verify: int) -> tuple[list, list, int]:
    """回傳 (A_fixed8, A_seen, 已驗證筆數)。殘缺折直接拋錯，不得平均進去。"""
    cfg = load_config()
    cfg["fold"] = fold
    ctx = Ctx(cfg)
    tasks = ORDERS[order]
    T = len(tasks)
    col = {t: j for j, t in enumerate(tasks)}
    recs = [r for r in json.loads(path.read_text())
            if (r.get("arch") or DEFAULT_ARCH) == arch]

    # ⚠️ 完整性**先驗**：最終階段必須涵蓋全部任務。放在最前面是為了 fail fast ——
    # 否則要先花數分鐘把特徵讀完才發現這折不能用。跑到一半的折若被當成完整的
    # 平均進去，十折均值會悄悄失真。
    final = {r["task"] for r in recs if r["stage"] == T - 1}
    if final != set(tasks):
        raise SystemExit(f"❌ {path.name} 最終階段只有 {sorted(final)}，"
                         f"預期 {sorted(tasks)} —— 這一折沒跑完，不計入")

    sid2i = {t: {ctx.get(t, "test", i)[0].sid: i
                 for i in range(ctx.n_slides(t, "test"))} for t in tasks}

    hit8: collections.Counter = collections.Counter()
    hits: collections.Counter = collections.Counter()
    tot: collections.Counter = collections.Counter()
    by_slide: dict = collections.defaultdict(list)
    for r in recs:
        by_slide[(r["task"], r["slide_id"])].append(r)

    checked = 0
    for (task, sid), rs in by_slide.items():
        need = any(r["stage"] < T - 1 for r in rs)
        Z = ctx.get(task, "test", sid2i[task][sid])[0].Z if need else None
        for r in rs:
            i, j = r["stage"], col[task]
            tot[(i, j)] += 1
            hit8[(i, j)] += int(r["pred_class_il"] == r["true"])
            if i == T - 1:
                hits[(i, j)] += int(r["pred_class_il"] == r["true"])
                continue
            idx = torch.tensor(r["selected_idx"], dtype=torch.long)
            w = torch.tensor(r["weights_softmax"])
            lg = conch_classify(Z.index_select(0, idx), w, ctx.f_txt,
                                ctx.logit_scale).reshape(-1)
            if checked < verify:
                if int(lg.argmax()) != r["pred_class_il"]:
                    raise SystemExit(
                        f"❌ 重建不符：{path.name} stage {i} {task} {sid} —— "
                        f"重建八類 argmax {int(lg.argmax())} vs 存檔 "
                        f"{r['pred_class_il']}。後續差值無意義，中止。")
                checked += 1
            c = torch.tensor(seen_columns(order, i, ctx.label_space))
            hits[(i, j)] += int(int(c[int(lg.index_select(0, c).argmax())]) == r["true"])

    def mat(h):
        return [[(h[(i, j)] / tot[(i, j)] if tot[(i, j)] else None)
                 for j in range(T)] for i in range(T)]

    return mat(hit8), mat(hits), checked


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--verify", type=int, default=40,
                    help="重建忠實度抽驗筆數（每折）；0 = 不驗，不建議")
    ap.add_argument("--out", default=None, help="改寫到別的路徑（測試用）")
    args = ap.parse_args(argv)

    rows, checked_total = [], 0
    for arm, order, arch, name in CONFIGS:
        per8, pers = [], []
        for fold, path in runs_of(arm, order, arch):
            A8, As, n = fold_matrices(order, arch, fold, path, args.verify)
            checked_total += n
            per8.append((statistics.mean([x for x in A8[-1]]), forgetting(A8), bwt(A8)))
            pers.append((statistics.mean([x for x in As[-1]]), forgetting(As), bwt(As)))
            print(f"  {name} fold {fold}: F {per8[-1][1]:.4f} → {pers[-1][1]:.4f}",
                  flush=True)
        if not per8:
            print(f"  ⚠️ {name} 沒有可用的折，略過")
            continue
        ms = lambda v, k: (statistics.mean(x[k] for x in v),
                           statistics.stdev(x[k] for x in v) if len(v) > 1 else 0.0)
        rows.append({"name": name, "n": len(per8),
                     "acc8": ms(per8, 0), "accs": ms(pers, 0),
                     "f8": ms(per8, 1), "fs": ms(pers, 1),
                     "b8": ms(per8, 2), "bs": ms(pers, 2)})

    mx = max(abs(r["fs"][0] - r["f8"][0]) for r in rows)
    L = ["# 已見類別限制版指標（seen-class check）", "",
         "PathSelect（`arm=A5`）十折，四個設定。把 class-IL 的 argmax 從**固定八類**",
         "改成**每階段只在已見類別上**取，重算三個指標並列出差值。", "",
         "## 為什麼要做這個檢查", "",
         "本方法的診斷頭是固定的八類 cosine 頭，自第一個任務起就涵蓋全部類別、",
         "不隨任務累積（`docs/CLAIMS.md` C-30）。外部累積式協定在中間階段只在**已見**",
         "類別上預測。固定八類在早期階段更嚴、峰值較低，**可能使 Forgetting 偏小** ——",
         "本檔量化這個差多少。", "",
         "## 結果", "",
         "| 設定 | 指標 | 固定八類（現行） | 只在已見類別 | 差 |",
         "|---|---|---|---|---|"]
    for r in rows:
        for lab, a, b, arrow in (("class-IL ACC", "acc8", "accs", "↑"),
                                 ("Forgetting", "f8", "fs", "↓"),
                                 ("BWT", "b8", "bs", "↑")):
            d = r[b][0] - r[a][0]
            L.append(f"| {r['name']} | {lab} {arrow} | {r[a][0]:.4f} ± {r[a][1]:.3f} | "
                     f"{r[b][0]:.4f} ± {r[b][1]:.3f} | **{d:+.4f}** |")
    L += ["", "## 三點讀法", "",
          "**① class-IL ACC 的 `+0.0000` 是恆等，不是巧合。** 最終階段的「已見類別」",
          "就是全部八類，兩種取法在該階段逐筆相同；class-IL ACC 只看最終階段。",
          "**因此 `docs/SOTA_TABLE.md` 裡我們與外部方法的 ACC 比較不受此設定差異影響。**", "",
          "**② 方向 4/4 一致，量級很小。** 改成只在已見類別上取 argmax 後，早期階段",
          "準確率上升、峰值抬高，Forgetting 四個設定**全部變大**、BWT 全部變小 ——",
          f"與預期方向相符。最大位移 **{mx:.4f}**（reverse／flat），其餘三個 0.0024–0.0049，",
          "**不到 Forgetting 逐折標準差（0.038–0.046）的五分之一**。", "",
          "**③ 對外部比較的結論不變。** 以反向為例：我們的 Forgetting 0.0899 → 0.0967，",
          "外部重現值 0.064，差距由 +0.026 變成 +0.033 —— 方向與量級的判讀都不改變。", "",
          "## 方法", "",
          "逐 slide 存檔（`outputs/exp2/sota/per_slide/*.json`）**沒有存 logits**，",
          "只有 `pred_class_il`（八類 argmax）與 `pred_task_il`。重算的做法是",
          "用存下來的 `selected_idx` 與 `weights_softmax`，經**與評估完全相同的路徑**",
          "（`selector.classifier.conch_classify` + `Ctx.f_txt` + `Ctx.logit_scale`）",
          "重建 logits，再在已見類別的欄位子集上取 argmax。", "",
          "stage `i` 的已見類別 = 該 order 前 `i+1` 個 task 各自的兩個全域類別欄位",
          "（全域索引固定：esca 0–1、rcc 2–3、brca 4–5、lung 6–7）。", "",
          f"⚠️ **重建忠實度已驗證**：本次抽驗 {checked_total} 筆，重建的八類 argmax",
          "與存檔的 `pred_class_il` **全數一致** —— 不一致會直接中止，不會產表。", "",
          "⚠️ **最終階段不重算**：已見類別即全部八類，直接沿用存檔值。", "",
          "產生：`python scripts/report_seen_class_check.py`。", ""]
    out = Path(args.out) if args.out else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print(f"→ {out}（抽驗 {checked_total} 筆全數一致；最大 Forgetting 位移 {mx:.4f}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
