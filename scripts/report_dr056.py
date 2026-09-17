#!/usr/bin/env python3
"""DR-056 第 7 步報表：outputs/exp4/DR056_PROBE.md（只讀 per_slide 與 logs/exp4）。

四個 probe run（A5、hier、accumulating、`--uold current`、B=8、fold 1、seed 1、|M|=128）：
  run 1  讀整張  reverse   run 2  只讀候選  reverse
  run 3  讀整張  forward   run 4  只讀候選  forward

⚠️ 判準在看到數字之前就凍結（PI 2026-09-17），本腳本不得改動 GATES。
⚠️ 本批與 outputs/exp3 不同機器且 baseline 未逐位元對齊 —— exp3 只能當參考，
   **不得與本批相減**。所有配對都在本批四個 run 之間。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_exp2 import ORDERS                                          # noqa: E402
from sota.metrics import all_metrics                                 # noqa: E402
from sota.report_sota import load_runs                               # noqa: E402

#: PI 於 2026-09-17 凍結；看到數字後不得調整。單位皆為百分點（r256 − rfull）。
GATES = {"acc_floor_pp": -2.0, "masked_floor_pp": -2.0, "forgetting_ceiling_pp": 3.0}
ORDER_LABEL = {"reverse": "reverse", "main": "forward（--order main）"}


def metrics_of(src: Path, order: str):
    if not src.is_dir():
        return None
    runs = load_runs(src, list(ORDERS))
    key = ("A5", "hier", order)
    if key not in runs:
        return None
    per = []
    for _rk, recs in sorted(runs[key].items()):
        try:
            per.append(all_metrics(recs, ORDERS[order]))
        except ValueError:
            pass
    if len(per) != 1:                       # probe 是單折單 seed，多於一筆代表目錄混了
        return {"_error": f"預期 1 個 run，實得 {len(per)}"}
    return per[0]


def wall_clock(logdir: Path, name: str):
    s, e = logdir / f"{name}.start", logdir / f"{name}.end"
    if not (s.is_file() and e.is_file()):
        return None
    return int(e.read_text().strip()) - int(s.read_text().strip())


def fmt(v, nd=4):
    return "—" if v is None else f"{v:.{nd}f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", default=str(ROOT / "outputs" / "exp4"))
    ap.add_argument("--logs", default=str(ROOT / "logs" / "exp4" / "b2"))
    ap.add_argument("--suffix", default="_b2",
                    help="tag 後綴；第二批是 _b2（第一批的 run 2／run 4 因側倉互相"
                         "覆寫而作廢，見 docs/ledger/DR-056.md）")
    ap.add_argument("--store-json", default=None, help="measure_dr056_store.py 的輸出")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    src, logs = Path(a.src), Path(a.logs)
    out = Path(a.out) if a.out else src / "DR056_PROBE.md"

    M = {(cfg, order): metrics_of(src / f"cand_m128_{cfg}{a.suffix}" / "per_slide", order)
         for cfg in ("rfull", "r256") for order in ("reverse", "main")}
    W = {n: wall_clock(logs, n) for n in
         ("run1_rfull_rev", "run2_r256_rev", "run3_rfull_fwd", "run4_r256_fwd")}

    L = ["# DR-056 第 7 步 — 候選級 replay 的 probe（|M| = 128、fold 1、seed 1）", "",
         "A5、hier、`--head accumulating`、`--uold current`、B = 8、c = 1、epochs 5。",
         "四個 run **同批同機**（見 [MACHINE.md](MACHINE.md)），四路平行、每 run 8 執行緒。", "",
         "⚠️ 這是**第二批**。第一批的兩個候選級 run 因共用側倉目錄互相覆寫候選特徵而",
         "作廢（證據與修法見 [DR-056](../../docs/ledger/DR-056.md) 的 2026-09-18 補記）。", "",
         "⚠️ **這不是「同方法換 buffer」，而是換了 replay 的定義**（DR-056）：分組原型改由",
         "候選子集計算，配額隨之改變。差值不得寫成無損壓縮的誤差。", "",
         "⚠️ **exp3 的數字只能當參考，不得與本批相減** —— 不同機器，baseline 未逐位元對齊。", "",
         "## 主表", "",
         "| 順序 | 設定 | ACC（class-IL） | Masked ACC（task-IL） | Forgetting ↓ | BWT | wall-clock |",
         "|---|---|---|---|---|---|---|"]
    NAME = {("rfull", "reverse"): "run1_rfull_rev", ("r256", "reverse"): "run2_r256_rev",
            ("rfull", "main"): "run3_rfull_fwd", ("r256", "main"): "run4_r256_fwd"}
    LAB = {"rfull": "讀整張（對照）", "r256": "只讀候選"}
    for order in ("reverse", "main"):
        for cfg in ("rfull", "r256"):
            m = M[(cfg, order)]
            w = W[NAME[(cfg, order)]]
            wtxt = "—" if w is None else f"{w:,} s = {w/60:.1f} min"
            if m is None:
                L.append(f"| {ORDER_LABEL[order]} | {LAB[cfg]} | 尚未 | | | | {wtxt} |"); continue
            if "_error" in m:
                L.append(f"| {ORDER_LABEL[order]} | {LAB[cfg]} | ⚠️ {m['_error']} | | | | {wtxt} |"); continue
            L.append(f"| {ORDER_LABEL[order]} | {LAB[cfg]} | {fmt(m['acc'])} | {fmt(m['masked_acc'])} | "
                     f"{fmt(m['forgetting'])} | {fmt(m['bwt'])} | {wtxt} |")

    L += ["", "## 配對差值（只讀候選 − 讀整張，同批同機、逐折相減）", "",
          "| 對照 | 順序 | ΔACC (pp) | ΔMasked (pp) | ΔForgetting (pp) | Δwall-clock |",
          "|---|---|---|---|---|---|"]
    deltas = {}
    for order, lab, na, nb in (("reverse", "run 2 − run 1", "run2_r256_rev", "run1_rfull_rev"),
                               ("main", "run 4 − run 3", "run4_r256_fwd", "run3_rfull_fwd")):
        ra, rb = M[("r256", order)], M[("rfull", order)]
        if not ra or not rb or "_error" in ra or "_error" in rb:
            L.append(f"| {lab} | {ORDER_LABEL[order]} | 尚未 | | | |"); continue
        d = {k: (ra[k] - rb[k]) * 100 for k in ("acc", "masked_acc", "forgetting")}
        deltas[order] = d
        wa, wb = W[na], W[nb]
        dw = "—" if wa is None or wb is None else f"{wa-wb:+,} s（{(wa-wb)/60:+.1f} min）"
        L.append(f"| {lab} | {ORDER_LABEL[order]} | {d['acc']:+.2f} | {d['masked_acc']:+.2f} | "
                 f"{d['forgetting']:+.2f} | {dw} |")

    L += ["", "## 判準（PI 於 2026-09-17 凍結，先於看到數字）", ""]
    if len(deltas) == 2:
        checks = []
        for order in ("reverse", "main"):
            d = deltas[order]
            checks.append((f"ACC ≥ {GATES['acc_floor_pp']:+.1f} pp（{ORDER_LABEL[order]}）",
                           d["acc"] >= GATES["acc_floor_pp"], f"{d['acc']:+.2f} pp"))
            checks.append((f"Masked ≥ {GATES['masked_floor_pp']:+.1f} pp（{ORDER_LABEL[order]}）",
                           d["masked_acc"] >= GATES["masked_floor_pp"], f"{d['masked_acc']:+.2f} pp"))
            checks.append((f"Forgetting ≤ {GATES['forgetting_ceiling_pp']:+.1f} pp（{ORDER_LABEL[order]}）",
                           d["forgetting"] <= GATES["forgetting_ceiling_pp"], f"{d['forgetting']:+.2f} pp"))
        sr, sf = deltas["reverse"]["acc"], deltas["main"]["acc"]
        consistent = not ((sr > 0 and sf < 0) or (sr < 0 and sf > 0))
        checks.append(("方向一致性：reverse 與 forward 的 ΔACC 不得一好一壞",
                       consistent, f"reverse {sr:+.2f} pp / forward {sf:+.2f} pp"))
        L += ["| 判準 | 實測 | 結果 |", "|---|---|---|"]
        for name, ok, val in checks:
            L.append(f"| {name} | {val} | {'✅ 通過' if ok else '❌ 未過'} |")
        allok = all(ok for _n, ok, _v in checks)
        L += ["", f"**整體：{'✅ 全數通過' if allok else '❌ 有判準未過'}**", "",
              "⚠️ **fold 1 單折、無變異數** —— probe 只能排除崩掉，**不能確認有效**。",
              "要宣稱效果必須擴到完整十折／二十折，且對照組與實驗組同批同機重跑。"]
    else:
        L.append("尚未有完整的四個 run。")

    # ── 同機決定性核對：兩批的 rfull 應逐位元相同（rfull 從不碰側倉）──
    L += ["", "## 同機決定性核對（第一批 vs 第二批的「讀整張」）", "",
          "「讀整張」的路徑從不碰側倉，兩批之間唯一的差別是執行時間。同一台機器、",
          "同一份程式、同一個 seed，per_slide 應**逐位元相同**；若不同，代表這台機器",
          "上的結果不可重現，整個配對比較都要重新檢討。", "",
          "| 順序 | 第一批檔案 | 第二批檔案 | 逐位元相同？ |", "|---|---|---|---|"]
    import hashlib
    for order in ("reverse", "main"):
        name = f"A5_{order}_seed1_M128_hier_acc_ucur.json"
        f1 = src / "cand_m128_rfull" / "per_slide" / name
        f2 = src / f"cand_m128_rfull{a.suffix}" / "per_slide" / name
        if not (f1.is_file() and f2.is_file()):
            L.append(f"| {ORDER_LABEL[order]} | {'有' if f1.is_file() else '無'} | "
                     f"{'有' if f2.is_file() else '無'} | 尚未 |")
            continue
        h1 = hashlib.sha256(f1.read_bytes()).hexdigest()
        h2 = hashlib.sha256(f2.read_bytes()).hexdigest()
        same = h1 == h2
        if same:
            note = f"✅ 相同（sha256 {h1[:12]}…）"
        else:
            r1 = json.loads(f1.read_text()); r2 = json.loads(f2.read_text())
            diff = sum(1 for x, y in zip(r1, r2)
                       if x.get("selected_idx") != y.get("selected_idx"))
            note = f"❌ 不同（{diff}/{min(len(r1), len(r2))} 筆 selected_idx 有差）"
        L.append(f"| {ORDER_LABEL[order]} | {f1.name} | {f2.name} | {note} |")

    L += ["", "## 資料量實測", ""]
    if a.store_json and Path(a.store_json).is_file():
        S = json.loads(Path(a.store_json).read_text())
        sd, fl, r = S["side_store"], S["full_slide"], S["ratio"]
        L += ["候選級 replay 的每一步只開**一個**側倉檔；現行 replay 的每一步開該 slide 的",
              "**完整特徵檔**。下表是同一批 slide 的逐筆配對實測。", "",
              "| 量 | 側倉（只讀候選） | 完整特徵檔（讀整張） | 倍率 |", "|---|---|---|---|",
              f"| 單一 replay 步驟 min | {sd['min']:,} B | {fl['min']:,} B | {r['min']:.1f}× |",
              f"| 單一 replay 步驟 mean | {sd['mean']:,.0f} B | {fl['mean']:,.0f} B | {r['mean']:.1f}× |",
              f"| 單一 replay 步驟 max | {sd['max']:,} B | {fl['max']:,} B | {r['max']:.1f}× |",
              f"| 整段訓練所需總量（{sd['n']} 筆） | **{sd['total_GiB']:.4f} GiB** | {fl['total_GiB']:.4f} GiB | {r['total']:.1f}× |",
              "",
              "「整段訓練所需總量」= |M| = 128 的側倉全部檔案大小總和 —— 候選級 replay 在",
              "**整個訓練期間**對舊任務資料的需求就只有這些位元組。",
              "（當前任務的特徵檔仍然要讀：那是 forward pass 的輸入，不是 replay 的需求。",
              "第 8 步以「把沒被任何 snapshot 指到的舊 task 特徵檔設為不可讀」直接驗證這一點。）"]
    else:
        L.append("尚未量測（執行 `scripts/measure_dr056_store.py`）。")

    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
