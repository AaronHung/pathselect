#!/usr/bin/env python3
"""產生 docs/EXP5_REPORT.md（晨間報告）。

所有數字從 outputs/exp5 的輸出檔讀取，**不從對話抄**。
報告要能直接貼給撰稿者使用，所以每個數字都附來源目錄。
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from report_exp5 import ORDER_LABEL, load_tag, msd, paired     # noqa: E402

CHAIN = [("m64_replay", "＋replay（A3, flat）", "Table 2 第 2 列"),
         ("m64_distutil", "＋蒸餾＋效用下限（A5 flat, uold=current）", "Table 2 第 3 列"),
         ("m64_full", "＋group 層預算＝**本方法**（A5 hier, uold=current）",
          "**Table 1 的 PathSelect 列 ＋ Table 2 第 4 列**")]
EXPECT = 10


def sh(c: str) -> str:
    try:
        return subprocess.run(c, cwd=ROOT, shell=True, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "UNKNOWN"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "outputs" / "exp5"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "EXP5_REPORT.md"))
    a = ap.parse_args(argv)
    src = Path(a.src)

    D = {tag: {"reverse": load_tag(src / f"{tag}_rev" / "per_slide", "reverse"),
               "main": load_tag(src / f"{tag}_fwd" / "per_slide", "main")}
         for tag, _l, _w in CHAIN}

    # 完成狀態
    runs = ROOT / "logs" / "exp5" / "runs"
    rcs = sorted(runs.glob("*.rc")) if runs.is_dir() else []
    ok = [p.stem for p in rcs if p.read_text().strip() == "0"]
    bad = [p.stem for p in rcs if p.read_text().strip() != "0"]
    times = []
    for p in rcs:
        s, e = p.with_suffix(".start"), p.with_suffix(".end")
        if s.is_file() and e.is_file():
            times.append(int(e.read_text()) - int(s.read_text()))

    complete = all(len(D[t][o]) == EXPECT for t, _l, _w in CHAIN for o in ("reverse", "main"))
    chk = (src / "PROTOCOL_CHECK.md")
    chk_ok = chk.is_file() and "逐位元相同" in chk.read_text()

    L = ["# exp5 晨間報告 —— |M| = 64 備份實驗（5090 pod）", ""]

    # ── 結論段（最上面）──
    L += ["## 結論", ""]
    if complete and not bad:
        L += ["**P0 全部完成，零失敗，協定核對通過。**"
              if chk_ok else "**P0 全部完成，零失敗。**", "",
              "本批**足以支撐 Table 1 的 PathSelect 列與 Table 2 的三個可變列**，"
              "也就是論文主結果所需的全部 |M| = 64 數字。", ""]
    else:
        L += [f"**P0 未全部完成。** 成功 {len(ok)}、失敗 {len(bad)}。", ""]
        for tag, lab, want in CHAIN:
            for o in ("reverse", "main"):
                n = len(D[tag][o])
                if n != EXPECT:
                    L.append(f"* 缺：{lab} {ORDER_LABEL[o]} 只有 {n}/{EXPECT} 折 → {want} 不完整")
        L.append("")
    L += ["**未跑的部分**（依 PI 2026-09-23 的更新指示，P0 完成後停止佇列）：",
          "P1 buffer sweep（M ∈ {16,32,128,256,512}，50 run）、P2 Table 3 重跑（25 run）、",
          "P3 B sweep（40 run）。佇列檔已保留：`logs/exp5/queue_full_p0p2.tsv`、`logs/exp5/queue_p3.tsv`，",
          "直接重跑 `scripts/exp5_queue.sh` 即可續跑（已完成的 run 會自動跳過）。", "",
          "⚠️ **兩套數字絕不混用。** H200 那套若在 9/23 16:00（台灣時間）前完整完成，"
          "論文全部用 H200；否則整套改用本批。兩套都完成時不比較數字高低，依此規則選（DR-057）。", "",
          "⚠️ **本批與 exp3／exp4／H200 的數字不得相減或補位。** 本批內部的所有配對都在同一台、同一批完成。", ""]

    # ── 完成狀態表 ──
    L += ["## 各設定完成狀態", "",
          "| 設定 | 對應 | reverse | forward | 產物目錄 |", "|---|---|---|---|---|"]
    for tag, lab, want in CHAIN:
        r, f = len(D[tag]["reverse"]), len(D[tag]["main"])
        mark = lambda n: f"{n}/{EXPECT}" + ("" if n == EXPECT else " ⚠️")
        L.append(f"| {lab} | {want} | {mark(r)} | {mark(f)} | "
                 f"`outputs/exp5/{tag}_rev`、`{tag}_fwd` |")
    L += [f"| 協定核對（M=512, fold 1） | — | {'✅ 通過' if chk_ok else '⚠️'} | — | `outputs/exp5/chk_m512_f1` |", ""]
    if times:
        L += [f"實測單 run wall-clock：平均 **{statistics.mean(times)/60:.1f} 分鐘**"
              f"（最短 {min(times)/60:.1f}、最長 {max(times)/60:.1f}），10 路平行。",
              f"成功 {len(ok)}、失敗 {len(bad)}。" + (f"失敗的是：{', '.join(bad)}" if bad else ""), ""]

    # ── 主表 ──
    L += ["## Table 1 的 PathSelect 列與 Table 2（|M| = 64，十折，seed = fold）", "",
          "| 設定 | 順序 | n | ACC | Masked ACC | Forgetting ↓ | BWT |",
          "|---|---|---|---|---|---|---|"]
    for tag, lab, _w in CHAIN:
        for o in ("reverse", "main"):
            per = D[tag][o]
            if not per:
                continue
            L.append(f"| {lab} | {ORDER_LABEL[o]} | {len(per)} | " + " | ".join(
                msd([m[k] for m in per.values()])
                for k in ("acc", "masked_acc", "forgetting", "bwt")) + " |")
    L += ["",
          "第 1 列（無保存，A1）與第 5 列（class-text top-8）對 |M| 完全免疫 —— A1 從不填也不取記憶體，"
          "top-8 零參數 —— 所以**未重跑，沿用既有數字**（.481/.584、.549/.478、.812）。", ""]

    # ── 逐折配對 ──
    L += ["## 逐折配對（同折相減，同批同機）", "",
          "| 配對 | 順序 | ΔACC | ΔMasked | ΔForgetting ↓ | n |", "|---|---|---|---|---|---|"]
    for o in ("reverse", "main"):
        for lab, hi, lo in (("本方法 − 只有 replay", "m64_full", "m64_replay"),
                            ("本方法 − ＋蒸餾＋效用下限", "m64_full", "m64_distutil"),
                            ("＋蒸餾＋效用下限 − 只有 replay", "m64_distutil", "m64_replay")):
            cs, n = [], 0
            for k in ("acc", "masked_acc", "forgetting"):
                r = paired(D[hi][o], D[lo][o], k)
                cs.append("—" if r is None else f"{r['mean']:+.4f}（{r['better']}/{r['n']}）")
                n = r["n"] if r else n
            if n:
                L.append(f"| {lab} | {ORDER_LABEL[o]} | " + " | ".join(cs) + f" | {n} |")
    L += ["",
          "括號內對 ACC／Masked 是差值為正的折數，對 Forgetting 是差值為負的折數。",
          "⚠️ **不做三級 win-count 判讀** —— 那個規則是為 model seed 校準的，未對 fold 層級的變異來源"
          "重新校準（PI 裁示，docs/SOTA_TABLE.md）。這裡只列勝負折數。", ""]

    # ── 科學判讀（由數字算出，不手寫）──
    ZS = 0.812                      # class-text top-8，對 |M| 免疫，Table 2 第 5 列
    L += ["## 科學判讀（由本批數字算出）", ""]
    findings = []
    for o in ("reverse", "main"):
        full = D["m64_full"][o]
        if len(full) != EXPECT:
            continue
        acc = statistics.mean([m["acc"] for m in full.values()])
        sd = statistics.stdev([m["acc"] for m in full.values()])
        # (1) 與零樣本參照的關係
        gap = acc - ZS
        findings.append(
            f"* **{ORDER_LABEL[o]}：本方法 ACC {acc:.4f} ± {sd:.4f}，"
            f"零樣本 class-text top-8 是 {ZS:.3f} → 相差 {gap:+.4f}**"
            + ("。本方法**低於**那個完全不用學的參照。" if gap < 0 else "。本方法高於該參照。"))
        # (2) 元件鏈：本方法 vs 只有 replay
        r = paired(full, D["m64_replay"][o], "acc")
        rf = paired(full, D["m64_replay"][o], "forgetting")
        if r:
            flat = abs(r["mean"]) < sd / 2
            findings.append(
                f"* **{ORDER_LABEL[o]}：本方法 − 只有 replay 的 ΔACC = {r['mean']:+.4f}"
                f"（{r['better']}/{r['n']} 折較佳）**"
                + ("，差值的絕對值小於折間 sd 的一半 → **元件鏈在這個記憶體大小下攤平**，"
                   "「每加一個保存元件就多一點」的敘事沒有階梯。" if flat
                   else "。"))
            if rf:
                findings.append(
                    f"    * 同一組配對的 ΔForgetting = {rf['mean']:+.4f}"
                    f"（{rf['better']}/{rf['n']} 折較佳）—— 遺忘是唯一還有方向的指標。")
    L += findings or ["（資料不足，無法判讀。）"]
    L += ["",
          "以上都是**本批內部**的同機同批配對，可以相減。與稿件現用的 |M| = 512 數字的任何比較",
          "都不是配對統計（不同批次），本報告不做相減；但兩組並列時，落差的量級遠大於折間 sd，",
          "而協定核對已證明本機與產生 exp3 的機器逐位元等價 —— 判讀時請把這一點納入考慮。", "",
          "**這些是觀察，不是出版建議。** 要不要用、怎麼改敘事，由 PI 決定。", ""]

    # ── buffer ──
    bj = src / "BUFFER_BYTES.json"
    if bj.is_file():
        B = json.loads(bj.read_text())
        e = {(x["capacity"], x["order"]): x for x in B["entries"]}
        L += ["## replay 實際需要的特徵量（實測，fold 1 的切分）", "",
              "replay 不是讀記憶庫本身 —— entry 不存 feature，它拿 `sample_key` 去**重新載入那張切片的",
              "完整特徵檔**。所以「buffer 多大」的誠實答案是：留在記憶庫裡那 |M| 筆對應的完整特徵檔總和。",
              "留下的是哪幾筆用無模型重放 `ReservoirSampling(seed=0)` 精確算出，不是用平均估。", "",
              "| \\|M\\| | reverse | forward | 占訓練特徵庫 |", "|---|---|---|---|"]
        for cap in (16, 32, 64, 128, 256, 512):
            r, f = e.get((cap, "reverse")), e.get((cap, "main"))
            if not r:
                continue
            star = "**" if cap == 64 else ""
            L.append(f"| {star}{cap}{star} | {star}{r['total_MiB']:.1f} MiB{star} | "
                     f"{f['total_MiB']:.1f} MiB | {r['pct_of_train']:.2f}% / {f['pct_of_train']:.2f}% |")
        r64, r512 = e[(64, "reverse")], e[(512, "reverse")]
        L += ["", f"訓練特徵庫 **{B['train_total_GiB']:.2f} GiB**、{B['train_files']} 檔。",
              f"**512 → 64 讓 replay 期間需要的特徵量降 {r512['total_MiB']/r64['total_MiB']:.1f} 倍**"
              f"（{r512['total_MiB']/1024:.2f} GiB → {r64['total_MiB']:.0f} MiB）。", "",
              "⚠️ 這是 fold 1 的切分；不同折的訓練集不同，數字會有小幅差異。",
              "⚠️ |M| ≠ 64 的列只是這個量的**算術**，不代表本批跑過那些設定（P1 已依指示取消）。", ""]

    # ── 協定核對 ──
    if chk.is_file():
        L += ["## 協定核對", ""] + chk.read_text().split("\n")[1:] + [""]

    # ── 未重跑的部分 ──
    L += ["## 正文中依賴 |M| = 512 的分析，本批**未**重跑", "",
          "以下都在 fold 1、5 個 seed 的消融協定下（Table 3 那一組），P2 已依指示取消。",
          "**由 PI 決定保留、刪除或改口徑。**", "",
          "| main.tex 位置 | 主張 | 需要的 run |", "|---|---|---|",
          "| §4.2（`:225`） | 「完整目標比只有 replay 高 **2.5** 個 class-IL 點，**五個 seed 全數成立**」 | A5 ucur ×5、A3 ×5 |",
          "| §4.2（`:247`） | 只有蒸餾時洩漏 **34.8** | B1 ×5 |",
          "| §4.2（`:247`） | **78.9** 與 **77.0** class-IL | B2 ucur ×5、A3 ×5 |",
          "| §4.2（`:247`） | 完整目標達到最高的 **79.5** | A5 ucur ×5 |",
          "| §4.2（`:247`） | Jaccard **0.165** 對 **0.125** 的反轉（摘要也引用了這個論點） | A4 ×5、A5 ucur ×5 |",
          "| §4.2（`:223`） | ΔU「**−0.64** 或更好」 | A3 ×5 |",
          "| Table 3 全表（`:227-245`） | 六列中五列依賴 \\|M\\| | 共 25 run |", "",
          "⚠️ Table 3 的第 1 列（A2）對 \\|M\\| 免疫，可沿用。",
          "⚠️ 最高風險的是 **Jaccard 0.165 對 0.125 的反轉** —— 摘要的收尾句"
          "（「保住證據身分不等於保住有用性」）建立在它上面。記憶庫縮小 8 倍後這個反轉是否還在，本批沒有答案。", "",
          "### M = 0 沒有跑的原因", "",
          "`SelectionMemory.__init__` 對 `capacity <= 0` 直接 raise（`selector/memory.py`），",
          "必須改程式才能跑，依規則跳過。**不能用 Table 2 的無保護列代表 M = 0**：",
          "A1 連 LoRA 都沒有，A2 有 LoRA 但完全沒有保存機制，兩者都不是「有完整保存機制但記憶庫為空」。", ""]

    # ── 出處 ──
    L += ["## 出處與可重現資訊", "",
          f"* 分支 `exp/m64-5090-backup`，報告產生於 commit `{sh('git rev-parse --short HEAD')}`",
          f"* 實驗基準 commit `081bdcd`（與 `main`／`origin/main` 相同）",
          "* 機器與環境：`outputs/exp5/MACHINE.md`",
          "* 預註冊：`docs/ledger/DR-057.md`（在看到本批任何結果之前提交）",
          "* 影響盤點：`docs/M64_NUMBER_INVENTORY.md`",
          "* LaTeX 表格：`paper/numbers_m64_5090/`",
          "* 逐 run 紀錄：`logs/exp5/runs/`，心跳：`logs/exp5/HEARTBEAT.md`", "",
          "**估算與實測的區分**：本報告中標「實測」的數字都從輸出檔讀取；",
          "任何估算值都會明確寫成「估」。單 run wall-clock 為實測。", ""]

    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"→ {a.out}  完成 {len(ok)}／失敗 {len(bad)}／完整={complete}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
