#!/usr/bin/env python3
"""S3 — Task-incremental 評估（純重算，不重跑任何模型）。

現行評估是 8-way argmax = class-incremental。在 class-IL 下「遺忘」總是看起來像
災難，但其中很大一部分是**跨任務混淆**，而不是「該任務內的鑑別力退化」。
這兩者是不同的科學主張，必須分開報。

重算方式：per_slide JSON 沒有存 8-way logits，但存了 `selected_idx` 與
`weights_softmax`。CONTRACT-4 的 head 是 frozen 且決定性的：

    logits = logit_scale * normalize(Σ_i w_i z_i) @ f_txt.T

z 來自特徵檔（slide_id → 檔案），f_txt 與 logit_scale 都不隨訓練改變，
所以 logits 可以**精確**重建。本檔會先逐筆驗證「重建 logits 的 8-way argmax
== 已存的 pred_softmax」，確認是無損重算後才往下算 task-IL。

不訓練、不呼叫任何 selector 網路。
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import OrderedDict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from selector.classifier import conch_classify                          # noqa: E402
from selector.evaluate import read_slide, slide_dataset                 # noqa: E402
from selector.text_encoder import build_f_txt, load_config              # noqa: E402

SEQFT_DIR = REPO_ROOT / "outputs" / "exp2" / "seqft"
OUT_PATH = SEQFT_DIR / "TASK_IL.md"
ORDERS = {
    "reverse": ["tcga_esca", "tcga_rcc", "tcga_brca", "tcga_lung"],
    "main": ["tcga_lung", "tcga_brca", "tcga_rcc", "tcga_esca"],
}
SLIDE_CACHE = 320


class Slides:
    """test slide 的有界快取（只讀特徵，不碰模型）。"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.tasks = list(cfg["tasks"])
        self._ds, self._lru = {}, OrderedDict()
        self._by_sid = {}

    def _dataset(self, task):
        if task not in self._ds:
            ds, shift = slide_dataset(self.cfg, task, self.tasks.index(task), "test")
            self._ds[task] = (ds, shift)
            self._by_sid[task] = {str(s): i for i, s in enumerate(ds.sids)}
        return self._ds[task]

    def n_patches(self, task) -> list[int]:
        ds, shift = self._dataset(task)
        return [self.get(task, str(s)).Z.shape[0] for s in ds.sids]

    def get(self, task, slide_id):
        key = (task, slide_id)
        if key in self._lru:
            self._lru.move_to_end(key)
            return self._lru[key]
        ds, shift = self._dataset(task)
        rec = read_slide(ds, shift, self._by_sid[task][slide_id])
        self._lru[key] = rec
        if len(self._lru) > SLIDE_CACHE:
            self._lru.popitem(last=False)
        return rec


def load_records() -> list[dict]:
    out = []
    for p in sorted((SEQFT_DIR / "per_slide").glob("*.json")):
        out += json.loads(p.read_text())
    return out


@torch.no_grad()
def rebuild_logits(slides, f_txt, logit_scale, rec) -> torch.Tensor:
    """從 selected_idx + weights_softmax 精確重建 8-way logits。"""
    Z = slides.get(rec["task"], rec["slide_id"]).Z
    idx = torch.tensor(rec["selected_idx"], dtype=torch.long)
    w = torch.tensor(rec["weights_softmax"], dtype=Z.dtype)
    return conch_classify(Z.index_select(0, idx), w, f_txt, logit_scale).reshape(-1)


def owned_rows(task, tasks) -> tuple[int, int]:
    p = tasks.index(task)
    return 2 * p, 2 * p + 1


def jaccard_reference(n_patches: list[int], K: int) -> float:
    """隨機重疊參照：從 n 個 patch 隨機抽兩次 K 個，E[Jaccard] = (K/n)/(2 − K/n)。"""
    vals = []
    for n in n_patches:
        f = min(K, n) / n
        vals.append(f / (2 - f))
    return statistics.mean(vals)


def ms(vals, fmt="{:.4f}"):
    vals = [v for v in vals if v == v]
    if not vals:
        return "—"
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f"{fmt.format(statistics.mean(vals))} ± {fmt.format(sd)}"


def main() -> int:
    cfg = load_config()
    label_space = list(cfg["tasks"])
    f_txt = torch.cat([build_f_txt(t, cfg).f_txt for t in label_space], 0)
    logit_scale = build_f_txt(label_space[0], cfg).logit_scale
    slides = Slides(cfg)

    records = load_records()
    print(f"載入 {len(records)} 筆逐 slide 記錄", flush=True)

    # ── 先驗證重建是無損的 ───────────────────────────────────────────────
    mismatch, checked = 0, 0
    for r in records:
        logits = rebuild_logits(slides, f_txt, logit_scale, r)
        r["_logits"] = logits
        lo, hi = owned_rows(r["task"], label_space)
        r["_pred_class_il"] = int(logits.argmax())
        r["_pred_task_il"] = lo + int(logits[lo:hi + 1].argmax())
        checked += 1
        if r["_pred_class_il"] != r["pred_softmax"]:
            mismatch += 1
    print(f"重建驗證：{checked} 筆，argmax 與已存 pred_softmax 不符 {mismatch} 筆",
          flush=True)
    if mismatch:
        print("⚠️ 重建不是無損的，停止。", flush=True)
        return 1

    seeds = sorted({r["seed"] for r in records})
    K = records[0]["B"]
    L = [
        "# S3 — Task-incremental 評估（從已存的逐 slide 記錄重算）",
        "",
        "SEQFT.md 的評估是 **8-way argmax = class-incremental**。在 class-IL 下"
        "「遺忘」總是看起來像災難，但其中很大一部分是**跨任務混淆**，而不是"
        "「該任務內的鑑別力退化」。這兩者是不同的科學主張，本檔把它們分開。",
        "",
        "**task-IL**：argmax 只在該 task 自己的兩列上取"
        "（esca rows 0-1、rcc 2-3、brca 4-5、lung 6-7），即 2-way，隨機基準 0.5000。",
        "",
        "## 重算的正當性",
        "",
        "per_slide JSON 沒有存 8-way logits，但存了 `selected_idx` 與 "
        "`weights_softmax`。CONTRACT-4 的 head 是 frozen 且決定性的：",
        "",
        "```",
        "logits = logit_scale * normalize(Σ_i w_i z_i) @ f_txt.T",
        "```",
        "",
        "z 來自特徵檔、f_txt 與 logit_scale 不隨訓練改變，所以 logits 可以精確重建。",
        f"**驗證**：{checked} 筆記錄全部重建，8-way argmax 與已存的 `pred_softmax` "
        f"不符 **{mismatch}** 筆 → 重建無損。**沒有重跑任何訓練或 selector 前向。**",
        "",
    ]

    for order_name, tasks in ORDERS.items():
        rs = [r for r in records if r["order"] == order_name]
        if not rs:
            continue
        short = [t.replace("tcga_", "") for t in tasks]
        L += [f"## order = {order_name}　（{' → '.join(short)}）", "",
              "### 表 1-IL：task-IL accuracy matrix（2-way，3 seeds mean ± std）", "",
              "| 學完 | " + " | ".join(f"eval {s}" for s in short) + " |",
              "|---" * (len(short) + 1) + "|"]

        def acc_il(stage, task, seed, key):
            sub = [r for r in rs if r["seed"] == seed and r["stage"] == stage
                   and r["task"] == task]
            return sum(r[key] == r["true"] for r in sub) / len(sub) if sub else float("nan")

        for stage in range(len(tasks)):
            cells = []
            for j, t in enumerate(tasks):
                cells.append("—" if j > stage else
                             ms([acc_il(stage, t, sd, "_pred_task_il") for sd in seeds]))
            L.append(f"| T{stage + 1} {short[stage]} | " + " | ".join(cells) + " |")

        last = len(tasks) - 1
        L += ["", "### 表 2-IL：class-IL vs task-IL 的 A1 forgetting 對照", "",
              "| task | n | class-IL A1 (pp) | task-IL A1 (pp) | "
              "class-IL acc @T4 | task-IL acc @T4 |", "|---|---|---|---|---|---|"]
        for i, t in enumerate(tasks):
            n = len({r["slide_id"] for r in rs if r["task"] == t})
            c_a1 = [(acc_il(i, t, sd, "_pred_class_il")
                     - acc_il(last, t, sd, "_pred_class_il")) * 100 for sd in seeds]
            t_a1 = [(acc_il(i, t, sd, "_pred_task_il")
                     - acc_il(last, t, sd, "_pred_task_il")) * 100 for sd in seeds]
            L.append(f"| {t} | {n} | {ms(c_a1, '{:+.2f}')} | {ms(t_a1, '{:+.2f}')} | "
                     f"{ms([acc_il(last, t, sd, '_pred_class_il') for sd in seeds])} | "
                     f"{ms([acc_il(last, t, sd, '_pred_task_il') for sd in seeds])} |")

        L += ["", "class-IL 隨機基準 0.1250（8 類）、task-IL 隨機基準 0.5000（2 類）。",
              "", "### 表 3-IL：跨任務洩漏率（學完 T4 後）", "",
              "預測落在**別的 task 的類別列**上的比例。這個數字直接量化 class-IL "
              "崩潰裡有多少是「跑錯 task」而非「task 內判錯」。", "",
              "| task | 洩漏率 | 落在自己列且判對 | 落在自己列但判錯 |",
              "|---|---|---|---|"]
        for t in tasks:
            lo, hi = owned_rows(t, label_space)
            leak, right, wrong = [], [], []
            for sd in seeds:
                sub = [r for r in rs if r["seed"] == sd and r["stage"] == last
                       and r["task"] == t]
                if not sub:
                    continue
                out = [r for r in sub if not (lo <= r["_pred_class_il"] <= hi)]
                inside = [r for r in sub if lo <= r["_pred_class_il"] <= hi]
                leak.append(len(out) / len(sub))
                right.append(sum(r["_pred_class_il"] == r["true"] for r in inside) / len(sub))
                wrong.append(sum(r["_pred_class_il"] != r["true"] for r in inside) / len(sub))
            L.append(f"| {t} | {ms(leak)} | {ms(right)} | {ms(wrong)} |")

        # Jaccard 隨機參照
        L += ["", "### Jaccard 的隨機重疊參照值", "",
              "從 n 個 patch 隨機抽兩次 K 個，期望 Jaccard = (K/n) / (2 − K/n)。"
              f"K = {K}，n 取該 task 每張 test slide 的實際 patch 數後平均。", "",
              "| task | 平均 n（patch/slide） | 隨機參照 Jaccard | "
              "SEQFT 實測 Jaccard | 實測 vs 參照 |", "|---|---|---|---|---|"]
        for i, t in enumerate(tasks):
            if i == last:
                continue
            npx = slides.n_patches(t)
            ref = jaccard_reference(npx, K)
            obs = []
            for sd in seeds:
                at_i = {r["slide_id"]: r for r in rs
                        if r["seed"] == sd and r["stage"] == i and r["task"] == t}
                at_e = {r["slide_id"]: r for r in rs
                        if r["seed"] == sd and r["stage"] == last and r["task"] == t}
                vals = []
                for sid, a in at_i.items():
                    if sid not in at_e:
                        continue
                    sa, sb = set(a["selected_idx"]), set(at_e[sid]["selected_idx"])
                    vals.append(len(sa & sb) / len(sa | sb) if (sa | sb) else 1.0)
                if vals:
                    obs.append(statistics.mean(vals))
            o = statistics.mean(obs)
            L.append(f"| {t} | {statistics.mean(npx):.0f} | {ref:.5f} | {o:.5f} | "
                     f"{'**低於**參照' if o < ref else '高於參照'} |")
        L.append("")

    L += ["## 產出", "",
          "本檔由 `python scripts/recompute_task_il.py` 從 "
          "`outputs/exp2/seqft/per_slide/*.json` 重算，與 `SEQFT.md` 並列，"
          "不覆蓋任何既有結果。", ""]
    OUT_PATH.write_text("\n".join(L) + "\n")
    print(f"→ {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
