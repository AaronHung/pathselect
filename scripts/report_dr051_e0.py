#!/usr/bin/env python3
"""DR-051 E0 —— 零訓練、只讀既有產物的三項抽取（Prompt 16）。

    python scripts/report_dr051_e0.py            # 產出三份報表
    python scripts/report_dr051_e0.py --check    # 只做引文行號自檢，不寫檔

E0a  `l_eq_fire_rate` 逐 stage：fold 1 五 seed 所有含 L_eq 的臂並列；另附十折。
E0b  正向順序 4×4 逐任務準確率矩陣（列 = 學完任務 t，欄 = 任務；十折 mean ± sd）、
     hier-full 的**評估期** b_j、hier 首次落後 flat 超過 0.010 的 (stage, task)。
E0c  ΔU 口徑：(i) 現行腳本的定義 (ii) 兩種逐切片平均 (iii) CE 的 logit scale
     (iv) flat 是否存並蒸餾 r_old —— (i)(iii)(iv) 是引文，**行號在產生時逐條自檢**：
     引的那一行必須真的含該片段，否則整支停下。

輸出 `outputs/exp2/dr051/E0A_FIRE_RATE.md`、`E0B_TRAJECTORY.md`、`E0C_DELTA_U.md`。

⚠️ 不自創算法：E0c 的現行口徑直接呼叫 `scripts/report_dr046.py::delta_utility`
與 `run_exp2.arm_metrics`；先自檢能重現 audit C1 的 −220.38／−300.84／−16.82，
不符就停下。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from report_dr046 import delta_utility                               # noqa: E402
from run_exp2 import ORDERS, arm_metrics                             # noqa: E402
from selector.grouping import NUM_GROUPS                             # noqa: E402
from selector.text_encoder import load_config                        # noqa: E402

OUT_DIR = ROOT / "outputs" / "exp2" / "dr051"
EXP2 = ROOT / "outputs" / "exp2"
TRAJ = ROOT / "docs" / "audit_C1_forward_trajectory"
SEEDS = [0, 1, 2, 3, 4]
FOLDS = list(range(1, 11))
#: audit C1 §4／§5 的既有數字：自檢用（容差 5e-3 / 5e-4）
AUDIT_DU_SUM = {"A1": -220.38, "A2": -300.84, "A5": -16.82}
AUDIT_DU_SLIDE = {"A1": -3.796, "A2": -4.705, "A5": -0.743}
AUDIT_FIRE = {1: 0.0169, 2: 0.0305, 3: 0.0740}
LAG_THRESHOLD = 0.010


#: 基準 commit（Prompt 16：「基準 commit 現 HEAD」= Fig.1 v3.0 之後的 HEAD）。
#: 寫死而不取 live HEAD：報表由測試重生時內容才不會隨每次 commit 漂移。
BASE_COMMIT = "a2734ee155e67bf55b142caeb549a89f61ca65f9"


def sha() -> str:
    """回傳基準 commit；若它不是目前 HEAD 的祖先就停下（資料基準錯了）。"""
    r = subprocess.run(["git", "merge-base", "--is-ancestor", BASE_COMMIT, "HEAD"],
                       cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"❌ 基準 commit {BASE_COMMIT[:7]} 不是 HEAD 的祖先")
    return BASE_COMMIT


def load(tag: str, arm: str, order: str, seed: int, suffix: str = "") -> list[dict] | None:
    p = EXP2 / tag / "per_slide" / f"{arm}_{order}_seed{seed}{suffix}.json"
    return json.loads(p.read_text()) if p.exists() else None


def msd(xs) -> str:
    xs = [x for x in xs if x is not None]
    if not xs:
        return "—"
    if len(xs) == 1:
        return f"{xs[0]:.4f}"
    return f"{statistics.mean(xs):.4f} ± {statistics.stdev(xs):.4f}"


# ══ E0a ══════════════════════════════════════════════════════════════════════

#: (顯示名, tag, arm, 檔名後綴)。fold 1、reverse、seeds 0–4。
E0A_ROWS = [
    ("B2  hinge-only（只 L_eq；無 replay、無 KD）", "ablation", "B2", ""),
    ("A5  full（replay + KD + hinge）flat —— audit C1 引用的臂", "main", "A5", ""),
    ("A5  full flat（ablation tag）", "ablation", "A5", ""),
    ("A5  full hier（main tag）", "main", "A5", "_hier"),
    ("A5  full hier（hier2 tag，階層主線）", "hier2", "A5", "_hier"),
    ("A5nG  replay + patch-KD + hinge（無 group-KD）hier", "hier2", "A5nG", "_hier"),
    ("A5H  full，半強度合併", "main", "A5H", ""),
    ("W1  warm-start + full", "main", "W1", ""),
    ("L2  single adapter + full", "main", "L2", ""),
    ("L2  single adapter + full hier", "main", "L2", "_hier"),
]


def fire_by_stage(recs: list[dict]) -> dict[int, tuple[float | None, int]]:
    """stage → (fire_rate, steps)。同一 stage 所有紀錄相同，取第一筆即可。"""
    out = {}
    for r in recs:
        st = r["stage"]
        if st not in out:
            out[st] = (r.get("l_eq_fire_rate"), r.get("l_eq_steps", 0))
    return out


def e0a() -> tuple[str, dict]:
    tasks = ORDERS["reverse"]
    L = ["# E0a — `l_eq_fire_rate` 逐 stage（fold 1、reverse、seeds 0–4）", "",
         f"commit `{sha()}`。來源：`outputs/exp2/<tag>/per_slide/`。",
         "fire rate = 該 stage 的 replay 步驟中 `max(0, U_old − U_new) > 0` 的比例",
         "（記錄點 `scripts/run_exp2.py::train_stage`，欄位 `l_eq_fire_rate`）。",
         "stage 0 記憶體為空、不進 `continual_terms`，無此值。**只列數字。**", ""]
    L += ["| 臂 | stage 1 " + tasks[1] + " | stage 2 " + tasks[2] + " | stage 3 " + tasks[3]
          + " | 逐 seed（s1 / s2 / s3） |", "|---|---|---|---|---|"]
    facts = {}
    for name, tag, arm, suf in E0A_ROWS:
        per_seed = {}
        same_as_main = tag != "main"
        for s in SEEDS:
            recs = load(tag, arm, "reverse", s, suf)
            if recs:
                per_seed[s] = fire_by_stage(recs)
            f_here = EXP2 / tag / "per_slide" / f"{arm}_reverse_seed{s}{suf}.json"
            f_main = EXP2 / "main" / "per_slide" / f"{arm}_reverse_seed{s}{suf}.json"
            if not (f_here.exists() and f_main.exists()
                    and f_here.read_bytes() == f_main.read_bytes()):
                same_as_main = False
        if same_as_main:
            name += "；**檔案與 main tag 逐位元相同，非獨立一跑**"
        if not per_seed:
            L.append(f"| {name} | 缺檔 | | | |")
            continue
        cells, detail = [], []
        for st in (1, 2, 3):
            vals = [per_seed[s][st][0] for s in per_seed if st in per_seed[s]]
            cells.append(msd(vals))
            detail.append("/".join(f"{v:.3f}" for v in vals))
        steps = sorted({per_seed[s][st][1] for s in per_seed for st in (1, 2, 3)
                        if st in per_seed[s]})
        L.append(f"| {name} | {cells[0]} | {cells[1]} | {cells[2]} | "
                 f"{' ; '.join(detail)} |")
        facts[(tag, arm, suf)] = {st: statistics.mean(
            [per_seed[s][st][0] for s in per_seed if st in per_seed[s]]) for st in (1, 2, 3)}
        facts[(tag, arm, suf)]["steps"] = steps
    # 自檢：A5 flat main 必須重現 audit C1 §5
    a5 = facts[("main", "A5", "")]
    for st, want in AUDIT_FIRE.items():
        if abs(a5[st] - want) > 5e-4:
            raise SystemExit(f"❌ E0a 自檢失敗：A5 flat stage {st} = {a5[st]:.4f}，"
                             f"audit C1 = {want}")
    L += ["", f"自檢：A5 flat（main）stage 1/2/3 = "
          f"{a5[1]:.4f} / {a5[2]:.4f} / {a5[3]:.4f}，與 audit C1 §5 的 "
          f"{AUDIT_FIRE[1]} / {AUDIT_FIRE[2]} / {AUDIT_FIRE[3]} 相符（容差 5e-4）✅。",
          f"各 stage 的 replay 步數（五 seed 相同）：{a5['steps']}。", ""]

    # 十折補充（DR-048 SOTA 協定，seed = fold）
    L += ["## 補充：十折（`outputs/exp2/sota/`，seed = fold）", "",
          "| 臂 | order | stage 1 | stage 2 | stage 3 |", "|---|---|---|---|---|"]
    for arch_suf, arch in (("", "flat"), ("_hier", "hier")):
        for order in ("reverse", "main"):
            t = ORDERS[order]
            per = defaultdict(list)
            for k in FOLDS:
                suf = ("" if k == 1 else f"_f{k}") + arch_suf
                recs = load("sota", "A5", order, k, suf)
                if not recs:
                    continue
                fb = fire_by_stage(recs)
                for st in (1, 2, 3):
                    if st in fb and fb[st][0] is not None:
                        per[st].append(fb[st][0])
            n = len(per[1])
            L.append(f"| A5 {arch} | {order}（{' → '.join(x[5:] for x in t)}）n={n} | "
                     f"{msd(per[1])} | {msd(per[2])} | {msd(per[3])} |")
            facts[("sota", "A5", order, arch)] = {st: statistics.mean(per[st]) for st in (1, 2, 3)}
    return "\n".join(L) + "\n", facts


# ══ E0b ══════════════════════════════════════════════════════════════════════

def read_csv(p: Path) -> list[dict]:
    with p.open() as f:
        return list(csv.DictReader(f))


def e0b() -> tuple[str, dict]:
    tasks = ORDERS["main"]                      # 正向：lung → brca → rcc → esca
    short = [t.replace("tcga_", "") for t in tasks]
    mats = {}
    for arch in ("flat", "hier"):
        rows = read_csv(TRAJ / f"A5_{arch}_forward_accuracy_matrix.csv")
        assert len(rows) == 100, f"{arch}: {len(rows)} 列（應為 100）"
        for key in ("class_il", "task_il"):
            m = defaultdict(list)                # (stage, task_idx) → [fold values]
            for r in rows:
                m[(int(r["stage"]), tasks.index(r["eval_task"]))].append(float(r[key]))
            for cell, v in m.items():
                assert len(v) == 10, f"{arch} {key} {cell}: {len(v)} 折"
            mats[(arch, key)] = m
    L = ["# E0b — 正向順序逐任務準確率矩陣（A5 full，十折 mean ± sd）", "",
         f"commit `{sha()}`。來源：`docs/audit_C1_forward_trajectory/*.csv`",
         "（audit C1 §7 自 `outputs/exp2/sota/per_slide/A5_main_*.json` 抽出，未重跑）。",
         "列 = 學完任務 t（stage），欄 = 被評估的任務；下三角。順序 " + " → ".join(short)
         + "。", "**描述性，不做判準。**", ""]
    facts = {}
    for key, label in (("class_il", "class-IL（8-way）"), ("task_il", "task-IL（2-way）")):
        for arch in ("flat", "hier"):
            m = mats[(arch, key)]
            L += [f"## {label} — A5 {arch}", "",
                  "| 學完 ↓ ＼ 任務 → | " + " | ".join(short) + " |",
                  "|---|" + "---|" * 4]
            for st in range(4):
                cells = []
                for j in range(4):
                    v = m.get((st, j))
                    cells.append(msd(v) if v else "—")
                L.append(f"| t={st} {short[st]} | " + " | ".join(cells) + " |")
            L.append("")
        # hier − flat 逐格差（十折平均）與首次落後
        L += [f"### hier − flat（{label}，十折平均之差）", "",
              "| 學完 ↓ ＼ 任務 → | " + " | ".join(short) + " |", "|---|" + "---|" * 4]
        first = None
        for st in range(4):
            cells = []
            for j in range(4):
                if (st, j) not in mats[("flat", key)]:
                    cells.append("—"); continue
                d = (statistics.mean(mats[("hier", key)][(st, j)])
                     - statistics.mean(mats[("flat", key)][(st, j)]))
                wins = sum(h > f for h, f in zip(mats[("hier", key)][(st, j)],
                                                 mats[("flat", key)][(st, j)]))
                cells.append(f"{d:+.4f}（{wins}/10）")
                if first is None and d < -LAG_THRESHOLD:
                    first = (st, j, d)
            L.append(f"| t={st} {short[st]} | " + " | ".join(cells) + " |")
        if first:
            st, j, d = first
            L += ["", f"**hier 首次落後 flat 超過 {LAG_THRESHOLD:.3f}（{label}）**："
                  f"stage t={st}（學完 {short[st]}）、任務 {short[j]}，差 {d:+.4f}。"
                  "（括號 = 十折中 hier > flat 的折數。）", ""]
        else:
            L += ["", f"**{label}：十折平均下 hier 沒有任何一格落後 flat 超過 "
                  f"{LAG_THRESHOLD:.3f}。**", ""]
        facts[key] = first
    # b_j（評估期）
    L += ["## hier-full 的 b_j（**eval-time**：評估時被選中 patch 在 8 組上的分佈，十折平均）",
          "", "⚠️ 訓練期配額未落檔（audit C1 NOT FOUND）；此表為評估期實際落點，"
          "每列 Σ b_j = 8。", "",
          "| 學完 ↓ | 任務 | " + " | ".join(f"b_{j}" for j in range(NUM_GROUPS)) + " | Σ |",
          "|---|---|" + "---|" * (NUM_GROUPS + 1)]
    rows = read_csv(TRAJ / "A5_hier_forward_group_quota.csv")
    assert len(rows) == 100
    q = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for j in range(NUM_GROUPS):
            q[(int(r["stage"]), r["eval_task"])][j].append(float(r[f"b_{j}"]))
    for st in range(4):
        for t in tasks[:st + 1]:
            means = [statistics.mean(q[(st, t)][j]) for j in range(NUM_GROUPS)]
            L.append(f"| t={st} {short[st]} | {t.replace('tcga_', '')} | "
                     + " | ".join(f"{v:.2f}" for v in means)
                     + f" | {sum(means):.2f} |")
    L.append("")
    return "\n".join(L) + "\n", facts


# ══ E0c ══════════════════════════════════════════════════════════════════════

#: (檔, 行號, 該行必須含的片段)。行號**在產生時自檢**，引錯就停。
CITES = {
    "logit_scale_doc": ("selector/text_encoder.py", 12, "logit_scale 直接取自 CONCH checkpoint"),
    "logit_scale_load": ("selector/text_encoder.py", 142, 'blob["logit_scale"]'),
    "ce_def": ("selector/utility.py", 44, "F.cross_entropy(flat, target"),
    "ce_logits": ("selector/utility.py", 71, "logits_cand = logit_scale * (E_cand @ f_txt.t())"),
    "ce_now": ("selector/utility.py", 53, "return logit_scale * (e @ f_txt.t())"),
    "du_def": ("selector/continual.py", 68, "ce = F.cross_entropy(logits_uniform.reshape(1, -1), target)"),
    "head_logits": ("selector/train.py", 67, "return logit_scale * (pooled @ f_txt.to(pooled.dtype).t())"),
    "r_flat": ("selector/rounds.py", 131, "r = f_group.score(grouping.prototypes, q_tau, state_feat,"),
    "r_store": ("selector/train.py", 308, "memory.add(make_entry(task, rec.sid, res.state, last.r, cand,"),
    "r_kd": ("selector/train.py", 337, "kd = l_kd(entry.r_old.to(last.r.dtype), last.r,"),
    "kd_w": ("scripts/run_exp2.py", 233, 'kd_group_weight=spec.get("kd_group_weight", 1.0))'),
    "du_script": ("scripts/report_dr046.py", 166, 'd = [per[t]["sum_u_at_end"] - per[t]["sum_u_at_learn"]'),
    "du_mean": ("scripts/report_dr046.py", 168, "return statistics.mean(d) if d else"),
    "sum_u": ("scripts/run_exp2.py", 620, '"sum_u_at_learn": sum(r["utility_total"] for r in at_i)'),
    "u_total": ("selector/utility.py", 98, "telescope"),
}


def check_cites() -> list[str]:
    bad = []
    for k, (f, n, frag) in CITES.items():
        lines = (ROOT / f).read_text().splitlines()
        if n > len(lines) or frag not in lines[n - 1]:
            bad.append(f"{k}: {f}:{n} 不含「{frag}」")
    return bad


def cite(k: str) -> str:
    f, n, _ = CITES[k]
    return f"`{f}:{n}`"


def per_slide_deltas(recs, arm, seed, tasks):
    """老任務逐切片 ΔU = U_end − U_learn；回傳 {task: [Δ...]}。"""
    last = len(tasks) - 1
    sub = [r for r in recs if r["arm"] == arm and r["seed"] == seed]
    out = {}
    for i, t in enumerate(tasks[:-1]):
        at_i = {r["slide_id"]: r["utility_total"] for r in sub
                if r["stage"] == i and r["task"] == t}
        at_e = {r["slide_id"]: r["utility_total"] for r in sub
                if r["stage"] == last and r["task"] == t}
        out[t] = [at_e[s] - at_i[s] for s in at_i if s in at_e]
    return out


def e0c() -> tuple[str, dict]:
    bad = check_cites()
    if bad:
        raise SystemExit("❌ 引文行號自檢失敗：\n  " + "\n  ".join(bad))
    cfg = load_config()
    label_space = list(cfg["tasks"])
    tasks = ORDERS["reverse"]
    import torch
    ls = float(torch.load(ROOT / "outputs" / "cache" / f"f_txt_{label_space[0]}.pt",
                          weights_only=False)["logit_scale"])
    L = ["# E0c — ΔU 口徑", "", f"commit `{sha()}`。以下引文的行號在產生本檔時逐條自檢"
         "（該行必須含所引片段）。", "",
         "## (i) 現行腳本的口徑", "",
         f"Table 3 的 ΔU 由 `scripts/report_dr046.py::delta_utility` 產生（{cite('du_script')}、"
         f"{cite('du_mean')}）：對每個舊任務取 `sum_u_at_end − sum_u_at_learn`，再對舊任務"
         f"取 `statistics.mean`。而 `sum_u_at_*` 是**該任務 test 切片的 `utility_total` 加總**"
         f"（{cite('sum_u')}）。", "",
         "**答：兩者都不是 —— 是「各任務先對切片加總、再跨任務平均」。**"
         "不是逐切片平均，也不是所有舊任務切片一起平均；片數多的任務（brca 93）主導量級。", "",
         "## (ii) 兩種逐切片平均（fold 1、reverse、seeds 0–4、flat；mean ± sd over seeds）", "",
         "* **S（現行）**：mean_task Σ_slide ΔU",
         "* **M1（各任務先平均再跨任務平均）**：mean_task mean_slide ΔU",
         "* **M2（所有舊任務切片一起）**：mean over all old-task slides of ΔU", "",
         "ΔU 逐切片 = `utility_total`(final stage) − `utility_total`(own stage)；"
         "`utility_total` = 等權證據下 log C − CE（telescoped counterfactual gain，"
         f"{cite('u_total')}）。", "",
         "| 臂 | S 現行 | M1 逐任務平均 | M2 全切片平均 | 逐任務 M1（esca / rcc / brca） |",
         "|---|---|---|---|---|"]
    facts = {}
    for tag, arm, suf, disp in (("main", "A1", "", "A1 flat"), ("main", "A2", "", "A2 flat"),
                                ("main", "A3", "", "A3 flat"), ("main", "A4", "", "A4 flat"),
                                ("main", "A5", "", "A5 flat"),
                                ("main", "A5", "_hier", "A5 hier（補充）")):
        S, M1, M2, per_t = [], [], [], defaultdict(list)
        for s in SEEDS:
            recs = load(tag, arm, "reverse", s, suf)
            if not recs:
                continue
            M = arm_metrics(recs, arm, tasks, s, label_space)
            S.append(delta_utility(M, tasks))
            d = per_slide_deltas(recs, arm, s, tasks)
            M1.append(statistics.mean([statistics.mean(v) for v in d.values()]))
            M2.append(statistics.mean([x for v in d.values() for x in v]))
            for t, v in d.items():
                per_t[t].append(statistics.mean(v))
        if not S:
            L.append(f"| {disp} | 缺檔 | | | |"); continue
        L.append(f"| {disp} | {msd(S)} | {msd(M1)} | {msd(M2)} | "
                 + " / ".join(f"{statistics.mean(per_t[t]):+.3f}" for t in tasks[:-1]) + " |")
        facts[disp] = dict(S=statistics.mean(S), M1=statistics.mean(M1), M2=statistics.mean(M2))
        if arm in AUDIT_DU_SUM and suf == "":
            if abs(statistics.mean(S) - AUDIT_DU_SUM[arm]) > 5e-3:
                raise SystemExit(f"❌ E0c 自檢：{arm} S={statistics.mean(S):.2f} ≠ audit {AUDIT_DU_SUM[arm]}")
            if abs(statistics.mean(M1) - AUDIT_DU_SLIDE[arm]) > 5e-4:
                raise SystemExit(f"❌ E0c 自檢：{arm} M1={statistics.mean(M1):.3f} ≠ audit {AUDIT_DU_SLIDE[arm]}")
    n_te = {t: None for t in tasks[:-1]}
    recs = load("main", "A5", "reverse", 0)
    for t in n_te:
        n_te[t] = len([r for r in recs if r["stage"] == 3 and r["task"] == t])
    L += ["", f"自檢：S 與 M1 的 A1／A2／A5 皆重現 audit C1 §4（−220.38／−300.84／−16.82 與 "
          "−3.796／−4.705／−0.743）✅。",
          f"test 片數 esca {n_te[tasks[0]]}／rcc {n_te[tasks[1]]}／brca {n_te[tasks[2]]}；"
          f"M2 以片數加權（brca 佔 93/184），M1 三任務等權。", "",
          "## (iii) CE 的 temperature／logit scale", "",
          f"* 沒有另設 temperature。所有 CE 都是 `F.cross_entropy` 直接吃 `logit_scale × cos`：",
          f"  * counterfactual gain／`utility_total`：`_ce` 定義於 {cite('ce_def')}；logits 由 "
          f"{cite('ce_logits')}（候選）與 {cite('ce_now')}（當前）產生。",
          f"  * hinge 的 U_new：{cite('du_def')}，其 `logits_uniform` 來自 `frozen_head` "
          f"{cite('head_logits')}。",
          f"* `logit_scale` 取自 CONCH checkpoint（已 exp），不自訂常數（{cite('logit_scale_doc')}；"
          f"載入於 {cite('logit_scale_load')}）。實際值（`outputs/cache/f_txt_*.pt`）= **{ls:.4f}**。", "",
          "## (iv) flat 是否存並蒸餾 r_old", "",
          f"**是。** `r = f_group.score(...)` 在 `run_rounds` 內**不分架構**都會算（{cite('r_flat')}）；"
          f"`fill_memory` 把 `last.r` 存進 entry 作 `r_old`（{cite('r_store')}）；"
          f"`continual_terms` 以 `l_kd(entry.r_old, last.r, ...)` 蒸餾（{cite('r_kd')}），"
          f"group 項係數 `kd_group_weight` 預設 1.0（{cite('kd_w')}），只有 A5nG 臂設 0。",
          "因此 flat 的 L_KD group 項是活的（audit C1 §1：flat 的 F_g 只從此項收梯度）；"
          "但 flat 下 r 不進入選取、也不進入 head（audit C1 §1），故此項不改變任何輸出。",
          ""]
    return "\n".join(L) + "\n", facts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只做引文行號自檢")
    a = ap.parse_args(argv)
    bad = check_cites()
    print("引文自檢：" + ("✅ 全部命中" if not bad else "❌\n  " + "\n  ".join(bad)))
    if bad:
        return 1
    if a.check:
        return 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, fn in (("E0A_FIRE_RATE.md", e0a), ("E0B_TRAJECTORY.md", e0b),
                     ("E0C_DELTA_U.md", e0c)):
        text, facts = fn()
        (OUT_DIR / name).write_text(text)
        print(f"→ {OUT_DIR / name}")
        for k, v in facts.items():
            print(f"   {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
