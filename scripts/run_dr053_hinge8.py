#!/usr/bin/env python3
"""DR-053：累積式頭下，hinge 改用「全 8 類」計算 U_old／U_new，其餘不變（runtime injection，主線零改）。

    python scripts/run_dr053_hinge8.py --arms B2 --seeds 0,1,2,3,4      # fold 1，寫到 outputs/exp3/dr053_hinge8
    python scripts/run_dr053_hinge8.py --trace --arms B2,A5 --seeds 0   # 第 1 項：逐筆核對 C_old 遮罩

機制：累積式頭下 `run_exp2.run_arm` 把 stage 的 C_t 傳給 `fill_memory(class_mask=C_t)`，
快照因此以 C_old 算 u_old 並存 `class_mask_old`；之後 `continual_terms` 用
`entry.class_mask_old` 遮罩 U_new。本檔只把 **fill_memory 收到的 class_mask 換成 None**：
u_old 以 8 類計、`class_mask_old=None` → U_new 也不遮罩（8 類）。L_diag、L_sem、
u_i teacher、評估的 argmax 仍照 `--head accumulating`（由 train_step / evaluate 的
class_mask 控制，本檔不動）。

生效性實測：注入後用合成資料呼叫 fill_memory 一次，entry.class_mask_old 必須為 None；
未注入時同一呼叫必須為 bool[8]。

--trace（DR-053 第 1 項）：包住 `continual_terms`，逐筆記錄 entry.class_mask_old、
U_new 實際使用的遮罩（攔截 mask_logits）、以及由 entry.tau 推得的「應有 C_old」，
三者逐筆比對；輸出 outputs/exp3/dr053/TRACE_<arm>_seed<s>.json。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_exp2 as R                                                     # noqa: E402
import selector.train as T                                               # noqa: E402
from selector.memory import SelectionMemory                              # noqa: E402

OUT_TRACE = ROOT / "outputs" / "exp3" / "dr053"


# ── 注入：hinge 全 8 類 ──────────────────────────────────────────────────────

def install_hinge8():
    orig = R.fill_memory

    def fill_memory_all8(*a, **kw):
        kw["class_mask"] = None            # u_old 以 8 類計、快照不存 C_old → U_new 不遮罩
        return orig(*a, **kw)

    fill_memory_all8.__wrapped_orig__ = orig
    R.fill_memory = fill_memory_all8
    return fill_memory_all8


def effectiveness_check(ctx) -> str:
    """同一組輸入：注入版 entry.class_mask_old 為 None；原版為 bool[8]。"""
    f_g, f_p = R.new_models(ctx, 0, True, 4)
    task = ctx.label_space[0]
    cm = R.class_mask_for(ctx.label_space, [task])
    spec = {**R.ARCH["flat"], "allocation": "per_budget"}
    mem_a, mem_b = SelectionMemory(capacity=8), SelectionMemory(capacity=8)
    R.fill_memory(mem_a, (f_g, f_p), task, ctx.cfg, ctx.f_txt, ctx.logit_scale, ctx.tissue,
                  budget=8, chunk=1, spec=spec, max_slides=2, class_mask=cm)
    R.fill_memory.__wrapped_orig__(mem_b, (f_g, f_p), task, ctx.cfg, ctx.f_txt, ctx.logit_scale,
                                   ctx.tissue, budget=8, chunk=1, spec=spec, max_slides=2,
                                   class_mask=cm)
    ea, eb = mem_a.sample(1, __import__("random").Random(0))[0], mem_b.sample(1, __import__("random").Random(0))[0]
    assert ea.class_mask_old is None, "注入無效：class_mask_old 不是 None"
    assert eb.class_mask_old is not None and int(eb.class_mask_old.sum()) == 2, "原版行為異常"
    # u_old 口徑不同：注入版 8 類、原版 2 類 → 值必須不同
    assert not torch.equal(ea.u_old, eb.u_old), "u_old 在兩個口徑下相同 —— 注入沒有改到 u_old"
    return "✅ 注入生效：class_mask_old=None、u_old 為 8 類口徑（原版 2 類、值不同）"


# ── 追蹤（第 1 項）────────────────────────────────────────────────────────────

def install_trace(ctx, tasks, log: list):
    orig_terms = T.continual_terms
    orig_mask = T.mask_logits

    def terms(entry, cfg, models, f_txt, logit_scale, tissue, **kw):
        seen = []

        def mask_capture(logits, class_mask):
            seen.append(None if class_mask is None else [int(v) for v in class_mask.tolist()])
            return orig_mask(logits, class_mask)
        T.mask_logits = mask_capture
        try:
            out = orig_terms(entry, cfg, models, f_txt, logit_scale, tissue, **kw)
        finally:
            T.mask_logits = orig_mask
        stage_of_tau = tasks.index(entry.tau)
        expected = [int(v) for v in R.class_mask_for(ctx.label_space, tasks[:stage_of_tau + 1]).tolist()]
        stored = None if entry.class_mask_old is None else [int(v) for v in entry.class_mask_old.tolist()]
        # 第一次 mask_logits 呼叫發生在 U_new（use_eq）；replay 的 l_diag 在其後
        used_for_unew = seen[0] if kw.get("use_eq", True) and seen else None
        log.append({"tau": entry.tau, "stored_C_old": stored, "used_for_U_new": used_for_unew,
                    "expected_C_old": expected, "current_C_t": None if kw.get("class_mask") is None
                    else [int(v) for v in kw["class_mask"].tolist()],
                    "ok": stored == used_for_unew == expected})
        return out
    T.continual_terms = terms
    R.continual_terms = terms


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arms", default="B2")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--tag", default="dr053_hinge8")
    ap.add_argument("--trace", action="store_true", help="第 1 項：逐筆核對，不注入 hinge8")
    ap.add_argument("--extra", default="", help="附加給 run_exp2 的參數（例：--max-train 20 --epochs 1）")
    a = ap.parse_args(argv)
    from selector.text_encoder import load_config
    cfg = load_config(); cfg["fold"] = 1
    ctx = R.Ctx(cfg)
    tasks = R.ORDERS["reverse"]
    argv2 = ["run_exp2.py", "--arms", a.arms, "--order", "reverse", "--arch", "flat", "--fold", "1",
             "--seeds", a.seeds, "--head", "accumulating", "--out-root", "outputs/exp3",
             "--tag", a.tag] + a.extra.split()
    if a.trace:
        log = []
        install_trace(ctx, tasks, log)
        sys.argv = argv2
        rc = R.main()
        OUT_TRACE.mkdir(parents=True, exist_ok=True)
        p = OUT_TRACE / f"TRACE_{a.arms}_seed{a.seeds}.json"
        n_ok = sum(r["ok"] for r in log)
        summary = {"arms": a.arms, "seeds": a.seeds, "n_calls": len(log), "n_ok": n_ok,
                   "by_tau": {t: sum(1 for r in log if r["tau"] == t) for t in tasks},
                   "sample": log[:5]}
        p.write_text(json.dumps(summary, indent=1))
        print(f"trace：continual_terms 呼叫 {len(log)} 筆，stored==used==expected 的筆數 {n_ok} → {p}")
        return 0 if (rc == 0 and n_ok == len(log) and log) else 1
    fn = install_hinge8()
    print(effectiveness_check(ctx), flush=True)
    sys.argv = argv2
    return R.main()


if __name__ == "__main__":
    raise SystemExit(main())
