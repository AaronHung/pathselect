#!/usr/bin/env python3
"""S4 smoke — reverse order 四個 task 跑完整條 CL pipeline，不落任何正式結果檔。

驗的是「接得起來、數字合理」，不是任何實驗結論：
  - 每個 task 結束後的 |M|
  - LoRA merge 前後的參數範數（ΔW 與 base W）
  - 三項 CL loss 的數值

    python scripts/smoke_cl.py --epochs 1 --max-train 6
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from selector.evaluate import read_slide, slide_dataset                  # noqa: E402
from selector.grouping import assign_groups, tissue_text_features        # noqa: E402
from selector.lora import (apply_lora, base_norm, delta_norm,            # noqa: E402
                           lora_parameters, merge_lora, n_parameters,
                           PerTaskLoRABank)
from selector.memory import MEMORY_CAPACITY, SelectionMemory             # noqa: E402
from selector.model import GroupSelector, PatchSelector                  # noqa: E402
from selector.priors import MAINLINE_PRIOR                               # noqa: E402
from selector.text_encoder import build_f_txt, load_config               # noqa: E402
from selector.train import (continual_terms, fill_memory, total_loss,    # noqa: E402
                            train_step)

ORDER = ["tcga_esca", "tcga_rcc", "tcga_brca", "tcga_lung"]
L3B = dict(use_query=False, use_state=False, hierarchy=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-train", type=int, default=6)
    ap.add_argument("--mem-slides", type=int, default=40)
    ap.add_argument("--replay-k", type=int, default=2)
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=1)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--beta-s", type=float, default=0.1)
    ap.add_argument("--beta-u", type=float, default=0.1)
    ap.add_argument("--lambda-kd", type=float, default=1.0)
    ap.add_argument("--lambda-eq", type=float, default=1.0)
    ap.add_argument("--lambda-replay", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    cfg = load_config()
    f_txt = torch.cat([build_f_txt(t, cfg).f_txt for t in cfg["tasks"]], 0)
    logit_scale = build_f_txt(cfg["tasks"][0], cfg).logit_scale
    tissue = tissue_text_features(cfg)

    f_g, f_p = GroupSelector(), PatchSelector()
    apply_lora(f_g, r=args.rank); apply_lora(f_p, r=args.rank)
    memory = SelectionMemory()
    bank = PerTaskLoRABank()
    rng = random.Random(args.seed)

    print(f"B={args.budget} c={args.chunk} rank={args.rank} epochs={args.epochs} "
          f"max_train={args.max_train} |M|<= {MEMORY_CAPACITY}")
    print(f"參數總量（含 LoRA）{n_parameters(f_g, f_p):,}；"
          f"只訓練 LoRA {sum(p.numel() for p in lora_parameters(f_g, f_p)):,}")
    print(f"λ_kd={args.lambda_kd} λ_eq={args.lambda_eq} λ_r={args.lambda_replay}  "
          f"beta_s={args.beta_s} beta_u={args.beta_u}\n")

    for stage, task in enumerate(ORDER):
        print(f"── stage {stage}: {task}")
        opt = torch.optim.Adam(lora_parameters(f_g, f_p), lr=args.lr,
                               weight_decay=1e-4)
        ds, shift = slide_dataset(cfg, task, cfg["tasks"].index(task), "train")
        n = min(args.max_train, len(ds)) if args.max_train > 0 else len(ds)
        last_parts = {}
        for epoch in range(args.epochs):
            for i in range(n):
                rec = read_slide(ds, shift, i)
                grp = assign_groups(rec.Z, tissue)
                l_ev, ev_parts, _res = train_step(
                    rec.Z, rec.label, torch.zeros(512), f_txt, logit_scale, f_g, f_p,
                    grouping=grp, budget=args.budget, chunk=args.chunk,
                    prior_kind=MAINLINE_PRIOR, beta_s=args.beta_s,
                    beta_u=args.beta_u, **L3B)

                kd = eq = replay = None
                if len(memory):                      # 第一個 task 時 M 還是空的
                    for entry in memory.sample(args.replay_k, rng):
                        k, e, r = continual_terms(
                            entry, cfg, (f_g, f_p), f_txt, logit_scale, tissue,
                            budget=args.budget, chunk=args.chunk, spec=L3B)
                        kd = k if kd is None else kd + k
                        eq = e if eq is None else eq + e
                        replay = r if replay is None else replay + r
                loss, cl_parts = total_loss(l_ev, kd, eq, replay,
                                            lambda_kd=args.lambda_kd,
                                            lambda_eq=args.lambda_eq,
                                            lambda_replay=args.lambda_replay)
                opt.zero_grad(); loss.backward(); opt.step()
                last_parts = {**ev_parts, **cl_parts, "L_total": float(loss.detach())}

        d_before, b_before = delta_norm(f_g, f_p), base_norm(f_g, f_p)
        bank.snapshot(task, f_g, f_p)                 # oracle upper bound 用，不是主方法
        merge_lora(f_g, f_p)
        d_after, b_after = delta_norm(f_g, f_p), base_norm(f_g, f_p)

        added = fill_memory(memory, (f_g, f_p), task, cfg, f_txt, logit_scale, tissue,
                            budget=args.budget, chunk=args.chunk, spec=L3B,
                            max_slides=args.mem_slides)

        fmt = lambda v: "—" if v is None else f"{v:.4f}"
        print(f"   loss  L_diag={fmt(last_parts.get('L_diag'))}  "
              f"L_sem={fmt(last_parts.get('L_sem'))}  "
              f"L_util={fmt(last_parts.get('L_util'))}  "
              f"| L_KD={fmt(last_parts.get('L_KD'))}  "
              f"L_eq={fmt(last_parts.get('L_eq'))}  "
              f"L_replay={fmt(last_parts.get('L_replay'))}  "
              f"→ L_total={fmt(last_parts.get('L_total'))}")
        print(f"   LoRA  merge 前 ‖ΔW‖={d_before:.6f} ‖W‖={b_before:.4f}   "
              f"merge 後 ‖ΔW‖={d_after:.6f} ‖W‖={b_after:.4f}")
        print(f"   記憶體 新增 {added} 筆 → |M|={len(memory)} "
              f"(n_seen={memory.n_seen}, tasks={[t.replace('tcga_', '') for t in memory.tasks()]})")
        print(f"   參數總量 {n_parameters(f_g, f_p):,}  "
              f"per-task LoRA bank {len(bank)} 份（oracle 用）\n")

    print("smoke 完成：沒有寫出任何結果檔。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
