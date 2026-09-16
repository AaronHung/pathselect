#!/usr/bin/env python3
"""DR-052 第 3 步的三批 run 清單（給 scripts/pod_run_exp3.sh 吃）。

    python scripts/pod_exp3_batches.py <1..12>

  B1 累積式・反向十折 hier+flat（20）   B2 固定頭・反向十折（20，同設定對照）
  B3 累積式・正向十折（20）             B4 固定頭・正向十折（20）
  B5 累積式・fold-1 五 seed 六臂（30）  B6 固定頭・同六臂（30）
  7 = B7-1 累積式・反向 A1/A3 十折（20）  8 = B7-2 累積式・正向 A1/A3 十折（20）

每行 = `run 名稱|run_exp2 參數`。所有 run 都 `--out-root outputs/exp3`：累積式落
outputs/exp3/{sota_acc,ablation_acc}/per_slide/（檔名加 `_acc`），固定頭對照落
outputs/exp3/{sota_fixed,ablation_fixed}/per_slide/（與 Mac 的 outputs/exp2 同設定，
但平台是 pod CPU x86 —— 見 DR-052 對數結果）。組件消融沿用 DR-046 的 flat 協定。
"""
from __future__ import annotations

import sys

ROOT_ARGS = "--out-root outputs/exp3"
ACC, FIX = f"--head accumulating {ROOT_ARGS}", f"--head fixed {ROOT_ARGS}"
ABL_ARMS = ("A2", "A3", "A4", "A5", "B1", "B2")


def tenfold(order: str, head_args: str, tag: str, label: str):
    return [(f"A5_{arch}_{label}_f{k}",
             f"--arms A5 --order {order} --arch {arch} --fold {k} --seeds {k} --tag {tag} {head_args}")
            for arch in ("hier", "flat") for k in range(1, 11)]


def ablation(head_args: str, tag: str, label: str):
    return [(f"{arm}_flat_rev_f1_s{s}_{label}",
             f"--arms {arm} --order reverse --arch flat --fold 1 --seeds {s} --tag {tag} {head_args}")
            for arm in ABL_ARMS for s in range(5)]


#: PI 裁定（Prompt 21 第 3 步，方案 1）：累積式與固定頭都在 pod 跑，同平台配對。
BATCHES = {
    1: tenfold("reverse", ACC, "sota_acc", "rev_acc"),        # 累積式・反向十折 hier+flat
    2: tenfold("reverse", FIX, "sota_fixed", "rev_fix"),      # 固定頭・反向十折（同設定對照）
    3: tenfold("main", ACC, "sota_acc", "fwd_acc"),           # 累積式・正向十折
    4: tenfold("main", FIX, "sota_fixed", "fwd_fix"),         # 固定頭・正向十折
    5: ablation(ACC, "ablation_acc", "acc"),                  # 累積式・fold-1 五 seed 六臂
    6: ablation(FIX, "ablation_fixed", "fix"),                # 固定頭・同六臂
    # Prompt 22（DR-052 附錄）：Table 2 鏈的前兩列在累積式頭下重跑，flat、十折。
    # 7 = B7-1 反向（A1、A3 各十折）；8 = B7-2 正向（同）。A5 用 B1/B3 的 flat 十折。
    7: [(f"{arm}_flat_rev_acc_f{k}",
         f"--arms {arm} --order reverse --arch flat --fold {k} --seeds {k} --tag sota_acc {ACC}")
        for arm in ("A1", "A3") for k in range(1, 11)],
    8: [(f"{arm}_flat_fwd_acc_f{k}",
         f"--arms {arm} --order main --arch flat --fold {k} --seeds {k} --tag sota_acc {ACC}")
        for arm in ("A1", "A3") for k in range(1, 11)],
    # DR-054：hinge 的 U_old 以當前 C_t 由 P_old 重算（--uold current）。
    # 9 = 第一階段 fold-1 五 seed A5/B2（10）；10 = 第二階段 A5 flat 十折兩順序（20，過門檻才跑）。
    9: [(f"{arm}_flat_rev_f1_s{s}_ucur",
         f"--arms {arm} --order reverse --arch flat --fold 1 --seeds {s} --tag ablation_ucur {ACC} --uold current")
        for arm in ("A5", "B2") for s in range(5)],
    10: [(f"A5_flat_{lab}_ucur_f{k}",
          f"--arms A5 --order {order} --arch flat --fold {k} --seeds {k} --tag sota_ucur {ACC} --uold current")
         for lab, order in (("rev", "reverse"), ("fwd", "main")) for k in range(1, 11)],
    # Prompt 24（DR-054 第三階段）：兩層選擇器（hier）＋合規效用下限，十折兩順序。
    11: [(f"A5_hier_{lab}_ucur_f{k}",
          f"--arms A5 --order {order} --arch hier --fold {k} --seeds {k} --tag sota_ucur {ACC} --uold current")
         for lab, order in (("rev", "reverse"), ("fwd", "main")) for k in range(1, 11)],
    # DR-055（Prompt 25）：evidence budget sweep。唯一變動是 --budget；因為檔名規約不含
    # budget，B=4 與 B=16 各自用獨立 tag，B=8 沿用 batch 11 的 sota_ucur。
    12: [(f"A5_hier_{lab}_ucur_b{b}_f{k}",
          f"--arms A5 --order {order} --arch hier --fold {k} --seeds {k} --budget {b} "
          f"--tag sota_ucur_b{b} {ACC} --uold current")
         for b in (4, 16) for lab, order in (("rev", "reverse"), ("fwd", "main"))
         for k in range(1, 11)],
}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1 or argv[0] not in {str(k) for k in BATCHES}:
        print(__doc__); return 2
    for name, args in BATCHES[int(argv[0])]:
        print(f"{name}|{args}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
