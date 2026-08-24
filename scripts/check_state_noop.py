"""G5 的前置 no-op 檢查 —— state 打開後，chunked loop 必須不再是 no-op。

PI 原文（PROMPT G345-ARCH-COMPLETENESS-20260824）：

    ⚠️ 先做一次 no-op 檢查：state 開啟後，c=1 八輪與 c=8 一輪的選取集合必須不同。
       若仍相同 → 停下來回報，代表 state 沒有真的進入計算。

對照組是 state OFF（現行主線），已知為 no-op（CLAIMS C-01）。兩組跑同一批
synthetic slide 與同一組初始化，唯一差異是 use_state。

結果寫進 `outputs/exp2/arch/noop_check.json`，報告從那裡讀 —— 不臨時算（憲法 §2.8）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch                                                          # noqa: E402
import torch.nn.functional as F                                       # noqa: E402

from selector.grouping import Grouping                                # noqa: E402
from selector.rounds import run_rounds                                # noqa: E402
from selector.train import make_models                                # noqa: E402

N_TRIALS, N, D, C, J, BUDGET = 20, 300, 512, 8, 8, 8
OUT = ROOT / "outputs" / "exp2" / "arch" / "noop_check.json"


def make_slide(seed: int):
    torch.manual_seed(seed)
    Z = F.normalize(torch.randn(N, D), dim=-1)
    assign = torch.randint(0, J, (N,))
    proto = torch.stack([Z[assign == j].mean(0) if bool((assign == j).any())
                         else torch.zeros(D) for j in range(J)])
    return Z, Grouping(assignment=assign, prototypes=proto,
                       mask=torch.stack([(assign == j).any() for j in range(J)]),
                       sizes=torch.stack([(assign == j).sum() for j in range(J)]))


@torch.no_grad()
def trial(seed: int, use_state: bool):
    Z, grp = make_slide(seed)
    torch.manual_seed(1000 + seed)
    f_g, f_p = make_models()
    f_g.eval(); f_p.eval()

    def sel(chunk):
        res = run_rounds(Z, grp, torch.zeros(D), f_g, f_p, budget=BUDGET,
                         chunk=chunk, allocation="per_budget", use_query=False,
                         use_state=use_state, hierarchy=True)
        return res.selected.tolist()

    a, b = sel(1), sel(BUDGET)
    return {"same_set": set(a) == set(b), "same_order": a == b,
            "n_rounds_c1": BUDGET, "n_selected": len(a)}


def run(use_state: bool):
    t = [trial(s, use_state) for s in range(N_TRIALS)]
    return {"use_state": use_state, "n_trials": N_TRIALS,
            "same_set": sum(x["same_set"] for x in t),
            "same_order": sum(x["same_order"] for x in t),
            "is_no_op": all(x["same_set"] for x in t)}


def main() -> int:
    off, on = run(False), run(True)
    verdict = ("PASS" if not on["is_no_op"] else "FAIL")
    out = {
        "prompt_id": "G345-ARCH-COMPLETENESS-20260824",
        "criterion": "state 開啟後，c=1 八輪與 c=8 一輪的選取集合必須不同",
        "config": {"n_trials": N_TRIALS, "n_patch": N, "budget": BUDGET,
                   "arch": "hier", "allocation": "per_budget",
                   "note": "synthetic slide + 未訓練的隨機初始化模型"},
        "state_off": off, "state_on": on, "verdict": verdict,
        "caveat": ("未訓練權重下 state 的影響偏小（同集合 "
                   f"{on['same_set']}/{N_TRIALS}）。這是下界，不是效果量的估計；"
                   "訓練後應以真實 slide 重做一次。"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    for r in (off, on):
        print(f"  use_state={r['use_state']!s:5s} 集合相同 {r['same_set']}/{N_TRIALS}"
              f"、順序相同 {r['same_order']}/{N_TRIALS}"
              f"  → {'no-op' if r['is_no_op'] else '非 no-op'}")
    print(f"判定：{verdict}  → {OUT}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
