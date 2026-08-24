"""G5 / G4 的前置檢查 —— 元件打開後必須真的改變輸出。

PI 原文（PROMPT G345-ARCH-COMPLETENESS-20260824）：

    ⚠️ 先做一次 no-op 檢查：state 開啟後，c=1 八輪與 c=8 一輪的選取集合必須不同。
       若仍相同 → 停下來回報，代表 state 沒有真的進入計算。

對照組是 state OFF（現行主線），已知為 no-op（CLAIMS C-01）。兩組跑同一批
synthetic slide 與同一組初始化，唯一差異是 use_state。

**G4 的對應檢查**：`use_query=False` 的實作是把 query 欄位**填零**
（`selector/model.py:44`），而 `run_exp2.Ctx.q0` 是 `torch.zeros(512)` ——
所以只把 use_query 打開、不接真正的 q_tau，結果會與關閉時**位元相同**，
G4 會是由構造保證的 null（憲法 §2.5）。這裡把兩件事都測出來：
零向量 query（等於沒接線）vs 真正的 task query。

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


@torch.no_grad()
def query_trial(seed: int, q_tau):
    """比較 use_query=False 與 use_query=True + 給定的 q_tau。"""
    Z, grp = make_slide(seed)
    torch.manual_seed(1000 + seed)
    f_g, f_p = make_models()
    f_g.eval(); f_p.eval()

    def sel(use_query):
        res = run_rounds(Z, grp, q_tau, f_g, f_p, budget=BUDGET, chunk=1,
                         allocation="per_budget", use_query=use_query,
                         use_state=False, hierarchy=True)
        return res.selected.tolist()

    a, b = sel(False), sel(True)
    return {"same_set": set(a) == set(b), "same_order": a == b}


def run_query(q_tau, label: str):
    t = [query_trial(s, q_tau) for s in range(N_TRIALS)]
    return {"query": label, "n_trials": N_TRIALS,
            "same_set": sum(x["same_set"] for x in t),
            "same_order": sum(x["same_order"] for x in t),
            "is_no_op": all(x["same_set"] for x in t)}


def run(use_state: bool):
    t = [trial(s, use_state) for s in range(N_TRIALS)]
    return {"use_state": use_state, "n_trials": N_TRIALS,
            "same_set": sum(x["same_set"] for x in t),
            "same_order": sum(x["same_order"] for x in t),
            "is_no_op": all(x["same_set"] for x in t)}


def main() -> int:
    off, on = run(False), run(True)

    # G4：零向量 query（= run_exp2 現行的 ctx.q0）vs 真正的 task query
    torch.manual_seed(4242)
    real_q = F.normalize(torch.randn(D), dim=-1)
    q_zero = run_query(torch.zeros(D), "zeros(512)（run_exp2 現行 ctx.q0）")
    q_real = run_query(real_q, "真正的 task query（非零）")

    verdict = ("PASS" if not on["is_no_op"] else "FAIL")
    q_verdict = ("PASS" if (q_zero["is_no_op"] and not q_real["is_no_op"])
                 else "FAIL")
    out = {
        "prompt_id": "G345-ARCH-COMPLETENESS-20260824",
        "criterion": "state 開啟後，c=1 八輪與 c=8 一輪的選取集合必須不同",
        "config": {"n_trials": N_TRIALS, "n_patch": N, "budget": BUDGET,
                   "arch": "hier", "allocation": "per_budget",
                   "note": "synthetic slide + 未訓練的隨機初始化模型"},
        "state_off": off, "state_on": on, "verdict": verdict,
        "query_zero": q_zero, "query_real": q_real, "query_verdict": q_verdict,
        "query_note": ("use_query=False 的實作是把 query 欄位填零"
                       "（selector/model.py:44），而 run_exp2.Ctx.q0 = zeros(512)。"
                       "因此只打開 use_query 而不接真正的 q_tau，結果與關閉時位元"
                       "相同 —— G4 必須由 run_arch_completeness.wire_task_queries "
                       "接上 TaskQueryBank 才成立。"),
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
    for r in (q_zero, q_real):
        print(f"  query={r['query']:38s} 集合相同 {r['same_set']}/{N_TRIALS}"
              f"  → {'no-op' if r['is_no_op'] else '非 no-op'}")
    print(f"判定：G5 {verdict}、G4 接線 {q_verdict}  → {OUT}")
    return 0 if verdict == "PASS" and q_verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
