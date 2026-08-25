"""G5 的訓練後 no-op 重測（PI 裁定 2 的未兌現承諾）。

`check_state_noop.py` 用的是**未訓練的隨機權重 + synthetic slide**，得到
「16/20 集合相同」。那是**下界**，不是效果量 —— state 對選取的影響取決於
F_g / F_p 學到多重視 e_t 與 B_tilde，隨機權重下的敏感度沒有理由代表訓練後的。

本檔用**訓練後的 G5 模型**在**真實 test slide** 上重做同一個比較。

⚠️ 這不會改變 G5 的落判（FAIL 已定，DR-043），只是把承諾補完。

## 為什麼可以跳過 evaluate

`run_arm` 每個 stage 都會跑一次完整評估，佔掉大半時間。評估在 `@torch.no_grad`
下執行、不改權重、也不消耗 `run_arm` 的 RNG（`memory.sample` 用的是獨立的
`random.Random(seed)`），因此跳過它得到的模型與正式 G5 跑出來的**應該相同**。
這不是假設 —— 腳本會用存檔的 G5 逐 slide 選取結果**逐筆驗證**，不符即報錯。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch                                                          # noqa: E402

import run_arch_completeness as G                                     # noqa: E402
import run_exp2 as R                                                  # noqa: E402
from selector.rounds import run_rounds                                # noqa: E402

SEED = 0
OUT = ROOT / "outputs" / "exp2" / "arch" / "noop_trained.json"
STORED = (ROOT / "outputs" / "exp2" / "arch" / "per_slide"
          / f"A5_reverse_seed{SEED}_hier_state.json")


def train_g5_model():
    """跑一次 G5 的序列訓練（跳過各 stage 的評估），回傳 (ctx, args, models)。"""
    G.inject_arch()
    stash = {}
    orig_eval = R.evaluate

    def skip_eval(ctx, models, task, *a, **kw):
        stash["ctx"], stash["models"] = ctx, models
        return []

    R.evaluate = skip_eval
    argv = ["--arms", "A5", "--arch", "hier_state", "--order", "reverse",
            "--seeds", str(SEED), "--allocation", "per_budget",
            "--tag", "arch_noop_trained", "--no-resume"]
    old_argv = sys.argv
    sys.argv = ["run_exp2.py", *argv]
    try:
        rc = R.main()
    finally:
        sys.argv = old_argv
        R.evaluate = orig_eval
    if rc != 0:
        raise SystemExit(f"訓練失敗 rc={rc}")
    return stash["ctx"], stash["models"]


@torch.no_grad()
def compare(ctx, models, spec, budget=8):
    """對每張真實 test slide 比較 c=1 跑八輪 vs c=8 跑一輪。"""
    f_g, f_p = models
    f_g.eval(); f_p.eval()
    rows = []
    for task in R.ORDERS["reverse"]:
        for i in range(ctx.n_slides(task, "test")):
            rec, grp = ctx.get(task, "test", i)

            def sel(chunk):
                res = run_rounds(rec.Z, grp, ctx.q0, f_g, f_p, budget=budget,
                                 chunk=chunk, allocation="per_budget", **spec)
                return res.selected.tolist()

            a, b = sel(1), sel(budget)
            rows.append({"task": task, "slide_id": rec.sid,
                         "same_set": set(a) == set(b), "same_order": a == b,
                         "n_overlap": len(set(a) & set(b)), "k": len(a),
                         "c1": a})
    return rows


def verify_against_stored(rows) -> dict:
    """用存檔的 G5 逐 slide 選取驗證模型與正式跑出來的一致。"""
    if not STORED.exists():
        return {"checked": 0, "note": "找不到 G5 存檔，未驗證"}
    recs = json.loads(STORED.read_text())
    last = max(r["stage"] for r in recs)
    by = {r["slide_id"]: r["selected_idx"] for r in recs if r["stage"] == last}
    hit = miss = 0
    for r in rows:
        want = by.get(r["slide_id"])
        if want is None:
            continue
        hit += int(list(want) == r["c1"])
        miss += int(list(want) != r["c1"])
    return {"checked": hit + miss, "identical": hit, "different": miss}


def main() -> int:
    print("訓練 G5 模型（hier + state，seed 0，跳過各 stage 評估）…", flush=True)
    ctx, models = train_g5_model()

    spec_on = dict(R.ARCH["hier_state"])
    spec_off = dict(R.ARCH["hier"])
    print("在真實 test slide 上比較 c=1×8 vs c=8×1 …", flush=True)
    rows_on = compare(ctx, models, spec_on)

    ver = verify_against_stored(rows_on)
    print(f"與 G5 存檔比對：{ver}", flush=True)
    if ver.get("checked") and ver.get("different"):
        raise SystemExit(f"❌ 模型與正式 G5 跑出來不一致（{ver['different']} 筆不同）"
                         "—— 跳過評估的假設不成立，停下來")

    # 對照：同一個訓練好的模型，把 state 關掉再比一次（應為 no-op）
    rows_off = compare(ctx, models, spec_off)

    def summarise(rows, label):
        n = len(rows)
        return {"label": label, "n_slides": n,
                "same_set": sum(r["same_set"] for r in rows),
                "same_order": sum(r["same_order"] for r in rows),
                "mean_overlap": (sum(r["n_overlap"] for r in rows) / n) if n else 0.0,
                "k": rows[0]["k"] if rows else 0,
                "is_no_op": all(r["same_set"] for r in rows)}

    on, off = summarise(rows_on, "state ON（G5）"), summarise(rows_off, "state OFF（主線）")
    out = {
        "prompt_id": "G345-VERDICT-20260826",
        "purpose": ("PI 裁定 2：未訓練權重下的數字是下界，以訓練後模型在真實 slide "
                    "的重測為準。本檔不改變 G5 的落判（FAIL 已定，DR-043）。"),
        "config": {"seed": SEED, "arch": "hier_state", "allocation": "per_budget",
                   "budget": 8, "split": "test", "n_slides": on["n_slides"],
                   "model": "G5 序列訓練後（4 個 stage 全跑完）的最終模型"},
        "verification_against_g5_dump": ver,
        "state_on": on, "state_off": off,
        "reading": (f"訓練後在 {on['n_slides']} 張真實 test slide 上，state 開啟時 "
                    f"c=1 八輪與 c=8 一輪的選取集合相同 {on['same_set']}/{on['n_slides']}"
                    f"（未訓練 synthetic 的對應數字是 16/20）。"),
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    for r in (off, on):
        print(f"  {r['label']:18s} 集合相同 {r['same_set']}/{r['n_slides']}"
              f"、順序相同 {r['same_order']}/{r['n_slides']}"
              f"、平均重疊 {r['mean_overlap']:.2f}/{r['k']}")
    print(f"→ {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
