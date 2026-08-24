"""G3 / G4 / G5 —— 架構完整性消融（hier-A5 baseline 之上，一次只開一個變因）。

三個實驗，各 5 個 seed，全部以 `outputs/exp2/hier2` 的 **hier-A5**（per_budget、
reverse、5 seeds）為對照：

    G5  use_state=True      證據狀態真的回饋到下一輪
    G4  use_query=True      task query 條件化
    G3  group L_sem         兩層語意 IB（beta_g=0.1）

⚠️ 本檔**不修改** run_exp2.py / rounds.py / train.py（憲法 §3.4，E1 跑中）。
   G5 / G4 用 runtime injection 往 `run_exp2.ARCH` 加設定；
   G3 用 wrapper 包住 `run_exp2.train_step`，在既有 loss 之上**加一項**。
   beta_g=0 時 wrapper 完全不碰 loss，與 baseline 位元相同（見
   tests/test_arch_completeness.py）。

⚠️ 一次只開一個變因 —— 三個實驗互不疊加。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch                                                          # noqa: E402

import run_exp2 as R                                                  # noqa: E402
from selector.priors import MAINLINE_PRIOR, semantic_prior            # noqa: E402
from selector.sem_loss import group_sem_term                          # noqa: E402

#: G5 / G4 注入的架構設定。baseline hier = 兩者皆 False。
INJECTED_ARCH = {
    "hier_state": dict(use_query=False, use_state=True, hierarchy=True),
    "hier_query": dict(use_query=True, use_state=False, hierarchy=True),
}

#: G3 的 arm。與 A5 完全相同，只是取不同名字讓產物檔名不與 baseline 撞。
G3_ARM = "A5g"

DEFAULT_BETA_G = 0.1
OUT_TAG = "arch"


def inject_arch() -> None:
    """把 G5 / G4 的架構設定注入 run_exp2。

    重複注入相同設定是允許的（同一個 process 內跑兩個實驗）；但若該 key 已存在
    且**內容不同**，代表 run_exp2 那邊已經有同名架構 —— 直接停下來，不覆蓋。
    """
    for name, spec in INJECTED_ARCH.items():
        assert R.ARCH.get(name, spec) == spec, \
            f"{name} 已存在於 run_exp2.ARCH 且設定不同，注入會覆蓋既有設定"
        R.ARCH[name] = spec


def inject_g3_arm() -> None:
    assert "A5" in R.ARMS, "找不到 A5，run_exp2.ARMS 結構已改變"
    spec = dict(R.ARMS["A5"], name="Ours + group-level L_sem（G3）")
    assert R.ARMS.get(G3_ARM, spec) == spec, \
        f"{G3_ARM} 已存在於 run_exp2.ARMS 且設定不同"
    R.ARMS[G3_ARM] = spec


def wrap_train_step(beta_g: float):
    """在 train_step 的 loss 上加 group 層語意錨。回傳被包過的函式。

    group prior 與 patch prior 用**同一個** kind / logit_scale，只是輸入從
    patch embedding 換成 group prototype（`semantic_prior` 對 [N, D] 泛用，
    內部自帶 L2 normalize，所以 prototype 不必先正規化）。

    ⚠️ 只作用在當前 slide 的 evidence loss —— 與 patch 層 L_sem 的作用範圍完全
       相同（L_replay 只有 l_diag，本來就不含 L_sem）。
    """
    orig = R.train_step

    def wrapped(Z, label, q_tau, f_txt, logit_scale, f_g, f_p, **kw):
        loss, parts, res = orig(Z, label, q_tau, f_txt, logit_scale, f_g, f_p, **kw)
        grp = kw.get("grouping")
        if beta_g == 0.0 or grp is None or not bool(grp.mask.any()):
            return loss, parts, res
        r = res.records[-1].r.index_select(0, grp.mask.nonzero().reshape(-1))
        p_g = semantic_prior(grp.prototypes[grp.mask], f_txt,
                             kind=kw.get("prior_kind", MAINLINE_PRIOR),
                             n_candidate_classes=f_txt.shape[0],
                             logit_scale=logit_scale)
        term = group_sem_term(r, p_g)
        parts["L_sem_group"] = float(term.detach())
        return loss + beta_g * term, parts, res

    wrapped.__wrapped_beta_g__ = beta_g
    return wrapped


def check_query_leakage(bank, tasks) -> dict:
    """PI 對 G4 的閘門：同 task 內 q_tau 位元相同、跨 task 不同。"""
    for t in tasks:
        assert torch.equal(bank.get(t), bank.get(t)), f"{t} 的 q_tau 不是決定性的"
    for i, a in enumerate(tasks):
        for b in tasks[i + 1:]:
            assert not torch.equal(bank.get(a), bank.get(b)), \
                f"{a} 與 {b} 的 q_tau 位元相同 —— 任務身分沒有進入 query"
    return {"n_tasks": len(tasks),
            "pairwise_cos": {f"{a}|{b}": round(float(bank.get(a) @ bank.get(b)), 6)
                             for i, a in enumerate(tasks) for b in tasks[i + 1:]}}


def wire_task_queries(cfg):
    """G4：把真正的 q_tau 接上。

    ⚠️ 這是 G4 能不能成立的關鍵 —— `run_exp2.Ctx` 的 `q0` 是 **torch.zeros(512)**，
       所有實驗一律傳這個零向量。只把 use_query=True 打開的話，selector 收到的
       是常數零，q_tau 不帶任何任務資訊，G4 會變成**由構造保證的 null**（憲法 §2.5）。

    四個注入點，都在 run_exp2 的模組全域，呼叫時查找，不必改檔：
      train_stage / evaluate  → 該 stage / 該 task 自己的 q_tau
      fill_memory             → 寫入記憶體時用該 task 的 q_tau
      continual_terms         → 用 **entry.tau** 的 q_tau（舊樣本屬於它自己的 task）
    """
    from selector.task_query import TaskQueryBank

    bank = TaskQueryBank(cfg)
    orig_stage, orig_eval = R.train_stage, R.evaluate
    orig_fill, orig_terms = R.fill_memory, R.continual_terms

    def stage(ctx, arm, models, tasks, seed, args, memory, rng):
        # A5 是 sequential，每個 stage 只訓練一個 task；joint 需要逐 sample 的
        # q_tau，單一 ctx.q0 表達不了 —— 直接擋下，不要靜默用錯的 query。
        assert len(tasks) == 1, f"G4 只支援單一 task 的 stage，收到 {tasks}"
        old = ctx.q0
        ctx.q0 = bank.get(tasks[0])
        try:
            return orig_stage(ctx, arm, models, tasks, seed, args, memory, rng)
        finally:
            ctx.q0 = old

    def evaluate(ctx, models, task, *a, **kw):
        old = ctx.q0
        ctx.q0 = bank.get(task)
        try:
            return orig_eval(ctx, models, task, *a, **kw)
        finally:
            ctx.q0 = old

    def fill_memory(memory, models, task, cfg_, *a, **kw):
        kw.setdefault("q_tau", bank.get(task))
        return orig_fill(memory, models, task, cfg_, *a, **kw)

    def terms(entry, cfg_, *a, **kw):
        kw.setdefault("q_tau", bank.get(entry.tau))
        return orig_terms(entry, cfg_, *a, **kw)

    R.train_stage, R.evaluate = stage, evaluate
    R.fill_memory, R.continual_terms = fill_memory, terms
    return bank


def build_argv(exp: str, seeds: str, extra: list[str]) -> list[str]:
    common = ["--order", "reverse", "--seeds", seeds,
              "--allocation", "per_budget", "--tag", OUT_TAG]
    if exp == "g5":
        return ["--arms", "A5", "--arch", "hier_state", *common, *extra]
    if exp == "g4":
        return ["--arms", "A5", "--arch", "hier_query", *common, *extra]
    if exp == "g3":
        return ["--arms", G3_ARM, "--arch", "hier", *common, *extra]
    raise ValueError(f"unknown exp: {exp}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("exp", choices=("g5", "g4", "g3"))
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--beta-g", type=float, default=DEFAULT_BETA_G)
    args, extra = ap.parse_known_args(argv)

    inject_arch()
    if args.exp == "g4":
        from selector.text_encoder import load_config
        cfg = load_config()
        bank = wire_task_queries(cfg)
        diag = check_query_leakage(bank, list(cfg["tasks"]))
        print(f"G4：q_tau 已接上（run_exp2 預設是 zeros(512)）。leakage 檢查通過，"
              f"{diag['n_tasks']} 個 task 兩兩不同。", flush=True)
        print(f"  兩兩 cos：{diag['pairwise_cos']}", flush=True)
    if args.exp == "g3":
        inject_g3_arm()
        R.train_step = wrap_train_step(args.beta_g)
        print(f"G3：group L_sem 已接上，beta_g={args.beta_g}", flush=True)
        if args.beta_g == 0.0:
            print("  ⚠️ beta_g=0 → wrapper 不改動 loss，這是位元相同性檢查模式", flush=True)

    sub = build_argv(args.exp, args.seeds, extra)
    print(f"→ run_exp2 {' '.join(sub)}", flush=True)
    torch.manual_seed(0)
    # ⚠️ run_exp2.main() 不吃 argv 參數，只讀 sys.argv —— 必須改寫 sys.argv 再呼叫。
    old_argv = sys.argv
    sys.argv = ["run_exp2.py", *sub]
    try:
        return R.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())
