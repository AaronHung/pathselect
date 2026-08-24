"""G3 / G4 / G5 的 runner —— 注入必須真的生效，且 beta_g=0 必須位元相同。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_arch_completeness as G                                     # noqa: E402
import run_exp2 as R                                                  # noqa: E402
from selector.grouping import Grouping                                # noqa: E402
from selector.train import make_models                                # noqa: E402

N, D, C, J = 300, 512, 8, 8


@pytest.fixture(autouse=True)
def restore():
    """每個測試都還原 run_exp2 的模組狀態 —— 注入是全域副作用。"""
    arch, arms, step = dict(R.ARCH), dict(R.ARMS), R.train_step
    yield
    R.ARCH.clear(); R.ARCH.update(arch)
    R.ARMS.clear(); R.ARMS.update(arms)
    R.train_step = step


def fixture(seed=0):
    torch.manual_seed(seed)
    Z = F.normalize(torch.randn(N, D), dim=-1)
    f_txt = F.normalize(torch.randn(C, D), dim=-1)
    assign = torch.randint(0, J, (N,))
    proto = torch.stack([Z[assign == j].mean(0) if bool((assign == j).any())
                         else torch.zeros(D) for j in range(J)])
    mask = torch.stack([(assign == j).any() for j in range(J)])
    sizes = torch.stack([(assign == j).sum() for j in range(J)])
    grp = Grouping(assignment=assign, prototypes=proto, mask=mask, sizes=sizes)
    return Z, f_txt, grp


def run_once(step_fn, seed=0, **kw):
    Z, f_txt, grp = fixture(seed)
    torch.manual_seed(100 + seed)
    f_g, f_p = make_models()
    torch.manual_seed(200 + seed)
    return step_fn(Z, torch.tensor(3), torch.zeros(D), f_txt, torch.tensor(56.3477),
                   f_g, f_p, grouping=grp, budget=8, chunk=1,
                   allocation="per_budget", use_query=False, use_state=False,
                   hierarchy=True, **kw)


# ── G3：beta_g=0 位元相同 ───────────────────────────────────────────────────

def test_beta_g_zero_is_bit_identical_to_the_hier_a5_baseline():
    """G3 的 beta_g=0 必須與 baseline 位元相同，否則 G3 與主表不可比。"""
    for seed in range(3):
        base = run_once(R.train_step, seed)[0]
        got = run_once(G.wrap_train_step(0.0), seed)[0]
        assert torch.equal(base, got), f"seed={seed} 位元不同"


def test_beta_g_zero_keeps_group_score_out_of_the_graph():
    _, parts, _ = run_once(G.wrap_train_step(0.0))
    assert "L_sem_group" not in parts, "beta_g=0 不該計算 group 項"


def test_non_zero_beta_g_actually_changes_the_loss():
    """§2.6：確認注入不是 no-op。"""
    base = run_once(R.train_step)[0]
    loss, parts, _ = run_once(G.wrap_train_step(0.1))
    assert not torch.equal(base, loss), "beta_g=0.1 沒有改變 loss → 注入是 no-op"
    assert parts["L_sem_group"] > 0.0
    assert float(loss.detach()) == pytest.approx(
        float(base.detach()) + 0.1 * parts["L_sem_group"], abs=1e-6)


def test_group_term_scales_with_beta_g():
    b0 = float(run_once(R.train_step)[0].detach())
    l1 = float(run_once(G.wrap_train_step(0.1))[0].detach())
    l2 = float(run_once(G.wrap_train_step(0.2))[0].detach())
    assert (l2 - b0) == pytest.approx(2 * (l1 - b0), rel=1e-5)


def test_group_term_responds_to_prototype_perturbation():
    """C-26 的反面：新版必須對 group prototype 敏感（舊版位元不變）。"""
    Z, f_txt, grp = fixture(0)
    step = G.wrap_train_step(0.1)

    def once(g):
        torch.manual_seed(100); f_g, f_p = make_models(); torch.manual_seed(200)
        return step(Z, torch.tensor(3), torch.zeros(D), f_txt,
                    torch.tensor(56.3477), f_g, f_p, grouping=g, budget=8, chunk=1,
                    allocation="per_budget", use_query=False, use_state=False,
                    hierarchy=True)[1]["L_sem_group"]

    torch.manual_seed(7)
    bad = Grouping(grp.assignment, F.normalize(torch.randn_like(grp.prototypes), dim=-1),
                   grp.mask, grp.sizes)
    assert once(grp) != once(bad), "擾動 group prototype 沒有改變 L_sem_group"


# ── G5 / G4：架構注入 ───────────────────────────────────────────────────────

def test_injected_arch_changes_exactly_one_flag_from_the_hier_baseline():
    """一次只開一個變因。"""
    base = R.ARCH["hier"]
    for name, spec in G.INJECTED_ARCH.items():
        diff = [k for k in base if base[k] != spec[k]]
        assert len(diff) == 1, f"{name} 改了 {diff}，不是單一變因"
        assert spec["hierarchy"] is True, f"{name} 必須維持階層"
    assert {tuple(sorted(d.items())) for d in G.INJECTED_ARCH.values()} != {
        tuple(sorted(base.items()))}


def test_inject_arch_is_idempotent_but_refuses_to_overwrite_a_different_setting():
    G.inject_arch()
    G.inject_arch()                       # 同一 process 跑兩個實驗 → 必須允許
    R.ARCH["hier_state"] = dict(use_query=True, use_state=True, hierarchy=True)
    with pytest.raises(AssertionError, match="注入會覆蓋"):
        G.inject_arch()


def test_inject_g3_arm_copies_a5_and_refuses_a_conflicting_definition():
    G.inject_g3_arm()
    a5 = dict(R.ARMS["A5"]); a5.pop("name")
    got = dict(R.ARMS[G.G3_ARM]); got.pop("name")
    assert got == a5, "G3 arm 必須與 A5 只差在名字"
    G.inject_g3_arm()
    R.ARMS[G.G3_ARM] = dict(R.ARMS["A3"], name="別的東西")
    with pytest.raises(AssertionError, match="設定不同"):
        G.inject_g3_arm()


def test_train_stage_looks_up_train_step_at_call_time():
    """monkey-patch 要生效，train_stage 必須在呼叫時查全域，而不是持有舊參照。"""
    import inspect
    src = inspect.getsource(R.train_stage)
    assert "train_step(" in src, "train_stage 不再直接呼叫 train_step，注入失效"

    # 全域查找的證據：train_step 不是 closure 變數，且 train_stage 的 globals
    # 就是 run_exp2 的模組字典 —— 所以 `R.train_step = wrapped` 會被看到。
    assert "train_step" not in R.train_stage.__code__.co_freevars
    assert R.train_stage.__globals__ is R.__dict__

    def boom(*a, **k):
        raise RuntimeError("injected")
    R.train_step = boom
    assert R.train_stage.__globals__["train_step"] is boom


# ── argv 組裝 ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("exp,arch,arm", [("g5", "hier_state", "A5"),
                                          ("g4", "hier_query", "A5"),
                                          ("g3", "hier", G.G3_ARM)])
def test_build_argv(exp, arch, arm):
    argv = G.build_argv(exp, "0,1,2,3,4", [])
    assert argv[argv.index("--arch") + 1] == arch
    assert argv[argv.index("--arms") + 1] == arm
    assert argv[argv.index("--allocation") + 1] == "per_budget"
    assert argv[argv.index("--order") + 1] == "reverse"
    assert argv[argv.index("--tag") + 1] == G.OUT_TAG


def test_group_prior_is_computed_from_group_prototypes_not_patch_features():
    """釘住 prior 的來源。

    ⚠️ 上一個測試不足以守住這件事：擾動 prototype 也會改變 group selector 的
    r，所以就算 prior 誤用 patch feature，L_sem_group 仍然會變。這裡直接把
    期望值算出來比對，才真的鎖住輸入。
    """
    from selector.priors import semantic_prior
    from selector.sem_loss import group_sem_term

    Z, f_txt, grp = fixture(0)
    _loss, parts, res = run_once(G.wrap_train_step(0.1))
    r = res.records[-1].r.index_select(0, grp.mask.nonzero().reshape(-1))
    want = group_sem_term(r, semantic_prior(grp.prototypes[grp.mask], f_txt,
                                            n_candidate_classes=C,
                                            logit_scale=torch.tensor(56.3477)))
    assert parts["L_sem_group"] == pytest.approx(float(want.detach()), abs=0, rel=1e-12)


# ── 真的把 run_exp2.main() 叫起來 ───────────────────────────────────────────

def test_main_actually_invokes_run_exp2_main_with_the_right_argv(monkeypatch):
    """⚠️ `--help` 測試碰不到這條路徑。

    run_exp2.main() 不吃 argv 參數（只讀 sys.argv）—— 用 `R.main(sub)` 呼叫會
    TypeError，而且只有真的跑起來才會炸。這裡把 main 換成 spy 直接走完整條路。
    """
    seen = {}

    def spy():
        seen["argv"] = list(sys.argv)
        return 0
    monkeypatch.setattr(R, "main", spy)

    before = list(sys.argv)
    assert G.main(["g5", "--seeds", "0,1"]) == 0
    assert sys.argv == before, "sys.argv 沒有還原"
    assert seen["argv"][0] == "run_exp2.py"
    assert seen["argv"][1:] == G.build_argv("g5", "0,1", [])


def test_main_restores_argv_even_when_run_exp2_raises(monkeypatch):
    def boom():
        raise RuntimeError("x")
    monkeypatch.setattr(R, "main", boom)
    before = list(sys.argv)
    with pytest.raises(RuntimeError):
        G.main(["g4"])
    assert sys.argv == before


def test_main_wires_the_group_loss_only_for_g3(monkeypatch):
    monkeypatch.setattr(R, "main", lambda: 0)
    base = R.train_step
    G.main(["g5"])
    assert R.train_step is base, "G5 不該動 train_step"
    G.main(["g3"])
    assert R.train_step is not base, "G3 沒有接上 group L_sem"
    assert R.train_step.__wrapped_beta_g__ == G.DEFAULT_BETA_G


# ── G4：q_tau 接線 ──────────────────────────────────────────────────────────

class FakeBank:
    def __init__(self, m): self.m = m
    def get(self, t): return self.m[t]


TASKS4 = ["tcga_esca", "tcga_rcc", "tcga_brca", "tcga_lung"]


def test_check_query_leakage_accepts_distinct_deterministic_queries():
    torch.manual_seed(0)
    bank = FakeBank({t: F.normalize(torch.randn(D), dim=-1) for t in TASKS4})
    out = G.check_query_leakage(bank, TASKS4)
    assert out["n_tasks"] == 4 and len(out["pairwise_cos"]) == 6


def test_check_query_leakage_rejects_two_tasks_sharing_a_query():
    torch.manual_seed(0)
    q = F.normalize(torch.randn(D), dim=-1)
    bank = FakeBank({t: (q if t in TASKS4[:2] else F.normalize(torch.randn(D), dim=-1))
                     for t in TASKS4})
    with pytest.raises(AssertionError, match="位元相同"):
        G.check_query_leakage(bank, TASKS4)


def test_zero_query_makes_use_query_a_bit_identical_no_op():
    """G4 若不接線就是由構造保證的 null（憲法 §2.5）。

    `use_query=False` 的實作是把 query 欄位填零，而 run_exp2.Ctx.q0 = zeros(512)
    —— 兩者輸入完全一樣。這條測試把「為什麼一定要接 TaskQueryBank」釘死。
    """
    from selector.rounds import run_rounds

    def sel(seed, use_query, q):
        Z, _f, grp = fixture(seed)
        torch.manual_seed(100 + seed)
        f_g, f_p = make_models()
        f_g.eval(); f_p.eval()
        with torch.no_grad():
            return run_rounds(Z, grp, q, f_g, f_p, budget=8, chunk=1,
                              allocation="per_budget", use_query=use_query,
                              use_state=False, hierarchy=True).selected.tolist()

    # ⚠️ 逐組判定會失真：真 query 只在少數 slide 上改變選取（實測約 4/20），
    #    單一組抽到「相同」是常態。要判的是**整批不是 no-op**。
    zero_same = real_same = 0
    for seed in range(10):
        off = sel(seed, False, torch.zeros(D))
        zero_same += sel(seed, True, torch.zeros(D)) == off
        torch.manual_seed(7000 + seed)
        real_same += sel(seed, True, F.normalize(torch.randn(D), dim=-1)) == off
    assert zero_same == 10, f"零向量下必須全部位元相同，實得 {zero_same}/10"
    assert real_same < 10, "真 query 完全沒有改變任何一組的選取 → G4 仍是 no-op"


def test_wire_task_queries_patches_all_four_call_sites(monkeypatch):
    torch.manual_seed(0)
    qs = {t: F.normalize(torch.randn(D), dim=-1) for t in TASKS4}
    monkeypatch.setattr(G, "wire_task_queries", G.wire_task_queries)
    import selector.task_query as tq
    monkeypatch.setattr(tq, "TaskQueryBank", lambda cfg, device="cpu": FakeBank(qs))

    before = (R.train_stage, R.evaluate, R.fill_memory, R.continual_terms)
    bank = G.wire_task_queries({"tasks": TASKS4})
    after = (R.train_stage, R.evaluate, R.fill_memory, R.continual_terms)
    assert all(a is not b for a, b in zip(before, after)), "有注入點沒被包到"
    assert bank.get(TASKS4[0]) is qs[TASKS4[0]]


def test_wired_evaluate_sets_and_restores_the_task_query(monkeypatch):
    torch.manual_seed(0)
    qs = {t: F.normalize(torch.randn(D), dim=-1) for t in TASKS4}
    import selector.task_query as tq
    monkeypatch.setattr(tq, "TaskQueryBank", lambda cfg, device="cpu": FakeBank(qs))

    seen = {}
    monkeypatch.setattr(R, "evaluate", lambda ctx, m, task, *a, **k: seen.update(
        q=ctx.q0.clone()) or [])
    monkeypatch.setattr(R, "train_stage", lambda *a, **k: {})
    G.wire_task_queries({"tasks": TASKS4})

    class C:
        q0 = torch.zeros(D)
    ctx = C()
    R.evaluate(ctx, None, "tcga_rcc")
    assert torch.equal(seen["q"], qs["tcga_rcc"]), "沒有換成該 task 的 q_tau"
    assert torch.equal(ctx.q0, torch.zeros(D)), "q0 沒有還原"


def test_wired_train_stage_refuses_multi_task_stages(monkeypatch):
    torch.manual_seed(0)
    qs = {t: F.normalize(torch.randn(D), dim=-1) for t in TASKS4}
    import selector.task_query as tq
    monkeypatch.setattr(tq, "TaskQueryBank", lambda cfg, device="cpu": FakeBank(qs))
    monkeypatch.setattr(R, "train_stage", lambda *a, **k: {})
    monkeypatch.setattr(R, "evaluate", lambda *a, **k: [])
    G.wire_task_queries({"tasks": TASKS4})

    class C:
        q0 = torch.zeros(D)
    with pytest.raises(AssertionError, match="只支援單一 task"):
        R.train_stage(C(), "A5", None, TASKS4, 0, None, None, None)


def test_wired_continual_terms_uses_the_entry_own_task(monkeypatch):
    """舊樣本要用**它自己 task** 的 q_tau，不是當前 task 的。"""
    torch.manual_seed(0)
    qs = {t: F.normalize(torch.randn(D), dim=-1) for t in TASKS4}
    import selector.task_query as tq
    monkeypatch.setattr(tq, "TaskQueryBank", lambda cfg, device="cpu": FakeBank(qs))
    seen = {}
    monkeypatch.setattr(R, "continual_terms",
                        lambda entry, cfg, *a, **k: seen.update(q=k.get("q_tau")))
    monkeypatch.setattr(R, "train_stage", lambda *a, **k: {})
    monkeypatch.setattr(R, "evaluate", lambda *a, **k: [])
    monkeypatch.setattr(R, "fill_memory", lambda *a, **k: 0)
    G.wire_task_queries({"tasks": TASKS4})

    class E:
        tau = "tcga_brca"
    R.continual_terms(E(), {}, None, None, None, None)
    assert torch.equal(seen["q"], qs["tcga_brca"])
