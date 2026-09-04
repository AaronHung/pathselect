"""DR-048 B6：OPCM-Merge (adapted) 的數學性質。

三件事必須釘住：

1. 遮罩生效時，`G(Δθ_t)` 與 `Δθ̃_{1:t−1}` 的 Frobenius 內積為 0（論文 Eq. 4 的
   全部理由）。
2. **遮罩失效時 `G` 退化成恆等**（`G(Δθ_t) ≡ Δθ_t`）—— 這正是官方程式
   `.diag().fill_(0)` 的實際後果。不釘住這條，我們就無法宣稱「本檔與官方行為不同」。
3. `Tensor.diag()` 對 2-D 回傳副本、`.diagonal()` 才是 view —— 上面第 2 點的成因。
   PyTorch 哪天改了語意，這條會先響。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from sota.opcm import _norm, _scaled, apply_delta, merge_sequence, project   # noqa: E402

M, N = 6, 4


def rand_delta(seed: int, scale: float = 1.0) -> list[dict]:
    g = torch.Generator().manual_seed(seed)
    return [{}, {"mlp.0.weight": torch.randn(M, N, generator=g) * scale,
                 "mlp.2.weight": torch.randn(1, M, generator=g) * scale}]


# ── 1. 遮罩生效 → 正交 ──────────────────────────────────────────────────────

def test_projection_is_frobenius_orthogonal_to_previous():
    g = torch.Generator().manual_seed(0)
    prev, cur = torch.randn(M, N, generator=g), torch.randn(M, N, generator=g)
    out = project(cur, prev, mask=True)
    assert float((out * prev).sum()) == pytest.approx(0.0, abs=1e-4)


def test_projection_actually_changes_the_task_vector():
    """正交化不能是「什麼都沒做」—— 否則上一條會因為巧合而通過。"""
    g = torch.Generator().manual_seed(1)
    prev, cur = torch.randn(M, N, generator=g), torch.randn(M, N, generator=g)
    assert not torch.allclose(project(cur, prev, mask=True), cur, atol=1e-4)


def test_projection_is_orthogonal_for_non_square_and_single_row():
    """1×N（`mlp.2.weight` 的形狀）也要成立 —— 對角線只有 1 格。"""
    g = torch.Generator().manual_seed(2)
    for shape in [(1, 8), (8, 1), (3, 7), (7, 3)]:
        prev, cur = torch.randn(*shape, generator=g), torch.randn(*shape, generator=g)
        out = project(cur, prev, mask=True)
        assert float((out * prev).sum()) == pytest.approx(0.0, abs=1e-4), shape


def test_zero_previous_returns_the_task_vector_unchanged():
    """沒有舊方向時不能把整個任務投影掉（論文未涵蓋的邊界）。"""
    cur = torch.randn(M, N)
    assert torch.equal(project(cur, torch.zeros(M, N), mask=True), cur)


# ── 2. 遮罩失效 → 恆等（官方程式的實際行為）──────────────────────────────────

def test_without_mask_projection_is_the_identity():
    """full SVD 下 U、V 都是正交方陣 → `U (Uᵀ X V) Vᵀ = X`。

    這條記錄的是**官方實作的實際行為**：它的零對角遮罩沒有生效，
    `merge_linear_weights` 因此與 `merge_other_parameters` 完全相同。
    """
    g = torch.Generator().manual_seed(3)
    prev, cur = torch.randn(M, N, generator=g), torch.randn(M, N, generator=g)
    assert torch.allclose(project(cur, prev, mask=False), cur, atol=1e-4)


def test_mask_flag_changes_the_merged_result():
    """兩種模式必須真的產生不同的合併結果，否則 `--no-mask` 是裝飾。"""
    deltas = [rand_delta(i) for i in range(4)]
    a = merge_sequence(deltas, mask=True)[-1][1]["mlp.0.weight"]
    b = merge_sequence(deltas, mask=False)[-1][1]["mlp.0.weight"]
    assert not torch.allclose(a, b, atol=1e-5)


def test_diag_is_a_copy_but_diagonal_is_a_view():
    """官方那行 no-op 的成因。語意變了要先在這裡響。"""
    A = torch.ones(4, 4)
    A.diag().fill_(0)
    assert float(A[0, 0]) == 1.0, "`.diag()` 竟然成了 view —— 重新檢視 sota/opcm.py 的說明"
    B = torch.ones(4, 4)
    B.diagonal().fill_(0)
    assert float(B[0, 0]) == 0.0


# ── 3. 序列合併的性質 ───────────────────────────────────────────────────────

def test_first_stage_is_the_first_task_vector_untouched():
    """θ̃_1 = θ_1，λ_1 = 1（與官方初始化一致）。"""
    deltas = [rand_delta(i) for i in range(3)]
    first = merge_sequence(deltas)[0]
    for k, v in deltas[0][1].items():
        assert torch.equal(first[1][k], v)


def test_sequence_length_matches_number_of_tasks():
    for T in (1, 2, 4):
        assert len(merge_sequence([rand_delta(i) for i in range(T)])) == T


def test_merged_norm_tracks_the_running_mean_of_task_norms():
    """λ 的目的：合併後的 task vector 範數 = 至今各任務 task vector 範數的平均。

    這是論文 Condition (b) 的整個重點 —— 合併模型不隨任務數愈飄愈遠。
    """
    import statistics
    deltas = [rand_delta(i, scale=1.0 + i) for i in range(4)]
    seq = merge_sequence(deltas)
    norms = [_norm(d) for d in deltas]
    for t in range(1, len(seq)):
        assert _norm(seq[t]) == pytest.approx(statistics.mean(norms[:t + 1]), rel=1e-4)


def test_norm_would_grow_without_the_lambda_rescaling():
    """反向對照：不做 λ 縮放時範數會膨脹，證明上一條不是恆真。"""
    import statistics
    deltas = [rand_delta(i) for i in range(4)]
    raw = [{k: v.clone() for k, v in sd.items()} for sd in deltas[0]]
    for t in range(1, 4):
        raw = [{k: raw[i].get(k, 0) + v for k, v in sd.items()}
               for i, sd in enumerate(deltas[t])]
    assert _norm(raw) > statistics.mean(_norm(d) for d in deltas) * 1.2


def test_scaled_and_apply_delta_round_trip():
    theta0 = [{}, {"mlp.0.weight": torch.zeros(M, N), "mlp.2.weight": torch.zeros(1, M)}]
    d = rand_delta(9)
    out = apply_delta(theta0, _scaled(d, 2.0))
    assert torch.allclose(out[1]["mlp.0.weight"], d[1]["mlp.0.weight"] * 2.0)


def test_all_zero_stage_is_rejected_not_silently_divided():
    zero = [{}, {"mlp.0.weight": torch.zeros(M, N)}]
    with pytest.raises(ValueError, match="全零"):
        merge_sequence([zero, zero])
