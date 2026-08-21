"""q_tau 不得洩漏 label。

q_tau 只能依賴 task identity 與該 task 的**全部** candidate class prompt。
只要它沾到單張 slide 的 label，所有 q ablation 與 joint 模式的結論都會失效。
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

from selector.rounds import run_rounds
from selector.task_query import QUERY_DIM, TaskQueryBank, encode_task_query
from selector.text_encoder import load_config

#: 任何形式的 label 參數名都不准出現在 selector 的輸入路徑上。
LABEL_LIKE = ("label", "labels", "y", "target", "targets", "gt",
              "ground_truth", "true_class", "class_id")

N_SLIDES = 20


def _cfg():
    return load_config()


def _dataset_ready(cfg, task) -> bool:
    return Path(cfg["dataset_root_dir"] + cfg["path_table"].format(
        task, task.upper())).exists()


def test_encode_task_query_signature_has_no_label_parameter():
    """用 inspect 檢查簽名 —— 這是硬性約束，不是靠 code review。"""
    sig = inspect.signature(encode_task_query)
    names = {p.lower() for p in sig.parameters}
    offending = names & set(LABEL_LIKE)
    assert not offending, f"encode_task_query 出現 label 類參數：{offending}"
    assert "task" in names


def test_selector_forward_path_takes_no_label():
    """整條 forward（run_rounds）也不得出現 label 類參數。"""
    for fn in (run_rounds, TaskQueryBank.get, TaskQueryBank.stack):
        names = {p.lower() for p in inspect.signature(fn).parameters}
        assert not (names & set(LABEL_LIKE)), (fn.__name__, names)


@pytest.mark.parametrize("task", ["tcga_lung"])
def test_same_task_different_labels_give_bit_identical_q_tau(task):
    """同一個 task 內，不同 label 的 20 張 slide → q_tau 位元相同。"""
    cfg = _cfg()
    if not _dataset_ready(cfg, task):
        pytest.skip("dataset not available")
    from selector.evaluate import iter_test_slides

    task_pos = cfg["tasks"].index(task)
    bank = TaskQueryBank(cfg)
    # split 是依 patient 排序的，直接取前 20 張會全是同一個 label ——
    # 改成每個 label 各取 10 張，確保真的跨 label。
    per_label = N_SLIDES // 2
    picked: dict[int, int] = {}
    labels, queries = [], []
    for rec in iter_test_slides(cfg, task, task_pos):
        if picked.get(rec.label, 0) >= per_label:
            continue
        picked[rec.label] = picked.get(rec.label, 0) + 1
        labels.append(rec.label)
        queries.append(encode_task_query(task, cfg))
        if len(queries) == N_SLIDES:
            break

    assert len(queries) == N_SLIDES
    assert len(set(labels)) >= 2, f"這 {N_SLIDES} 張都是同一個 label，測不出東西"
    first = queries[0]
    for i, q in enumerate(queries[1:], 1):
        assert torch.equal(first, q), f"slide {i} 的 q_tau 與第 0 張不同"
    # 走 bank 的快取路徑也一樣
    assert torch.equal(first, bank.get(task))


def test_different_tasks_give_different_q_tau():
    cfg = _cfg()
    tasks = list(cfg["tasks"])
    qs = {t: encode_task_query(t, cfg) for t in tasks}
    for t, q in qs.items():
        assert q.shape == (QUERY_DIM,)
    for i, a in enumerate(tasks):
        for b in tasks[i + 1:]:
            assert not torch.equal(qs[a], qs[b]), f"{a} 與 {b} 的 q_tau 相同"
            assert float((qs[a] - qs[b]).abs().max()) > 1e-4, (a, b)


def test_q_tau_is_deterministic_across_calls():
    cfg = _cfg()
    a = encode_task_query("tcga_brca", cfg)
    b = encode_task_query("tcga_brca", cfg)
    assert torch.equal(a, b)


def test_bank_stack_gives_each_sample_its_own_query():
    """joint 模式：一個 batch 裡每個 sample 帶自己 task 的 q_tau。"""
    cfg = _cfg()
    bank = TaskQueryBank(cfg)
    tasks = ["tcga_lung", "tcga_brca", "tcga_lung", "tcga_esca"]
    Q = bank.stack(tasks)
    assert Q.shape == (4, QUERY_DIM)
    assert torch.equal(Q[0], Q[2])                 # 同 task
    assert not torch.equal(Q[0], Q[1])             # 不同 task
