"""共用評估核心：載 test slide、套用選取政策、走同一條下游分類路徑。

所有 policy（learned / similarity / random / grid）都必須走這裡的
`select_and_classify`，確保「選法」以外的每一步完全相同：

    selected patches → 權重 → L2 normalize → conch_classify → argmax
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import torch

from .classifier import conch_classify, softmax_weights
from .multiround import top_k_select


@dataclass
class SlideRecord:
    sid: str
    Z: torch.Tensor          # [n, D]
    label: int               # 全域 label（已加 shift）


def iter_test_slides(cfg: dict, task: str, task_pos: int,
                     limit: int = 0) -> Iterator[SlideRecord]:
    """v9 用的同一組 test slide（fold 1、shuffle=False、batch_size=1）。

    label 已加上 shift = 2 * task_pos，對應 8-way label space 的列索引。
    """
    from data.table_utils import read_datasplit_npz
    from data.wsi_dataset import WSIClf

    root = cfg["dataset_root_dir"]
    _, _, pids_test = read_datasplit_npz(root + cfg["path_split"].format(task, cfg["fold"]))
    ds = WSIClf(pids_test,
                root + cfg["path_feat"].format(task, cfg["conch_path_feat"]),
                root + cfg["path_table"].format(task, task.upper()),
                cfg["feat_format"])
    shift = 2 * task_pos
    n = len(ds) if limit <= 0 else min(limit, len(ds))
    for i in range(n):
        _idx, feats, label = ds[i]
        Z = feats.float()
        yield SlideRecord(str(ds.sids[i]), Z.squeeze(0) if Z.dim() == 3 else Z,
                          int(label.view(-1)[0]) + shift)


@torch.no_grad()
def select_and_classify(Z: torch.Tensor, idx: torch.Tensor,
                        f_txt: torch.Tensor, logit_scale,
                        scores: Optional[torch.Tensor] = None,
                        weighting: str = "softmax") -> tuple[int, torch.Tensor]:
    """共用下游路徑。回傳 (predicted class, logits)。

    weighting="softmax"（主線，與訓練一致）需要整張 slide 的 `scores`；
    weighting="uniform" 是 selection-only ablation，也是沒有分數的 policy
    （random / grid）唯一能用的權重。
    """
    if weighting == "softmax":
        if scores is None:
            raise ValueError('weighting="softmax" 需要 scores')
        w = softmax_weights(scores, idx)
    elif weighting == "uniform":
        w = None
    else:
        raise ValueError(f"unknown weighting: {weighting}")
    logits = conch_classify(Z.index_select(0, idx), w, f_txt, logit_scale)
    return int(logits.reshape(-1).argmax()), logits


@torch.no_grad()
def score_based_indices(scores: torch.Tensor, k: int) -> torch.Tensor:
    """分數型 policy 的選取：單次 top-K。"""
    return top_k_select(scores, k)
