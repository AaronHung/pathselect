"""q_tau — task query，512 維，**絕不接觸 label**。

定義：該 task 全部 candidate class prompt 的文字特徵取平均後 L2 normalize。

    q_tau = normalize( mean_c f_txt[c] )    for c in 該 task 的所有類別

為什麼這樣定：
  - 它只依賴 task identity 與 candidate class 集合，與任何一張 slide 的 label
    無關 —— 同 task 內不同 label 的 slide 拿到 **位元相同** 的 q_tau。
  - 不同 task 的 candidate class 不同，q_tau 自然不同。
  - 不需要訓練，per_task 與 joint 兩種模式都能用同一個定義。

⚠️ `encode_task_query` 的簽名不得出現 label / y / target 之類的參數，
   由 tests/test_leakage.py 用 inspect 強制檢查。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .text_encoder import build_f_txt

QUERY_DIM = 512


def encode_task_query(task: str, cfg: dict | None = None, *,
                      device: str | torch.device = "cpu") -> torch.Tensor:
    """[512]：該 task 的 query 向量。只吃 task 名稱，不吃任何樣本層資訊。"""
    f_txt = build_f_txt(task, cfg, device=device).f_txt        # [C, D]
    return F.normalize(f_txt.mean(0), dim=-1)


class TaskQueryBank:
    """task → q_tau 的快取。joint 模式下每個 sample 依自己的 task 取值。"""

    def __init__(self, cfg: dict | None = None, device="cpu"):
        self.cfg = cfg
        self.device = device
        self._cache: dict[str, torch.Tensor] = {}

    def get(self, task: str) -> torch.Tensor:
        if task not in self._cache:
            self._cache[task] = encode_task_query(task, self.cfg, device=self.device)
        return self._cache[task]

    def stack(self, tasks: list[str]) -> torch.Tensor:
        """[B, 512]：一個 batch 的 q_tau，每個 sample 帶自己 task 的那一條。"""
        return torch.stack([self.get(t) for t in tasks], dim=0)
