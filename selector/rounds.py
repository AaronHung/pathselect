"""CONTRACT-1 — chunked sequential loop。

B = 64、chunk c = 8、共 B / c = 8 rounds。每一輪：

  1. F_g 依 [g_j ; q_tau ; e_t ; B_tilde_t] 給每個 group 一個 r_j
  2. r_j → 非空 group 上 softmax → 乘 c → largest-remainder，得 b_j，sum(b_j) = c
  3. F_p 依 [x_i ; q_tau ; e_t ; B_tilde_t] 給每個 patch 一個 s_i
  4. 每個 group 內在「仍可選」的 patch 中取 top-b_j
  5. 選中的 patch 併入 evidence，並從候選移除

同一個 group 可以跨輪重複被選；已選過的 patch 不會再進候選。
每一輪的 sum_j b_j = c（不是 B）。

⚠️ e_t 在輪與輪之間 detach（CONTRACT-2）：每一輪只依當前狀態決策，
   不對後續輪次做任何預估。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .allocation import allocate
from .grouping import Grouping, group_capacity
from .model import GroupSelector, PatchSelector, straight_through_topk, topk_indices
from .state import EvidenceState
from .utility import CANDIDATE_SIZE, top_candidates

DEFAULT_BUDGET = 64
DEFAULT_CHUNK = 8

#: F_g 的梯度路徑。
#:
#: ⚠️ **凍結契約沒有規定這件事，預設關閉，等 PI 裁定。**
#: r_j 經過 softmax → 乘 c → largest-remainder 取整才變成 b_j，取整是不可微的，
#: 所以 L_diag 的梯度到不了 F_g。本輪的選項：
#:   "none"            契約字面：F_g 這輪不從 L_diag 收梯度（若之後由 L_util /
#:                     Selection Memory 的 r_old 監督 r_j，這就是正確行為）
#:   "ste_allocation"  把 a_j = softmax(r)_j 以 a_j / a_j.detach() 注入該 group
#:                     的選取 mask：forward 恆等於 1（CONTRACT-4 的 head 完全
#:                     不受影響），backward 讓 r_j 收到梯度
GROUP_GRAD_MODES = ("none", "ste_allocation")
DEFAULT_GROUP_GRAD = "none"


@dataclass
class RoundRecord:
    """單一輪的紀錄。"""
    t: int
    b: torch.Tensor              # [J] 這一輪各 group 分到幾個
    picked: torch.Tensor         # [<=c] 這一輪選到的 patch index
    r: torch.Tensor              # [J] group 分數
    s: torch.Tensor              # [n] patch 分數
    cand_idx: torch.Tensor       # [<=256] 當輪候選集合
    ste_mask: torch.Tensor       # [n] straight-through 選取 mask
    n_selected_after: int = 0


@dataclass
class RoundsResult:
    selected: torch.Tensor                       # [<=B] 依觀察順序
    records: list[RoundRecord] = field(default_factory=list)
    state: EvidenceState | None = None

    @property
    def n_rounds(self) -> int:
        return len(self.records)

    def allocation_table(self) -> list[list[int]]:
        return [rec.b.tolist() for rec in self.records]


def run_rounds(Z: torch.Tensor, grouping: Grouping, q_tau: torch.Tensor,
               f_group: GroupSelector, f_patch: PatchSelector, *,
               budget: int = DEFAULT_BUDGET, chunk: int = DEFAULT_CHUNK,
               temperature: float = 1.0,
               candidate_size: int = CANDIDATE_SIZE,
               group_grad: str = DEFAULT_GROUP_GRAD,
               state: EvidenceState | None = None) -> RoundsResult:
    """跑完 ceil(budget / chunk) 輪，回傳完整 trace。

    group_grad 見 GROUP_GRAD_MODES —— 預設 "none"，F_g 不從 L_diag 收梯度。
    """
    if chunk <= 0:
        raise ValueError(f"chunk must be positive, got {chunk}")
    if group_grad not in GROUP_GRAD_MODES:
        raise ValueError(f"unknown group_grad: {group_grad}; {GROUP_GRAD_MODES}")
    state = state or EvidenceState(Z, budget)
    res = RoundsResult(selected=torch.empty(0, dtype=torch.long, device=Z.device),
                       state=state)

    n_rounds = -(-budget // chunk)          # ceil
    for t in range(n_rounds):
        if state.B_t <= 0 or int(state.available_mask.sum()) == 0:
            break
        state_feat = state.feature()
        r = f_group.score(grouping.prototypes, q_tau, state_feat)          # [J]
        s = f_patch.score(Z, q_tau, state_feat)                            # [n]

        cap = group_capacity(grouping, state.available_mask)
        c_this = min(chunk, state.B_t)
        b = allocate(r, c_this, grouping.mask, cap)                        # [J]

        a_soft = None
        if group_grad == "ste_allocation":
            active = grouping.mask & (cap > 0)
            a_soft = torch.zeros_like(r)
            if bool(active.any()):
                idx_a = active.nonzero(as_tuple=False).reshape(-1)
                a_soft = a_soft.index_copy(
                    0, idx_a, torch.softmax(r.index_select(0, idx_a), dim=0))

        picks, ste = [], torch.zeros_like(s)
        for j in range(grouping.num_groups):
            k = int(b[j])
            if k <= 0:
                continue
            member = (grouping.assignment == j) & state.available_mask     # [n]
            picks.append(topk_indices(s, k, mask=member))
            m_j = straight_through_topk(s, k, temperature=temperature, mask=member)
            if a_soft is not None:
                # forward 恆等於 1，只把梯度注入 r_j；head 的數值完全不變
                m_j = m_j * (a_soft[j] / a_soft[j].detach().clamp_min(1e-12))
            ste = ste + m_j
        picked = (torch.cat(picks) if picks
                  else torch.empty(0, dtype=torch.long, device=Z.device))

        cand_idx = top_candidates(s.detach(), state.available_mask, candidate_size)
        res.records.append(RoundRecord(t=t, b=b, picked=picked, r=r, s=s,
                                       cand_idx=cand_idx, ste_mask=ste))
        if picked.numel() == 0:
            break
        state.update(Z.index_select(0, picked), picked)
        res.records[-1].n_selected_after = state.n_selected

    res.selected = torch.tensor(state.selected, dtype=torch.long, device=Z.device)
    return res
