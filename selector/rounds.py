"""CONTRACT-1 — chunked sequential loop。

操作點 B = 8、chunk c = 1，共 B / c = 8 rounds（PI 裁定 A；原契約的 B=64/c=8
在 Exp 0 的 K 曲線出來後改成這個，rounds 數不變）。每一輪：

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

#: 操作點（PI 裁定 A）。configs/pathselect.yaml 才是權威來源，這裡是模組層 fallback。
#: B=8 是 Exp 0 完整 K 曲線的峰值（不是飽和點）；c=1 讓 e_t 的槓桿最大。
#: B / c 皆為 CLI 參數；B=8 時 c=8 即為 one-shot。
DEFAULT_BUDGET = 8
DEFAULT_CHUNK = 1

#: F_g 的梯度路徑。
#:
#: r_j 經過 softmax → 乘 c → largest-remainder 取整才變成 b_j，取整不可微，
#: L_diag 的梯度到不了 F_g。原契約沒有考慮到取整會截斷梯度；PI 裁定以契約
#: **意圖**為準：
#:   "ste_allocation"  （主線，預設）把 a_j = softmax(r)_j 以 a_j / a_j.detach()
#:                     注入該 group 的選取 mask：forward 恆等於 1（CONTRACT-4 的
#:                     head 數值完全不受影響），backward 讓 r_j 收到梯度
#:   "none"            **僅供 ablation**：F_g 完全不接收梯度，等於固定的隨機函數。
#:                     用來證明 group 層確實需要學習；不可當作主線跑 L5。
GROUP_GRAD_MODES = ("ste_allocation", "none")
DEFAULT_GROUP_GRAD = "ste_allocation"

#: 配額口徑（DR-025）。
#:   "per_chunk"  舊版：每一輪對 chunk c 配額。c=1 時 largest-remainder 只有一個
#:                名額可發、必然給 argmax(r)；r 逐輪不變 ⇒ 每輪同一組 ⇒ 退化為
#:                「先挑一組再取該組 top-c」。G1 實測 84.5% 的 slide 只用一組。
#:   "per_budget" 新版：每一輪對**剩餘預算 B_t** 配額得到 b_j，逐輪累計每組已取
#:                數 taken_j，該輪從「taken_j < b_j 且 r_j 最高」的組取 patch。
#:                配額用完的組會讓位給次高的組，因此預算會攤到多個 group 上 ——
#:                對齊架構圖「tumor 12 / lymph 8 / stroma 4 / necrosis 2」的示例。
ALLOCATION_MODES = ("per_budget", "per_chunk")
DEFAULT_ALLOCATION = "per_budget"


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
               use_query: bool = True, use_state: bool = True,
               hierarchy: bool = True,
               allocation: str = DEFAULT_ALLOCATION,
               state: EvidenceState | None = None) -> RoundsResult:
    """跑完 ceil(budget / chunk) 輪，回傳完整 trace。

    group_grad 見 GROUP_GRAD_MODES —— 預設 "ste_allocation"，F_g 從 L_diag 收梯度。

    消融階梯的三個開關（關掉的輸入區塊填零，架構與參數量不變）：
      use_query   q_tau 是否進入輸入（L3 關、L4+ 開）
      use_state   e_t / B_tilde_t 是否進入輸入（L6 才開）
      hierarchy   True = Group → Patch 兩層配額（L5+）；
                  False = flat，直接在全部候選上取 top-c（L3 / L4）

    use_state=False 時 r 與 s 在輪與輪之間是常數，只算一次再重用 —— 這是純粹的
    計算重用，數值與逐輪重算完全相同（同一組輸入、同一個網路）。
    """
    if chunk <= 0:
        raise ValueError(f"chunk must be positive, got {chunk}")
    if group_grad not in GROUP_GRAD_MODES:
        raise ValueError(f"unknown group_grad: {group_grad}; {GROUP_GRAD_MODES}")
    if allocation not in ALLOCATION_MODES:
        raise ValueError(f"unknown allocation: {allocation}; {ALLOCATION_MODES}")
    state = state or EvidenceState(Z, budget)
    res = RoundsResult(selected=torch.empty(0, dtype=torch.long, device=Z.device),
                       state=state)

    n_rounds = -(-budget // chunk)          # ceil
    cached: tuple[torch.Tensor, torch.Tensor] | None = None
    taken = torch.zeros(grouping.num_groups, dtype=torch.long, device=Z.device)
    for t in range(n_rounds):
        if state.B_t <= 0 or int(state.available_mask.sum()) == 0:
            break
        state_feat = state.feature()
        if cached is not None:
            r, s = cached
        else:
            r = f_group.score(grouping.prototypes, q_tau, state_feat,
                              use_query=use_query, use_state=use_state)     # [J]
            s = f_patch.score(Z, q_tau, state_feat,
                              use_query=use_query, use_state=use_state)     # [n]
            if not use_state:
                cached = (r, s)      # 無狀態時分數逐輪不變，重用即可

        cap = group_capacity(grouping, state.available_mask)
        c_this = min(chunk, state.B_t)
        if hierarchy and allocation == "per_budget":
            # 對**整個 budget** 做一次 largest-remainder 配額，逐輪追蹤各組已取數；
            # 本輪從「尚未滿額（taken_j < quota_j）且 r_j 最高」的組取 patch。
            #
            # 這是「每輪對剩餘預算 B_t 配額」的**累計等價形式**：任一時點各組的
            # 剩餘配額為 quota_j − taken_j，其總和恰為 B_t。實作成對 B_0 配額一次
            # 是為了避免逐輪重新取整的漂移（逐輪重配會讓配額提前用完、預算花不完）。
            # sum(quota) = B_0 且每輪取 1，故八輪剛好取滿、不會提前結束。
            quota = allocate(r, budget, grouping.mask,
                             group_capacity(grouping,
                                            torch.ones_like(state.available_mask)))
            b = torch.zeros_like(cap)
            room = (taken < quota) & grouping.mask & (cap > 0)
            for _ in range(c_this):
                if not bool(room.any()):     # 全滿額或無容量 → 放寬配額限制
                    room = grouping.mask & ((cap - b) > 0)
                    if not bool(room.any()):
                        break
                j = int(torch.where(room, r, torch.full_like(r, float("-inf"))).argmax())
                b[j] += 1
                taken[j] += 1
                room = (taken < quota) & grouping.mask & ((cap - b) > 0)
        elif hierarchy:
            b = allocate(r, c_this, grouping.mask, cap)                    # [J]
        else:
            # flat：不分組，直接在全部候選上取 top-c；b 僅供紀錄
            b = torch.zeros_like(cap)

        a_soft = None
        if group_grad == "ste_allocation":
            active = grouping.mask & (cap > 0)
            a_soft = torch.zeros_like(r)
            if bool(active.any()):
                idx_a = active.nonzero(as_tuple=False).reshape(-1)
                a_soft = a_soft.index_copy(
                    0, idx_a, torch.softmax(r.index_select(0, idx_a), dim=0))

        picks, ste = [], torch.zeros_like(s)
        if not hierarchy:
            flat_idx = topk_indices(s, c_this, mask=state.available_mask)
            picks.append(flat_idx)
            ste = ste + straight_through_topk(s, c_this, temperature=temperature,
                                              mask=state.available_mask)
            onehot = torch.zeros_like(cap)
            for j in grouping.assignment.index_select(0, flat_idx).tolist():
                onehot[j] += 1
            b = onehot
        for j in (range(grouping.num_groups) if hierarchy else ()):
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
