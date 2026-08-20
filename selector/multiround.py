"""N1 — Sequential Budgeted Observation (the agentic core of the CNL).

把舊的「單步 Top-K selector」升級為「序列、累積證據」的觀察迴圈：
每一輪選一小批 patch -> 用 Observation State 累積已看證據 -> 下一輪的選擇
取決於已看到的東西（redundancy-aware），可在 backbone 信心達標時提早停。

設計（對齊 docs/wiki/05 介面契約 R1/R2/R3 與 STORYLINE §6）：
- 不訓練 backbone（frozen），只透過 predict(subset) 取信心。
- 重用既有 navigation skill（NSM 的 per-task MicroRouterV0）當「基礎評分器」。
- one-shot 模式（redundancy_weight=0 且 step_size>=budget）會退化成舊 Top-K
  -> 直接拿來做 N2 的 sequential vs one-shot ablation。

本檔 device-agnostic、不 import QPMIL / train_router_v0，可單獨單元測試。
backbone 鴨子型別需提供：class_text_features(), prototype_features(),
aggregate_and_predict(Z_subset) -> (logits, ...)。

對應 spec：specs/01_sop_navipath-cl_phase0.md (N1)。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from .routers import MicroRouterV0, summary_feats, top_k_select
from .continual_agent import ContextGate, NavigationSkillBank


@dataclass
class ObserveConfig:
    """序列觀察的超參數。

    one-shot baseline：step_size>=budget 且 redundancy_weight=0 -> 等價舊 Top-K。

    N6 路線 A（讓 sequential 真的 != one-shot）：
    - normalize_base：把 base_score z-score 正規化，讓 redundancy 懲罰與分數同尺度
      （否則 router 分數尺度大、0.5*cos 幾乎不改排序 -> seq==oneshot 的主因）。z-score
      單調，故 one-shot（redundancy=0、單輪）的 top-K 結果不變。
    - redundancy_mode：
        "maxsim"  = MMR 式：懲罰「與已選『任一』patch 的最大相似度」-> 真正逐步探索新區域（預設）。
        "centroid"= 舊版：懲罰「與已看『平均』的相似度」（向心，效果弱，保留供對照）。
    """
    budget: int = 64                       # 總觀察上限 K
    step_size: int = 16                    # 每輪看幾個 patch
    redundancy_weight: float = 0.5         # 對「與已看區域過於相似」的懲罰（驅動序列性）
    confidence_threshold: Optional[float] = None  # backbone 信心達標即提早停（None=不早停）
    normalize_base: bool = True            # base_score z-score 正規化（路線 A）
    redundancy_mode: str = "maxsim"        # "maxsim"(MMR, 預設) | "centroid"(舊向心)

    @property
    def is_one_shot(self) -> bool:
        return self.redundancy_weight == 0.0 and self.step_size >= self.budget


class ObservationState:
    """Agent 讀單張 slide 時的短期記憶：累積「已看到什麼證據」。

    保存觀察順序（= Navigation Trace 的來源）、已看 patch 的聚合特徵、
    覆蓋率，以及最近一次 backbone 的預測信心。
    """

    def __init__(self, Z: torch.Tensor):
        self.Z = Z
        self.n = int(Z.shape[0])
        self.seen: list[int] = []                 # 依觀察順序
        self._mask = torch.zeros(self.n, dtype=torch.bool, device=Z.device)
        self.confidence: float = 0.0
        self.last_logits: Optional[torch.Tensor] = None

    def add(self, idx: torch.Tensor) -> None:
        for i in idx.tolist():
            if not self._mask[i]:
                self._mask[i] = True
                self.seen.append(int(i))

    @property
    def seen_mask(self) -> torch.Tensor:
        return self._mask

    @property
    def coverage(self) -> float:
        return len(self.seen) / self.n if self.n else 0.0

    def aggregate(self) -> Optional[torch.Tensor]:
        """已看 patch 的 normalized 平均特徵；尚未看任何 patch 時回 None。"""
        if not self.seen:
            return None
        idx = torch.tensor(self.seen, device=self.Z.device)
        return F.normalize(self.Z.index_select(0, idx).mean(0), dim=-1)

    def summary(self) -> dict:
        return {
            "n_seen": len(self.seen),
            "coverage": round(self.coverage, 4),
            "confidence": round(float(self.confidence), 4),
        }


@dataclass
class ObservationResult:
    """一次序列觀察的完整輸出。"""
    selected: torch.Tensor                 # 已看 patch index（依觀察順序）= trace
    trace: list[list[int]] = field(default_factory=list)  # 每一輪選了哪些
    n_rounds: int = 0
    stopped_early: bool = False
    confidence: float = 0.0
    logits: Optional[torch.Tensor] = None
    state: Optional[ObservationState] = None


class SequentialBudgetedObserver:
    """序列、有預算的觀察器（純粹、可測，不依賴 backbone 內部）。

    輸入：
      base_score: [n] 每個 patch 的靜態重要度（來自 NSM 的 navigation skill）。
      predict_fn: callable(Z_subset[m,D]) -> logits[C]，包住 frozen backbone。
    """

    def __init__(self, config: Optional[ObserveConfig] = None):
        self.cfg = config or ObserveConfig()

    @torch.no_grad()
    def observe(self, Z: torch.Tensor, base_score: torch.Tensor,
                predict_fn: Callable[[torch.Tensor], torch.Tensor]) -> ObservationResult:
        cfg = self.cfg
        n = int(Z.shape[0])
        budget = n if cfg.budget <= 0 else min(cfg.budget, n)
        Zn = F.normalize(Z, dim=-1)
        state = ObservationState(Z)
        res = ObservationResult(selected=torch.empty(0, dtype=torch.long, device=Z.device),
                                state=state)

        # 路線 A：base_score 正規化（z-score，單調 -> 不改 one-shot top-K），讓
        # redundancy 懲罰與分數同尺度，真正影響排序。
        norm_base = base_score
        if cfg.normalize_base and n > 1:
            std = base_score.std()
            if float(std) > 1e-6:
                norm_base = (base_score - base_score.mean()) / (std + 1e-6)
        # MMR 式冗餘：max over 已選 patch 的相似度（逐輪增量更新；初始 0 -> 第一輪無懲罰）
        max_sim_seen = torch.zeros(n, device=Z.device)

        while len(state.seen) < budget:
            # 依已看證據調整分數：懲罰與已看區域過於相似者（驅動覆蓋/探索 -> 序列性）
            adj = norm_base.clone()
            if cfg.redundancy_weight != 0.0 and state.seen:
                if cfg.redundancy_mode == "centroid":
                    agg = state.aggregate()
                    if agg is not None:
                        adj = adj - cfg.redundancy_weight * (Zn @ agg)
                else:                                        # "maxsim" (MMR, 預設)
                    adj = adj - cfg.redundancy_weight * max_sim_seen
            adj = adj.masked_fill(state.seen_mask, float("-inf"))  # 不重選

            k_this = min(cfg.step_size, budget - len(state.seen))
            if k_this <= 0:
                break
            pick = top_k_select(adj, k_this)
            if pick.numel() == 0:
                break

            state.add(pick)
            res.trace.append(pick.tolist())
            res.n_rounds += 1

            # 增量更新 MMR 冗餘：每個 patch 對「已選集合」的最大相似度
            if cfg.redundancy_weight != 0.0 and cfg.redundancy_mode != "centroid":
                sims_new = Zn @ Zn.index_select(0, pick).t()     # [n, k_this]
                max_sim_seen = torch.maximum(max_sim_seen, sims_new.amax(dim=1))

            # 只有需要「信心早停」時才每輪呼叫 backbone（否則白做；省大量算力）。
            # 不早停時，最終子集只需在迴圈外預測一次 -> sequential 成本 ≈ one-shot。
            if cfg.confidence_threshold is not None:
                sel = torch.tensor(state.seen, device=Z.device)
                logits = predict_fn(Z.index_select(0, sel))
                state.last_logits = logits
                state.confidence = float(F.softmax(logits.reshape(-1), dim=-1).max())
                if state.confidence >= cfg.confidence_threshold:
                    res.stopped_early = True
                    break

        # 最終在已選子集上預測一次（涵蓋無早停情況，並確保 logits 對應最終子集）
        if state.seen:
            sel = torch.tensor(state.seen, device=Z.device)
            logits = predict_fn(Z.index_select(0, sel))
            state.last_logits = logits
            state.confidence = float(F.softmax(logits.reshape(-1), dim=-1).max())

        res.selected = torch.tensor(state.seen, dtype=torch.long, device=Z.device)
        res.confidence = state.confidence
        res.logits = state.last_logits
        return res


class ContinualSequentialNavigationAgent:
    """CNL（序列版）：frozen backbone + NSM + Context Gate + 序列觀察。

    與 continual_agent.ContinualWSINavigationAgent 的差別：
    後者一次 Top-K（單步）；本類做多輪、累積證據的序列觀察並產出 trace。
    """

    def __init__(self, backbone, skill_bank: NavigationSkillBank,
                 gate: Optional[ContextGate] = None,
                 config: Optional[ObserveConfig] = None, device=None,
                 policy_mode: str = "router"):
        if policy_mode not in ("router", "zero_shot"):
            raise ValueError(f"unknown policy_mode: {policy_mode}")
        self.backbone = backbone
        self.skill_bank = skill_bank
        self.gate = gate or ContextGate("oracle")
        self.observer = SequentialBudgetedObserver(config)
        self.device = device
        self.policy_mode = policy_mode  # "router"=trained NSM skill; "zero_shot"=frozen-FM text sim
        self._router_cache: dict[int, MicroRouterV0] = {}

    def _router_for(self, task_id: int) -> MicroRouterV0:
        if task_id not in self._router_cache:
            self._router_cache[task_id] = self.skill_bank.build_router(task_id, self.device)
        return self._router_cache[task_id]

    @torch.no_grad()
    def _base_score(self, Z: torch.Tensor, task_id: int) -> torch.Tensor:
        f_txt = self.backbone.class_text_features()
        F_p = self.backbone.prototype_features()
        if self.device is not None:
            f_txt, F_p = f_txt.to(self.device), F_p.to(self.device)
        if self.policy_mode == "zero_shot":
            # SPEC-07: zero-shot navigation — 不訓練、不查 skill bank，
            # 直接用 frozen FM 的 patch-text 相似度 (ZeroSlide 精神搬到 navigation)。
            _, sim_txt_max = summary_feats(Z, f_txt, F_p)
            return sim_txt_max
        router = self._router_for(task_id)
        score, _ = router(Z, f_txt, F_p)
        return score

    @torch.no_grad()
    def observe(self, Z: torch.Tensor, *, task_id: Optional[int] = None) -> ObservationResult:
        skill_id = self.gate.select(task_id=task_id, Z=Z, backbone=self.backbone)
        base_score = self._base_score(Z, skill_id)
        predict_fn = lambda S: self.backbone.aggregate_and_predict(S)[0]  # noqa: E731
        return self.observer.observe(Z, base_score, predict_fn)

    @torch.no_grad()
    def predict(self, Z: torch.Tensor, *, task_id: Optional[int] = None):
        res = self.observe(Z, task_id=task_id)
        return res.logits, res.selected
