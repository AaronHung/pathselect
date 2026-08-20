"""ZeroNav router — CONCH text-image alignment only, no prototype features.

ZeroSlide-inspired: uses max cosine similarity between patch embeddings
and class text embeddings (CONCH) as the navigation foundation.

Key difference from MicroRouterV0:
  summary: 4-D [max_txt, txt_ent, max_proto, mean_proto]
        →  2-D [max_txt, txt_ent]          (prototype terms removed)
  MLP input: 516-D → 514-D
  backbone call: prototype_features() removed entirely
"""
from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def text_nav_feats(Z: torch.Tensor, f_txt: torch.Tensor) -> torch.Tensor:
    """2-D per-patch text-image alignment summary (ZeroSlide-inspired).

    Uses CONCH text-patch cosine similarity only — no prototype features.

    Z:     [n, D=512]  CONCH patch embeddings
    f_txt: [C, D=512]  CONCH class text embeddings
    returns [n, 2]: [max_text_sim, entropy(text_sim_distribution)]
    """
    z = F.normalize(Z, dim=-1)
    t = F.normalize(f_txt, dim=-1)
    txt = z @ t.t()                                               # [n, C]
    txt_prob = F.softmax(txt, dim=-1)
    txt_ent = -(txt_prob * (txt_prob + 1e-9).log()).sum(-1, keepdim=True)  # [n,1]
    return torch.cat([txt.amax(-1, keepdim=True), txt_ent], dim=-1)        # [n, 2]


def zeroslide_score(Z: torch.Tensor, f_txt: torch.Tensor) -> torch.Tensor:
    """ZeroSlide zero-shot patch importance score.

    max cosine similarity between CONCH patch embeddings and class text
    embeddings — analogous to ZeroSlide's patch scoring with TITAN/CONCH.
    No training required.

    Z: [n, D], f_txt: [C, D]  →  score [n]
    """
    z = F.normalize(Z, dim=-1)
    t = F.normalize(f_txt, dim=-1)
    return (z @ t.t()).amax(-1)                                   # [n]


class TextNavRouter(nn.Module):
    """ZeroNav learned navigation router.

    ZeroSlide-inspired: CONCH patch embedding + 2-D text-alignment summary
    → learned importance score.  No prototype features from QPMIL.

    Architecture: Linear(D+2 → 256) → GELU → Linear(256 → 1)
    Default D=512 → 514-D input, ~130 K params (cf. MicroRouterV0 ~132 K).

    forward(Z, f_txt) — no F_p argument.
    """

    def __init__(self, feat_dim: int = 512, hidden: int = 256):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden = hidden
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim + 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, Z: torch.Tensor, f_txt: torch.Tensor) -> torch.Tensor:
        """
        Z:     [n, D]  CONCH patch embeddings
        f_txt: [C, D]  CONCH class text embeddings
        returns score [n] — no prototype_features() call.
        """
        s = text_nav_feats(Z, f_txt)          # [n, 2]
        u = torch.cat([Z, s], dim=-1)         # [n, D+2]
        return self.mlp(u).squeeze(-1)        # [n]


class ZeroNavSkillBank:
    """Navigation Skill Memory for ZeroNav (TextNavRouter).

    task_id → TextNavRouter state_dict.
    Drop-in replacement for NavigationSkillBank without prototype dependency.
    """

    def __init__(self, feat_dim: int = 512, hidden: int = 256):
        self.feat_dim = feat_dim
        self.hidden = hidden
        self._skills: dict[int, dict] = {}

    # ── skill management ────────────────────────────────────────────────────

    def add_skill(self, task_id: int, router_or_state) -> None:
        state = (router_or_state.state_dict()
                 if hasattr(router_or_state, "state_dict") else router_or_state)
        self._skills[int(task_id)] = copy.deepcopy(state)

    def has(self, task_id: int) -> bool:
        return int(task_id) in self._skills

    def task_ids(self) -> list[int]:
        return sorted(self._skills)

    def __len__(self) -> int:
        return len(self._skills)

    def build_router(self, task_id: int, device=None) -> TextNavRouter:
        router = TextNavRouter(feat_dim=self.feat_dim, hidden=self.hidden)
        router.load_state_dict(self._skills[int(task_id)])
        if device is not None:
            router = router.to(device)
        return router.eval()

    def weight_vectors(self) -> dict[int, torch.Tensor]:
        """Flat parameter vector per task (for cosine similarity analysis)."""
        return {
            int(tid): torch.cat([v.reshape(-1) for v in state.values()])
            for tid, state in self._skills.items()
        }

    # ── persistence ─────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        torch.save({
            "type": "zeronav_v1",
            "feat_dim": self.feat_dim,
            "hidden": self.hidden,
            "skills": self._skills,
        }, path)
        print(f"[ZeroNavSkillBank] saved {len(self)} skills → {path}")

    @classmethod
    def load(cls, path: str, map_location: str = "cpu") -> "ZeroNavSkillBank":
        blob = torch.load(path, map_location=map_location)
        bank = cls(feat_dim=blob["feat_dim"], hidden=blob["hidden"])
        bank._skills = blob["skills"]
        print(f"[ZeroNavSkillBank] loaded {len(bank)} skills from {path}")
        return bank
