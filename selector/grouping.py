"""Tissue grouping — 用固定的 tissue prompt 把 patch 分成 J 個語意 group。

流程：
  1. 8 條 tissue prompt（下方常數，**不動態組**）→ CONCH text 編碼 → t_j [J, D]
  2. 每個 patch 取 argmax cosine 指派到唯一一個 group
  3. group prototype g_j = 該組 patch embedding 的平均
  4. 空 group 保留（維持 J 固定、index 穩定）但標記 mask=False，不參與 budget 分配

group 數 J = 8，與 chunk c = 8 沒有關係，只是剛好同數。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .text_encoder import encode_prompt_groups

#: 8 個 tissue group 的 prompt ensemble（pre-registered 常數，不得動態生成）。
#: 涵蓋 H&E WSI 上的常見組織成分，用途是「把 patch 分區」而非診斷分類 ——
#: 這些字串與任何 task 的 class label 無關，四個 task 共用同一組。
TISSUE_PROMPTS: tuple[tuple[str, ...], ...] = (
    ("tumor tissue.", "an H&E image of tumor tissue.", "malignant epithelial cells."),
    ("stroma.", "an H&E image of fibrous stroma.", "desmoplastic stromal tissue."),
    ("lymphocytes.", "an H&E image of lymphocytic infiltrate.", "immune cell infiltration."),
    ("necrosis.", "an H&E image of necrotic tissue.", "necrotic debris."),
    ("normal epithelium.", "an H&E image of benign epithelium.", "non-neoplastic epithelial tissue."),
    ("blood vessels.", "an H&E image of blood vessels.", "red blood cells."),
    ("adipose tissue.", "an H&E image of fat tissue.", "adipocytes."),
    ("background.", "an H&E image of slide background.", "blank glass slide area."),
)
TISSUE_GROUP_NAMES: tuple[str, ...] = (
    "tumor", "stroma", "lymphocyte", "necrosis",
    "normal_epithelium", "vessel", "adipose", "background",
)
NUM_GROUPS = len(TISSUE_PROMPTS)
assert len(TISSUE_GROUP_NAMES) == NUM_GROUPS


@dataclass
class Grouping:
    """一張 slide 的分組結果。"""
    assignment: torch.Tensor      # [n] long，每個 patch 的 group index
    prototypes: torch.Tensor      # [J, D]，空 group 為零向量
    mask: torch.Tensor            # [J] bool，True = 非空
    sizes: torch.Tensor           # [J] long

    @property
    def num_groups(self) -> int:
        return int(self.mask.shape[0])

    def members(self, j: int) -> torch.Tensor:
        return (self.assignment == j).nonzero(as_tuple=False).reshape(-1)


def tissue_text_features(cfg: dict | None = None, *, device="cpu",
                         refresh: bool = False) -> torch.Tensor:
    """[J, D]：tissue prompt 的 CONCH text 特徵（已 L2 normalize，有 cache）。"""
    return encode_prompt_groups([list(p) for p in TISSUE_PROMPTS], cfg,
                                device=device, cache_name="f_tissue",
                                refresh=refresh)


@torch.no_grad()
def assign_groups(Z: torch.Tensor, t_tissue: torch.Tensor) -> Grouping:
    """argmax cosine 指派 + group prototype。

    Z:        [n, D] patch embedding
    t_tissue: [J, D] tissue text 特徵
    """
    if Z.dim() != 2 or t_tissue.dim() != 2:
        raise ValueError(f"Z {tuple(Z.shape)}, t_tissue {tuple(t_tissue.shape)}")
    J, D = t_tissue.shape
    z = F.normalize(Z, dim=-1)
    t = F.normalize(t_tissue, dim=-1)
    assignment = (z @ t.t()).argmax(dim=-1)                        # [n]

    onehot = F.one_hot(assignment, num_classes=J).to(Z.dtype)      # [n, J]
    sizes = onehot.sum(0)                                          # [J]
    summed = onehot.t() @ Z                                        # [J, D]
    mask = sizes > 0
    prototypes = torch.zeros_like(summed)
    if bool(mask.any()):
        prototypes[mask] = summed[mask] / sizes[mask].unsqueeze(-1)
    return Grouping(assignment=assignment, prototypes=prototypes,
                    mask=mask, sizes=sizes.to(torch.long))


def group_capacity(grouping: Grouping, available: torch.Tensor) -> torch.Tensor:
    """[J] long：每個 group 目前還剩幾個「可選」patch（扣掉已選的）。"""
    J = grouping.num_groups
    onehot = F.one_hot(grouping.assignment, num_classes=J)
    return (onehot * available.reshape(-1, 1)).sum(0).to(torch.long)
