"""Device-agnostic helper：Mac M1 (MPS) 開發、CUDA (RunPod) 跑重活共用同一份 code。

用法：
    from navipath_moe.device import get_device, setup_mps
    setup_mps()                       # 在 import torch 後、建模型前呼叫一次
    device = get_device()             # 自動選 cuda > mps > cpu
    model.to(device); Z = Z.to(device)

MPS 兩個坑：
  1. 不支援 float64 — 一律用 float32（本專案特徵本來就是 float32）。
  2. 少數 op 尚未實作 — setup_mps() 會開 PYTORCH_ENABLE_MPS_FALLBACK 自動退回 CPU。
"""
from __future__ import annotations

import os
import torch


def setup_mps():
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def get_device(prefer: str = "auto") -> torch.device:
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer in ("cuda", "auto") and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer in ("mps", "auto") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
