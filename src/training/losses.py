"""Binary classification loss.

BCEWithLogitsLoss takes a raw logit and applies sigmoid internally (numerically
stable). `pos_weight` up-weights the rare burr class (label=1) to handle the
expected ok/burr imbalance; its value comes from `conf/training/*.yaml`.
"""

from __future__ import annotations

import torch
from omegaconf import DictConfig


def build_criterion(cfg: DictConfig) -> torch.nn.Module:
    pos_weight_val = float(cfg.training.pos_weight)
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32)
    return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
