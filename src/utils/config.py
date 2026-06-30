"""Hydra / OmegaConf helpers shared across the pipeline.

Everything that needs to read values from the composed config goes through
these helpers so the rest of the code stays config-agnostic and testable.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU + all CUDA) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cfg_to_container(cfg: DictConfig, resolve: bool = True) -> dict[str, Any]:
    """Convert a DictConfig to a plain dict (useful for MLflow param logging)."""
    return OmegaConf.to_container(cfg, resolve=resolve)  # type: ignore[return-value]


def get(cfg: DictConfig, key_path: str, default: Any = None) -> Any:
    """Dotted-path getter with a default, e.g. get(cfg, 'training.pos_weight', 1.0)."""
    node: Any = cfg
    for part in key_path.split("."):
        if node is None:
            return default
        if isinstance(node, DictConfig):
            if part not in node:
                return default
            node = node[part]
        elif isinstance(node, dict):
            if part not in node:
                return default
            node = node[part]
        else:
            return default
    return node
