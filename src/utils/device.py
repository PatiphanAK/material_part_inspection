"""Device resolution and multi-GPU (DDP) setup.

All device decisions route through here so nothing in the training loop
hardcodes a device string. The env config (`conf/env/*.yaml`) drives both the
target device and whether DDP is used.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from omegaconf import DictConfig

from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class DeviceInfo:
    device: torch.device
    local_rank: int
    world_size: int
    use_ddp: bool


def resolve_device(cfg: DictConfig) -> torch.device:
    """Resolve a single torch.device from cfg.env.device (no DDP bookkeeping)."""
    requested = cfg.env.device
    if requested.startswith("cuda") and not torch.cuda.is_available():
        log.warning("CUDA requested but unavailable -> falling back to CPU")
        return torch.device("cpu")
    return torch.device(requested)


def setup_ddp(cfg: DictConfig) -> DeviceInfo:
    """Prepare the device + DDP world info for training.

    Returns a DeviceInfo even when DDP is disabled (world_size=1, local_rank=0),
    so the trainer can branch on `use_ddp` without caring about the env shape.

    When DDP is enabled this expects the launcher (torchrun / spawn) to have set
    RANK / WORLD_SIZE / LOCAL_RANK in the environment.
    """
    use_ddp = bool(cfg.env.get("use_ddp", False)) and torch.cuda.is_available()

    if not use_ddp:
        return DeviceInfo(device=resolve_device(cfg), local_rank=0, world_size=1, use_ddp=False)

    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        log.warning("DDP requested but RANK/WORLD_SIZE not set -> running single-process")
        return DeviceInfo(device=resolve_device(cfg), local_rank=0, world_size=1, use_ddp=False)

    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    log.info("DDP world_size=%d rank=%d local_rank=%d", world_size, rank, local_rank)
    return DeviceInfo(device=device, local_rank=local_rank, world_size=world_size, use_ddp=True)
