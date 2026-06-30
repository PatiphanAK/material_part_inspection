"""Image transforms for train / val / test splits.

Augmentation is applied to the TRAIN split only; val and test get a
deterministic resize + normalize so metrics are comparable across runs.
"""

from __future__ import annotations

from typing import Any

import torchvision.transforms as T
from omegaconf import DictConfig


def _normalize(cfg: DictConfig) -> T.Normalize:
    data_cfg = cfg.data
    return T.Normalize(mean=list(data_cfg.normalize.mean), std=list(data_cfg.normalize.std))


def build_train_transform(cfg: DictConfig) -> T.Compose:
    """Train transform: resize + (optional) augmentation + normalize."""
    size = int(cfg.data.image_size)
    aug = cfg.data.augment
    steps: list[Any] = [T.Resize((size, size))]

    if bool(aug.enabled):
        if bool(aug.horizontal_flip):
            steps.append(T.RandomHorizontalFlip(p=0.5))
        if bool(aug.vertical_flip):
            steps.append(T.RandomVerticalFlip(p=0.5))
        steps.append(T.RandomRotation(degrees=float(aug.rotation_degrees)))
        cj = aug.color_jitter
        steps.append(
            T.ColorJitter(
                brightness=float(cj.brightness),
                contrast=float(cj.contrast),
                saturation=float(cj.saturation),
                hue=float(cj.hue),
            )
        )

    steps.extend([T.ToTensor(), _normalize(cfg)])
    return T.Compose(steps)


def build_eval_transform(cfg: DictConfig) -> T.Compose:
    """Deterministic transform for val/test (no augmentation)."""
    size = int(cfg.data.image_size)
    return T.Compose([T.Resize((size, size)), T.ToTensor(), _normalize(cfg)])
