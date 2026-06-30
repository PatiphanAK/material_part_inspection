"""PyTorch Dataset + DataLoader builders for binary burr classification.

Label mapping is declared once in `conf/data/*.yaml` (ok=0, burr=1) and read
from config here, so no other module hardcodes the label semantics.

Expected raw layout (after lakeFS pull into cfg.env.raw_dir):

    <raw_dir>/
        ok/   *.jpg / *.png ...
        burr/ *.jpg / *.png ...

`prepare_splits` walks that tree once, deterministically splits by seed, and
caches the file lists. `build_loaders` wires up the three DataLoaders.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

from src.data.transforms import build_eval_transform, build_train_transform
from src.utils.logger import get_logger

log = get_logger(__name__)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


@dataclass
class SplitPaths:
    """The three split directories produced by `prepare_splits`.

    For the simplest flow we keep a single ImageFolder root and split by index;
    `prepare_splits` returns the per-split index lists instead of copying files.
    """

    root: Path
    train_idx: list[int]
    val_idx: list[int]
    test_idx: list[int]


def _label_map(cfg: DictConfig) -> dict[str, int]:
    """ok=0, burr=1 (or whatever is declared in conf/data)."""
    return {str(k): int(v) for k, v in cfg.data.label_map.items()}


class BinaryImageFolder(Dataset):
    """ImageFolder-style dataset that yields (image_tensor, label) with label from config.

    label is a float tensor shaped [1] so it drops straight into BCEWithLogitsLoss.
    """

    def __init__(self, root: str, label_map: dict[str, int], transform) -> None:
        self.root = Path(root)
        self.transform = transform
        self.samples: list[tuple[str, int]] = []
        for class_name, label in label_map.items():
            class_dir = self.root / class_name
            if not class_dir.is_dir():
                continue
            for p in sorted(class_dir.rglob("*")):
                if p.suffix.lower() in IMAGE_EXTS:
                    self.samples.append((str(p), label))
        if not self.samples:
            raise RuntimeError(f"No images found under {self.root} for classes {list(label_map)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        from PIL import Image

        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        target = torch.tensor([float(label)], dtype=torch.float32)  # [1] for BCEWithLogitsLoss
        return img, target


class _SubsetDataset(Dataset):
    """Wrap a base dataset and expose only a subset of indices."""

    def __init__(self, base: BinaryImageFolder, indices: Sequence[int]) -> None:
        self.base = base
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.base[self.indices[idx]]


def prepare_splits(cfg: DictConfig) -> tuple[BinaryImageFolder, BinaryImageFolder, BinaryImageFolder]:
    """Build train/val/test datasets over the raw image tree, split deterministically by seed.

    We use two transform instances (train vs eval) and split indices rather than
    copying files. Splits are reproducible because cfg.data.split.seed is fixed.
    """
    label_map = _label_map(cfg)
    root = cfg.env.raw_dir
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"Raw data dir not found: {root}. Run the lakeFS pull (scripts/sync_lakefs.sh) "
            "or set env to point at an existing data root."
        )

    # Enumerate every sample once to split indices deterministically.
    full = BinaryImageFolder(root, label_map, transform=build_eval_transform(cfg))
    n = len(full)
    g = torch.Generator().manual_seed(int(cfg.data.split.seed))
    perm = torch.randperm(n, generator=g).tolist()

    s = cfg.data.split
    n_train = int(n * float(s.train))
    n_val = int(n * float(s.val))
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]

    train_ds = BinaryImageFolder(root, label_map, transform=build_train_transform(cfg))
    val_ds = full  # shares the eval transform; we subset it below

    # Subsets share the underlying samples but use their own transform by being
    # distinct BinaryImageFolder instances for train vs val/test.
    return (
        _SubsetDataset(train_ds, train_idx),  # type: ignore[return-value]
        _SubsetDataset(val_ds, val_idx),  # type: ignore[return-value]
        _SubsetDataset(val_ds, test_idx),  # type: ignore[return-value]
    )


def build_loaders(
    cfg: DictConfig,
) -> tuple[DataLoader, DataLoader, DataLoader] | tuple[DataLoader, DataLoader]:
    """Build the DataLoaders. Train returns (train, val); eval returns (val, test) by default.

    For training we return (train, val); test is held back for src/evaluate.py.
    """
    train_ds, val_ds, test_ds = prepare_splits(cfg)
    bs = int(cfg.training.batch_size)
    nw = int(cfg.training.num_workers)

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)

    log.info("splits: train=%d val=%d test=%d", len(train_ds), len(val_ds), len(test_ds))
    return train_loader, val_loader, test_loader


def save_split_manifest(cfg: DictConfig, out_path: str) -> None:
    """Materialize the deterministic split manifest (train/val/test file lists).

    The preprocess DVC stage calls this so `data/processed/` is a real, cacheable
    artifact and the exact split (seed-locked) is reproducible without re-training.
    """
    train_ds, val_ds, test_ds = prepare_splits(cfg)
    import json

    manifest = {
        "seed": int(cfg.data.split.seed),
        "label_map": {str(k): int(v) for k, v in cfg.data.label_map.items()},
        "counts": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
        # Subset indices into the (deterministically enumerated) raw tree.
        "train_idx": train_ds.indices,
        "val_idx": val_ds.indices,
        "test_idx": test_ds.indices,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("Wrote split manifest -> %s (train=%d val=%d test=%d)", out_path, len(train_ds), len(val_ds), len(test_ds))


def compute_pos_weight(cfg: DictConfig, loader: DataLoader) -> float:
    """Compute pos_weight = #ok / #burr from a loader's labels (burr is the positive class=1).

    Use this to sanity-check cfg.training.pos_weight, or to set it automatically.
    """
    n_pos = n_neg = 0
    for _, y in loader:
        n_pos += int((y == 1).sum().item())
        n_neg += int((y == 0).sum().item())
    if n_pos == 0:
        return float(cfg.training.pos_weight)
    return round(n_neg / n_pos, 4)
