"""Model definitions for binary burr classification.

The head produces a SINGLE output logit per image. There is NO softmax/sigmoid
in the model — `BCEWithLogitsLoss` (src/training/losses.py) applies the sigmoid
internally for numerical stability. At inference, threshold the logit at 0.

Backbones are torchvision models with the classifier swapped for a 1-neuron head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.utils.logger import get_logger

log = get_logger(__name__)

_BACKBONES = {"resnet50", "resnet18", "resnet34", "mobilenet_v3_large", "mobilenet_v3_small"}


class BinaryHead(nn.Module):
    """1-neuron linear head -> single logit. No activation."""

    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(1)  # (B,) logits


class BinaryClassifier(nn.Module):
    """backbone -> BinaryHead. Exposes `backbone` so GradCAM can target its conv layers."""

    def __init__(self, backbone: nn.Module, head: BinaryHead) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def _build_backbone(cfg: DictConfig) -> tuple[nn.Module, int]:
    name = cfg.model.name
    weights = "DEFAULT" if bool(cfg.model.pretrained) else None
    if name not in _BACKBONES:
        raise ValueError(f"Unknown backbone {name!r}. Supported: {sorted(_BACKBONES)}")
    import torchvision.models as tvm

    ctor = getattr(tvm, name)
    net = ctor(weights=weights)

    # ResNet family
    if name.startswith("resnet"):
        in_features = net.fc.in_features
        net.fc = nn.Identity()
        return net, in_features
    # MobileNetV3 family
    if name.startswith("mobilenet_v3"):
        in_features = net.classifier[-1].in_features
        net.classifier = nn.Identity()
        return net, in_features

    raise ValueError(f"Backbone {name!r} wiring not implemented")


def build_model(cfg: DictConfig) -> nn.Module:
    """Build the full model: backbone -> BinaryHead. Returns logit (B,)."""
    backbone, in_features = _build_backbone(cfg)
    model = BinaryClassifier(backbone, BinaryHead(in_features))

    if bool(cfg.model.freeze_backbone):
        for p in backbone.parameters():
            p.requires_grad = False
        log.info("Backbone frozen; training head only")

    log.info("Built model %s (pretrained=%s) with single-logit binary head", cfg.model.name, cfg.model.pretrained)
    return model


def last_conv_layer(model: nn.Module) -> nn.Module | None:
    """Return the last Conv2d in the model's backbone (GradCAM target)."""
    backbone = getattr(model, "backbone", model)
    last = None
    for m in backbone.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last
