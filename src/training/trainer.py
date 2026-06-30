"""Training loop ONLY.

This module owns the train/val loop and MLflow logging. It deliberately contains
NO test-set evaluation, confusion-matrix, GradCAM or drift logic — that all
lives in src/evaluate.py, which the trainer never imports. Keeping them separate
makes the evaluate step independently testable and matches the DVC `evaluate`
stage.

Entry point: `run_training(cfg)` (called from main.py when mode=train).
"""

from __future__ import annotations

import os
from typing import Any

import torch
from omegaconf import DictConfig

from src.data.dataset import build_loaders, compute_pos_weight, save_split_manifest
from src.models.model import build_model
from src.training.losses import build_criterion
from src.utils.config import set_seed
from src.utils.device import setup_ddp
from src.utils.logger import get_logger
from src.utils.lakefs_client import resolve_dataset_commit, sync_from_lakefs

log = get_logger(__name__)


def _run_epoch(model, loader, criterion, optimizer, device, train: bool) -> tuple[float, float]:
    model.train(mode=train)
    total_loss = 0.0
    correct = 0
    seen = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, targets in loader:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).squeeze(1)  # (B,)

            logits = model(imgs)  # (B,) raw logits
            loss = criterion(logits, targets)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds = (torch.sigmoid(logits) > 0.5).long()
            correct += (preds == targets.long()).sum().item()
            seen += imgs.size(0)
    return total_loss / max(seen, 1), correct / max(seen, 1)


def run_training(cfg: DictConfig) -> None:
    """Compose model + data + optimizer + MLflow and run the training loop."""
    set_seed(int(cfg.training.seed))

    # Always pull the exact dataset version from lakeFS first (env-gated), so a
    # clean checkout (DVC repro, Colab) has data/raw before splitting or training.
    if bool(cfg.env.get("sync_data", False)):
        sync_from_lakefs(cfg)

    # Preprocess-only mode: materialize the deterministic split manifest and exit.
    # Used by the DVC `preprocess` stage so data/processed is a real cacheable artifact.
    if bool(cfg.data.get("only_preprocess", False)):
        out_path = os.path.join(cfg.env.processed_dir, "split_manifest.json")
        save_split_manifest(cfg, out_path)
        log.info("only_preprocess=true -> wrote manifest and exiting without training")
        return

    import mlflow
    import mlflow.pytorch

    mlflow.set_tracking_uri(cfg.env.mlflow_tracking_uri)
    mlflow.set_experiment("Part_Inspection_Defect_Classification")

    info = setup_ddp(cfg)
    device = info.device
    if info.use_ddp:
        import torch.distributed as dist

        dist.init_process_group(backend="nccl")
        model = build_model(cfg).to(device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[info.local_rank])
    else:
        model = build_model(cfg).to(device)

    criterion = build_criterion(cfg).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
    )

    train_loader, val_loader, _ = build_loaders(cfg)

    # Optionally auto-set pos_weight from the actual class ratio (info-only here).
    if bool(cfg.training.get("auto_pos_weight", False)):
        pw = compute_pos_weight(cfg, train_loader)
        log.info("Computed pos_weight from data: %.4f (config had %.4f)", pw, float(cfg.training.pos_weight))

    os.makedirs(cfg.env.outputs_dir, exist_ok=True)

    with mlflow.start_run() as run:
        # Reproducibility payload: seed + exact dataset version + full config.
        mlflow.log_param("mode", cfg.mode.name)
        mlflow.log_param("seed", int(cfg.training.seed))
        mlflow.log_param("environment", cfg.env.name)
        mlflow.log_param("architecture", cfg.model.name)
        mlflow.log_param("batch_size", int(cfg.training.batch_size))
        mlflow.log_param("lr", float(cfg.training.lr))
        mlflow.log_param("pos_weight", float(cfg.training.pos_weight))
        mlflow.log_param("lakefs_commit", resolve_dataset_commit(cfg))
        mlflow.log_param("use_ddp", info.use_ddp)
        mlflow.log_param("world_size", info.world_size)

        best_val_acc = -1.0
        for epoch in range(int(cfg.training.epochs)):
            train_loss, train_acc = _run_epoch(model, train_loader, criterion, optimizer, device, train=True)
            val_loss, val_acc = _run_epoch(model, val_loader, criterion, optimizer, device, train=False)

            # Per-epoch metrics -> MLflow's UI plots the loss curve automatically.
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_acc, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)
            log.info("epoch %d/%d  train_loss=%.4f val_loss=%.4f val_acc=%.4f", epoch + 1, cfg.training.epochs, train_loss, val_loss, val_acc)

            # Save the best checkpoint (weights only; evaluation is a separate step).
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                ckpt = os.path.join(cfg.env.outputs_dir, "model.pt")
                state = model.module.state_dict() if info.use_ddp else model.state_dict()
                torch.save({"epoch": epoch, "state_dict": state, "val_acc": val_acc}, ckpt)

        # Final model artifact to the MLflow registry. We use pickle serialization
        # (not pt2/TorchScript) since our model is a plain nn.Sequential — pickle is
        # robust and needs no input_example/TensorSpec ceremony. The canonical
        # reproducible weights are outputs/model.pt (state dict) logged as an artifact.
        model_to_log = model.module if info.use_ddp else model
        mlflow.pytorch.log_model(
            model_to_log,
            "model",
            serialization_format="pickle",
            pip_requirements=["torch", "torchvision"],
        )
        mlflow.log_artifact(os.path.join(cfg.env.outputs_dir, "model.pt"))
        log.info("Logged run to MLflow: %s", run.info.run_id)

    # Write a DVC metrics file so the `train` stage's declared metrics output
    # is produced. Final test metrics live in src/evaluate.py; here we record
    # the training-side summary (best val acc + the config that produced it).
    import json

    metrics_path = os.path.join(cfg.env.outputs_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "best_val_accuracy": float(best_val_acc),
                "final_train_loss": float(train_loss),
                "final_val_loss": float(val_loss),
                "epochs": int(cfg.training.epochs),
                "seed": int(cfg.training.seed),
                "architecture": cfg.model.name,
            },
            f,
            indent=2,
        )
    log.info("Wrote metrics -> %s", metrics_path)

    if info.use_ddp:
        import torch.distributed as dist

        dist.destroy_process_group()
