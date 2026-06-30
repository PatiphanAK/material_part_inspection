"""Standalone evaluation: metrics + confusion matrix + GradCAM + Evidently drift.

This module is fully separated from training. The trainer never imports it and
never calls it. It is invoked through the single entrypoint:

    python main.py mode=evaluate env=local_rtx2080 model=resnet50

It owns four things:
  1. Test-set metrics (accuracy / precision / recall / F1) on the held-out split.
  2. Confusion matrix figure (ok=0, burr=1).
  3. GradCAM heatmaps verifying the model attends to the burr region.
  4. Evidently AI drift report comparing training vs. new-batch embeddings.

Each section degrades gracefully if its optional input is missing, so you can
run a partial evaluation (metrics only) without the drift CSVs in place.
"""

from __future__ import annotations

import os
from typing import Any

import torch
from omegaconf import DictConfig

from src.data.dataset import build_loaders
from src.models.model import build_model
from src.utils.config import set_seed
from src.utils.device import resolve_device
from src.utils.lakefs_client import resolve_dataset_commit, sync_from_lakefs
from src.utils.logger import get_logger

log = get_logger(__name__)


def _load_checkpoint(model: torch.nn.Module, cfg: DictConfig, device: torch.device) -> torch.nn.Module:
    ckpt_path = os.path.join(cfg.env.outputs_dir, "model.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path}. Train first with `python main.py mode=train ...`"
        )
    state = torch.load(ckpt_path, map_location=device)
    state_dict = state.get("state_dict", state)
    model.load_state_dict(state_dict)
    log.info("Loaded checkpoint from %s", ckpt_path)
    return model


@torch.no_grad()
def _test_inference(model: torch.nn.Module, loader, device: torch.device, threshold: float):
    """Run inference on the test loader, returning (y_true, y_pred_logit, y_pred_label)."""
    model.eval()
    y_true: list[int] = []
    logits_all: list[float] = []
    for imgs, targets in loader:
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs)  # (B,) raw logits (BinaryHead already squeezes to 1D)
        if logits.dim() > 1:
            logits = logits.squeeze(-1)
        logits_all.extend(logits.cpu().tolist())
        y_true.extend(targets.view(-1).long().cpu().tolist())
    y_pred = [1 if torch.sigmoid(torch.tensor(l)).item() > threshold else 0 for l in logits_all]
    return y_true, logits_all, y_pred


def _log_metrics(y_true, y_pred) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix (ok=0, burr=1)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for (i, j), v in __import__("numpy").ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    return {"accuracy": float(acc), "precision": float(precision), "recall": float(recall), "f1": float(f1)}, fig, cm


def _gradcam(model: torch.nn.Module, loader, cfg: DictConfig, device: torch.device, out_dir: str) -> None:
    """Generate GradCAM heatmaps for a handful of test images.

    Verifies the classifier focuses on the defect (burr) region rather than
    background. Falls back to a log message if grad-cam isn't installed.
    """
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError:
        log.warning("grad-cam not installed; skipping GradCAM heatmaps")
        return

    # Target the last conv layer of the backbone (GradCAM needs a real conv module).
    from src.models.model import last_conv_layer

    target_conv = last_conv_layer(model)
    if target_conv is None:
        log.warning("Could not locate a target conv layer for GradCAM")
        return
    target_layers = [target_conv]

    os.makedirs(out_dir, exist_ok=True)
    cam = GradCAM(model=model, target_layers=target_layers)

    # Our model emits a single binary logit (B,). grad-cam's default target
    # resolver assumes a multi-class output and chokes on 1D, so we supply an
    # explicit target whose __call__ returns the logit to maximize.
    class _BinaryLogitTarget:
        def __call__(self, output):
            return output  # maximize the single logit (burr probability)

    target_fn = _BinaryLogitTarget()
    shown = 0
    max_imgs = int(cfg.training.get("gradcam_samples", 8))
    import numpy as np

    for imgs, _ in loader:
        for b in range(imgs.size(0)):
            if shown >= max_imgs:
                log.info("Wrote %d GradCAM heatmaps to %s", shown, out_dir)
                return
            inp = imgs[b:b + 1].to(device)
            grayscale = cam(input_tensor=inp, targets=[target_fn])
            np.save(os.path.join(out_dir, f"cam_{shown:03d}.npy"), grayscale[0])
            shown += 1
    log.info("Wrote %d GradCAM heatmaps to %s", shown, out_dir)


def _evidently_drift(cfg: DictConfig, out_path: str) -> None:
    """Compare training vs. new-batch embedding CSVs with Evidently AI."""
    ref_csv = os.path.join(cfg.env.processed_dir, "metadata_embeddings.csv")
    cur_csv = os.path.join(cfg.env.raw_dir, "new_batch", "metadata_embeddings.csv")
    if not (os.path.isfile(ref_csv) and os.path.isfile(cur_csv)):
        log.info("Skipping Evidently drift (need %s and %s)", ref_csv, cur_csv)
        # Always emit a stub so DVC's declared `outs:` is satisfied; the real
        # report is generated once embedding CSVs are in place.
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            f.write(
                "<html><body><h1>Drift report not generated</h1>"
                "<p>Provide data/processed/metadata_embeddings.csv and "
                "data/raw/new_batch/metadata_embeddings.csv to generate it.</p></body></html>"
            )
        return
    import pandas as pd
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset
    from evidently.report import Report

    ref = pd.read_csv(ref_csv)
    cur = pd.read_csv(cur_csv)
    report = Report(metrics=[DataQualityPreset(), DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    report.save_html(out_path)
    log.info("Evidently drift report -> %s", out_path)


def run_evaluation(cfg: DictConfig) -> None:
    """End-to-end evaluation: metrics + confusion matrix + GradCAM + drift."""
    set_seed(int(cfg.training.seed))

    if bool(cfg.env.get("sync_data", False)):
        sync_from_lakefs(cfg)

    device = resolve_device(cfg)
    model = build_model(cfg).to(device)
    model = _load_checkpoint(model, cfg, device)

    _, _, test_loader = build_loaders(cfg)
    threshold = float(cfg.training.threshold)

    y_true, _logits, y_pred = _test_inference(model, test_loader, device, threshold)
    metrics, cm_fig, _ = _log_metrics(y_true, y_pred)
    log.info("test metrics: %s", {k: round(v, 4) for k, v in metrics.items()})

    os.makedirs(cfg.env.reports_dir, exist_ok=True)
    cm_fig.savefig(os.path.join(cfg.env.reports_dir, "confusion_matrix.png"))

    _gradcam(model, test_loader, cfg, device, os.path.join(cfg.env.reports_dir, "gradcam"))
    _evidently_drift(cfg, os.path.join(cfg.env.reports_dir, "drift_report.html"))

    # Log evaluation metrics to MLflow alongside the training run's artifacts.
    import mlflow

    mlflow.set_tracking_uri(cfg.env.mlflow_tracking_uri)
    with mlflow.start_run(run_name=f"evaluate-{cfg.model.name}") as run:
        mlflow.log_param("mode", cfg.mode.name)
        mlflow.log_param("architecture", cfg.model.name)
        mlflow.log_param("lakefs_commit", resolve_dataset_commit(cfg))
        mlflow.log_param("threshold", threshold)
        for k, v in metrics.items():
            mlflow.log_metric(f"test_{k}", v)
        mlflow.log_artifact(os.path.join(cfg.env.reports_dir, "confusion_matrix.png"))
    log.info("Evaluation complete. Reports in %s", cfg.env.reports_dir)
