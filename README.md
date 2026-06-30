# Material Part Inspection - ML Experimental

This repository contains the experimental MLOps framework for the **Material Part Inspection** image classification project. It's designed as a portable, highly-repeatable research loop that runs identically whether you're on a local GPU workstation or a **Google Colab notebook/CLI**, integrating data versioning, experiment tracking, config management, pipeline reproducibility, and data drift monitoring.

This environment is built strictly for **experimental and development phases** (no model-serving API or production deployment pipelines are included in this initial iteration). Every run is reproducible by design: fixed seed, exact dataset version (lakeFS commit), and exact config (Hydra) are logged together with every result, so anyone cloning this repo can reproduce the reported numbers exactly.

---

## 🖥️ Supported Environments

The pipeline is environment-agnostic by design — all paths, device selection, and remote endpoints are resolved through config rather than hardcoded, so the same `src/` code runs in any of these:

| Environment | Notes |
|---|---|
| **Local Workstation Main** | 2x NVIDIA GeForce RTX 2080 (16GB VRAM each), CUDA 13 |
| **Local Workstation C** | 1x NVIDIA GeForce RTX 3050 CUDA 13|
| **Google Colab (CLI/notebook)** | Single GPU runtime (T4/A100 depending on tier), ephemeral disk — requires lakeFS for data persistence since `data/` is wiped on disconnect |

Device selection, MLflow tracking URI, and lakeFS endpoint are all pulled from Hydra config (`conf/`), so switching environments is a one-line override, not a code change.

---

## 🏗️ Experimental Stack

```
                  +-----------------------------------------+
                  |               lakeFS                    |
                  |  (Raw & Versioned Image Data Lake)      |
                  +--------------------+--------------------+
                                       |
                                       v (S3 API Client)
+--------------------------------------+--------------------------------------+
|                                                                             |
|  Pipeline Orchestration (runs identically: local GPU workstation or Colab)  |
|                                                                             |
|  +-------------------------+                 +---------------------------+  |
|  |        Hydra            |                 |        Evidently AI       |  |
|  |  (Config composition:   |                 |  (Data Drift, Embeddings  |  |
|  |   env / model / train)  |                 |   & Feature Quality)      |  |
|  +------------+------------+                 +-------------+-------------+  |
|               |                                            ^                |
|               v                                            |                |
|  +------------+------------+                 +-------------+-------------+  |
|  |           DVC           |                 |                           |  |
|  |  (Pipeline DAG &        |---------------->|         MLflow            |  |
|  |   Cached Artifacts)     |                 |   (Experiment Tracking &  |  |
|  +------------+------------+                 |    Artifact Registry)     |  |
|               |                               +---------------------------+  |
|               v                                                             |
|  +------------+------------+                                                |
|  |       PyTorch CLI       |                                                |
|  |   (Model Training,      |                                                |
|  |    GPU via CUDA)        |                                                |
|  +-------------------------+                                                |
+-----------------------------------------------------------------------------+
```

1. **lakeFS (Data Lake Versioning):**
   Acts as the version control layer for raw and processed image datasets, with Git-like branching (`main`, `experiment/new-dataset-v2`) on top of object storage. This is also what makes Colab runs viable — since Colab disk is ephemeral, every run pulls data fresh from a lakeFS branch+commit rather than relying on local cache.
2. **Hydra (Config Composition):**
   Composes environment, model, and training config from small YAML fragments (`conf/env/`, `conf/model/`, `conf/training/`) instead of hardcoded flags or scattered constants. This is the layer that makes "run anywhere" practical — `env=colab` swaps device/paths/endpoints without touching code.
3. **DVC (Data Version Control & Pipeline DAG):**
   Manages the execution pipeline (`dvc.yaml`). Tracks lightweight pointers to artifacts and ensures reproducible steps (Preprocessing → Training → Evaluation), parameterized by the active Hydra config.
4. **MLflow (Experiment Tracking & Model Registry):**
   Logs parameters, training hyperparameters, runs, metrics, confusion matrices, and final weights to a **shared remote tracking server** (not localhost), so runs from different workstations and Colab sessions land in the same dashboard.
5. **Evidently AI (Data & Model Drift Analysis):**
   Performs offline evaluations comparing training data against new validation/test sets — image quality metrics and feature embedding drift — to catch semantic drift before registering models.
6. **GradCAM (Explainability)**
   Generates class activation heatmaps to visualize which image regions contribute most to the model's predictions. Used to verify that the classifier focuses on the relevant defect area (e.g., burr) rather than irrelevant background features, supporting qualitative error analysis and model interpretability. Runs inside `src/evaluate.py` as part of the standalone evaluation step.
---

## 📂 Project Structure

```text
part_inspection/
├── agent.md                    # Agent brief: PyTorch + Hydra experimental scope
├── main.py                     # Thin Hydra entrypoint — dispatches via mode= (no training logic)
├── pyproject.toml              # uv-managed deps (torch, hydra-core, mlflow, grad-cam, evidently, ...)
├── docker-compose.yaml         # lakeFS + MLflow services
├── README.md
│
├── conf/                       # Hydra config root — every experiment is config-driven
│   ├── config.yaml             # defaults list composes: mode + env + model + training + data
│   ├── mode/
│   │   ├── train.yaml          # mode=train    → runs src/training/trainer.py
│   │   └── evaluate.yaml       # mode=evaluate → runs src/evaluate.py (GradCAM + drift)
│   ├── env/                    # {local_rtx2080, local_rtx3050, colab}.yaml — device/paths/endpoints
│   ├── model/                  # resnet50.yaml, mobilenetv3.yaml — binary head: 1 logit
│   ├── training/               # default.yaml — epochs, batch_size, lr, seed, pos_weight
│   └── data/                   # default.yaml — image size, augment strength, split ratios
│
├── data/                       # git-ignored; synced from lakeFS branch+commit
│   ├── raw/                    # original images organized as ok/ and burr/
│   └── processed/             # resized + normalized train/val/test splits
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── dataset.py          # PyTorch Dataset, folder→label map: ok=0, burr=1
│   │   └── transforms.py       # train/val/test transforms (augmentation on train only)
│   ├── models/                 # NOTE: plural, per project convention
│   │   └── model.py            # backbone builders + binary head (single output logit, NO softmax)
│   ├── training/
│   │   ├── trainer.py          # training loop ONLY (no eval logic) — MLflow logging, DDP
│   │   └── losses.py           # BCEWithLogitsLoss + pos_weight for class imbalance
│   ├── evaluate.py             # standalone: metrics + confusion matrix + GradCAM + Evidently drift
│   └── utils/
│       ├── config.py           # Hydra/Omegaconf helpers (resolve env, seed setup)
│       ├── device.py           # torch.device resolution; multi-GPU/DDP device prep
│       ├── lakefs_client.py    # S3-compatible lakeFS client, env-aware
│       └── logger.py           # structured logging setup
│
├── scripts/                    # setup_colab.sh, sync_lakefs.sh (env bootstrap helpers)
├── notebooks/                  # EDA + colab_runner.ipynb
├── docs/
│   └── experimental.md         # experimental design (already exists)
├── assets/                     # figures (Figure 1, etc.)
└── dvc.yaml                    # pipeline DAG: preprocess → train → evaluate(drift)
```

### Single-entrypoint dispatch (`main.py` is the only entrypoint)

`main.py` is intentionally thin — it contains **no training or evaluation logic**, only a dispatch on `cfg.mode.name` to the relevant module:

```python
# main.py
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    if cfg.mode.name == "train":
        from src.training.trainer import run_training
        run_training(cfg)
    elif cfg.mode.name == "evaluate":
        from src.evaluate import run_evaluation
        run_evaluation(cfg)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode.name}")

if __name__ == "__main__":
    main()
```

```bash
# Run an experiment end-to-end, zero code changes:
python main.py mode=train    env=local_rtx2080 model=resnet50 training=default data=default
python main.py mode=evaluate env=local_rtx2080 model=resnet50

# One-line override of anything, on any machine:
python main.py mode=train env=colab training.epochs=20 training.batch_size=16
```

### Binary-classification design (locked into the structure)

| Concern | Realized in |
|---|---|
| Single output **logit**, no softmax/sigmoid in the model | `src/models/model.py` (1-neuron head) |
| `BCEWithLogitsLoss` + `pos_weight` for class imbalance | `src/training/losses.py` (`pos_weight` from `conf/training/`) |
| Label mapping `ok=0`, `burr=1` (declared once, in config) | `src/data/dataset.py` (map lives in `conf/data/`) |
| Evaluation fully separated, never mixed into trainer | `src/evaluate.py` (trainer never imports it) |

### System goals → where each is realized

| Goal | Realized in |
|---|---|
| Reproducible (seed + lakeFS commit logged together) | `src/utils/config.py` + MLflow `log_param("lakefs_commit")` in `trainer.py` |
| Fully config-driven, no code changes per experiment | `conf/` composition + `main.py` dispatch |
| Multi-GPU training (PyTorch DDP) | `src/utils/device.py` (DDP world setup) + `src/training/trainer.py` |
| lakeFS dataset versioning | `src/utils/lakefs_client.py` (env-aware endpoint) |
| DVC pipeline integration | `dvc.yaml` stages invoke `main.py mode=...` |

Key change from a single-machine layout: nothing in `src/` reads a hardcoded path, host, or device string — everything routes through `conf/env/*.yaml`. Adding a new machine (or a Colab Pro session with different disk paths) means adding one new env file, not touching pipeline code.

---

## 📊 Results

| Model | Seed | Test Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Run |
|---|---|---|---|---|---|---|
| ResNet50 | 42 | — | — | — | — | [MLflow run link] |

*Run `dvc repro` to reproduce this row exactly — config, seed, and dataset commit for each result are logged in the linked MLflow run. Table will be filled in as runs complete.*

## 🚀 Reproduce

```bash
git clone https://github.com/your-org/part_inspection.git
cd part_inspection
uv sync      # installs deps from pyproject.toml into a managed venv
```

This pulls the exact dataset version from lakeFS, trains with the fixed seed, and logs everything (config, loss curve, eval metrics, confusion matrix, model weights) to MLflow — no manual bookkeeping required to reproduce a result.

### Option A — Local Workstation

```bash
# Pick the env file matching this machine (single entrypoint, mode= dispatches)
python main.py mode=train    env=local_rtx2080 model=resnet50 training=default data=default
python main.py mode=evaluate env=local_rtx2080 model=resnet50
```

### Option B — Google Colab (CLI-style)

Run from a Colab cell (or `!` CLI in a notebook):

```bash
!git clone https://github.com/your-org/part_inspection.git
%cd part_inspection
!bash scripts/setup_colab.sh        # installs deps, configures lakeFS creds from Colab secrets

!python main.py mode=train env=colab model=resnet50 training=default data=default
```

`scripts/setup_colab.sh` handles the parts that differ on Colab: installing dependencies into the ephemeral runtime, pulling lakeFS credentials from `google.colab.userdata` instead of a local `.env`, and pointing the MLflow tracking URI at the shared remote server so the run shows up alongside local runs.

*Verifying GPU in either environment:*
```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Device Count: {torch.cuda.device_count()}")
print(f"Current Device Name: {torch.cuda.get_device_name(0)}")
```

---

## 🔧 Component Integration & Workflow

### Step 1: Data Lake Operations with lakeFS

```python
# src/utils/lakefs_client.py
import os
import boto3

s3 = boto3.client(
    's3',
    endpoint_url=os.getenv('LAKEFS_ENDPOINT'),   # set per-env: local URL or remote/Colab-reachable URL
    aws_access_key_id=os.getenv('LAKEFS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('LAKEFS_SECRET_ACCESS_KEY')
)

bucket_name = "part-inspection"
branch = "experiment-v1"

response = s3.list_objects_v2(Bucket=bucket_name, Prefix=f"{branch}/data/raw/")
```

> ⚠️ On Colab, `LAKEFS_ENDPOINT` must be a publicly reachable URL (or tunneled), since Colab can't see `localhost` on your workstation. This is exactly why the endpoint lives in `conf/env/colab.yaml` rather than a `.env` shared across machines.

*Working workflow (same on every environment):*
1. `lakectl branch create lakefs://part-inspection/experiment-v1 --source lakefs://part-inspection/main`
2. Upload new raw images to `lakefs://part-inspection/experiment-v1/data/raw/`
3. Commit via lakeFS CLI or UI — every environment now resolves the same data by branch+commit, not by local file state.

### Step 2: Config Composition with Hydra

```yaml
# conf/config.yaml
defaults:
  - mode: train              # train | evaluate
  - env: local_rtx2080
  - model: resnet50
  - training: default
  - data: default
  - _self_

hydra:
  job:
    chdir: false             # keep relative DVC out: paths stable
```

```yaml
# conf/mode/train.yaml
name: train
# routes main.py → src/training/trainer.py
```

```yaml
# conf/mode/evaluate.yaml
name: evaluate
# routes main.py → src/evaluate.py (GradCAM + Evidently drift)
```

```yaml
# conf/data/default.yaml
image_size: 224
label_map:                   # declared once, in config — ok=0, burr=1
  ok: 0
  burr: 1
split:
  train: 0.7
  val: 0.15
  test: 0.15
augment:
  strength: 1.0              # flip/rotate/color jitter scale (train split only)
```

```yaml
# conf/env/colab.yaml
device: "cuda:0"
data_root: "/content/data"
lakefs_endpoint: ${oc.env:LAKEFS_ENDPOINT}
mlflow_tracking_uri: ${oc.env:MLFLOW_TRACKING_URI}
```

Override anything at the CLI without touching files:
```bash
python main.py mode=train env=colab training.epochs=20 training.batch_size=16
```

### Step 3: Reproducible Pipeline Orchestration with DVC

```yaml
# dvc.yaml
stages:
  preprocess:
    cmd: python main.py mode=train env=${env} data=default data.only_preprocess=true
    deps:
      - src/data
      - conf/env/${env}.yaml
      - conf/data/default.yaml
    outs:
      - data/processed

  train:
    cmd: python main.py mode=train env=${env} model=resnet50 training=default data=default
    deps:
      - data/processed
      - src/training/trainer.py
      - conf/training/default.yaml
      - conf/model/resnet50.yaml
    outs:
      - outputs/model.pt
    metrics:
      - metrics.json:
          cache: false

  evaluate:
    cmd: python main.py mode=evaluate env=${env} model=resnet50
    deps:
      - data/processed
      - outputs/model.pt
      - src/evaluate.py
    outs:
      - reports/drift_report.html:
          cache: false
      - reports/gradcam:
          cache: false
```

Run via `dvc repro` — `${env}` is a DVC param so the same `dvc.yaml` reproduces correctly whether `env` resolves to `local_rtx2080` or `colab`. The single `main.py` entrypoint drives every stage; `mode=` selects the code path.

### Step 4: Experiment Tracking with MLflow (shared server)

```python
# src/training/trainer.py — training loop ONLY (no eval logic here)
import mlflow
import mlflow.pytorch
import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from src.utils.config import set_seed
from src.utils.device import resolve_device, setup_ddp
from src.utils.lakefs_client import resolve_dataset_commit
from src.training.losses import build_criterion
from src.models.model import build_model
from src.data.dataset import build_loaders


def run_training(cfg: DictConfig):  # called from main.py when mode=train
    set_seed(cfg.training.seed)
    mlflow.set_tracking_uri(cfg.env.mlflow_tracking_uri)   # shared remote, not localhost
    mlflow.set_experiment("Part_Inspection_Defect_Classification")

    device, local_rank, world_size = setup_ddp(cfg)        # multi-GPU when available
    model = build_model(cfg).to(device)
    criterion = build_criterion(cfg)                       # BCEWithLogitsLoss(pos_weight=...)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr)
    train_loader, val_loader = build_loaders(cfg)

    with mlflow.start_run() as run:
        # Config + exact dataset version, so the run is fully reproducible/citable
        mlflow.log_param("mode", cfg.mode.name)
        mlflow.log_param("seed", cfg.training.seed)
        mlflow.log_param("environment", cfg.env.name)
        mlflow.log_param("architecture", cfg.model.architecture)
        mlflow.log_param("batch_size", cfg.training.batch_size)
        mlflow.log_param("pos_weight", cfg.training.pos_weight)
        mlflow.log_param("lakefs_commit", resolve_dataset_commit(cfg))  # exact data version used

        for epoch in range(cfg.training.epochs):
            train_loss = run_train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = run_val_epoch(model, val_loader, criterion, device)
            # this is what becomes the loss curve in MLflow's UI automatically
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)

        # NOTE: final test-set metrics + confusion matrix live in src/evaluate.py,
        #       never in the trainer. Trainer only persists weights.
        mlflow.pytorch.log_model(model, "model")
        print(f"Logged run successfully to MLflow: {run.info.run_id}")
```

```python
# src/training/losses.py — binary loss with class-imbalance handling
import torch
from torch import nn
from omegaconf import DictConfig

def build_criterion(cfg: DictConfig) -> nn.Module:
    # Binary classification: single logit in, BCEWithLogitsLoss applies sigmoid internally.
    # pos_weight up-weights the rare burr class (label=1); value comes from conf/training/.
    pos_weight = torch.tensor([cfg.training.pos_weight], dtype=torch.float32)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

> Logging `seed` and `lakefs_commit` together means anyone (including future-you writing the paper) can reconstruct exactly what data + config produced this run. The loss curve comes for free from per-epoch `log_metric` calls — MLflow's UI plots `train_loss`/`val_loss` over `step` automatically, no extra charting needed.

> Note `hydra.job.chdir=false` should be set in `conf/config.yaml` (or passed at the CLI) so Hydra's working-directory behavior doesn't break the relative `outs:` paths DVC expects.

> **Why log this much per run?** This is what makes the `## Results` table at the top actually verifiable — anyone cloning the repo can click the run link and see the exact seed, dataset commit, and config that produced each number, not just a reported metric they have to trust.

Because `mlflow_tracking_uri` now points at a shared server instead of `http://localhost:5000`, runs from RTX 2080 workstation A, workstation B, and a Colab session all appear in the same MLflow UI:
```bash
mlflow ui --port 5000 --backend-store-uri <shared-store>
```

### Step 5: Evaluation, GradCAM & Drift with Evidently AI

`src/evaluate.py` is **fully standalone** — the trainer never imports or calls it. It owns the test-set metrics, the confusion matrix, GradCAM explainability heatmaps, and the Evidently drift report. It is invoked through the same single entrypoint: `python main.py mode=evaluate ...`.

```python
# src/evaluate.py — standalone; called from main.py when mode=evaluate
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from omegaconf import DictConfig

from src.models.model import build_model
from src.utils.device import resolve_device
from src.utils.lakefs_client import resolve_dataset_commit


def run_evaluation(cfg: DictConfig):
    device = resolve_device(cfg)
    model = build_model(cfg).to(device).eval()

    y_true, y_pred = run_test_inference(model, cfg, device)  # threshold logit @ 0.5 → {0=ok, 1=burr}

    # --- Metrics (binary) ---
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    print(f"test_accuracy={acc}  precision={precision}  recall={recall}  f1={f1}")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])   # rows=ok/burr order per conf/data label_map
    fig, ax = plt.subplots()
    ax.imshow(cm); ax.set_title("Confusion Matrix (ok=0, burr=1)")
    fig.savefig("reports/confusion_matrix.png")

    # --- GradCAM: verify the model focuses on the burr region, not background ---
    generate_gradcam(model, cfg, device, out_dir="reports/gradcam")

    # --- Evidently drift report (training embeddings vs. new batch embeddings) ---
    import pandas as pd
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset

    ref_data = pd.read_csv("data/processed/metadata_embeddings.csv")
    current_data = pd.read_csv("data/new_batch/metadata_embeddings.csv")
    drift = Report(metrics=[DataQualityPreset(), DataDriftPreset()])
    drift.run(reference_data=ref_data, current_data=current_data)
    drift.save_html("reports/drift_report.html")
    print("Evaluation complete: reports/confusion_matrix.png, reports/gradcam/, reports/drift_report.html")
```

Open `reports/drift_report.html` in your browser to inspect visual and statistical shifts in image features, and `reports/gradcam/` for the class-activation heatmaps that confirm the classifier is attending to the defect region.

---

## 🔄 The Experimental Lifecycle (Developer Checklist)

1. 📂 **lakeFS:** Branch out `lakectl branch create lakefs://part-inspection/experiment-batch-X`.
2. ⚙️ **Hydra:** Pick or override `env=` for the machine you're on (`local_rtx2080`, `local_rtx3050`, `colab`), and `mode=` (`train`/`evaluate`).
3. 🔄 **DVC:** Sync images into `data/raw/` and trigger `dvc repro`.
4. 🧪 **MLflow:** Train models — `python main.py mode=train ...` — multi-GPU via PyTorch DDP locally, single-GPU on Colab, all logged to the shared tracking server.
5. 📈 **Evaluate:** Run `python main.py mode=evaluate ...` (drift via Evidently, explainability via GradCAM). If the new batch is representative, merge the lakeFS branch back to `main`.
