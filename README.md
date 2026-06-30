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
   Generates class activation heatmaps to visualize which image regions contribute most to the model's predictions. Used to verify that the classifier focuses on the relevant defect area (e.g., burr) rather than irrelevant background features, supporting qualitative error analysis and model interpretability.
---

## 📂 Project Structure

```text
part_inspection/
├── .github/workflows/         # Basic pipeline/CI automation
├── conf/                      # Hydra config root
│   ├── config.yaml            # Top-level composition (defaults list)
│   ├── env/
│   │   ├── local_rtx2080.yaml # device=cuda:0/1, local lakeFS endpoint, local mlflow URI
│   │   ├── local_rtx3050.yaml
│   │   └── colab.yaml         # device=cuda:0, remote lakeFS+mlflow endpoints, /content paths
│   ├── model/
│   │   └── resnet50.yaml
│   └── training/
│       └── default.yaml       # epochs, batch_size, lr, seed, etc.
├── data/                      # Local git-ignored, ephemeral on Colab
│   ├── raw/                   # Synced from lakeFS branch at run start
│   └── processed/             # Preprocessed images ready for training
├── notebooks/                 # EDA + Colab entrypoint notebook
│   └── colab_runner.ipynb     # Clones repo, installs deps, runs CLI with env=colab
├── scripts/
│   ├── setup_colab.sh         # One-shot Colab bootstrap (deps + lakeFS auth + mount)
│   └── sync_lakefs.sh         # Shared shell helper for pulling a branch into data/raw
├── src/
│   ├── __init__.py
│   ├── dataset.py             # PyTorch Dataset & DataLoader implementation
│   ├── train.py                # Hydra-decorated entrypoint for training + MLflow logging
│   ├── evaluate.py            # Evidently AI report generation
│   └── utils/
│       ├── lakefs_client.py   # lakeFS S3-compatible client, env-aware
│       └── device.py          # Resolves torch.device from Hydra env config
├── dvc.yaml                   # DVC Pipeline DAG, parameterized by conf/
├── pyproject.toml             # Python dependencies and build system configuration
└── README.md                  # Project documentation (this file)
```

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
pip install -r requirements.txt
```

This pulls the exact dataset version from lakeFS, trains with the fixed seed, and logs everything (config, loss curve, eval metrics, confusion matrix, model weights) to MLflow — no manual bookkeeping required to reproduce a result.

### Option A — Local Workstation

```bash
# Pick the env file matching this machine
python src/train.py env=local_rtx2080 training=default model=resnet50
```

### Option B — Google Colab (CLI-style)

Run from a Colab cell (or `!` CLI in a notebook):

```bash
!git clone https://github.com/your-org/part_inspection.git
%cd part_inspection
!bash scripts/setup_colab.sh        # installs deps, configures lakeFS creds from Colab secrets

!python src/train.py env=colab training=default model=resnet50
```

`scripts/setup_colab.sh` handles the parts that differ on Colab: installing `requirements.txt` into the ephemeral runtime, pulling lakeFS credentials from `google.colab.userdata` instead of a local `.env`, and pointing the MLflow tracking URI at the shared remote server so the run shows up alongside local runs.

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
  - env: local_rtx2080
  - model: resnet50
  - training: default
  - _self_
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
python src/train.py env=colab training.epochs=20 training.batch_size=16
```

### Step 3: Reproducible Pipeline Orchestration with DVC

```yaml
# dvc.yaml
stages:
  preprocess:
    cmd: python src/dataset.py --config-name config env=${env}
    deps:
      - src/dataset.py
      - conf/env/${env}.yaml
    outs:
      - data/processed

  train:
    cmd: python src/train.py env=${env} training=default model=resnet50
    deps:
      - data/processed
      - src/train.py
      - conf/training/default.yaml
      - conf/model/resnet50.yaml
    outs:
      - outputs/model.pt
    metrics:
      - metrics.json:
          cache: false

  drift_analysis:
    cmd: python src/evaluate.py --reference data/processed --current data/new_batch
    deps:
      - data/processed
      - outputs/model.pt
      - src/evaluate.py
    outs:
      - reports/drift_report.html:
          cache: false
```

Run via `dvc repro` — `${env}` is a DVC param so the same `dvc.yaml` reproduces correctly whether `env` resolves to `local_rtx2080` or `colab`.

### Step 4: Experiment Tracking with MLflow (shared server)

```python
# src/train.py
import hydra
import mlflow
import mlflow.pytorch
import torch
from omegaconf import DictConfig
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    set_seed(cfg.training.seed)
    mlflow.set_tracking_uri(cfg.env.mlflow_tracking_uri)  # shared remote, not localhost
    mlflow.set_experiment("Part_Inspection_Defect_Classification")

    device = torch.device(cfg.env.device if torch.cuda.is_available() else "cpu")

    with mlflow.start_run() as run:
        # Config + exact dataset version, so the run is fully reproducible/citable
        mlflow.log_param("seed", cfg.training.seed)
        mlflow.log_param("environment", cfg.env.name)
        mlflow.log_param("architecture", cfg.model.architecture)
        mlflow.log_param("batch_size", cfg.training.batch_size)
        mlflow.log_param("lakefs_commit", cfg.env.lakefs_commit)  # exact data version used

        for epoch in range(cfg.training.epochs):
            train_loss = run_train_epoch(...)   # your training loop
            val_loss, val_acc = run_val_epoch(...)
            # this is what becomes the loss curve in MLflow's UI automatically
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)

        # Final evaluation on held-out test set
        y_true, y_pred = run_test_inference(...)
        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1", f1)

        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        ax.imshow(cm)
        ax.set_title("Confusion Matrix")
        fig.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")

        mlflow.pytorch.log_model(model, "model")
        print(f"Logged run successfully to MLflow: {run.info.run_id}")

if __name__ == "__main__":
    main()
```

> Logging `seed` and `lakefs_commit` together means anyone (including future-you writing the paper) can reconstruct exactly what data + config produced this run. The loss curve comes for free from per-epoch `log_metric` calls — MLflow's UI plots `train_loss`/`val_loss` over `step` automatically, no extra charting needed.

> Note `hydra.job.chdir=false` should be set in `conf/config.yaml` (or passed at the CLI) so Hydra's working-directory behavior doesn't break the relative `outs:` paths DVC expects.

> **Why log this much per run?** This is what makes the `## Results` table at the top actually verifiable — anyone cloning the repo can click the run link and see the exact seed, dataset commit, and config that produced each number, not just a reported metric they have to trust.

Because `mlflow_tracking_uri` now points at a shared server instead of `http://localhost:5000`, runs from RTX 2080 workstation A, workstation B, and a Colab session all appear in the same MLflow UI:
```bash
mlflow ui --port 5000 --backend-store-uri <shared-store>
```

### Step 5: Data & Model Drift with Evidently AI

```python
# src/evaluate.py
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset

ref_data = pd.read_csv("data/processed/metadata_embeddings.csv")
current_data = pd.read_csv("data/new_batch/metadata_embeddings.csv")

data_drift_report = Report(metrics=[DataQualityPreset(), DataDriftPreset()])
data_drift_report.run(reference_data=ref_data, current_data=current_data)
data_drift_report.save_html("reports/drift_report.html")
print("Data Drift report generated successfully at reports/drift_report.html")
```

Open `reports/drift_report.html` in your browser to inspect visual and statistical shifts in image features.

---

## 🔄 The Experimental Lifecycle (Developer Checklist)

1. 📂 **lakeFS:** Branch out `lakectl branch create lakefs://part-inspection/experiment-batch-X`.
2. ⚙️ **Hydra:** Pick or override `env=` for the machine you're on (`local_rtx2080`, `local_rtx3050`, `colab`).
3. 🔄 **DVC:** Sync images into `data/raw/` and trigger `dvc repro`.
4. 🧪 **MLflow:** Train models — multi-GPU via PyTorch DDP locally, single-GPU on Colab — all logged to the shared tracking server.
5. 📈 **Evidently AI:** Run `src/evaluate.py` to check image quality and embedding drift. If representative, merge the lakeFS branch back to `main`.
