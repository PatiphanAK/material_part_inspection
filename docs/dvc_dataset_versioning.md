# DVC + Dataset Versioning Setup Guide

This guide walks you through setting up **dataset version control** and the **reproducible pipeline** for the Material Part Inspection experiment. It covers two tools that work together:

- **lakeFS** — versions your *raw image data* (Git-like branches/commits on top of object storage). This is what makes Colab runs viable: every run pulls a specific `branch + commit`, not whatever happens to be on local disk.
- **DVC** — versions your *pipeline + artifacts* (`dvc.yaml`) and caches heavy outputs (`data/processed/`, `outputs/model.pt`) without bloating Git.

They are complementary: **lakeFS = data versioning, DVC = pipeline versioning**. Git still tracks code + config (`conf/`, `src/`, `dvc.yaml`); DVC tracks artifact pointers; lakeFS tracks the actual image bytes.

---

## 0. Why both? (the mental model)

| Layer | What it versions | Where bytes live |
|---|---|---|
| **Git** | code (`src/`, `main.py`) + config (`conf/`) + pipeline (`dvc.yaml`) | GitHub |
| **DVC** | artifact pointers + the pipeline DAG; caches `model.pt`, `data/processed/` | DVC remote (S3/local) |
| **lakeFS** | raw + processed image datasets, branchable like Git | object storage (S3/minio) |

The reproducibility contract is: **seed (in `conf/training`) + `lakefs_commit` (logged to MLflow) + config (Hydra) + code (Git commit)** together reconstruct any reported number exactly. That's why `trainer.py` logs `lakefs_commit` alongside the seed on every run.

---

## 1. Prerequisites

Already installed via `uv sync`:
```bash
uv sync                       # installs dvc, boto3, mlflow, torch, ...
```

You also need:
- **Docker** for the lakeFS + MLflow + postgres stack (`docker-compose.yaml`).
- A **DVC remote** for artifact caching. For experiments this can be a local directory or an S3 bucket; for shared/Colab runs use S3 so all machines see the same cache.

Verify:
```bash
dvc --version                # >= 3.55
docker compose version
```

---

## 2. lakeFS setup (data versioning)

### 2.1 Start the stack (postgres + lakeFS + MLflow)

The compose file runs postgres (shared by lakeFS + MLflow), lakeFS (local blockstore), and MLflow (postgres-backed, artifact proxy enabled).

```bash
docker compose up -d
# Services:
#   postgres  -> localhost:5433   (lakeFS DB + MLflow DB)
#   lakefs    -> localhost:8000   (UI + S3 gateway)
#   mlflow    -> localhost:5000   (tracking UI)
```

> **lakeFS runs as `user: 0:0`** in the compose so the local blockstore `/data` (a root-owned named volume) is writable. Without this lakeFS fatal-exits with `path provided is not writable`.

> **MLflow artifacts are proxied** via `--default-artifact-root mlflow-artifacts:/part-inspection --serve-artifacts`. This is mandatory — a plain `--default-artifact-root /mlruns` makes the host client try to write `/mlruns` literally and fail with `PermissionError`. If you ever see that error, the experiment has a stale filesystem `artifact_location`; drop & recreate the `mlflow` postgres DB so fresh experiments pick up the proxy URI.

### 2.2 Initialize lakeFS admin + create the repo

```bash
# One-time: create the admin user (prints access_key_id + secret_access_key)
docker exec lakefs lakefs setup --user-name admin

# Capture those creds into .env (see .env.example), then:
set -a; source .env; set +a

# Create the repo the env configs point at (conf/env/*.yaml -> lakefs_repo: part-inspection)
# Via the REST API (or lakectl if installed):
curl -X POST http://localhost:8000/api/v1/repositories \
  -u "$LAKEFS_ACCESS_KEY_ID:$LAKEFS_SECRET_ACCESS_KEY" \
  -H "Content-Type: application/json" \
  -d '{"name":"part-inspection","storage_namespace":"local://part-inspection","default_branch":"main"}'
```

> On Colab, `LAKEFS_ENDPOINT` must be a **publicly reachable URL** (Colab can't see your `localhost`). Tunnel it or host lakeFS on a real endpoint and set the env var / Colab secret.

### 2.3 Upload raw data + commit (the "git commit" of your dataset)

Organize images as `ok/` and `burr/` folders, then upload to the repo's `main` branch via the S3-compatible gateway. The repo's `src/utils/lakefs_client.py` uses boto3 for this:

```python
# upload_to_lakefs.py (ad-hoc helper)
import boto3, glob, os
s3 = boto3.client("s3", endpoint_url=os.environ["LAKEFS_ENDPOINT"],
                  aws_access_key_id=os.environ["LAKEFS_ACCESS_KEY_ID"],
                  aws_secret_access_key=os.environ["LAKEFS_SECRET_ACCESS_KEY"])
for cls in ["ok", "burr"]:
    for f in sorted(glob.glob(f"data/raw/{cls}/*.png")):
        s3.upload_file(f, "part-inspection", f"main/data/raw/{cls}/{os.path.basename(f)}")
# Then commit the dataset version (commit ID is what gets logged to MLflow)
import requests
requests.post("http://localhost:8000/api/v1/repositories/part-inspection/branches/main/commits",
              auth=(os.environ["LAKEFS_ACCESS_KEY_ID"], os.environ["LAKEFS_SECRET_ACCESS_KEY"]),
              json={"message": "experiment-v1: initial ok/burr images"})
```

Grab the commit SHA and pin it in **both** `.env` (for direct `python main.py` runs) and `params.yaml` (for `dvc repro` runs — DVC tracks this so changing it forces a re-pull):
```bash
SHA=$(curl -s http://localhost:8000/api/v1/repositories/part-inspection/refs/main \
  -u "$LAKEFS_ACCESS_KEY_ID:$LAKEFS_SECRET_ACCESS_KEY" | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)
sed -i "s|^LAKEFS_COMMIT=.*|LAKEFS_COMMIT=$SHA|" .env
sed -i "s|^lakefs_commit:.*|lakefs_commit: $SHA|" params.yaml
```

> **Why both?** `LAKEFS_COMMIT` in `.env` is invisible to DVC; changing only `.env` and running `dvc repro` would silently reuse stale cached data. `params.yaml` is a DVC-tracked dependency (interpolated into each stage's `cmd:`), so editing it correctly invalidates the pipeline.

### 2.4 Pull a specific dataset version into `data/raw/`

`run_training` and `run_evaluation` call `sync_from_lakefs(cfg)` automatically (gated by `env.sync_data: true`) before loading data, pulling the **exact commit** in `LAKEFS_COMMIT`:

```bash
set -a; source .env; set +a
python main.py mode=train env=local_rtx3050 model=resnet50
# -> "Syncing from lakeFS commit <sha> -> data/raw"
# -> "Synced N objects from commit <sha>"
```

To pull manually (e.g. for inspection):
```bash
python -c "
import os
from hydra import compose, initialize_config_dir
from src.utils.lakefs_client import sync_from_lakefs
with initialize_config_dir(version_base=None, config_dir=os.path.abspath('conf')):
    cfg = compose(config_name='config', overrides=['env=local_rtx3050'])
    sync_from_lakefs(cfg)
"
```

> **Colab note:** `scripts/setup_colab.sh` should `source` the Colab-secrets `.env` and do this pull at startup since `/content/data` is wiped on disconnect. Same code, different `env=colab` endpoint.

---

## 3. DVC setup (pipeline versioning)

### 3.1 Initialize DVC (one-time)

```bash
git init            # if not already a repo
dvc init            # creates dvc.yaml, dvc.lock, .dvc/, adds them to .gitignore appropriately
```

`dvc.yaml` and `params.yaml` already exist in this repo — `dvc init` just wires up the `.dvc/` cache.

### 3.2 Configure the DVC remote (artifact cache)

The DVC remote is where heavy artifacts (`model.pt`, `data/processed/`) get pushed/pulled. For local experiments use a local path; for shared/Colab use S3.

**Local (single workstation):**
```bash
mkdir -p /home/tatar025/dvc_cache
dvc remote add -d localstorage /home/tatar025/dvc_cache
```

**S3 (shared across workstations + Colab):**
```bash
dvc remote add -d s3remote s3://my-dvc-bucket/part-inspection
dvc remote modify s3remote endpointurl https://s3.example.com
# creds come from AWS_* / boto3 config, same as lakeFS
```

### 3.3 The pipeline DAG

Already defined in `dvc.yaml`:

```
preprocess  →  train  →  evaluate
```

Each stage calls the single entrypoint `python main.py mode=...`, and `${env}` is a DVC param (`params.yaml`) so the same DAG runs on any machine:

| Stage | Command | Produces |
|---|---|---|
| `preprocess` | `python main.py mode=train env=${env} data.only_preprocess=true` | `data/processed/` (split manifest) |
| `train` | `python main.py mode=train env=${env} model=resnet50 ...` | `outputs/model.pt`, `metrics.json` |
| `evaluate` | `python main.py mode=evaluate env=${env} model=resnet50` | `reports/{confusion_matrix.png, gradcam/, drift_report.html}` |

---

## 4. The experiment workflow (day-to-day)

### 4.1 Run the whole pipeline on a machine

```bash
set -a; source .env; set +a     # load lakeFS + MLflow creds/endpoints
dvc repro                       # runs preprocess -> train -> evaluate, caching each stage
```

Switch machines by setting `env` in `params.yaml` (DVC reads it from there), then reproduce — no code change:

```bash
# RTX 3050 single-GPU
sed -i 's/^env: .*/env: local_rtx3050/' params.yaml && dvc repro

# RTX 2080 multi-GPU
sed -i 's/^env: .*/env: local_rtx2080/' params.yaml && dvc repro

# Colab
sed -i 's/^env: .*/env: colab/' params.yaml && dvc repro
```

### 4.2 Run just one stage / iterate without DVC

```bash
dvc repro train             # only the train stage (+ its deps)
dvc repro evaluate          # only evaluate

# Or skip DVC and run the entrypoint directly (useful while iterating):
set -a; source .env; set +a
python main.py mode=train env=local_rtx3050 model=resnet50 training.epochs=3
python main.py mode=evaluate env=local_rtx3050 model=resnet50
```

### 4.3 Push / pull artifacts between machines

```bash
dvc push                    # upload cached artifacts to the remote
dvc pull                    # download them on another machine / Colab
git push                    # code + dvc.yaml + dvc.lock still go to Git
git pull && dvc pull        # clone workflow on a new machine
```

### 4.4 Compare experiments

```bash
dvc exp show                  # table of experiments with metrics
dvc exp diff <exp-a> <exp-b>  # metric + param diff between two runs
```

### 4.5 Iterate on the dataset (the lakeFS loop)

This is the experiment loop that keeps `main` clean:

```bash
# 1. Branch the dataset (REST API; or use lakectl if installed)
curl -X POST http://localhost:8000/api/v1/repositories/part-inspection/branches \
  -u "$LAKEFS_ACCESS_KEY_ID:$LAKEFS_SECRET_ACCESS_KEY" \
  -H "Content-Type: application/json" \
  -d '{"name":"experiment-v2","source":"main"}'

# 2. Add/replace images on the branch (S3 gateway), then commit
python -c "
import boto3, os, glob, requests
s3 = boto3.client('s3', endpoint_url=os.environ['LAKEFS_ENDPOINT'],
                  aws_access_key_id=os.environ['LAKEFS_ACCESS_KEY_ID'],
                  aws_secret_access_key=os.environ['LAKEFS_SECRET_ACCESS_KEY'])
for f in glob.glob('new_burr/*.png'):
    s3.upload_file(f, 'part-inspection', f'experiment-v2/data/raw/burr/{os.path.basename(f)}')
requests.post('http://localhost:8000/api/v1/repositories/part-inspection/branches/experiment-v2/commits',
              auth=(os.environ['LAKEFS_ACCESS_KEY_ID'], os.environ['LAKEFS_SECRET_ACCESS_KEY']),
              json={'message': 'more burr samples'})
"

# 3. Train against the new dataset version (pin the new commit in BOTH .env and params.yaml)
sed -i "s|^LAKEFS_COMMIT=.*|LAKEFS_COMMIT=<new-sha>|" .env
sed -i "s|^lakefs_commit:.*|lakefs_commit: <new-sha>|" params.yaml
dvc repro train

# 4. If the new batch is representative (check Evidently drift in reports/drift_report.html),
#    merge it back into main:
curl -X POST "http://localhost:8000/api/v1/repositories/part-inspection/refs/main/merge/experiment-v2" \
  -u "$LAKEFS_ACCESS_KEY_ID:$LAKEFS_SECRET_ACCESS_KEY"
```

---

## 5. Reproducibility checklist

Before claiming a result in the README's `## Results` table, confirm:

- [ ] `git status` clean (code + config committed)
- [ ] `dvc.lock` present (exact deps/outs pinned)
- [ ] `LAKEFS_COMMIT` set to a real commit SHA (not a branch name) and logged to MLflow
- [ ] `training.seed` logged to MLflow
- [ ] `dvc repro` reproduces the metrics from a clean checkout on another machine

Anyone who clones the repo + `dvc pull` + sets `LAKEFS_COMMIT` will then get the same number.

---

## 6. Common pitfalls

| Symptom | Cause / fix |
|---|---|
| `dvc repro` re-runs train every time | a `deps:` path changed, or `metrics.json` has `cache: false` and is being touched — check `dvc status` |
| Colab can't reach lakeFS | `LAKEFS_ENDPOINT` is `localhost` — set it to a public URL / tunnel |
| Different metric on another machine | `seed` differs, or `LAKEFS_COMMIT` is a branch name that moved — pin the SHA |
| `dvc repro` says "didn't change, skipping" after a new dataset commit | you changed `LAKEFS_COMMIT` in `.env` only. DVC can't see `.env`; pin the SHA in `params.yaml` (tracked by DVC) so the pipeline invalidates, or run `dvc repro -f` |
| `data/processed` re-preprocessed every run | it's not in `outs:` or DVC cache is off — ensure `outs:` lists it |
| MLflow runs don't show up | `mlflow_tracking_uri` points at localhost on Colab — set `MLFLOW_TRACKING_URI` to the shared server |
| `PermissionError: /mlruns` from client | MLflow experiment has a stale filesystem `artifact_location`; drop & recreate the `mlflow` postgres DB so fresh experiments use the `mlflow-artifacts:/` proxy |
| lakeFS fatal: `path provided is not writable` | local blockstore `/data` is root-owned; run lakeFS as `user: 0:0` (set in compose) |
| `torchvision` import is broken after `uv sync` | left in a namespace-package state; `uv pip install --reinstall torchvision` |
| `mlflow.pytorch.log_model` raises on TensorSpec | pt2 serialization needs a TensorSpec signature; use `serialization_format="pickle"` (already set in `trainer.py`) |
| GradCAM `numpy.int64 not iterable` | default target resolver assumes multi-class; `evaluate.py` passes an explicit binary-logit target for the 1D output |
| Hydra changes the working dir | set `hydra.job.chdir: false` (already set in `conf/config.yaml`) |

---

## 7. Verified end-to-end example

This exact sequence was run on 2026-06-30 against the docker-compose stack with a synthetic 40-ok / 30-burr dataset committed to lakeFS as `synthetic-v1`:

```bash
# 1. Stack up
docker compose up -d
docker exec lakefs lakefs setup --user-name admin          # capture creds -> .env

# 2. Create lakeFS repo, upload images, commit, pin SHA in .env AND params.yaml
#    (see 2.2 / 2.3; set LAKEFS_COMMIT to the returned commit SHA)

# 3. Train (auto-pulls from lakeFS at LAKEFS_COMMIT, logs to MLflow)
set -a; source .env; set +a
python main.py mode=train env=local_rtx3050 model=resnet50 training.epochs=3 training.batch_size=8 training.num_workers=0
# -> Synced 70 objects from commit aa0750b1...
# -> splits: train=49 val=10 test=11
# -> epoch 3/3 train_loss=0.0037 val_acc=1.0000
# -> Logged run to MLflow: <run-id>

# 4. Evaluate (metrics + confusion matrix + GradCAM; drift skipped without CSVs)
python main.py mode=evaluate env=local_rtx3050 model=resnet50 training.num_workers=0
# -> test metrics: {'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
# -> Wrote 8 GradCAM heatmaps to reports/gradcam
# -> Evaluation complete. Reports in reports
```

MLflow UI: `http://localhost:5000` — the training run logs `seed`, `lakefs_commit`, `pos_weight`, per-epoch `train_loss`/`val_loss`/`val_accuracy`, the `model` artifact, and `outputs/model.pt`; the evaluate run logs `test_*` metrics and `reports/confusion_matrix.png`.

---

## 8. Quick reference

```bash
# lakeFS
lakectl branch create lakefs://part-inspection/exp --source lakefs://part-inspection/main
lakectl commit  lakefs://part-inspection/exp --message "..."
lakectl merge   lakefs://part-inspection/exp lakefs://part-inspection/main

# DVC
dvc repro                              # run pipeline (env comes from params.yaml)
sed -i 's/^env:.*/env: colab/' params.yaml && dvc repro   # run on a different env
dvc push / dvc pull                    # sync artifacts
dvc exp show / dvc exp diff a b        # compare experiments

# Entry point
python main.py mode=train    env=local_rtx3050 model=resnet50 training.epochs=20
python main.py mode=evaluate env=local_rtx3050 model=resnet50
```
