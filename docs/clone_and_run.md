# For Contributors / Supervisor — From Clone to Experiment

> **Goal:** clone this repo on a fresh machine (workstation or VM) and run a training + evaluation experiment, end to end. No prior knowledge of the codebase assumed.
>
> **What this project is:** an experimental ML pipeline for binary image classification (`ok` vs `burr` defects on small parts). You run training, it logs everything to MLflow, you compare runs. There is no serving/production component — it's a research loop.
>
> **Time:** ~20–30 min first time (mostly Docker + data upload). Subsequent experiments are one command.

---

## What you need on the machine

- **Docker + Docker Compose** (`docker compose version` should work)
- **A GPU + NVIDIA driver** (CUDA). Check with `nvidia-smi`.
- **Python 3.12+** and **uv** (installed below), OR just use `uv` — it manages Python for you.
- **Git access** to the repo: `git@github.com:PatiphanAK/material_part_inspection.git`
- **Your image data** organized as `ok/` and `burr/` folders (see Part A).

> No data and no secrets are in the repo. Data lives in **lakeFS** (a data lake we start here); secrets go in a local `.env` file that is git-ignored.

---

## Step 0 — Clone & install dependencies

```bash
git clone git@github.com:PatiphanAK/material_part_inspection.git
cd material_part_inspection

# Install uv if you don't have it (manages Python + deps for us)
curl -LsSf https://astral.sh/uv/install.sh | sh
# (or: pip install uv)

# Install all dependencies into a project venv (.venv/)
uv sync

# Activate the venv for every shell session from here on:
source .venv/bin/activate
```

> **Important:** `source .venv/bin/activate` must be run in every new terminal. The pipeline commands (`python main.py ...`, `dvc repro`) need it, otherwise `python` won't find torch/mlflow.

---

## Step 1 — Start the infrastructure (one time)

This starts three services: **postgres** (shared DB), **lakeFS** (data versioning), **MLflow** (experiment tracking).

```bash
docker compose up -d
```

Wait ~20s, then verify all three are healthy:

```bash
curl -s -o /dev/null -w "lakefs: %{http_code}\n" http://localhost:8000/_health   # expect 200
curl -s -o /dev/null -w "mlflow:  %{http_code}\n" http://localhost:5000/health   # expect 200
```

If not 200, check `docker compose logs lakefs` / `docker compose logs mlflow`.

---

## Step 2 — Set up lakeFS + MLflow credentials (one time)

```bash
# Initialize the lakeFS admin user. This PRINTS your access key + secret — copy them.
docker exec lakefs lakefs setup --user-name admin
# You'll see lines like:
#   credentials:
#     access_key_id: AKIA...
#     secret_access_key: ....
```

Copy `.env.example` to `.env` and paste those credentials in:

```bash
cp .env.example .env
# Edit .env and fill in:
#   LAKEFS_ACCESS_KEY_ID=<from setup output>
#   LAKEFS_SECRET_ACCESS_KEY=<from setup output>
#   LAKEFS_ENDPOINT=http://localhost:8000
#   MLFLOW_TRACKING_URI=http://localhost:5000
#   LAKEFS_COMMIT=<leave for now, set in Part A step A.3>
```

Then create the lakeFS repository the configs point at:

```bash
set -a; source .env; set +a
curl -s -X POST http://localhost:8000/api/v1/repositories \
  -u "$LAKEFS_ACCESS_KEY_ID:$LAKEFS_SECRET_ACCESS_KEY" \
  -H "Content-Type: application/json" \
  -d '{"name":"part-inspection","storage_namespace":"local://part-inspection","default_branch":"main"}'
```

Done. Infrastructure is ready. You only repeat Step 2 if you wipe the Docker volumes.

---

## Step 3 — Add your dataset (do this once per new batch of images)

### 3.1 Organize images

Put your images in two folders named **exactly** `ok` and `burr`:

```
my_images/
├── ok/      *.jpg / *.png ...   (parts with no defect)
└── burr/    *.jpg / *.png ...   (parts with burr defects)
```

> The folder names ARE the labels (`ok=0`, `burr=1`). Don't rename them.

### 3.2 Upload to lakeFS and commit

```bash
set -a; source .env; set +a

# Create a branch for this dataset version
BRANCH="exp-$(date +%Y%m%d)"
curl -s -X POST http://localhost:8000/api/v1/repositories/part-inspection/branches \
  -u "$LAKEFS_ACCESS_KEY_ID:$LAKEFS_SECRET_ACCESS_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"name\":\"$BRANCH\",\"source\":\"main\"}"

# Upload images (change my_images/ to your folder path)
python - <<PY
import boto3, glob, os
s3 = boto3.client("s3", endpoint_url=os.environ["LAKEFS_ENDPOINT"],
                  aws_access_key_id=os.environ["LAKEFS_ACCESS_KEY_ID"],
                  aws_secret_access_key=os.environ["LAKEFS_SECRET_ACCESS_KEY"])
branch = "$BRANCH"
n = 0
for cls in ["ok", "burr"]:
    for f in sorted(glob.glob(f"my_images/{cls}/*.png")) + sorted(glob.glob(f"my_images/{cls}/*.jpg")):
        s3.upload_file(f, "part-inspection", f"{branch}/data/raw/{cls}/{os.path.basename(f)}")
        n += 1
print(f"uploaded {n} images to branch {branch}")
PY

# Commit the dataset version and capture the commit SHA
COMMIT_SHA=$(curl -s -X POST \
  "http://localhost:8000/api/v1/repositories/part-inspection/branches/$BRANCH/commits" \
  -u "$LAKEFS_ACCESS_KEY_ID:$LAKEFS_SECRET_ACCESS_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"message\":\"$BRANCH: ok/burr images\"}" | grep -o '"id":"[0-9a-f]*"' | head -1 | cut -d'"' -f4)
echo "COMMIT_SHA=$COMMIT_SHA  (branch=$BRANCH)"
```

### 3.3 Pin the commit (IMPORTANT — this makes runs reproducible)

The dataset commit must be pinned in **two** places, depending on how you run:

- **`.env`** — used when you run `python main.py ...` directly
- **`params.yaml`** — used when you run `dvc repro` (DVC tracks this as a dependency, so changing it correctly invalidates the pipeline and re-pulls the new data)

```bash
# Pin in .env (for direct `python main.py` runs)
sed -i "s|^LAKEFS_COMMIT=.*|LAKEFS_COMMIT=$COMMIT_SHA|" .env
# Pin in params.yaml (for `dvc repro` runs)
sed -i "s|^lakefs_commit:.*|lakefs_commit: $COMMIT_SHA|" params.yaml
grep LAKEFS_COMMIT .env; grep '^lakefs_commit:' params.yaml
```

> **Always pin a commit SHA, never a branch name.** A SHA is frozen; a branch can move. The trainer logs this SHA to MLflow so any reported metric can be traced back to the exact data that produced it.
>
> **Why both files?** DVC only re-runs a stage when one of its tracked dependencies changes. `LAKEFS_COMMIT` in `.env` is invisible to DVC, so changing only `.env` and running `dvc repro` would silently reuse stale cached data. `params.yaml` IS tracked by DVC (the commit is interpolated into each stage's command), so editing it forces a correct re-pull.

---

## Step 4 — Run an experiment

### 4.1 Pick your `env=` (which machine am I on?)

| Your machine | Use `env=` |
|---|---|
| 1 GPU (e.g. RTX 3050, single-GPU VM) | `env=local_rtx3050` |
| 2 GPUs (e.g. RTX 2080) | `env=local_rtx2080` |
| Google Colab | `env=colab` |

> Don't see your exact GPU? Use `local_rtx3050` for any single-GPU machine — the GPU model in the name is cosmetic; what matters is `use_ddp` (multi-GPU) vs not. To add a new machine, copy `conf/env/local_rtx3050.yaml` to a new file and adjust `device`/`num_devices`/`use_ddp`.

### 4.2 Run the full pipeline (recommended — one command)

First, set your environment in `params.yaml` (DVC reads `env` from there):

```bash
# Edit params.yaml and set:  env: local_rtx3050   (or local_rtx2080 / colab)
# or do it with sed:
sed -i 's/^env: .*/env: local_rtx3050/' params.yaml
grep '^env:' params.yaml
```

Then run the whole pipeline:

```bash
set -a; source .env; set +a
source .venv/bin/activate

dvc repro
```

This runs **preprocess → train → evaluate** automatically, caching stages so re-runs skip unchanged work. It:
1. Pulls the exact dataset version (`LAKEFS_COMMIT`) from lakeFS into `data/raw/`
2. Splits train/val/test deterministically (seed-locked, reproducible)
3. Trains a binary classifier (single logit, `BCEWithLogitsLoss` with `pos_weight` for class imbalance)
4. Logs to MLflow: `seed`, `lakefs_commit`, `pos_weight`, per-epoch loss/accuracy, the model
5. Evaluates on the held-out test set: accuracy/precision/recall/F1, confusion matrix, GradCAM heatmaps

### 4.3 Or run stages manually (when iterating)

```bash
set -a; source .env; set +a
source .venv/bin/activate

# Train (override anything at the CLI — no code changes):
python main.py mode=train env=local_rtx3050 model=resnet50 training.epochs=10 training.batch_size=16

# Evaluate:
python main.py mode=evaluate env=local_rtx3050 model=resnet50
```

Common CLI overrides:

| Flag | What it does |
|---|---|
| `training.epochs=10` | number of epochs |
| `training.batch_size=16` | batch size |
| `training.lr=1e-4` | learning rate |
| `training.pos_weight=2.0` | up-weight the rare burr class (>1 if burr is minority) |
| `training.seed=123` | change the seed (different split/init) |
| `model=resnet50` | backbone (resnet18/34/50, mobilenet_v3_large/small) |

---

## Step 5 — Look at the results

### MLflow UI (metrics + parameters + model)
Open in a browser: **http://localhost:5000**
- Experiment: `Part_Inspection_Defect_Classification`
- Each run shows: `seed`, `lakefs_commit`, `pos_weight`, loss curves, the model artifact
- Sort by `val_accuracy` or `test_f1` to find the best run
- The evaluate run (named `evaluate-resnet50`) holds the test metrics + confusion matrix

### Reports on disk
```bash
ls reports/
# confusion_matrix.png   — ok=0 / burr=1
# gradcam/cam_*.npy       — heatmaps of which image region drove each prediction
# drift_report.html       — data drift (stub unless you add embedding CSVs)
```

`confusion_matrix.png` — quick visual of errors. `gradcam/` — sanity-check the model is looking at the burr, not background.

### Compare experiments
```bash
dvc metrics show               # training summary table
# or in MLflow UI: compare runs side by side
```

---

## Step 6 — If this dataset version is good, merge to `main`

```bash
set -a; source .env; set +a
curl -s -X POST \
  "http://localhost:8000/api/v1/repositories/part-inspection/refs/main/merge/$BRANCH" \
  -u "$LAKEFS_ACCESS_KEY_ID:$LAKEFS_SECRET_ACCESS_KEY"
```

---

## Quick reference (the whole loop in 7 lines)

```bash
git clone git@github.com:PatiphanAK/material_part_inspection.git && cd material_part_inspection
uv sync && docker compose up -d
docker exec lakefs lakefs setup --user-name admin          # copy creds -> .env
cp .env.example .env                                       # fill in creds + create lakeFS repo (Step 2)
# ...upload ok/ + burr/ to a lakeFS branch, commit, pin SHA as LAKEFS_COMMIT (Step 3)
set -a; source .env; set +a; source .venv/bin/activate
# edit params.yaml -> env: local_rtx3050, then:
dvc repro                                              # train + evaluate (Step 4)
# open http://localhost:5000 and reports/ (Step 5)
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `python: command not found` | You forgot `source .venv/bin/activate` |
| `No images found under data/raw` | `LAKEFS_COMMIT` in `.env` is wrong/stale — re-copy the SHA from Step 3.2 |
| `Synced 0 objects from commit` | the commit SHA doesn't contain `data/raw/...` — check the upload landed on that branch/commit |
| Changed dataset but `dvc repro` says "didn't change, skipping" / pulled old data | you changed `LAKEFS_COMMIT` in `.env` but not in `params.yaml`. DVC only sees `params.yaml`. Pin the SHA in `params.yaml` too (Step 3.3), or run `dvc repro -f` to force |
| lakeFS / MLflow not reachable | `docker compose up -d`, wait 20s, re-check health (Step 1) |
| `PermissionError: /mlruns` | (only if you reset the MLflow DB) — the experiment has a stale artifact path; the server already handles this via artifact proxy, but if it recurs see `docs/dvc_dataset_versioning.md` §6 |
| Model predicts everything as `ok` (low recall on burr) | burr is the minority class — raise `training.pos_weight` (≈ `#ok / #burr`) |
| Want a different backbone | `model=resnet18` / `model=mobilenet_v3_large` (must have a matching `conf/model/<name>.yaml`) |
| `nvidia-smi` shows GPU but training says CPU | `device: cuda` should be in your `conf/env/*.yaml`; check `torch.cuda.is_available()` in the venv |

---

## Where to read more (if you want the "why")

- `docs/experimental.md` — the experimental design (what's being classified, imaging setup)
- `docs/dvc_dataset_versioning.md` — why lakeFS + DVC, the full data-versioning mechanics
- `docs/runbook_dataset_experiment.md` — the day-to-day "add data → run experiment" runbook
- `README.md` — architecture overview + project structure

If anything in this guide doesn't work on a fresh clone, that's a bug — report it.
