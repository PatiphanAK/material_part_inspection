# Runbook: Add a New Dataset & Run an Experiment

> Audience: a tech/note-taker joining the project. This is the **copy-paste runbook** for the two things you'll do most: (1) add a new batch of images to the data lake, and (2) run a training/evaluation experiment against it. The full background is in [`dvc_dataset_versioning.md`](./dvc_dataset_versioning.md); this doc is the short version.
>
> TL;DR — every experiment is **config + a pinned dataset commit**. You never edit code to run an experiment; you add data to lakeFS, pin the commit (in `.env` for direct runs, in `params.yaml` for `dvc repro`), and override Hydra flags at the CLI.

---

## 0. One-time setup (do this once on your machine)

```bash
# 1. Install deps
uv sync

# 2. Start the stack (postgres + lakeFS + MLflow) — already configured in docker-compose.yaml
docker compose up -d
#    Verify: http://localhost:8000 (lakeFS)  http://localhost:5000 (MLflow)

# 3. Initialize lakeFS admin and capture creds (prints access_key_id + secret_access_key)
docker exec lakefs lakefs setup --user-name admin

# 4. Copy .env.example -> .env and paste those creds + endpoint
cp .env.example .env
#    Fill in LAKEFS_ACCESS_KEY_ID, LAKEFS_SECRET_ACCESS_KEY from step 3

# 5. Create the lakeFS repo the configs point at (one-time)
set -a; source .env; set +a
curl -s -X POST http://localhost:8000/api/v1/repositories \
  -u "$LAKEFS_ACCESS_KEY_ID:$LAKEFS_SECRET_ACCESS_KEY" \
  -H "Content-Type: application/json" \
  -d '{"name":"part-inspection","storage_namespace":"local://part-inspection","default_branch":"main"}'
```

Done. From now on you only do **Part A** (add data) and **Part B** (run experiment).

> **Sanity check (any time):** `curl -s -o /dev/null -w "lakefs:%{http_code}\n" http://localhost:8000/_health && curl -s -o /dev/null -w "mlflow:%{http_code}\n" http://localhost:5000/health` → both should be `200`.

---

## Part A — Add a new dataset (or new batch of images)

### A.1 Organize your images

Put your images in two folders named exactly `ok` and `burr` (these map to labels `0` and `1`, declared in `conf/data/default.yaml`):

```
my_new_batch/
├── ok/      *.jpg / *.png ...   (no-defect parts)
└── burr/    *.jpg / *.png ...   (burr-defect parts)
```

> The folder names `ok` / `burr` are the source of truth for labels. Don't rename them. If you need different class names later, change `conf/data/default.yaml:label_map` — no code change.

### A.2 Upload to lakeFS and commit

You can either **add to a new branch** (recommended — keeps `main` clean while you test) or **add directly to `main`**.

**Option 1 — branch out for an experiment (recommended):**

```bash
set -a; source .env; set +a

# Create a branch from main
BRANCH="exp-batch-$(date +%Y%m%d)"
curl -s -X POST http://localhost:8000/api/v1/repositories/part-inspection/branches \
  -u "$LAKEFS_ACCESS_KEY_ID:$LAKEFS_SECRET_ACCESS_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"name\":\"$BRANCH\",\"source\":\"main\"}"

# Upload images into the branch via the S3 gateway
.venv/bin/python - <<PY
import boto3, glob, os
s3 = boto3.client("s3", endpoint_url=os.environ["LAKEFS_ENDPOINT"],
                  aws_access_key_id=os.environ["LAKEFS_ACCESS_KEY_ID"],
                  aws_secret_access_key=os.environ["LAKEFS_SECRET_ACCESS_KEY"])
branch = "$BRANCH"
for cls in ["ok", "burr"]:
    for f in sorted(glob.glob(f"my_new_batch/{cls}/*.png")) + sorted(glob.glob(f"my_new_batch/{cls}/*.jpg")):
        s3.upload_file(f, "part-inspection", f"{branch}/data/raw/{cls}/{os.path.basename(f)}")
print("uploaded")
PY

# Commit the dataset version on the branch
COMMIT_SHA=$(curl -s -X POST \
  "http://localhost:8000/api/v1/repositories/part-inspection/branches/$BRANCH/commits" \
  -u "$LAKEFS_ACCESS_KEY_ID:$LAKEFS_SECRET_ACCESS_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"message\":\"$BRANCH: new ok/burr images\"}" | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)
echo "COMMIT_SHA=$COMMIT_SHA  (branch=$BRANCH)"
```

**Option 2 — add directly to `main`:** same upload step but use the string `"main"` everywhere `$BRANCH` appears, and commit on `main`.

### A.3 Pin the commit (in BOTH .env and params.yaml)

This is the **most important step** — it's what makes the run reproducible. Pin the SHA in two places:

```bash
# .env  -> for direct `python main.py ...` runs
sed -i "s|^LAKEFS_COMMIT=.*|LAKEFS_COMMIT=$COMMIT_SHA|" .env
# params.yaml -> for `dvc repro` runs (DVC tracks this; changing it forces a re-pull)
sed -i "s|^lakefs_commit:.*|lakefs_commit: $COMMIT_SHA|" params.yaml
grep LAKEFS_COMMIT .env; grep '^lakefs_commit:' params.yaml
```

> **Always pin a commit SHA, never a branch name.** A branch can move; a SHA is frozen. The trainer logs this SHA to MLflow on every run, so a reported number can always be traced back to the exact data.
>
> **Why both?** `LAKEFS_COMMIT` in `.env` is invisible to DVC, so changing only `.env` and running `dvc repro` would silently reuse stale cached data. `params.yaml` is tracked by DVC (the commit is interpolated into each stage's command), so editing it correctly invalidates the pipeline.

That's it for adding data. Move to Part B to train against it.

---

## Part B — Run an experiment

> **First:** activate the project venv so `python` resolves with torch/mlflow installed:
> ```bash
> source .venv/bin/activate     # or prefix every command with:  uv run
> ```
> (DVC runs `python` directly, so the venv must be active for `dvc repro` too.)

### B.1 Train

```bash
set -a; source .env; set +a    # load lakeFS creds + pinned LAKEFS_COMMIT + MLflow URI
source .venv/bin/activate       # project venv (torch, mlflow, ...)

# Default experiment (uses conf/ defaults: resnet50, 20 epochs, seed 42)
python main.py mode=train env=local_rtx3050 model=resnet50

# Override anything at the CLI — NO code changes:
python main.py mode=train env=local_rtx3050 \
    model=resnet50 \
    training.epochs=10 \
    training.batch_size=16 \
    training.lr=1e-4 \
    training.pos_weight=2.0          # >1 if burr is the minority class
```

What happens automatically:
1. Pulls the exact dataset version (`LAKEFS_COMMIT`) from lakeFS into `data/raw/`
2. Splits train/val/test deterministically (seed-locked)
3. Trains a **single-logit binary classifier** with `BCEWithLogitsLoss(pos_weight=...)`
4. Logs to MLflow: `seed`, `lakefs_commit`, `pos_weight`, per-epoch loss/accuracy, the model artifact, `outputs/model.pt`

Watch it live: `http://localhost:5000` → experiment **Part_Inspection_Defect_Classification**.

### B.2 Evaluate

```bash
set -a; source .env; set +a
python main.py mode=evaluate env=local_rtx3050 model=resnet50
```

Produces, in `reports/`:
- `confusion_matrix.png` — ok=0 / burr=1
- `gradcam/cam_*.npy` — heatmaps showing which image region drove each prediction (sanity-check it's the burr, not background)
- `drift_report.html` — Evidently drift report *(only if you've added `metadata_embeddings.csv` files — skipped otherwise, that's normal)*

Evaluation metrics (`test_accuracy`, `test_precision`, `test_recall`, `test_f1`) are logged to a separate MLflow run named `evaluate-resnet50`.

### B.3 Pick the right `env=` for your machine

| Machine | `env=` | GPU |
|---|---|---|
| RTX 2080 workstation (2 GPUs) | `env=local_rtx2080` | multi-GPU via DDP |
| RTX 3050 workstation (1 GPU) | `env=local_rtx3050` | single GPU |
| Google Colab | `env=colab` | single Colab GPU |

Switching machines is a one-word override — no code change. On Colab, `LAKEFS_ENDPOINT` / `MLFLOW_TRACKING_URI` must be public URLs (Colab can't reach your `localhost`).

### B.4 One-command full pipeline (DVC)

Instead of running train + evaluate by hand, run the whole DAG in one go (preprocess → train → evaluate), cached so unchanged stages are skipped:

```bash
set -a; source .env; set +a
source .venv/bin/activate
dvc repro                       # run all stages (cached if nothing changed)
# To run on a different machine: edit params.yaml -> env: local_rtx2080, then `dvc repro`
dvc metrics show                # print the training summary table
```

---

## Part C — After the experiment

### C.1 If the new dataset is good → merge it into `main`

```bash
set -a; source .env; set +a
curl -s -X POST \
  "http://localhost:8000/api/v1/repositories/part-inspection/refs/main/merge/$BRANCH" \
  -u "$LAKEFS_ACCESS_KEY_ID:$LAKEFS_SECRET_ACCESS_KEY"
```

### C.2 Compare experiments

```bash
# In the MLflow UI (http://localhost:5000): sort runs by val_accuracy / test_f1.
# Each run shows the exact seed + lakefs_commit that produced it.
```

### C.3 Record the result

When a run is worth keeping, add a row to the `## Results` table in `README.md` with the model, seed, metrics, and the MLflow run link. The `seed` + `lakefs_commit` in that run are enough for anyone to reproduce it.

---

## Cheat sheet (the whole thing in 6 lines)

```bash
# Add data
set -a; source .env; set +a
#   ...upload ok/ + burr/ to a lakeFS branch, commit, pin SHA in .env AND params.yaml
# Run experiment
python main.py mode=train    env=local_rtx3050 model=resnet50 training.epochs=10
python main.py mode=evaluate env=local_rtx3050 model=resnet50
# Check results
#   open http://localhost:5000  and  reports/confusion_matrix.png  and  reports/gradcam/
```

---

## Troubleshooting (quick)

| Symptom | Fix |
|---|---|
| `No images found under data/raw` | `LAKEFS_COMMIT` in `.env` is wrong/stale — re-pull the SHA from lakeFS, or the upload didn't land on that commit |
| `Synced 0 objects from commit` | same — the commit SHA doesn't contain `data/raw/...` |
| Changed dataset but `dvc repro` skipped / pulled old data | you changed `.env` but not `params.yaml`. DVC only re-runs when `params.yaml` changes. Pin the SHA in `params.yaml` too (A.3), or `dvc repro -f` to force |
| `PermissionError: /mlruns` | MLflow experiment has a stale artifact path; recreate the `mlflow` postgres DB (see `dvc_dataset_versioning.md` §6) |
| lakeFS / MLflow unreachable | `docker compose up -d` and health-check (see §0 note) |
| Burrs predicted as ok (low recall) | burr is the minority class — raise `training.pos_weight` (e.g. `#ok / #burr`) |
| Want to try a different backbone | add `conf/model/<name>.yaml` + pass `model=<name>` (supported: resnet18/34/50, mobilenet_v3_large/small) |
