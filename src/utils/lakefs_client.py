"""Env-aware lakeFS client (S3-compatible API).

lakeFS exposes an S3-compatible gateway, so we talk to it with boto3 using an
endpoint URL resolved from the active env config (`conf/env/*.yaml`). On Colab
the endpoint must be publicly reachable, which is exactly why it lives in
config rather than a shared .env.
"""

from __future__ import annotations

import os
from typing import Any

from omegaconf import DictConfig

from src.utils.logger import get_logger

log = get_logger(__name__)


def _build_s3(cfg: DictConfig):
    """Build a boto3 S3 client pointed at the configured lakeFS endpoint."""
    import boto3  # imported lazily so non-lakeFS runs don't require boto3

    endpoint = cfg.env.lakefs_endpoint
    access_key = os.getenv("LAKEFS_ACCESS_KEY_ID")
    secret_key = os.getenv("LAKEFS_SECRET_ACCESS_KEY")

    if not access_key or not secret_key:
        log.warning("LAKEFS_ACCESS_KEY_ID / LAKEFS_SECRET_ACCESS_KEY not set")

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )


def resolve_dataset_commit(cfg: DictConfig) -> str:
    """Return the exact lakeFS commit (data version) used by this run.

    Logged to MLflow alongside the seed so any run can be reconstructed:
    seed + lakefs_commit + config => fully reproducible result.
    """
    return str(cfg.env.get("lakefs_commit", cfg.env.get("lakefs_branch", "main")))


def pull_raw(cfg: DictConfig, dest_dir: str) -> None:
    """Pull raw images for the active branch/commit from lakeFS into dest_dir.

    Kept simple: lists objects under <branch>/data/raw/ and downloads each.
    """
    s3 = _build_s3(cfg)
    bucket = cfg.env.lakefs_repo
    branch = cfg.env.lakefs_branch
    prefix = f"{branch}/data/raw/"
    os.makedirs(dest_dir, exist_ok=True)

    log.info("Listing lakeFS objects: s3://%s/%s", bucket, prefix)
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key: str = obj["Key"]
            rel = key[len(prefix):]
            if not rel:
                continue
            local_path = os.path.join(dest_dir, rel)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            log.debug("Downloading %s -> %s", key, local_path)
            s3.download_file(bucket, key, local_path)
    log.info("lakeFS pull complete -> %s", dest_dir)


def list_objects(cfg: DictConfig, prefix: str | None = None) -> list[dict[str, Any]]:
    """List objects under a prefix in the configured lakeFS repo/branch."""
    s3 = _build_s3(cfg)
    full_prefix = prefix or f"{cfg.env.lakefs_branch}/data/raw/"
    objs: list[dict[str, Any]] = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=cfg.env.lakefs_repo, Prefix=full_prefix):
        objs.extend(page.get("Contents", []))
    return objs


def sync_from_lakefs(cfg: DictConfig, dest_dir: str | None = None) -> str:
    """Pull raw images for the pinned commit into dest_dir (defaults to cfg.env.raw_dir).

    Pulls from the exact commit in cfg.env.lakefs_commit so the run is reproducible
    regardless of whatever is currently on the branch. Returns the commit used.
    """
    target = dest_dir or cfg.env.raw_dir
    s3 = _build_s3(cfg)
    bucket = cfg.env.lakefs_repo
    commit = str(cfg.env.lakefs_commit)
    # In lakeFS's S3 gateway, you read a specific commit by using it as the "branch"
    # prefix: <commit>/data/raw/...
    prefix = f"{commit}/data/raw/"
    os.makedirs(target, exist_ok=True)

    log.info("Syncing from lakeFS commit %s -> %s", commit, target)
    found = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key: str = obj["Key"]
            rel = key[len(prefix):]
            if not rel or rel.endswith("/"):
                continue
            local_path = os.path.join(target, rel)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            s3.download_file(bucket, key, local_path)
            found += 1
    if found == 0:
        log.warning("No objects found under s3://%s/%s — is LAKEFS_COMMIT correct?", bucket, prefix)
    else:
        log.info("Synced %d objects from commit %s", found, commit)
    return commit
