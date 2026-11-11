"""Stage 1: raw CSVs -> windowed feature matrix, one parquet shard per condition folder (resumable)."""
import logging
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from . import config
from .catalog import build_catalog, summarize
from .features import condition_features
from .loader import iter_conditions

log = logging.getLogger(__name__)

META_COLS = ["Shaft_Hz", "N_Channels", "Condition_ID", "Variant", "Severity_Level", "Fault_Type",
             "Condition", "Speed_Percent", "Motor_ID", "Modality", "Recording_ID", "Window_Idx"]


def _shard_path(cond_id: str):
    return config.SHARD_DIR / (re.sub(r"[^A-Za-z0-9]+", "_", cond_id).strip("_") + ".parquet")


def _process_condition(files: pd.DataFrame) -> tuple:
    cond_id = files["Condition_ID"].iloc[0]
    t0 = time.time()
    meta, signals, rec_idx = next(iter_conditions(files))
    df = condition_features(meta, signals, rec_idx)
    tmp = _shard_path(cond_id).with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, _shard_path(cond_id))        # atomic: an interrupted run never leaves half a shard
    return cond_id, len(df), signals.shape, time.time() - t0


def build(workers: int = 4, limit: int = None, rebuild: bool = False) -> dict:
    config.SHARD_DIR.mkdir(parents=True, exist_ok=True)
    catalog = build_catalog()
    log.info("Catalog:\n%s", summarize(catalog))
    catalog.to_csv(config.FEATURE_DIR / "catalog.csv", index=False)

    groups = [g for _, g in catalog.groupby("Condition_ID", sort=True)]
    if limit:
        groups = groups[:: max(1, len(groups) // limit)][:limit]
    todo = [g for g in groups if rebuild or not _shard_path(g["Condition_ID"].iloc[0]).exists()]
    log.info("%d condition folders, %d already cached, %d to process with %d workers",
             len(groups), len(groups) - len(todo), len(todo), workers)

    t0, done_mb = time.time(), 0.0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_condition, g): g for g in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            g = futures[fut]
            try:
                cond_id, n_rows, shape, secs = fut.result()
                done_mb += g["Size_MB"].sum()
                log.info("[%d/%d] %-55s rec x ch x samples=%s -> %d windows (%.0fs, %.0f MB/s overall)",
                         i, len(todo), cond_id, shape, n_rows, secs, done_mb / (time.time() - t0))
            except Exception:
                log.exception("FAILED %s", g["Condition_ID"].iloc[0])

    # Consolidate shards into one matrix per modality
    wanted = {_shard_path(g["Condition_ID"].iloc[0]) for g in groups}
    shards = [pd.read_parquet(p) for p in sorted(wanted) if p.exists()]
    out = {}
    for mod in config.MODALITIES:
        parts = [s for s in shards if s["Modality"].iloc[0] == mod]
        if not parts:
            continue
        df = pd.concat(parts, ignore_index=True)
        path = config.FEATURE_DIR / f"{mod.lower()}_features.parquet"
        df.to_parquet(path, index=False)
        out[mod] = path
        log.info("%s feature matrix: %d windows x %d features from %d recordings -> %s", mod, len(df),
                 df.shape[1] - len(META_COLS), df["Recording_ID"].nunique(), path)
    return out
