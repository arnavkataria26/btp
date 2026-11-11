"""Memory-bounded raw-signal loading.

Only one condition folder (K channel files, ~0.3-1.5 GB of CSV text) is materialised at a time, as
float32. A full pass over the ~100 GB dataset therefore never holds more than a few hundred MB per worker.
"""
import logging
from typing import Iterator, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

from . import config

log = logging.getLogger(__name__)


def _is_junk_column(name: str, col: pa.ChunkedArray) -> bool:
    """Trailing commas yield unnamed / all-null columns."""
    return name.strip() == "" or name.startswith("Unnamed") or col.null_count == len(col)


def read_channel_file(path: str) -> np.ndarray:
    """Read one channel CSV -> float32 array of shape (n_samples, n_recordings), time column dropped."""
    try:
        table = pacsv.read_csv(path, convert_options=pacsv.ConvertOptions(strings_can_be_null=True))
        cols = [
            c for name, c in zip(table.column_names, table.columns)
            if not _is_junk_column(name, c) and name.strip().lower() != "time"
        ]
        if cols and all(pa.types.is_floating(c.type) or pa.types.is_integer(c.type) for c in cols):
            data = np.column_stack([c.to_numpy().astype(np.float32, copy=False) for c in cols])
        else:
            raise TypeError("non-numeric column(s)")
    except (pa.ArrowInvalid, TypeError) as exc:
        # Slow but forgiving path: stray text cells, ragged rows, etc.
        log.warning("pyarrow fallback for %s (%s)", path, exc)
        df = pd.read_csv(path, engine="python", on_bad_lines="skip")
        df = df.loc[:, ~df.columns.astype(str).str.match(r"^(Unnamed|\s*$)")]
        df = df.drop(columns=[c for c in df.columns if str(c).strip().lower() == "time"])
        df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
        data = df.to_numpy(dtype=np.float32)

    # Trim trailing all-NaN rows (truncated final line)
    valid_rows = ~np.isnan(data).all(axis=1)
    if not valid_rows.all():
        data = data[: np.flatnonzero(valid_rows)[-1] + 1]
    return data


def load_condition(files: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Load all channel files of one condition folder.

    Returns
    -------
    signals : float32 (n_recordings, n_channels, n_samples)
    rec_idx : int array of the original column index of each kept recording
    """
    files = files.sort_values("Channel")
    per_channel = [read_channel_file(p) for p in files["Path"]]
    n_samples = min(a.shape[0] for a in per_channel)
    n_recs = min(a.shape[1] for a in per_channel)
    if len({a.shape[1] for a in per_channel}) > 1:
        log.warning("%s: channel files disagree on recording count %s; using first %d",
                    files["Condition_ID"].iloc[0], [a.shape[1] for a in per_channel], n_recs)
    stacked = np.stack([a[:n_samples, :n_recs] for a in per_channel], axis=0)  # (C, T, R)
    signals = np.ascontiguousarray(stacked.transpose(2, 0, 1))               # (R, C, T)
    del stacked, per_channel

    nan_frac = np.isnan(signals).mean(axis=2).max(axis=1)
    keep = nan_frac <= config.MAX_NAN_FRACTION
    if not keep.all():
        log.warning("%s: dropping %d recordings with >%.0f%% NaN", files["Condition_ID"].iloc[0],
                    (~keep).sum(), config.MAX_NAN_FRACTION * 100)
    signals, rec_idx = signals[keep], np.flatnonzero(keep)
    if np.isnan(signals).any():  # sparse gaps: fill with the per-channel mean
        means = np.nanmean(signals, axis=2, keepdims=True)
        signals = np.where(np.isnan(signals), means, signals)
    return signals, rec_idx


def iter_conditions(catalog: pd.DataFrame) -> Iterator[Tuple[dict, np.ndarray, np.ndarray]]:
    """Generator over condition folders: yields (metadata, signals, rec_idx)."""
    for cond_id, files in catalog.groupby("Condition_ID", sort=False):
        meta = files.iloc[0][["Modality", "Motor_ID", "Speed_Percent", "Condition", "Fault_Type",
                              "Severity_Level", "Variant", "Condition_ID"]].to_dict()
        signals, rec_idx = load_condition(files)
        yield meta, signals, rec_idx


def window(signals: np.ndarray, fs: int = config.SAMPLE_RATE_HZ,
           seconds: float = config.WINDOW_SECONDS) -> np.ndarray:
    """(R, C, T) -> (R, W, C, L) non-overlapping windows (a view, no copy); trailing remainder dropped."""
    L = int(round(fs * seconds))
    R, C, T = signals.shape
    W = T // L
    return signals[:, :, : W * L].reshape(R, C, W, L).transpose(0, 2, 1, 3)
