"""Walk the dataset tree, catalog every channel CSV and parse ground-truth labels from paths."""
import logging
import re
from pathlib import Path

import pandas as pd

from . import config

log = logging.getLogger(__name__)

# e.g. Vibration_Motor-4_70_time-coupling 2D-ch3.csv
FILE_RE = re.compile(
    r"^(?P<modality>Electric|Vibration)_(?P<motor>Motor-\d+)_(?P<speed>\d+)_time-(?P<condition>.+)-ch(?P<channel>\d+)\.csv$",
    re.IGNORECASE,
)
# e.g. "bearing bpfi 3", "coupling 2D", "loose foot motor", "healthy noise"
CONDITION_RE = re.compile(r"^(?P<stem>[a-z][a-z ]*?)(?:\s+(?P<level>\d+)(?P<variant>[a-z]*))?$", re.IGNORECASE)


def _pretty(stem: str) -> str:
    return " ".join(config.ACRONYMS.get(w, w.capitalize()) for w in stem.split())


def parse_condition(condition: str):
    """Map a condition folder name to (Fault_Type, Severity_Level, Variant)."""
    m = CONDITION_RE.match(condition.strip())
    if not m:
        raise ValueError(f"Unparseable condition folder: {condition!r}")
    stem = m["stem"].lower().strip()
    if stem in config.HEALTHY_STEMS:
        return config.HEALTHY_LABEL, 0, stem
    level = int(m["level"]) if m["level"] else config.DEFAULT_FAULT_SEVERITY
    return _pretty(stem), level, (m["variant"] or "").upper()


def build_catalog(root: Path = config.DATASET_ROOT) -> pd.DataFrame:
    """One row per channel CSV with parsed labels; folder and filename metadata are cross-checked."""
    records = []
    for path in sorted(root.rglob("*.csv")):
        rel = path.relative_to(root).parts
        if len(rel) != 5:
            log.warning("Skipping unexpected path depth: %s", path)
            continue
        modality, motor, speed, condition, fname = rel
        m = FILE_RE.match(fname)
        if not m:
            log.warning("Skipping file with unexpected name: %s", path)
            continue
        if (m["modality"].lower(), m["motor"], m["speed"], m["condition"]) != (
            modality.lower(), motor, speed, condition
        ):
            log.warning("Folder/filename metadata disagree, trusting folders: %s", path)
        fault, severity, variant = parse_condition(condition)
        records.append({
            "Modality": modality.capitalize(),
            "Motor_ID": motor,
            "Speed_Percent": int(speed),
            "Condition": condition,
            "Fault_Type": fault,
            "Severity_Level": severity,
            "Variant": variant,
            "Channel": int(m["channel"]),
            "Path": str(path),
            "Size_MB": path.stat().st_size / 1e6,
        })
    cat = pd.DataFrame(records)
    cat["Condition_ID"] = cat["Modality"] + "/" + cat["Motor_ID"] + "/" + cat["Speed_Percent"].astype(str) + "/" + cat["Condition"]
    return cat


def summarize(cat: pd.DataFrame) -> str:
    lines = [
        f"{len(cat)} channel files, {cat['Condition_ID'].nunique()} condition folders, {cat['Size_MB'].sum() / 1e3:.1f} GB",
    ]
    for mod, g in cat.groupby("Modality"):
        conds = g.drop_duplicates("Condition_ID")
        lines.append(
            f"  {mod}: {len(g)} files, {len(conds)} conditions, channels/condition="
            f"{sorted(g.groupby('Condition_ID')['Channel'].nunique().unique().tolist())}, "
            f"{conds['Fault_Type'].nunique()} fault types, severities={sorted(conds['Severity_Level'].unique().tolist())}"
        )
    return "\n".join(lines)
