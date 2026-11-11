"""Generalisation to an unseen operating speed (leave-one-speed-out).

Only Motor-2 was run at several speeds (50 / 75 / 100 %), so the experiment uses Motor-2 data: train on two
speeds, test on the third, for each speed in turn. 75 % tests interpolation between known speeds; 50 % and
100 % test extrapolation beyond them. Motor-4 (single speed) is left out so the test isolates the speed effect.
"""
import logging
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

from . import config
from .aggregation import aggregate
from .evaluate import fault_metrics, load_modality, severity_metrics
from .models import MODEL_FACTORIES
from .uncertainty import FAULT_PREFIX, SEV_PREFIX

log = logging.getLogger(__name__)
MOTOR = "Motor-2"


def run(models=("hrf", "xgb", "mlp")) -> pd.DataFrame:
    rows, frames = [], []
    for modality in config.MODALITIES:
        path = config.FEATURE_DIR / f"{modality.lower()}_features.parquet"
        if not path.exists():
            continue
        df, X, _ = load_modality(modality)
        df = df.reset_index(drop=True)
        enc = LabelEncoder().fit(df["Fault_Type"])
        y_fault, y_sev = enc.transform(df["Fault_Type"]), df["Severity_Level"].to_numpy(int)
        n_fault, max_sev = len(enc.classes_), int(y_sev.max())
        groups = df["Recording_ID"].to_numpy()
        on_motor = (df["Motor_ID"] == MOTOR).to_numpy()
        speeds = sorted(df.loc[on_motor, "Speed_Percent"].unique())
        if len(speeds) < 2:
            continue
        for test_speed in speeds:
            tr = np.flatnonzero(on_motor & (df["Speed_Percent"] != test_speed).to_numpy())
            te = np.flatnonzero(on_motor & (df["Speed_Percent"] == test_speed).to_numpy())
            train_speeds = sorted(df.loc[tr, "Speed_Percent"].unique())
            unseen = sorted(set(df.loc[te, "Fault_Type"]) - set(df.loc[tr, "Fault_Type"]))
            log.info("== %s %s: train on %s %% (%d windows), test on %d %% (%d windows)%s", modality, MOTOR,
                     train_speeds, len(tr), test_speed, len(te),
                     f"; fault types never seen in training: {unseen}" if unseen else "")
            for key in models:
                Model = MODEL_FACTORIES[key]
                t0 = time.time()
                model = Model().fit(X[tr], y_fault[tr], y_sev[tr], groups[tr], n_fault=n_fault, max_sev=max_sev)
                pf, ps = model.predict(X[te])
                prob_f, prob_s = model.predict_proba(X[te])
                frame = pd.concat([
                    pd.DataFrame({"Modality": modality, "Model": Model.name, "Test_Speed": test_speed,
                                  "Fold": 0, "Recording_ID": groups[te], "Window_Idx": df.loc[te, "Window_Idx"].to_numpy(),
                                  "Fault_True": df.loc[te, "Fault_Type"].to_numpy(),
                                  "Fault_Pred": enc.inverse_transform(pf),
                                  "Severity_True": y_sev[te], "Severity_Pred": ps}),
                    pd.DataFrame(prob_f, columns=[f"{FAULT_PREFIX}{c}]" for c in enc.classes_]),
                    pd.DataFrame(prob_s, columns=[f"{SEV_PREFIX}{s}]" for s in range(max_sev + 1)]),
                ], axis=1)
                frames.append(frame)
                rec = aggregate(frame)
                row = {"Modality": modality, "Model": Model.name, "Test speed (%)": test_speed,
                       "Train speeds (%)": "+".join(map(str, train_speeds)), "Test windows": len(te),
                       "Test recordings": rec.shape[0],
                       **fault_metrics(y_fault[te], pf), **severity_metrics(y_sev[te], ps),
                       "recording_fault_accuracy": (rec.Fault_True == rec.Fault_Pred).mean(),
                       "recording_severity_exact_acc": (rec.Severity_True == rec.Severity_Pred).mean(),
                       "unseen_fault_types": ", ".join(unseen), "fit_seconds": time.time() - t0}
                rows.append(row)
                log.info("   %-22s test %3d%%  acc=%.4f macroF1=%.4f sevMAE=%.4f | per-recording acc=%.4f (%.0fs)",
                         Model.name, test_speed, row["fault_accuracy"], row["fault_macro_f1"],
                         row["severity_mae"], row["recording_fault_accuracy"], row["fit_seconds"])
    metrics = pd.DataFrame(rows)
    metrics.to_csv(config.REPORT_DIR / "cross_speed_metrics.csv", index=False)
    pd.concat(frames, ignore_index=True).to_parquet(config.REPORT_DIR / "cross_speed_predictions.parquet", index=False)
    return metrics


def in_distribution_baseline(oof: pd.DataFrame) -> pd.DataFrame:
    """Same Motor-2 windows, scored by the main CV models (which had seen that speed during training)."""
    parts = oof["Recording_ID"].str.split("/", expand=True)
    d = oof[parts[1] == MOTOR].assign(Speed=parts.loc[parts[1] == MOTOR, 2].astype(int))
    return (d.groupby(["Modality", "Model", "Speed"])[["Fault_True", "Fault_Pred", "Severity_True", "Severity_Pred"]]
             .apply(lambda g: pd.Series({
                 "seen_accuracy": (g.Fault_True == g.Fault_Pred).mean(),
                 "seen_macro_f1": f1_score(g.Fault_True, g.Fault_Pred, average="macro", zero_division=0),
                 "seen_severity_mae": (g.Severity_True - g.Severity_Pred).abs().mean()}))
             .reset_index())
