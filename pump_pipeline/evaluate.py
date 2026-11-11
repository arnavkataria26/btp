"""Stage 2: grouped + stratified K-fold training of all models per modality, metrics, and figures."""
import json
import logging
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, mean_absolute_error,
                             mean_squared_error, precision_recall_fscore_support)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

from . import config
from .build_features import META_COLS
from .models import MODEL_FACTORIES
from .report import write_all
from .style import GRID, INK, INK_2, SEQ_BLUE, SERIES, SURFACE, plt
from .uncertainty import FAULT_PREFIX, SEV_PREFIX

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- data
def load_modality(modality: str):
    df = pd.read_parquet(config.FEATURE_DIR / f"{modality.lower()}_features.parquet")
    feat_cols = [c for c in df.columns if c not in META_COLS]
    # Operating context known at inference time: speed and which motor/pump set.
    ctx = pd.DataFrame({"ctx_speed_pct": df["Speed_Percent"].astype(np.float32),
                        "ctx_motor4": (df["Motor_ID"] == "Motor-4").astype(np.float32)})
    X = np.hstack([df[feat_cols].to_numpy(np.float32), ctx.to_numpy()])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return df, X, feat_cols + list(ctx.columns)


def make_folds(df: pd.DataFrame, group_level: str, n_splits: int):
    """StratifiedGroupKFold: whole recordings (or whole condition folders) stay on one side of every split,
    so no window of a test recording is ever seen in training; stratified on fault x severity so minority
    fault classes appear in each fold wherever their group count allows."""
    groups = df["Recording_ID" if group_level == "recording" else "Condition_ID"].to_numpy()
    strat = (df["Fault_Type"] + "|" + df["Severity_Level"].astype(str)).to_numpy()
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)
    return groups, list(sgkf.split(np.zeros(len(df)), strat, groups))


# --------------------------------------------------------------------------- metrics
def fault_metrics(y, p):
    out = {"fault_accuracy": accuracy_score(y, p)}
    for avg in ("macro", "micro"):
        pr, rc, f1, _ = precision_recall_fscore_support(y, p, average=avg, zero_division=0)
        out.update({f"fault_{avg}_precision": pr, f"fault_{avg}_recall": rc, f"fault_{avg}_f1": f1})
    return out


def severity_metrics(y, p):
    return {"severity_mae": mean_absolute_error(y, p),
            "severity_rmse": float(np.sqrt(mean_squared_error(y, p))),
            "severity_exact_acc": accuracy_score(y, p)}


# --------------------------------------------------------------------------- plots
def _annotate_heatmap(ax, M, fmt, counts=None):
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if (counts if counts is not None else M)[i, j] == 0:
                continue
            ax.text(j, i, format(M[i, j], fmt), ha="center", va="center", fontsize=6.5,
                    color="#ffffff" if M[i, j] > 0.55 * M.max() else INK)


def plot_fault_confusion(y, p, classes, title, path):
    cm = confusion_matrix(y, p, labels=range(len(classes)))
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    n = len(classes)
    fig, ax = plt.subplots(figsize=(0.42 * n + 3, 0.42 * n + 2))
    im = ax.imshow(norm, cmap=SEQ_BLUE, vmin=0, vmax=1)
    _annotate_heatmap(ax, norm, ".2f", counts=cm)
    ax.set_xticks(range(n), classes, rotation=60, ha="right")
    ax.set_yticks(range(n), classes)
    ax.set_xlabel("Predicted fault type")
    ax.set_ylabel("True fault type")
    ax.set_title(title, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Recall (row-normalised)")
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_severity_errors(y, p, title, path, max_sev):
    err = p - y
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.2, 1]})
    lo, hi = -max_sev, max_sev
    vals = np.arange(lo, hi + 1)
    counts = np.array([(err == v).sum() for v in vals])
    a1.bar(vals, counts / len(err) * 100, width=0.8, color=SERIES[0], edgecolor=SURFACE, linewidth=2)
    a1.set_yscale("log")
    a1.set_xticks(vals)
    a1.set_xlabel("Predicted − true severity")
    a1.set_ylabel("Share of windows (%, log scale)")
    a1.yaxis.grid(True, color=GRID, linewidth=0.6)
    a1.set_axisbelow(True)
    a1.set_title("Error distribution", loc="left")
    a1.text(0.98, 0.95, f"MAE {np.abs(err).mean():.3f}\nRMSE {np.sqrt((err ** 2).mean()):.3f}\n"
                        f"Exact {np.mean(err == 0) * 100:.1f}%",
            transform=a1.transAxes, ha="right", va="top", color=INK_2)

    levels = np.union1d(np.unique(y), np.unique(p))     # only levels that occur, no empty rows
    cm = confusion_matrix(y, p, labels=levels)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    im = a2.imshow(norm, cmap=SEQ_BLUE, vmin=0, vmax=1)
    _annotate_heatmap(a2, norm, ".2f", counts=cm)
    a2.set_xticks(range(len(levels)), levels)
    a2.set_yticks(range(len(levels)), levels)
    a2.set_xlabel("Predicted severity")
    a2.set_ylabel("True severity")
    a2.set_title("True vs predicted severity", loc="left")
    for s in a2.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=a2, fraction=0.045, pad=0.03)
    cb.outline.set_visible(False)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_model_comparison(summary: pd.DataFrame, path):
    """Two panels (never two y-axes): fault macro-F1 and severity MAE, models as categorical series."""
    models = list(dict.fromkeys(summary["Model"]))
    mods = list(dict.fromkeys(summary["Modality"]))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    w = 0.8 / len(models)
    for ax, (metric, label) in zip(axes, [("fault_macro_f1", "Fault type macro-F1 (higher is better)"),
                                          ("severity_mae", "Severity MAE (lower is better)")]):
        for k, m in enumerate(models):
            sub = summary[summary["Model"] == m].set_index("Modality").loc[mods]
            x = np.arange(len(mods)) + (k - (len(models) - 1) / 2) * w
            mean, std = sub[f"{metric}_mean"].to_numpy(), sub[f"{metric}_std"].fillna(0).to_numpy()
            yerr = np.vstack([np.minimum(std, mean), std])          # whiskers never cross zero
            ax.bar(x, mean, w, yerr=yerr, color=SERIES[k], label=m,
                   edgecolor=SURFACE, linewidth=2, error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 2})
            for xi, v, e in zip(x, mean, std):
                ax.text(xi, v + e, f"{v:.3f}", ha="center", va="bottom", fontsize=7, color=INK_2)
        ax.set_xticks(range(len(mods)), mods)
        ax.set_title(label, loc="left")
        ax.yaxis.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=0)
    axes[0].set_ylim(0, 1.1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=len(models), bbox_to_anchor=(0.5, 0.995))
    fig.text(0.01, 0.01, "Bars: mean over CV folds; whiskers: ±1 std across folds.", color=INK_2, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- driver
def run(models=("hrf", "xgb", "mlp"), group_level="recording", n_splits=config.N_SPLITS):
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fold_rows, oof_frames = [], []

    for modality in config.MODALITIES:
        path = config.FEATURE_DIR / f"{modality.lower()}_features.parquet"
        if not path.exists():
            log.warning("No feature matrix for %s, skipping", modality)
            continue
        df, X, feat_names = load_modality(modality)
        df = df.reset_index(drop=True)          # row i of df == row i of X == row i of the probability arrays
        enc = LabelEncoder().fit(df["Fault_Type"])
        y_fault = enc.transform(df["Fault_Type"])
        y_sev = df["Severity_Level"].to_numpy(int)
        n_fault, max_sev = len(enc.classes_), int(y_sev.max())
        groups, folds = make_folds(df, group_level, n_splits)
        log.info("== %s: %d windows, %d features, %d fault types, severity 0..%d, %d %s groups",
                 modality, len(df), X.shape[1], n_fault, max_sev, len(np.unique(groups)), group_level)
        for k, (tr, te) in enumerate(folds):
            log.info("   fold %d: train %d / test %d windows; test classes %d/%d", k, len(tr), len(te),
                     len(np.unique(y_fault[te])), n_fault)

        for key in models:
            Model = MODEL_FACTORIES[key]
            pf, ps, fold_id = np.full(len(df), -1), np.full(len(df), -1), np.full(len(df), -1)
            prob_f = np.zeros((len(df), n_fault), np.float32)
            prob_s = np.zeros((len(df), max_sev + 1), np.float32)
            for k, (tr, te) in enumerate(folds):
                t0 = time.time()
                model = Model().fit(X[tr], y_fault[tr], y_sev[tr], groups[tr], n_fault=n_fault, max_sev=max_sev)
                pf[te], ps[te] = model.predict(X[te])
                prob_f[te], prob_s[te] = model.predict_proba(X[te])   # for conformal confidence sets
                fold_id[te] = k
                row = {"Modality": modality, "Model": Model.name, "Fold": k, "n_test": len(te),
                       **fault_metrics(y_fault[te], pf[te]), **severity_metrics(y_sev[te], ps[te]),
                       "fit_seconds": time.time() - t0}
                fold_rows.append(row)
                log.info("   %-22s fold %d  acc=%.4f macroF1=%.4f  sevMAE=%.4f sevExact=%.4f (%.0fs)",
                         Model.name, k, row["fault_accuracy"], row["fault_macro_f1"], row["severity_mae"],
                         row["severity_exact_acc"], row["fit_seconds"])

            oof_frames.append(pd.concat([
                pd.DataFrame({
                    "Modality": modality, "Model": Model.name, "Fold": fold_id, "Recording_ID": df["Recording_ID"],
                    "Window_Idx": df["Window_Idx"], "Fault_True": df["Fault_Type"],
                    "Fault_Pred": enc.inverse_transform(pf), "Severity_True": y_sev, "Severity_Pred": ps}),
                pd.DataFrame(prob_f, columns=[f"{FAULT_PREFIX}{c}]" for c in enc.classes_]),
                pd.DataFrame(prob_s, columns=[f"{SEV_PREFIX}{s}]" for s in range(max_sev + 1)]),
            ], axis=1))
            slug = f"{modality.lower()}_{key}"
            plot_fault_confusion(y_fault, pf, list(enc.classes_),
                                 f"{Model.name} · {modality} · fault type (out-of-fold)",
                                 config.FIGURE_DIR / f"cm_fault_{slug}.png")
            plot_severity_errors(y_sev, ps, f"{Model.name} · {modality} · severity level (out-of-fold)",
                                 config.FIGURE_DIR / f"severity_errors_{slug}.png", max_sev)

    folds_df = pd.DataFrame(fold_rows)
    metric_cols = [c for c in folds_df.columns if c.startswith(("fault_", "severity_"))]
    summary = folds_df.groupby(["Modality", "Model"], sort=False)[metric_cols].agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index()
    oof = pd.concat(oof_frames, ignore_index=True)

    folds_df.to_csv(config.REPORT_DIR / "metrics_per_fold.csv", index=False)
    summary.to_csv(config.REPORT_DIR / "metrics_summary.csv", index=False)
    oof.to_parquet(config.REPORT_DIR / "oof_predictions.parquet", index=False)
    plot_model_comparison(summary, config.FIGURE_DIR / "model_comparison.png")
    run_info = {"group_level": group_level, "n_splits": n_splits, "models": list(models)}
    (config.REPORT_DIR / "run_info.json").write_text(json.dumps(run_info, indent=2))
    print(write_all(summary, oof, run_info))
    return summary
