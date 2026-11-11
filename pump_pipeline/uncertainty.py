"""Uncertainty at 90 / 95 / 99 % confidence, computed from out-of-fold predictions (no retraining).

1. Metric confidence intervals -- "we are 95 % confident accuracy lies in [a, b]".
   Recording-level (cluster) bootstrap: whole recordings are resampled with replacement, because the
   windows of one recording are correlated and resampling single windows would make intervals too narrow.

2. Per-prediction conformal sets -- "we are 95 % sure this window's fault is in {A, B} and its severity
   is in [2, 4]". Split-conformal calibration with a guaranteed marginal coverage of >= level, provided
   test data resemble the calibration data. Each CV fold is calibrated on the out-of-fold scores of the
   *other* folds, so a window is never used to calibrate its own set.
     * Fault type: LAC score 1 - p(true fault); the set is every fault with p >= 1 - q (top-1 always kept).
     * Severity: always a contiguous range ("2 to 4"), as an ordinal scale requires. Two constructions:
         - "ordinal": grow from the most likely level towards the more probable neighbour; the score is
           the probability mass at which the true level joins (ordinal APS, Lu et al., 2022);
         - "hull": keep every level with p >= 1 - q, then fill the gaps between them.
       Which is tighter depends on the model, so for each fold the method with the narrower ranges on
       the *calibration* folds is used -- the test fold never influences the choice.
"""
import json
import logging

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger(__name__)

LEVELS = (0.90, 0.95, 0.99)
N_BOOTSTRAP = 1000
FAULT_PREFIX, SEV_PREFIX = "P_fault[", "P_sev["
SET_SEP = ", "                     # fault names never contain commas


# =========================================================================== 1. metric CIs
def _metrics_weighted(y_f, p_f, y_s, p_s, w, n_classes):
    """Fault/severity metrics with per-window weights (bootstrap multiplicities)."""
    W = w.sum()
    cm = np.bincount(y_f * n_classes + p_f, weights=w, minlength=n_classes ** 2).reshape(n_classes, n_classes)
    tp, pred, true = np.diag(cm), cm.sum(0), cm.sum(1)
    present = (pred + true) > 0                      # sklearn macro: labels in y_true U y_pred
    prec = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
    rec = np.divide(tp, true, out=np.zeros_like(tp), where=true > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    err = p_s - y_s
    return {
        "Fault accuracy": tp.sum() / W,
        "Fault macro precision": prec[present].mean(),
        "Fault macro recall": rec[present].mean(),
        "Fault macro F1": f1[present].mean(),
        "Severity MAE": (w * np.abs(err)).sum() / W,
        "Severity RMSE": np.sqrt((w * err ** 2).sum() / W),
        "Severity exact-match acc": (w * (err == 0)).sum() / W,
    }


def bootstrap_metric_cis(oof: pd.DataFrame, n_boot=N_BOOTSTRAP, levels=LEVELS) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(config.RANDOM_STATE)
    for (mod, model), d in oof.groupby(["Modality", "Model"], sort=False):
        classes = np.union1d(d["Fault_True"].unique(), d["Fault_Pred"].unique())
        y_f, p_f = np.searchsorted(classes, d["Fault_True"]), np.searchsorted(classes, d["Fault_Pred"])
        y_s, p_s = d["Severity_True"].to_numpy(float), d["Severity_Pred"].to_numpy(float)
        rec, _ = pd.factorize(d["Recording_ID"])
        n_rec = rec.max() + 1
        point = _metrics_weighted(y_f, p_f, y_s, p_s, np.ones(len(d)), len(classes))
        boots = {k: np.empty(n_boot) for k in point}
        for b in range(n_boot):
            counts = np.bincount(rng.integers(0, n_rec, n_rec), minlength=n_rec)
            m = _metrics_weighted(y_f, p_f, y_s, p_s, counts[rec].astype(float), len(classes))
            for k, v in m.items():
                boots[k][b] = v
        for k, est in point.items():
            row = {"Modality": mod, "Model": model, "Metric": k, "Estimate": est}
            for lv in levels:
                a = (1 - lv) / 2
                row[f"CI{lv * 100:.0f}_lo"], row[f"CI{lv * 100:.0f}_hi"] = np.quantile(boots[k], [a, 1 - a])
            rows.append(row)
        log.info("bootstrap CIs done: %s / %s (%d recordings, %d resamples)", mod, model, n_rec, n_boot)
    return pd.DataFrame(rows)


# =========================================================================== 2. conformal sets
def conformal_quantile(scores: np.ndarray, level: float) -> float:
    """Finite-sample-corrected quantile: the ceil((n+1)*level)-th smallest score."""
    n = len(scores)
    k = int(np.ceil((n + 1) * level))
    return np.inf if k > n else float(np.sort(scores)[k - 1])


def ordinal_inclusion_mass(P: np.ndarray) -> np.ndarray:
    """M[i, s] = cumulative probability at which level s joins the interval grown from the mode."""
    n, S = P.shape
    r = np.arange(n)
    lo = hi = P.argmax(1)
    M = np.full((n, S), np.inf)
    cum = P[r, lo].copy()
    M[r, lo] = cum
    for _ in range(S - 1):
        left = np.where(lo > 0, P[r, np.maximum(lo - 1, 0)], -1.0)
        right = np.where(hi < S - 1, P[r, np.minimum(hi + 1, S - 1)], -1.0)
        go_left = left >= right
        new = np.where(go_left, lo - 1, hi + 1)
        cum = cum + np.where(go_left, left, right)
        M[r, new] = cum
        lo, hi = np.where(go_left, new, lo), np.where(go_left, hi, new)
    return M


def hull_scores(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Smallest q whose gap-filled set {p >= 1 - q} reaches y: 1 - min p between the mode and y."""
    mode = P.argmax(1)
    lo, hi = np.minimum(mode, y), np.maximum(mode, y)
    levels = np.arange(P.shape[1])
    between = (levels >= lo[:, None]) & (levels <= hi[:, None])
    return 1 - np.where(between, P, np.inf).min(1)


def hull_intervals(P: np.ndarray, q: float):
    inc = P >= 1 - q - 1e-12
    inc[np.arange(len(P)), P.argmax(1)] = True
    levels = np.arange(P.shape[1])
    return np.where(inc, levels, P.shape[1]).min(1), np.where(inc, levels, -1).max(1)


def fault_sets(P: np.ndarray, q: float) -> np.ndarray:
    S = P >= 1 - q - 1e-12
    S[np.arange(len(P)), P.argmax(1)] = True        # never return an empty set
    return S


def severity_intervals(P: np.ndarray, q: float):
    M = ordinal_inclusion_mass(P)
    inc = M <= q + 1e-12
    inc[np.arange(len(P)), P.argmax(1)] = True
    levels = np.arange(P.shape[1])
    lo = np.where(inc, levels, P.shape[1]).min(1)
    hi = np.where(inc, levels, -1).max(1)
    return lo, hi


SEVERITY_METHODS = {
    "ordinal": (lambda P, y: ordinal_inclusion_mass(P)[np.arange(len(P)), y], severity_intervals),
    "hull": (hull_scores, hull_intervals),
}


def _calibrate_severity(P, y, level):
    """Pick the severity construction with the narrower ranges on this calibration data."""
    best = None
    for name, (score_fn, interval_fn) in SEVERITY_METHODS.items():
        q = conformal_quantile(score_fn(P, y), level)
        lo, hi = interval_fn(P, q)
        width = (hi - lo + 1).mean()
        if best is None or width < best[2]:
            best = (name, q, width)
    return best[0], best[1]


def _prob_matrices(d: pd.DataFrame):
    fcols = [c for c in d.columns if c.startswith(FAULT_PREFIX) and d[c].notna().all()]
    scols = sorted([c for c in d.columns if c.startswith(SEV_PREFIX) and d[c].notna().all()],
                   key=lambda c: int(c[len(SEV_PREFIX):-1]))
    classes = np.array([c[len(FAULT_PREFIX):-1] for c in fcols])
    return classes, d[fcols].to_numpy(float), d[scols].to_numpy(float)


def has_probabilities(oof: pd.DataFrame) -> bool:
    return any(c.startswith(FAULT_PREFIX) for c in oof.columns) and "Fold" in oof.columns


def conformal_analysis(oof: pd.DataFrame, levels=LEVELS):
    """Returns (coverage summary, per-window sets, deployable thresholds)."""
    summary, per_window, thresholds = [], [], {}
    for (mod, model), d in oof.groupby(["Modality", "Model"], sort=False):
        classes, Pf, Ps = _prob_matrices(d)
        y_f = np.searchsorted(classes, d["Fault_True"].to_numpy())
        y_s = d["Severity_True"].to_numpy(int)
        folds = d["Fold"].to_numpy()
        score_f = 1 - Pf[np.arange(len(d)), y_f]
        out = d[["Modality", "Model", "Recording_ID", "Window_Idx", "Fault_True", "Fault_Pred",
                 "Severity_True", "Severity_Pred"]].copy()
        out["Fault_Confidence"] = Pf.max(1)
        thresholds.setdefault(mod, {})[model] = {}
        for lv in levels:
            tag = f"{lv * 100:.0f}"
            in_set = np.zeros(len(d), bool)
            set_size = np.zeros(len(d), int)
            set_str = np.empty(len(d), object)
            lo_all, hi_all = np.zeros(len(d), int), np.zeros(len(d), int)
            methods = []
            for k in np.unique(folds):
                te = folds == k
                qf = conformal_quantile(score_f[~te], lv)
                method, qs = _calibrate_severity(Ps[~te], y_s[~te], lv)
                methods.append(method)
                S = fault_sets(Pf[te], qf)
                in_set[te] = S[np.arange(te.sum()), y_f[te]]
                set_size[te] = S.sum(1)
                order = np.argsort(-Pf[te], axis=1)
                set_str[te] = [SET_SEP.join(classes[o][s[o]]) for o, s in zip(order, S)]
                lo_all[te], hi_all[te] = SEVERITY_METHODS[method][1](Ps[te], qs)
            sev_in = (y_s >= lo_all) & (y_s <= hi_all)
            per_class = pd.Series(in_set).groupby(d["Fault_True"].to_numpy()).mean()
            summary.append({
                "Modality": mod, "Model": model, "Level": lv,
                "Fault coverage": in_set.mean(),
                "Fault avg set size": set_size.mean(),
                "Fault single-answer share": (set_size == 1).mean(),
                "Worst-class coverage": per_class.min(),
                "Worst class": per_class.idxmin(),
                "Classes below level-5pp": int((per_class < lv - 0.05).sum()),
                "Severity coverage": sev_in.mean(),
                "Severity avg width": (hi_all - lo_all + 1).mean(),
                "Severity single-level share": (hi_all == lo_all).mean(),
                "Severity method": max(set(methods), key=methods.count),
            })
            out[f"Fault_Set_{tag}"] = set_str
            out[f"Severity_Low_{tag}"], out[f"Severity_High_{tag}"] = lo_all, hi_all
            # Deployment thresholds: calibrated on every out-of-fold score
            method, qs = _calibrate_severity(Ps, y_s, lv)
            thresholds[mod][model][tag] = {"fault_min_probability": 1 - conformal_quantile(score_f, lv),
                                           "severity_method": method, "severity_threshold_q": qs}
        per_window.append(out)
        log.info("conformal sets done: %s / %s", mod, model)
    return pd.DataFrame(summary), pd.concat(per_window, ignore_index=True), thresholds


def pick_examples(sets: pd.DataFrame, model: str, level_tag="95", per_kind=2) -> pd.DataFrame:
    """A few illustrative windows per modality: confident, ambiguous, and top-1 wrong but covered."""
    rows = []
    for mod, d in sets[sets["Model"] == model].groupby("Modality", sort=False):
        size = d[f"Fault_Set_{level_tag}"].str.count(SET_SEP) + 1
        wrong = d["Fault_Pred"] != d["Fault_True"]
        covered = [t in s.split(SET_SEP) for t, s in zip(d["Fault_True"], d[f"Fault_Set_{level_tag}"])]
        kinds = [("confident", (size == 1) & ~wrong), ("ambiguous", (size >= 2) & ~wrong),
                 ("top-1 wrong, set covers it", wrong & np.array(covered))]
        for kind, mask in kinds:
            pick = d[mask.to_numpy()]
            if len(pick):
                rows.append(pick.sample(min(per_kind, len(pick)), random_state=config.RANDOM_STATE).assign(Case=kind))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run(oof: pd.DataFrame, prefix: str = ""):
    """Compute everything and write CSV / parquet / JSON next to the other reports.
    prefix="recording_" is used for the recording-level (aggregated) predictions."""
    R = config.REPORT_DIR
    ci = bootstrap_metric_cis(oof)
    ci.to_csv(R / f"{prefix}metric_confidence_intervals.csv", index=False)
    conf = sets = None
    if has_probabilities(oof):
        conf, sets, thresholds = conformal_analysis(oof)
        conf.to_csv(R / f"{prefix}conformal_coverage.csv", index=False)
        sets.to_parquet(R / f"{prefix}conformal_prediction_sets.parquet", index=False)
        (R / f"{prefix}conformal_thresholds.json").write_text(json.dumps(thresholds, indent=2))
    else:
        log.warning("Saved predictions have no class probabilities; per-prediction conformal sets need a "
                    "retrain (run_pipeline.py --stage train). Metric CIs were still computed.")
    return ci, conf, sets


def apply_thresholds(P_fault: np.ndarray, P_sev: np.ndarray, classes, thresholds: dict, level_tag="95"):
    """Deployment helper: probabilities from model.predict_proba on new windows -> fault sets + severity ranges.

    thresholds = json.load(open("conformal_thresholds.json"))[modality][model_name]
    """
    t = thresholds[level_tag]
    S = fault_sets(P_fault, 1 - t["fault_min_probability"])
    order = np.argsort(-P_fault, axis=1)
    classes = np.asarray(classes)
    lo, hi = SEVERITY_METHODS[t["severity_method"]][1](P_sev, t["severity_threshold_q"])
    return [list(classes[o][s[o]]) for o, s in zip(order, S)], lo, hi
