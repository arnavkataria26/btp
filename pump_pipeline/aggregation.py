"""Recording-level decisions: one diagnosis per recording instead of one per 1-second window.

A monitoring system can listen for several seconds before deciding. The window probabilities of a recording
are averaged (soft voting), and the decision is the most probable fault and the median severity of the
averaged distributions (the standard decision for an ordinal scale). Without saved probabilities it falls back to a majority vote / median severity.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from . import uncertainty
from .style import GRID, INK_2, model_color, plt

KEYS = ["Modality", "Model", "Recording_ID"]


def _cols(d: pd.DataFrame, prefix: str):
    cols = [c for c in d.columns if c.startswith(prefix) and d[c].notna().all()]
    if prefix == uncertainty.SEV_PREFIX:
        cols.sort(key=lambda c: int(c[len(prefix):-1]))
    return cols


def aggregate(oof: pd.DataFrame, n_windows: int = None) -> pd.DataFrame:
    """Window-level out-of-fold predictions -> one row per recording, same columns as the window frame.
    n_windows: only use each recording's first n windows (n seconds); None = all windows."""
    d_all = oof if n_windows is None else oof[oof["Window_Idx"] < n_windows]
    out = []
    for _, d in d_all.groupby(["Modality", "Model"], sort=False):
        g = d.groupby(KEYS, sort=False)
        rec = g[["Fold", "Fault_True", "Severity_True"]].first() if "Fold" in d else \
            g[["Fault_True", "Severity_True"]].first()
        rec["Window_Idx"] = g.size()                       # number of windows (seconds) used
        fcols, scols = _cols(d, uncertainty.FAULT_PREFIX), _cols(d, uncertainty.SEV_PREFIX)
        if fcols and scols:
            pf, ps = g[fcols].mean(), g[scols].mean()
            classes = np.array([c[len(uncertainty.FAULT_PREFIX):-1] for c in fcols])
            rec["Fault_Pred"] = classes[pf.to_numpy().argmax(1)]
            # median of the averaged ordinal distribution: first level with cumulative probability >= 50 %
            rec["Severity_Pred"] = (np.cumsum(ps.to_numpy(), axis=1) >= 0.5 - 1e-9).argmax(1)
            rec = pd.concat([rec, pf, ps], axis=1)
        else:
            rec["Fault_Pred"] = g["Fault_Pred"].agg(lambda s: s.mode().iloc[0])
            rec["Severity_Pred"] = g["Severity_Pred"].median().round().astype(int)
        out.append(rec.reset_index())
    return pd.concat(out, ignore_index=True)


def seconds_curve(oof: pd.DataFrame) -> pd.DataFrame:
    """Accuracy / macro-F1 / severity exact-match when deciding after the first n seconds."""
    rows = []
    for n in range(1, int(oof["Window_Idx"].max()) + 2):
        rec = aggregate(oof, n)
        for (mod, model), d in rec.groupby(["Modality", "Model"], sort=False):
            if n > d["Window_Idx"].max():                  # modality has fewer windows per recording
                continue
            rows.append({"Modality": mod, "Model": model, "Seconds": n,
                         "Fault accuracy": (d.Fault_True == d.Fault_Pred).mean(),
                         "Fault macro F1": f1_score(d.Fault_True, d.Fault_Pred, average="macro", zero_division=0),
                         "Severity exact-match acc": (d.Severity_True == d.Severity_Pred).mean()})
    return pd.DataFrame(rows)


def plot_seconds_curve(curve: pd.DataFrame, path):
    """One panel per modality (each with its own y-range); models are categorical series, direct-labelled."""
    mods = list(dict.fromkeys(curve["Modality"]))
    fig, axes = plt.subplots(1, len(mods), figsize=(5.2 * len(mods), 3.8), squeeze=False)
    for ax, mod in zip(axes[0], mods):
        d = curve[curve["Modality"] == mod]
        for model, dm in d.groupby("Model", sort=False):
            c = model_color(model)
            ax.plot(dm["Seconds"], dm["Fault macro F1"] * 100, color=c, linewidth=2, marker="o", markersize=4,
                    label=model)
            last = dm.iloc[-1]
            ax.annotate(f"{last['Fault macro F1'] * 100:.1f}", (last["Seconds"], last["Fault macro F1"] * 100),
                        xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK_2)
        ax.set_title(f"{mod}", loc="left")
        ax.set_xlabel("Seconds of signal used per decision")
        ax.set_ylabel("Fault macro-F1 (%)")
        ax.yaxis.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_xticks(sorted(d["Seconds"].unique()))
        ax.set_xlim(right=d["Seconds"].max() + 1.5)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.0))
    fig.text(0.01, 0.01, "Out-of-fold predictions; window probabilities averaged over the first n seconds of "
                         "each recording.", color=INK_2, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.9))
    fig.savefig(path, dpi=150)
    plt.close(fig)
