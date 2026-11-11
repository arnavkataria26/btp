"""Builds evaluation_report.md and a self-contained evaluation_report.html from saved results, so the
report (including confidence intervals) can be regenerated without retraining: --stage report."""
import base64
import json
import logging

import markdown
import pandas as pd
from sklearn.metrics import classification_report

from . import aggregation, config, uncertainty

log = logging.getLogger(__name__)

FAULT_COLS = ["fault_accuracy", "fault_macro_precision", "fault_macro_recall", "fault_macro_f1",
              "fault_micro_precision", "fault_micro_recall", "fault_micro_f1"]
SEV_COLS = ["severity_mae", "severity_rmse", "severity_exact_acc"]
MODEL_SLUGS = {"Hierarchical RF": "hrf", "Multi-Output XGBoost": "xgb", "Multi-Task MLP": "mlp"}


def _pct(x):
    return f"{x * 100:.1f}%"


def _short_set(s: str, keep=4) -> str:
    """Most probable `keep` faults of a set, then a count of the rest (sets are stored most-probable first)."""
    items = s.split(uncertainty.SET_SEP)
    return s if len(items) <= keep else uncertainty.SET_SEP.join(items[:keep]) + f" … (+{len(items) - keep} more)"


def _ci_section(ci: pd.DataFrame):
    lv = [f"{v * 100:.0f}" for v in uncertainty.LEVELS]
    lines = [
        "## Confidence intervals for the metrics",
        "",
        f"Recording-level bootstrap ({uncertainty.N_BOOTSTRAP} resamples of whole recordings). "
        "*Estimate* is computed on all pooled out-of-fold predictions, so it can differ slightly from the "
        "mean-of-folds tables above. Read a row as: *we are 95 % confident the true value lies in the "
        "95 % interval*. Narrower = more certain.",
        "",
    ]
    for mod, dm in ci.groupby("Modality", sort=False):
        lines += [f"### {mod}", "",
                  "| Model | Metric | Estimate | " + " | ".join(f"{v}% CI" for v in lv) + " |",
                  "|---|---|---|" + "---|" * len(lv)]
        for _, r in dm.iterrows():
            cis = " | ".join(f"[{r[f'CI{v}_lo']:.4f}, {r[f'CI{v}_hi']:.4f}]" for v in lv)
            lines.append(f"| {r.Model} | {r.Metric} | {r.Estimate:.4f} | {cis} |")
        lines.append("")
    return lines


def _conformal_section(conf: pd.DataFrame, sets: pd.DataFrame):
    lines = [
        "## Per-prediction confidence: conformal prediction sets",
        "",
        "Instead of a single answer, every window gets a **set of fault types** and a **severity range** that "
        "contain the truth with the chosen confidence. *Coverage* is how often that actually happened on "
        "held-out recordings; it should be at or just above the level. Smaller sets at the same coverage "
        "mean a more useful model. A set of size 1 is a confident single answer.",
        "",
        "Coverage is guaranteed on average over all windows, not for every fault type separately: "
        "*Worst-class coverage* shows the weakest fault type. When a model's single answer is already right "
        "more often than the level asks for, sets cannot shrink below one fault, so coverage exceeds the level.",
        "",
        "| Modality | Model | Level | Fault coverage | Avg faults in set | Single-answer share | "
        "Worst-class coverage | Severity coverage | Avg severity range (levels) | Single-level share | Range method |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in conf.iterrows():
        lines.append(
            f"| {r.Modality} | {r.Model} | {r.Level * 100:.0f}% | {_pct(r['Fault coverage'])} | "
            f"{r['Fault avg set size']:.2f} | {_pct(r['Fault single-answer share'])} | "
            f"{_pct(r['Worst-class coverage'])} ({r['Worst class']}) | {_pct(r['Severity coverage'])} | "
            f"{r['Severity avg width']:.2f} | {_pct(r['Severity single-level share'])} | {r['Severity method']} |")
    best = "Multi-Output XGBoost" if "Multi-Output XGBoost" in set(sets["Model"]) else sets["Model"].iloc[0]
    ex = uncertainty.pick_examples(sets, best)
    if len(ex):
        lines += ["", f"### Example predictions ({best}, out-of-fold)", "",
                  "Fault sets are listed most-probable first; the percentage is the model's probability "
                  "for its top answer.", "",
                  "| Modality | Case | True fault / severity | Predicted | 90% fault set | 95% fault set | "
                  "99% fault set | 90% / 95% / 99% severity range |",
                  "|---|---|---|---|---|---|---|---|"]
        for _, r in ex.iterrows():
            rng = " / ".join(f"{r[f'Severity_Low_{t}']}–{r[f'Severity_High_{t}']}" for t in ("90", "95", "99"))
            lines.append(
                f"| {r.Modality} | {r.Case} | {r.Fault_True} / {r.Severity_True} | "
                f"{r.Fault_Pred} / {r.Severity_Pred} ({_pct(r.Fault_Confidence)}) | {_short_set(r.Fault_Set_90)} | "
                f"{_short_set(r.Fault_Set_95)} | {_short_set(r.Fault_Set_99)} | {rng} |")
    lines += ["", "All sets for every window: `conformal_prediction_sets.parquet`. Thresholds for applying "
                  "the same sets to new data: `conformal_thresholds.json`.", ""]
    return lines


def _ci_cell(ci: pd.DataFrame, metric: str, digits=3, level="95"):
    r = ci[ci["Metric"] == metric].iloc[0]
    return f"{r.Estimate:.{digits}f} [{r[f'CI{level}_lo']:.{digits}f}, {r[f'CI{level}_hi']:.{digits}f}]"


def _recording_section(oof, rec, rec_ci, rec_conf, curve):
    lines = [
        "## Recording-level decisions (one diagnosis per recording)",
        "",
        "Instead of judging each 1-second window on its own, the window probabilities of a whole recording "
        "(12 s vibration, 15 s electric) are averaged and one decision is made. Values in brackets are 95 % "
        "bootstrap confidence intervals over recordings.",
        "",
        "| Modality | Model | Recordings | Fault accuracy | Macro F1 | Severity MAE | Severity exact-match | "
        "Per-window accuracy (for comparison) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for (mod, model), d in rec.groupby(["Modality", "Model"], sort=False):
        c = rec_ci[(rec_ci.Modality == mod) & (rec_ci.Model == model)]
        w = oof[(oof.Modality == mod) & (oof.Model == model)]
        lines.append(f"| {mod} | {model} | {len(d)} | {_ci_cell(c, 'Fault accuracy')} | "
                     f"{_ci_cell(c, 'Fault macro F1')} | {_ci_cell(c, 'Severity MAE')} | "
                     f"{_ci_cell(c, 'Severity exact-match acc')} | {(w.Fault_True == w.Fault_Pred).mean():.3f} |")
    if rec_conf is not None:
        lines += ["", "### Per-recording confidence sets", "",
                  "Conformal sets as above, but for the recording-level decision (calibrated on recordings of "
                  "the other folds).", "",
                  "| Modality | Model | Level | Fault coverage | Avg faults in set | Single-answer share | "
                  "Severity coverage | Avg severity range (levels) |", "|---|---|---|---|---|---|---|---|"]
        for _, r in rec_conf.iterrows():
            lines.append(f"| {r.Modality} | {r.Model} | {r.Level * 100:.0f}% | {_pct(r['Fault coverage'])} | "
                         f"{r['Fault avg set size']:.2f} | {_pct(r['Fault single-answer share'])} | "
                         f"{_pct(r['Severity coverage'])} | {r['Severity avg width']:.2f} |")
    marks = [s for s in (1, 2, 3, 5, 8, 10, 12, 15) if s in set(curve["Seconds"])]
    lines += ["", "### How many seconds are needed?", "",
              "Fault macro-F1 when the decision uses only the first *n* seconds of each recording.", "",
              "| Modality | Model | " + " | ".join(f"{s} s" for s in marks) + " |",
              "|---|---|" + "---|" * len(marks)]
    for (mod, model), d in curve.groupby(["Modality", "Model"], sort=False):
        v = d.set_index("Seconds")["Fault macro F1"]
        lines.append(f"| {mod} | {model} | " + " | ".join(f"{v[s]:.3f}" if s in v else "–" for s in marks) + " |")
    return lines + [""]


def _cross_speed_section(metrics, baseline, xs_ci):
    lines = [
        "## Generalisation to an unseen operating speed (Motor-2)",
        "",
        "Leave-one-speed-out: models are trained on Motor-2 at two speeds and tested on the third, which they "
        "have never seen. 75 % tests interpolation between known speeds; 50 % and 100 % test extrapolation. "
        "*Seen-speed* is the main cross-validation on the same windows, where that speed was in training. "
        "Brackets: 95 % bootstrap confidence intervals.",
        "",
        "| Modality | Model | Test speed | Trained on | Accuracy, unseen speed | Accuracy, seen speed | "
        "Macro F1, unseen | Macro F1, seen | Severity MAE, unseen | Severity MAE, seen | Per-recording accuracy, unseen |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in metrics.iterrows():
        b = baseline[(baseline.Modality == r.Modality) & (baseline.Model == r.Model) &
                     (baseline.Speed == r["Test speed (%)"])]
        c = xs_ci[(xs_ci.Modality == r.Modality) & (xs_ci.Model == f"{r.Model} @ {r['Test speed (%)']}")]
        seen = (lambda k, f=".3f": format(b[k].iloc[0], f) if len(b) else "–")
        lines.append(
            f"| {r.Modality} | {r.Model} | {r['Test speed (%)']}% | {r['Train speeds (%)'].replace('+', ' + ')}% | "
            f"{_ci_cell(c, 'Fault accuracy')} | {seen('seen_accuracy')} | {_ci_cell(c, 'Fault macro F1')} | "
            f"{seen('seen_macro_f1')} | {r.severity_mae:.3f} | {seen('seen_severity_mae')} | "
            f"{r.recording_fault_accuracy:.3f} |")
    unseen = metrics[metrics["unseen_fault_types"].fillna("") != ""].drop_duplicates(["Modality", "Test speed (%)"])
    for _, r in unseen.iterrows():
        lines.append(f"\n*{r.Modality}, test {r['Test speed (%)']}%: fault types absent from the training speeds "
                     f"(cannot be predicted): {r.unseen_fault_types}.*")
    return lines + [""]


def render_markdown(summary, oof, run_info, ci, conf, sets, extra=()) -> str:
    def cell(r, m):
        return f"{r[m + '_mean']:.4f} ± {r[m + '_std']:.4f}"

    lines = [
        "# Evaluation report",
        "",
        f"{run_info['n_splits']}-fold StratifiedGroupKFold (groups = {run_info['group_level']}, strata = "
        "fault × severity). Values are mean ± std across folds. Severity metrics are end-to-end, so for the "
        "hierarchical model they include errors carried over from a wrong Stage-1 fault prediction.",
        "",
        "## Fault type (classification)",
        "",
        "| Modality | Model | Accuracy | Macro P | Macro R | Macro F1 | Micro P | Micro R | Micro F1 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in summary.iterrows():
        lines.append(f"| {r.Modality} | {r.Model} | " + " | ".join(cell(r, c) for c in FAULT_COLS) + " |")
    lines += ["", "## Severity level (ordinal)", "",
              "| Modality | Model | MAE | RMSE | Exact-match acc |", "|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        lines.append(f"| {r.Modality} | {r.Model} | " + " | ".join(cell(r, c) for c in SEV_COLS) + " |")
    lines += [""] + _ci_section(ci)
    if conf is not None:
        lines += _conformal_section(conf, sets)
    else:
        lines += ["## Per-prediction confidence: conformal prediction sets", "",
                  "*Not available yet: the saved predictions have no class probabilities. Re-run "
                  "`run_pipeline.py --stage train` once to produce them.*", ""]
    for section in extra:
        lines += section
    lines += ["## Per-class reports (pooled out-of-fold predictions)", ""]
    for (mod, model), d in oof.groupby(["Modality", "Model"], sort=False):
        labels = sorted(d["Fault_True"].unique())
        lines.append(f"### {model} — {mod}\n```\n" + classification_report(
            d["Fault_True"], d["Fault_Pred"], labels=labels, digits=4, zero_division=0) + "```\n")
    return "\n".join(lines)


def render_html(md_text: str, models) -> str:
    body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])

    def img(name, cap):
        path = config.FIGURE_DIR / name
        if not path.exists():
            return ""
        b = base64.b64encode(path.read_bytes()).decode()
        return f'<figure><img src="data:image/png;base64,{b}" alt="{cap}"><figcaption>{cap}</figcaption></figure>'

    figs = ["<h2>Model comparison</h2>",
            img("model_comparison.png", "Fault macro-F1 and severity MAE, mean ± std over folds")]
    for mod in config.MODALITIES:
        figs.append(f"<h2>{mod} — severity error plots</h2>")
        for name in models:
            slug = MODEL_SLUGS.get(name, name)
            figs.append(f"<h3>{name}</h3>")
            figs.append(img(f"severity_errors_{mod.lower()}_{slug}.png", f"{name} · {mod} · severity error distribution"))
    body = body.replace("<h2>Confidence intervals", "\n".join(figs) + "\n<h2>Confidence intervals", 1)
    seconds_fig = img("seconds_needed.png", "Fault macro-F1 vs. seconds of signal per decision (out-of-fold)")
    body = body.replace("<h2>Generalisation", seconds_fig + "\n<h2>Generalisation", 1) if "<h2>Generalisation" in body \
        else body.replace("<h2>Per-class reports", seconds_fig + "\n<h2>Per-class reports", 1)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Pump Fault Evaluation</title>
<style>
:root{{--bg:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--rule:#e4e3df}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Helvetica Neue",sans-serif}}
main{{max-width:1100px;margin:0 auto;padding:32px 16px 64px}}
h1{{font-size:28px;margin:0 0 8px}} h2{{margin-top:40px;padding-bottom:6px;border-bottom:1px solid var(--rule)}}
p{{color:var(--ink2);max-width:80ch}}
table{{border-collapse:collapse;width:100%;font-size:13px;background:var(--surface);margin:12px 0;display:block;overflow-x:auto}}
th,td{{padding:7px 10px;border-bottom:1px solid var(--rule);text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}}
th:nth-child(-n+2),td:nth-child(-n+2){{text-align:left}} th{{color:var(--ink2);font-weight:600}}
pre{{background:var(--surface);border:1px solid var(--rule);border-radius:6px;padding:12px;overflow-x:auto;font-size:12px}}
figure{{margin:16px 0 28px;background:var(--surface);border:1px solid var(--rule);border-radius:8px;padding:12px}}
img{{max-width:100%;height:auto;display:block;margin:0 auto}} figcaption{{color:var(--ink2);font-size:13px;margin-top:6px}}
@media print{{body{{background:#fff}} figure,table,pre{{break-inside:avoid}}}}
</style></head><body><main>{body}</main></body></html>"""


def write_all(summary=None, oof=None, run_info=None):
    """Load anything not passed in from artifacts/reports, compute uncertainty, write md + html."""
    R = config.REPORT_DIR
    summary = summary if summary is not None else pd.read_csv(R / "metrics_summary.csv")
    oof = oof if oof is not None else pd.read_parquet(R / "oof_predictions.parquet")
    if run_info is None:
        info_path = R / "run_info.json"
        run_info = json.loads(info_path.read_text()) if info_path.exists() else \
            {"group_level": "recording", "n_splits": config.N_SPLITS}
    ci, conf, sets = uncertainty.run(oof)
    extra = []

    # Recording-level decisions + seconds-needed curve
    rec = aggregation.aggregate(oof)
    rec.to_parquet(R / "recording_predictions.parquet", index=False)
    rec_ci, rec_conf, _ = uncertainty.run(rec, prefix="recording_")
    curve = aggregation.seconds_curve(oof)
    curve.to_csv(R / "seconds_needed.csv", index=False)
    aggregation.plot_seconds_curve(curve, config.FIGURE_DIR / "seconds_needed.png")
    extra.append(_recording_section(oof, rec, rec_ci, rec_conf, curve))

    # Unseen-speed experiment (written by --stage speed)
    if (R / "cross_speed_metrics.csv").exists():
        from .cross_speed import in_distribution_baseline      # lazy: cross_speed imports evaluate -> report
        xs = pd.read_csv(R / "cross_speed_metrics.csv")
        xs_pred = pd.read_parquet(R / "cross_speed_predictions.parquet")
        xs_pred = xs_pred.assign(Model=xs_pred["Model"] + " @ " + xs_pred["Test_Speed"].astype(str))
        xs_ci = uncertainty.bootstrap_metric_cis(xs_pred)
        xs_ci.to_csv(R / "cross_speed_confidence_intervals.csv", index=False)
        extra.append(_cross_speed_section(xs, in_distribution_baseline(oof), xs_ci))

    md_text = render_markdown(summary, oof, run_info, ci, conf, sets, extra)
    (R / "evaluation_report.md").write_text(md_text)
    (R / "evaluation_report.html").write_text(render_html(md_text, list(dict.fromkeys(summary["Model"]))))
    log.info("Report written: %s and %s", R / "evaluation_report.md", R / "evaluation_report.html")
    return md_text
