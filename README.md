# Pump fault diagnosis: fault type and severity from motor current and vibration

An end-to-end pipeline that turns the raw *Motor Current and Vibration Monitoring Dataset for various Faults in an
E-motor-driven Centrifugal Pump* (Bruinsma et al., CC0) into windowed features. It then trains three models to
predict the **fault type** (21 classes) and its **severity** (0–6), separately for the electric and vibration
measurements, and evaluates them with confidence intervals.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Put the raw dataset at `Dataset/Dataset/{Electric,Vibration}/Motor-*/<speed>/<condition>/*.csv`. The original
`README.txt` from the dataset authors is included for reference.

On macOS, XGBoost needs an OpenMP runtime. The pipeline loads the copy that ships with PyTorch, so Homebrew's
`libomp` is not required.

## Running

```bash
.venv/bin/python run_pipeline.py                    # everything: features -> training -> speed test -> report
.venv/bin/python run_pipeline.py --stage features   # raw CSVs -> feature matrices (~4 min, cached and resumable)
.venv/bin/python run_pipeline.py --stage train      # 5-fold cross-validation of all models + report
.venv/bin/python run_pipeline.py --stage speed      # unseen-speed test (Motor-2, leave one speed out)
.venv/bin/python run_pipeline.py --stage report     # rebuild the report from saved predictions, no retraining
```

Useful options: `--models hrf,xgb`, `--folds 5`, `--group-level condition` (hold out whole condition folders),
`--workers 4`, `--limit 6` (smoke test), `--rebuild`.

The results go to `artifacts/reports/`, mainly `evaluation_report.html`, which is self-contained and opens in a
browser. Keep the machine plugged in and awake while training; on battery with sleep enabled, runs can take hours.

## How it works

| Module | Role |
|---|---|
| `pump_pipeline/catalog.py` | Walks the dataset and parses labels from folder names (e.g. `bearing bpfi 3` → Bearing BPFI, severity 3) |
| `pump_pipeline/loader.py` | Loads one condition folder at a time (memory-bounded) and cuts recordings into 1-second windows |
| `pump_pipeline/features.py` | ~280 features per window: signal statistics, spectral bands, shaft-order and envelope (vibration), supply frequency, harmonics, sidebands, phase balance (electric) |
| `pump_pipeline/build_features.py` | Parallel, resumable feature extraction into `artifacts/features/*.parquet` |
| `pump_pipeline/models.py` | Hierarchical Random Forest, multi-output XGBoost, multi-task MLP (PyTorch) |
| `pump_pipeline/evaluate.py` | Stratified group K-fold (whole recordings stay on one side of each split), metrics, figures |
| `pump_pipeline/uncertainty.py` | 90/95/99 % bootstrap confidence intervals for metrics; conformal fault sets and severity ranges per prediction |
| `pump_pipeline/aggregation.py` | Recording-level decisions (averaged window probabilities) and the "seconds needed" curve |
| `pump_pipeline/cross_speed.py` | Generalisation to an operating speed not seen in training |
| `pump_pipeline/report.py` | Markdown and HTML report |

## Dataset notes

- Each CSV holds **one sensor channel**. Its columns are repeated recordings of that channel, so column *r*
  across a folder's `ch1…chN` files is one synchronous recording.
- Vibration: 5 accelerometers, 12 s at 20 kHz. Electric: 3 phase currents and 3 voltages, 15 s at 20 kHz.
- `healthy 1/2/3`, `healthy noise` and `new motor` are all healthy baselines (severity 0). `coupling 2D` is
  coupling severity 2. Faults without a number get severity 1.
- "Motor-2" and "Motor-4" are setup labels: by speed and supply frequency, Motor-2 is the 4-pole machine.

## Early exploration scripts

`index_dataset.py`, `build_human_readable_catalog.py`, `inspect_signal.py` and `extract_feature_dataset.py` are
the first exploration scripts. They are superseded by `pump_pipeline/`. `extract_feature_dataset.py` treats the
recording columns as sensor channels, so its output should not be used for modelling.
