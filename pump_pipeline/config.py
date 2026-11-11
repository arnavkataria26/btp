"""Central configuration: paths, label conventions, signal-processing constants.

Dataset facts (verified against the raw files and Appendices/Other/measurement overview.xlsx):
  * One CSV per *physical sensor channel* (``...-ch1.csv`` .. ``-ch6.csv``), NOT one CSV per recording.
  * Inside a CSV: column 0 = time [s]; columns 1..N = N *repeated recordings* of that same sensor.
    N varies per condition folder (2..68) but is identical across the channel files of a folder,
    so column ``r`` of ``ch1..chK`` forms one synchronous K-channel recording.
  * Vibration: 5 accelerometers [g], 12 s @ 20 kHz (240,000 samples).
  * Electric:  3 phase currents [A] + 3 phase voltages [V], 15 s @ 20 kHz (300,000 samples).
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "Dataset" / "Dataset"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
FEATURE_DIR = ARTIFACT_DIR / "features"
SHARD_DIR = FEATURE_DIR / "shards"
REPORT_DIR = ARTIFACT_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

SAMPLE_RATE_HZ = 20_000
WINDOW_SECONDS = 1.0          # non-overlapping windows -> 12 (vib) / 15 (elec) windows per recording
MAX_NAN_FRACTION = 0.05       # drop a recording if any channel has more NaNs than this

MODALITIES = ("Electric", "Vibration")

# Nominal shaft speed per (Motor_ID, Speed_Percent), from the measurement overview (RPM).
SHAFT_RPM = {
    ("Motor-4", 70): 2070,
    ("Motor-2", 100): 1480,
    ("Motor-2", 75): 1110,
    ("Motor-2", 50): 740,
}

# Folder stems that are all healthy machine states. The trailing digit on "healthy 1/2/3" is a
# repeat-baseline index, not a severity; "healthy noise" is healthy with a neighbouring pump
# running; "new motor" is the healthy baseline of the replacement motor used for stator shorts.
HEALTHY_STEMS = {"healthy", "healthy noise", "new motor"}
HEALTHY_LABEL = "Healthy"

# Faults recorded at a single, unnumbered level get this severity.
DEFAULT_FAULT_SEVERITY = 1

ACRONYMS = {"bpfi": "BPFI", "bpfo": "BPFO", "bsf": "BSF"}

# Feature-extraction constants
N_LOG_BANDS = 24                       # log-spaced PSD bands, 2 Hz .. Nyquist
SHAFT_ORDERS = (0.5, 1.0, 2.0, 3.0, 4.0)
ORDER_TOLERANCE = 0.03                 # +-3 % search window around each shaft order
ENVELOPE_HIGHPASS_HZ = 1_000           # demodulation band for bearing-fault envelope analysis
ENVELOPE_BAND_EDGES_HZ = (5, 10, 20, 35, 50, 75, 100, 150, 200, 300, 400, 600, 1000)
LINE_FREQ_SEARCH_HZ = (5, 120)         # VFD-driven: estimate supply frequency per window
N_HARMONICS = 9

RANDOM_STATE = 42
N_SPLITS = 5
