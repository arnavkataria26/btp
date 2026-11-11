"""Shared matplotlib styling (reference palette from the dataviz skill): light surface, ink tokens,
a single-hue sequential blue ramp, and categorical slots in fixed order (one per model)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", [SURFACE, "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
MODEL_ORDER = ["Hierarchical RF", "Multi-Output XGBoost", "Multi-Task MLP"]   # colour follows the model

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.grid": False,
})


def model_color(name: str) -> str:
    return SERIES[MODEL_ORDER.index(name) % len(SERIES)] if name in MODEL_ORDER else INK_2
