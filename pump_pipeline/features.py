"""Per-window, per-channel condition-monitoring features (vectorised over windows x channels).

Every function takes x with shape (W, C, L) -- W windows, C channels, L samples -- and returns a dict
{feature_name: array(W, C)}. Channel-level dicts are flattened to ``ch{k}_{name}`` columns later.
"""
from typing import Dict

import numpy as np
import pandas as pd

from . import config
from .loader import window

EPS = 1e-12


# --------------------------------------------------------------------------- time domain
def time_features(x: np.ndarray) -> Dict[str, np.ndarray]:
    mean = x.mean(-1)
    xc = x - mean[..., None]
    m2 = (xc ** 2).mean(-1) + EPS
    absx = np.abs(x)
    rms = np.sqrt((x ** 2).mean(-1)) + EPS
    peak = absx.max(-1)
    abs_mean = absx.mean(-1) + EPS
    return {
        "mean": mean,
        "std": np.sqrt(m2),
        "rms": rms,
        "peak": peak,
        "p2p": x.max(-1) - x.min(-1),
        "skew": (xc ** 3).mean(-1) / m2 ** 1.5,
        "kurtosis": (xc ** 4).mean(-1) / m2 ** 2,
        "crest": peak / rms,
        "shape": rms / abs_mean,
        "impulse": peak / abs_mean,
        "clearance": peak / (np.sqrt(absx).mean(-1) ** 2 + EPS),
    }


# --------------------------------------------------------------------------- spectral helpers
def _amplitude_spectrum(x: np.ndarray, fs: int):
    L = x.shape[-1]
    win = np.hanning(L)
    xc = x - x.mean(-1, keepdims=True)
    amp = np.abs(np.fft.rfft(xc * win, axis=-1)) * (2.0 / win.sum())
    return np.fft.rfftfreq(L, 1.0 / fs), amp


def _band_log_power(psd: np.ndarray, freqs: np.ndarray, edges_hz) -> np.ndarray:
    """log10 power in each [edge_i, edge_i+1) band; band edges forced to span >= 1 bin."""
    b = np.searchsorted(freqs, edges_hz)
    for i in range(1, len(b)):
        b[i] = max(b[i], b[i - 1] + 1)
    b = np.minimum(b, psd.shape[-1])
    cs = np.concatenate([np.zeros(psd.shape[:-1] + (1,)), np.cumsum(psd, axis=-1)], axis=-1)
    return np.log10(cs[..., b[1:]] - cs[..., b[:-1]] + EPS)


def _peak_near(amp: np.ndarray, center_bin, half_width) -> np.ndarray:
    """Max amplitude within center_bin +- half_width (center_bin: scalar or array(W, C))."""
    center = np.broadcast_to(np.asarray(center_bin), amp.shape[:-1])
    offsets = np.arange(-half_width, half_width + 1)
    idx = np.clip(center[..., None] + offsets, 0, amp.shape[-1] - 1)
    return np.take_along_axis(amp, idx.astype(np.int64), axis=-1).max(-1)


def spectral_features(freqs: np.ndarray, amp: np.ndarray, fs: int) -> Dict[str, np.ndarray]:
    psd = amp ** 2
    total = psd.sum(-1) + EPS
    p = psd / total[..., None]
    centroid = (p * freqs).sum(-1)
    out = {
        "spec_centroid": centroid,
        "spec_spread": np.sqrt((p * (freqs - centroid[..., None]) ** 2).sum(-1)),
        "spec_entropy": -(p * np.log(p + EPS)).sum(-1) / np.log(p.shape[-1]),
    }
    edges = np.geomspace(4.0, fs / 2, config.N_LOG_BANDS + 1)
    bands = _band_log_power(psd, freqs, edges)
    for i in range(bands.shape[-1]):
        out[f"band{i:02d}_{edges[i]:.0f}Hz"] = bands[..., i]
    return out


# --------------------------------------------------------------------------- vibration-specific
def order_features(freqs: np.ndarray, amp: np.ndarray, shaft_hz: float) -> Dict[str, np.ndarray]:
    """Amplitude at shaft-speed orders (unbalance=1x, misalignment=2x, looseness=0.5x/harmonics)."""
    df = freqs[1] - freqs[0]
    out = {}
    for k in config.SHAFT_ORDERS:
        f = k * shaft_hz
        half = max(1, int(np.ceil(f * config.ORDER_TOLERANCE / df)))
        out[f"order_{k:g}x"] = _peak_near(amp, int(round(f / df)), half)
    return out


def envelope_features(x: np.ndarray, fs: int) -> Dict[str, np.ndarray]:
    """High-pass + Hilbert envelope spectrum: exposes bearing defect repetition frequencies."""
    L = x.shape[-1]
    xc = x - x.mean(-1, keepdims=True)
    X = np.fft.rfft(xc, axis=-1)
    freqs = np.fft.rfftfreq(L, 1.0 / fs)
    X[..., freqs < config.ENVELOPE_HIGHPASS_HZ] = 0
    Z = np.zeros(x.shape[:-1] + (L,), dtype=np.complex128)   # one-sided -> analytic signal
    n_pos = (L + 1) // 2
    Z[..., 1:n_pos] = 2 * X[..., 1:n_pos]
    if L % 2 == 0:
        Z[..., L // 2] = X[..., L // 2]
    env = np.abs(np.fft.ifft(Z, axis=-1))
    env_c = env - env.mean(-1, keepdims=True)
    m2 = (env_c ** 2).mean(-1) + EPS
    env_freqs, env_amp = _amplitude_spectrum(env, fs)
    bands = _band_log_power(env_amp ** 2, env_freqs, config.ENVELOPE_BAND_EDGES_HZ)
    out = {"env_rms": np.sqrt(m2), "env_kurtosis": (env_c ** 4).mean(-1) / m2 ** 2}
    for i, lo in enumerate(config.ENVELOPE_BAND_EDGES_HZ[:-1]):
        out[f"env_band{i:02d}_{lo}Hz"] = bands[..., i]
    return out


# --------------------------------------------------------------------------- electric-specific
def electric_channel_features(freqs: np.ndarray, amp: np.ndarray) -> Dict[str, np.ndarray]:
    """Supply fundamental, harmonic distortion and slip sidebands (MCSA: rotor bars, stator shorts)."""
    df = freqs[1] - freqs[0]
    lo, hi = (int(f / df) for f in config.LINE_FREQ_SEARCH_HZ)
    f1_bin = lo + amp[..., lo:hi].argmax(-1)
    a1 = _peak_near(amp, f1_bin, 1) + EPS
    harm = np.stack([_peak_near(amp, k * f1_bin, 1) for k in range(2, config.N_HARMONICS + 1)], -1)
    side = np.maximum(
        np.stack([_peak_near(amp, f1_bin - d, 0) for d in range(3, 11)], -1).max(-1),
        np.stack([_peak_near(amp, f1_bin + d, 0) for d in range(3, 11)], -1).max(-1),
    )
    return {
        "line_freq": f1_bin * df,
        "fund_amp": a1,
        "thd": np.sqrt((harm ** 2).sum(-1)) / a1,
        "h3_ratio": harm[..., 1] / a1,
        "h5_ratio": harm[..., 3] / a1,
        "h7_ratio": harm[..., 5] / a1,
        "sideband_db": 20 * np.log10(side / a1 + EPS),
    }


def electric_cross_features(x: np.ndarray, rms: np.ndarray) -> Dict[str, np.ndarray]:
    """Phase balance and power features; channels 1-3 = currents, 4-6 = voltages."""
    if x.shape[1] < 6:
        return {}
    i_rms, v_rms = rms[:, :3], rms[:, 3:6]
    p_real = (x[:, :3] * x[:, 3:6]).mean(-1).sum(-1)
    s_app = (i_rms * v_rms).sum(-1) + EPS
    return {
        "current_unbalance": (i_rms.max(-1) - i_rms.min(-1)) / (i_rms.mean(-1) + EPS),
        "voltage_unbalance": (v_rms.max(-1) - v_rms.min(-1)) / (v_rms.mean(-1) + EPS),
        "current_mean_rms": i_rms.mean(-1),
        "voltage_mean_rms": v_rms.mean(-1),
        "real_power": p_real,
        "power_factor": p_real / s_app,
    }


# --------------------------------------------------------------------------- assembly
def window_features(x: np.ndarray, modality: str, shaft_hz: float, fs: int = config.SAMPLE_RATE_HZ) -> pd.DataFrame:
    """x: (W, C, L) one recording's windows -> DataFrame (W rows)."""
    x = x.astype(np.float64)
    per_channel = time_features(x)
    freqs, amp = _amplitude_spectrum(x, fs)
    per_channel.update(spectral_features(freqs, amp, fs))
    cross = {}
    if modality == "Vibration":
        per_channel.update(order_features(freqs, amp, shaft_hz))
        per_channel.update(envelope_features(x, fs))
    else:
        per_channel.update(electric_channel_features(freqs, amp))
        cross = electric_cross_features(x, per_channel["rms"])

    cols = {}
    for c in range(x.shape[1]):
        for name, arr in per_channel.items():
            cols[f"ch{c + 1}_{name}"] = arr[:, c]
    cols.update(cross)
    return pd.DataFrame(cols).astype(np.float32)


def shaft_frequency(motor: str, speed_pct: int, signals: np.ndarray, fs: int) -> float:
    rpm = config.SHAFT_RPM.get((motor, speed_pct))
    if rpm is not None:
        return rpm / 60.0
    # Fallback: dominant vibration peak in 5-60 Hz of the first recording/channel
    freqs, amp = _amplitude_spectrum(signals[0, 0, : fs * 4].astype(np.float64), fs)
    band = (freqs >= 5) & (freqs <= 60)
    return float(freqs[band][amp[band].argmax()])


def condition_features(meta: dict, signals: np.ndarray, rec_idx: np.ndarray) -> pd.DataFrame:
    """All windows of all recordings of one condition folder -> labelled feature rows."""
    fs = config.SAMPLE_RATE_HZ
    shaft_hz = shaft_frequency(meta["Motor_ID"], meta["Speed_Percent"], signals, fs)
    windows = window(signals, fs)  # (R, W, C, L)
    frames = []
    for r, col in enumerate(rec_idx):
        df = window_features(windows[r], meta["Modality"], shaft_hz, fs)
        df.insert(0, "Window_Idx", np.arange(len(df)))
        df.insert(0, "Recording_ID", f"{meta['Condition_ID']}#rec{col:02d}")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    for k in ("Modality", "Motor_ID", "Speed_Percent", "Condition", "Fault_Type", "Severity_Level",
              "Variant", "Condition_ID"):
        out.insert(0, k, meta[k])
    out.insert(0, "N_Channels", signals.shape[1])
    out.insert(0, "Shaft_Hz", np.float32(shaft_hz))
    return out
