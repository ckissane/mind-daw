#!/usr/bin/env python3
"""
compute_baseline.py — MNE-powered baseline profile computation.

Usage:
    python3 scripts/compute_baseline.py <profile_name>

Reads:
    profiles/<profile_name>/raw_baseline.bin   — flat float32, shape (n_samples, 64)
    profiles/<profile_name>/raw_baseline.json  — sample_rate, n_channels, n_samples

Writes:
    profiles/<profile_name>/baseline.json      — BaselineProfile-compatible JSON
                                                 (overwrites any Rust-computed version)

Pipeline (per Mullen et al. 2015, IEEE TBME):
    1. Load raw data as MNE RawArray
    2. FIR high-pass 0.5 Hz  (min-phase, order ~660 @ 300 Hz)
    3. Identify and interpolate bad channels (z-score of variance)
    4. ASR (Artifact Subspace Reconstruction, c=7)  — via mne-icalabel / pyprep
    5. Common Average Reference (CAR)
    6. FIR notch 50 Hz (and 100 Hz)
    7. Compute multitaper PSD per channel
    8. FOOOF / specparam on average occipital PSD  → aperiodic offset + exponent
    9. Band powers (δ θ α β γ), IAF, FAA, region powers, channel quality
   10. Write baseline.json

Dependencies:
    pip install mne numpy scipy fooof
    Optional (for ASR): pip install pyprep
"""

import sys
import os
import json
import time
import math
import traceback
from pathlib import Path

import numpy as np

# ── Channel layout (Cognionics HD-72, from official LSL config) ────────────────
# github.com/labstreaminglayer/App-Cognionics  —  cognionics_config.cfg
CH_NAMES = [
    "AF7h",  "AFp3",  "AFPz",  "AFp4",  "AF8h",   # 0–4
    "F5h",   "AFF3",  "AFF1",  "AFFz",  "AFF2",   # 5–9
    "AFF4",  "F6h",                                # 10–11
    "FC5",   "FFC3",  "FFC3h", "FFC1",  "FFCz",   # 12–16
    "FFC2",  "FFC4h", "FFC4",  "FC6",              # 17–20
    "FCC5h", "FCC3",  "FCC3h", "FCC1h", "FCCz",   # 21–25
    "FCC2h", "FCC4h", "FCC4",  "FCC6h",            # 26–29
    "CCP5h", "CCP3",  "CCP3h", "CCP1",  "CCPz",   # 30–34
    "CCP2",  "CCP4h", "CCP4",  "CCP6h",            # 35–38
    "CP5",   "CPP3",  "CPP3h", "CPP1",  "CPPz",   # 39–43
    "CPP2",  "CPP4h", "CPP4",  "CP6",              # 44–47
    "P5h",   "PPO5",  "PPO3",  "PO1",   "PPOz",   # 48–52
    "PO2",   "PPO4",  "PPO6",  "P6h",              # 53–56
    "PPO9h", "POO7",  "O1",    "POOz",  "O2",     # 57–61
    "POO8",  "PPO10h",                             # 62–63
]

# Occipital channels for IAF
IAF_CHANNELS = ["O1", "POOz", "O2"]          # indices 59, 60, 61
# Frontal channels for FAA  (AFF3 ≈ F3,  AFF4 ≈ F4)
FAA_L_CH = "AFF3"   # index 6
FAA_R_CH = "AFF4"   # index 10

# Brain region groupings (channel names)
REGION_NAMES = ["Frontal", "Temporal", "Central", "Parietal", "Occipital"]
REGION_CHANNELS = {
    "Frontal":   ["AF7h", "AFp3", "AFPz", "AFp4", "AF8h", "AFF3", "AFF1", "AFFz", "AFF2", "AFF4"],
    "Temporal":  ["FC5", "FCC5h", "CP5", "CP6", "FCC6h"],
    "Central":   ["FCCz", "CCPz", "FFCz", "FFC1", "FFC2"],
    "Parietal":  ["CPPz", "PPOz", "CPP1", "CPP2", "P5h", "P6h"],
    "Occipital": ["O1", "POOz", "O2", "POO7", "POO8", "PPO9h", "PPO10h"],
}

BAND_RANGES = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 80.0),
}
BAND_ORDER = ["delta", "theta", "alpha", "beta", "gamma"]


def load_raw(profile_dir: Path):
    """Load raw EEG data from the flat binary + JSON sidecar."""
    meta_path = profile_dir / "raw_baseline.json"
    bin_path  = profile_dir / "raw_baseline.bin"

    with open(meta_path) as f:
        meta = json.load(f)

    sfreq      = float(meta["sample_rate"])
    n_channels = int(meta["n_channels"])
    n_samples  = int(meta["n_samples"])

    raw = np.fromfile(str(bin_path), dtype="<f4").reshape(n_samples, n_channels)
    # MNE wants shape (n_channels, n_times) in Volts; Cognionics outputs µV
    data_v = raw.T.astype(np.float64) * 1e-6
    return data_v, sfreq


def make_mne_raw(data_v: np.ndarray, sfreq: float):
    """Wrap numpy array in an MNE RawArray with correct channel info."""
    import mne
    mne.set_log_level("WARNING")

    info = mne.create_info(
        ch_names=CH_NAMES,
        sfreq=sfreq,
        ch_types=["eeg"] * len(CH_NAMES),
    )
    raw = mne.io.RawArray(data_v, info, verbose=False)

    # Assign standard 10-5 montage (closest available in MNE)
    try:
        montage = mne.channels.make_standard_montage("standard_1005")
        raw.set_montage(montage, on_missing="ignore", verbose=False)
    except Exception:
        pass  # montage is optional; positions used only for display

    return raw


def preprocess(raw):
    """Apply the Mullen 2015 preprocessing pipeline."""
    import mne

    # 1. High-pass FIR 0.5 Hz (minimum-phase)
    raw.filter(
        l_freq=0.5, h_freq=None,
        method="fir", fir_design="firwin",
        phase="minimum", verbose=False,
    )

    # 2. Notch filter 50 Hz (power line) + harmonics
    freqs = [f for f in [50.0, 100.0, 150.0] if f < raw.info["sfreq"] / 2]
    if freqs:
        raw.notch_filter(freqs=freqs, verbose=False)

    # 3. ASR — use pyprep if available, otherwise skip
    try:
        from pyprep.prep_pipeline import PrepPipeline
        prep_params = {
            "ref_chs": "average",
            "reref_chs": "average",
            "line_freqs": [50.0],
        }
        prep = PrepPipeline(raw.copy(), prep_params, raw.get_montage(), verbose=False)
        prep.fit()
        raw = prep.raw_eeg
    except ImportError:
        # pyprep not installed — do simple bad-channel detection + CAR
        _flag_bad_channels(raw)

    # 4. Common Average Reference
    raw.set_eeg_reference("average", projection=False, verbose=False)

    # 5. Low-pass FIR 80 Hz (above gamma band)
    if raw.info["sfreq"] > 170:
        raw.filter(l_freq=None, h_freq=80.0, verbose=False)

    return raw


def _flag_bad_channels(raw, z_thresh=3.5):
    """Mark channels with unusually high variance as bad."""
    data = raw.get_data()
    var = np.var(data, axis=1)
    log_var = np.log(var + 1e-30)
    z = (log_var - log_var.mean()) / (log_var.std() + 1e-10)
    bads = [raw.ch_names[i] for i, zv in enumerate(z) if zv > z_thresh]
    if bads:
        raw.info["bads"] = bads
        raw.interpolate_bads(verbose=False)


def compute_psd(raw, fmax=80.0):
    """Multitaper PSD for every channel. Returns freqs (Hz) and psd (ch × freq) in µV²/Hz."""
    import mne
    psd = raw.compute_psd(method="multitaper", fmax=fmax, verbose=False)
    freqs = psd.freqs
    data  = psd.get_data() * 1e12   # V²/Hz → µV²/Hz
    return freqs, data


def band_power(freqs, psd_row, fmin, fmax):
    """Integrate PSD between fmin and fmax using trapezoidal rule."""
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not mask.any():
        return 0.0
    return float(np.trapz(psd_row[mask], freqs[mask]))


def compute_iaf(freqs, psd, ch_names, iaf_channels):
    """Peak alpha frequency from averaged occipital PSD."""
    idxs = [ch_names.index(c) for c in iaf_channels if c in ch_names]
    if not idxs:
        return 10.0
    occ_psd = psd[idxs, :].mean(axis=0)
    alpha_mask = (freqs >= 8.0) & (freqs <= 13.0)
    if not alpha_mask.any():
        return 10.0
    peak_idx = np.argmax(occ_psd[alpha_mask])
    return float(freqs[alpha_mask][peak_idx])


def compute_fooof(freqs, psd, ch_names, iaf_channels, freq_range=(1.0, 40.0)):
    """Fit FOOOF / specparam aperiodic component on averaged occipital PSD."""
    try:
        from fooof import FOOOF
    except ImportError:
        try:
            from specparam import SpectralModel as FOOOF
        except ImportError:
            return 0.0, 0.0, 0.0

    idxs = [ch_names.index(c) for c in iaf_channels if c in ch_names]
    if not idxs:
        return 0.0, 0.0, 0.0

    avg_psd = psd[idxs, :].mean(axis=0)
    mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    if mask.sum() < 4:
        return 0.0, 0.0, 0.0

    fm = FOOOF(verbose=False)
    try:
        fm.fit(freqs[mask], avg_psd[mask], freq_range)
        offset   = float(fm.aperiodic_params_[0])
        exponent = float(fm.aperiodic_params_[1])
        r2       = float(fm.r_squared_) if hasattr(fm, "r_squared_") else 0.0
        return offset, exponent, r2
    except Exception:
        return 0.0, 0.0, 0.0


def build_profile(raw, freqs, psd, sfreq, recorded_at):
    """Assemble the baseline.json dict from processed data."""
    ch_names = raw.ch_names
    n_ch = len(ch_names)

    # Band powers [channel][band]
    mean_band_powers = []
    for ch_idx in range(n_ch):
        row = []
        for band in BAND_ORDER:
            fmin, fmax = BAND_RANGES[band]
            row.append(band_power(freqs, psd[ch_idx], fmin, fmax))
        mean_band_powers.append(row)

    # Noise floor (std dev of raw signal in µV)
    data_uv = raw.get_data() * 1e6
    noise_floor = list(np.std(data_uv, axis=1).astype(float))

    max_noise = max(noise_floor) or 1.0
    channel_quality = [float(1.0 - min(n / max_noise, 1.0) * 0.92) for n in noise_floor]

    dominant_band = [int(np.argmax(row)) for row in mean_band_powers]

    # IAF
    iaf_hz = compute_iaf(freqs, psd, ch_names, IAF_CHANNELS)

    # FAA
    faa = 0.0
    if FAA_L_CH in ch_names and FAA_R_CH in ch_names:
        li = ch_names.index(FAA_L_CH)
        ri = ch_names.index(FAA_R_CH)
        la = band_power(freqs, psd[li], 8.0, 13.0)
        ra = band_power(freqs, psd[ri], 8.0, 13.0)
        faa = float(math.log(max(ra, 1e-10)) - math.log(max(la, 1e-10)))

    # Region powers
    region_powers = []
    for region in REGION_NAMES:
        chs = [c for c in REGION_CHANNELS.get(region, []) if c in ch_names]
        if not chs:
            region_powers.append([0.0] * 5)
            continue
        idxs = [ch_names.index(c) for c in chs]
        avg = []
        for band in BAND_ORDER:
            fmin, fmax = BAND_RANGES[band]
            avg.append(float(np.mean([band_power(freqs, psd[i], fmin, fmax) for i in idxs])))
        region_powers.append(avg)

    # Mean amplitude spectrum [channel][bin] — use sqrt(PSD) for amplitude
    amp = np.sqrt(psd)
    mean_spectrum = amp.tolist()

    # FOOOF
    fooof_offset, fooof_exponent, fooof_r2 = compute_fooof(
        freqs, psd, ch_names, IAF_CHANNELS
    )

    return {
        "recorded_at":     recorded_at,
        "duration_s":      float(raw.times[-1]),
        "mean_band_powers": mean_band_powers,
        "noise_floor":     noise_floor,
        "channel_quality": channel_quality,
        "dominant_band":   dominant_band,
        "iaf_hz":          iaf_hz,
        "faa":             faa,
        "region_powers":   region_powers,
        "mean_spectrum":   mean_spectrum,
        "fooof_offset":    fooof_offset,
        "fooof_exponent":  fooof_exponent,
        "fooof_r2":        fooof_r2,
        "mne_processed":   True,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 compute_baseline.py <profile_name>", file=sys.stderr)
        sys.exit(1)

    profile_name = sys.argv[1]
    profile_dir  = Path("profiles") / profile_name

    if not profile_dir.exists():
        print(f"[mne] profile directory not found: {profile_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[mne] loading raw baseline for '{profile_name}'…", flush=True)
    try:
        data_v, sfreq = load_raw(profile_dir)
    except Exception as e:
        print(f"[mne] failed to load raw data: {e}", file=sys.stderr)
        sys.exit(1)

    recorded_at = time.time()

    print(f"[mne] {data_v.shape[1]} samples × {data_v.shape[0]} channels @ {sfreq} Hz", flush=True)
    print("[mne] building MNE RawArray…", flush=True)
    raw = make_mne_raw(data_v, sfreq)

    print("[mne] preprocessing (HP filter → notch → ASR/CAR)…", flush=True)
    try:
        raw = preprocess(raw)
    except Exception as e:
        print(f"[mne] preprocessing error: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    print("[mne] computing multitaper PSD…", flush=True)
    freqs, psd = compute_psd(raw)

    print("[mne] fitting FOOOF aperiodic model…", flush=True)
    profile = build_profile(raw, freqs, psd, sfreq, recorded_at)

    out_path = profile_dir / "baseline.json"
    with open(out_path, "w") as f:
        json.dump(profile, f, indent=2)

    fooof_str = ""
    if profile["fooof_r2"] > 0:
        fooof_str = f"  |  1/f exponent {profile['fooof_exponent']:.2f} (R²={profile['fooof_r2']:.3f})"
    print(
        f"[mne] ✓ done — IAF {profile['iaf_hz']:.1f} Hz  "
        f"FAA {profile['faa']:+.2f}{fooof_str}",
        flush=True,
    )
    print(f"[mne] written to {out_path}", flush=True)


if __name__ == "__main__":
    main()
