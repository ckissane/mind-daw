#!/usr/bin/env python3
"""Convert a mind-daw recording session to an MNE-compatible FIF file.

Reads the JSON epoch files saved by the Rust app under:
    sessions/{session_id}/epochs/*.json

Writes:
    sessions/{session_id}/exports/{session_id}-epo.fif   — MNE EpochsArray
    sessions/{session_id}/exports/{session_id}-epo.csv   — feature summary (optional)

Usage:
    # Single session by ID
    python scripts/session_to_fif.py sessions/<session_id>

    # Auto-discover the most recent session
    python scripts/session_to_fif.py

    # With ICA artifact removal
    python scripts/session_to_fif.py sessions/<session_id> --ica

    # Also plot ERP averages per class
    python scripts/session_to_fif.py sessions/<session_id> --plot

Requirements:
    pip install mne numpy
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import mne
    from mne import EpochsArray, create_info
    from mne.preprocessing import ICA
except ImportError:
    print("ERROR: MNE-Python not found.  Install it with:\n  pip install mne", file=sys.stderr)
    sys.exit(1)

# ── Cognionics HD-72 channel layout ──────────────────────────────────────────
# Standard 10-20 names for a 64-channel cap.  Adjust if your montage differs.
_CH_NAMES_64 = [
    "Fp1", "Fp2", "F7",  "F3",  "Fz",  "F4",  "F8",  "FC5",
    "FC1", "FC2", "FC6", "T7",  "C3",  "Cz",  "C4",  "T8",
    "TP9", "CP5", "CP1", "CP2", "CP6", "TP10","P7",  "P3",
    "Pz",  "P4",  "P8",  "PO9", "O1",  "Oz",  "O2",  "PO10",
    "AF7", "AF3", "AF4", "AF8", "F5",  "F1",  "F2",  "F6",
    "FT9", "FT7", "FC3", "FC4", "FT8", "FT10","C5",  "C1",
    "C2",  "C6",  "TP7", "CP3", "CPz", "CP4", "TP8", "P5",
    "P1",  "P2",  "P6",  "PO7", "PO3", "POz", "PO4", "PO8",
]


def load_epochs_from_session(session_dir: Path) -> list[dict]:
    """Return a list of raw epoch dicts loaded from JSON files."""
    epoch_dir = session_dir / "epochs"
    if not epoch_dir.exists():
        print(f"ERROR: no epochs/ directory found in {session_dir}", file=sys.stderr)
        sys.exit(1)

    epoch_files = sorted(epoch_dir.glob("*.json"))
    if not epoch_files:
        print(f"ERROR: no .json epoch files found in {epoch_dir}", file=sys.stderr)
        sys.exit(1)

    epochs = []
    for f in epoch_files:
        try:
            ep = json.loads(f.read_text())
            epochs.append(ep)
        except Exception as e:
            print(f"  [skip] {f.name}: {e}", file=sys.stderr)

    # Sort by timestamp so epoch order is chronological
    epochs.sort(key=lambda e: e.get("timestamp", 0.0))
    print(f"Loaded {len(epochs)} epochs from {epoch_dir}")
    return epochs


def build_mne_epochs(epoch_dicts: list[dict]) -> EpochsArray:
    """Convert a list of JSON epoch dicts to an MNE EpochsArray."""
    # Infer metadata from the first epoch
    first = epoch_dicts[0]
    sample_rate = float(first["sample_rate"])
    n_channels  = len(first["samples"][0])   # samples[time][channel]
    pre_samples = int(first.get("pre_samples", 60))
    tmin        = -pre_samples / sample_rate  # seconds before stimulus

    # Build channel info
    if n_channels == 64:
        ch_names = _CH_NAMES_64
    else:
        ch_names = [f"EEG{i:03d}" for i in range(n_channels)]

    info = create_info(ch_names=ch_names, sfreq=sample_rate, ch_types="eeg")
    info.set_montage("standard_1020", match_case=False, on_missing="ignore")

    # Build unique integer event IDs for each label
    all_labels  = [ep["label"] for ep in epoch_dicts]
    unique_labels = sorted(set(all_labels))
    label_to_id = {lbl: i + 1 for i, lbl in enumerate(unique_labels)}
    print("Event IDs:", {v: k for k, v in label_to_id.items()})

    # Stack data: shape (n_epochs, n_channels, n_times)
    data_list  = []
    events_list = []

    for idx, ep in enumerate(epoch_dicts):
        # samples: list[list[float]] → shape (n_times, n_channels)
        arr = np.array(ep["samples"], dtype=np.float32)  # (n_times, n_channels)
        # Pad or trim to a consistent length based on the first epoch
        n_times_expected = len(first["samples"])
        if arr.shape[0] < n_times_expected:
            pad = np.zeros((n_times_expected - arr.shape[0], n_channels), dtype=np.float32)
            arr = np.vstack([arr, pad])
        elif arr.shape[0] > n_times_expected:
            arr = arr[:n_times_expected]

        # MNE expects (n_channels, n_times)
        data_list.append(arr.T)

        # Events row: [sample_number, prev_id, event_id]
        # Use index as the sample number (onset within the concatenated timeline)
        events_list.append([idx * n_times_expected, 0, label_to_id[ep["label"]]])

    data   = np.stack(data_list, axis=0)          # (n_epochs, n_channels, n_times)
    events = np.array(events_list, dtype=np.int32)

    # MNE wants volts; Cognionics data is in µV → convert
    data = data * 1e-6

    epochs = EpochsArray(
        data,
        info,
        events=events,
        event_id=label_to_id,
        tmin=tmin,
        baseline=None,   # caller can apply baseline later
        verbose=False,
    )
    return epochs


def apply_ica(epochs: EpochsArray, n_components: int = 20) -> EpochsArray:
    """Run ICA on the epochs and interactively exclude artifact components."""
    print(f"\nRunning ICA with {n_components} components...")
    ica = ICA(n_components=n_components, random_state=42, max_iter="auto")

    # ICA works better on filtered data; fit on a copy
    epochs_for_fit = epochs.copy().filter(1.0, 40.0, verbose=False)
    ica.fit(epochs_for_fit)

    # Automatically flag EOG-like components using Fp1/Fp2
    eog_channels = [ch for ch in ["Fp1", "Fp2"] if ch in epochs.ch_names]
    if eog_channels:
        eog_idx, eog_scores = ica.find_bads_eog(epochs, ch_name=eog_channels, verbose=False)
        print(f"  Auto-detected EOG components: {eog_idx}")
        ica.exclude = eog_idx

    print(f"  Excluding {len(ica.exclude)} component(s): {ica.exclude}")
    epochs_clean = ica.apply(epochs.copy())
    return epochs_clean


def plot_erps(epochs: EpochsArray) -> None:
    """Plot ERP averages per stimulus class."""
    print("\nPlotting ERP averages (close windows to continue)...")
    for label, event_id in epochs.event_id.items():
        subset = epochs[label]
        if len(subset) == 0:
            continue
        evoked = subset.average()
        evoked.plot(titles=f"ERP: {label}  (n={len(subset)})", show=False)
    mne.viz.plot_compare_evokeds(
        {lbl: epochs[lbl].average() for lbl in epochs.event_id},
        title="All classes — ERP comparison",
    )
    import matplotlib.pyplot as plt
    plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

def find_latest_session() -> Path:
    sessions_root = Path("sessions")
    if not sessions_root.exists():
        print("ERROR: no sessions/ directory found.  Run from the mind-daw project root.", file=sys.stderr)
        sys.exit(1)
    candidates = [d for d in sessions_root.iterdir() if d.is_dir() and (d / "epochs").exists()]
    if not candidates:
        print("ERROR: no sessions with epoch data found.", file=sys.stderr)
        sys.exit(1)
    # Pick most recently modified
    latest = max(candidates, key=lambda d: d.stat().st_mtime)
    print(f"Auto-selected session: {latest}")
    return latest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a mind-daw session to an MNE FIF file."
    )
    parser.add_argument(
        "session_dir",
        nargs="?",
        help="Path to the session directory (e.g. sessions/<id>). "
             "Omit to use the most recently modified session.",
    )
    parser.add_argument(
        "--ica",
        action="store_true",
        help="Run ICA to remove eye/muscle artifacts before saving.",
    )
    parser.add_argument(
        "--ica-components",
        type=int,
        default=20,
        metavar="N",
        help="Number of ICA components (default: 20).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show ERP plots per stimulus class after conversion.",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Skip the 1–40 Hz bandpass filter applied before saving.",
    )
    args = parser.parse_args()

    session_dir = Path(args.session_dir) if args.session_dir else find_latest_session()
    if not session_dir.exists():
        print(f"ERROR: directory not found: {session_dir}", file=sys.stderr)
        sys.exit(1)

    # Load
    epoch_dicts = load_epochs_from_session(session_dir)
    epochs = build_mne_epochs(epoch_dicts)

    # Apply baseline correction (-200 ms → 0 ms = pre-stimulus window)
    epochs.apply_baseline(baseline=(None, 0), verbose=False)

    # Optional bandpass filter
    if not args.no_filter:
        print("Applying 1–40 Hz bandpass filter...")
        epochs.filter(1.0, 40.0, verbose=False)

    # Optional ICA
    if args.ica:
        epochs = apply_ica(epochs, n_components=args.ica_components)

    # Save
    export_dir = session_dir / "exports"
    export_dir.mkdir(exist_ok=True)
    session_id = session_dir.name
    out_path   = export_dir / f"{session_id}-epo.fif"

    epochs.save(str(out_path), overwrite=True)
    print(f"\nSaved: {out_path}")
    print(f"  {len(epochs)} epochs  |  {len(epochs.ch_names)} channels  |  {epochs.info['sfreq']} Hz")
    print(f"  Classes: {dict(sorted((k, sum(epochs.events[:,2] == v) for k, v in epochs.event_id.items())))}")
    print(f"\nLoad in Python with:\n  import mne\n  epochs = mne.read_epochs('{out_path}')")

    # Optional plots
    if args.plot:
        plot_erps(epochs)


if __name__ == "__main__":
    main()
