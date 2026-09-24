"""Load and explore CTG records from WFDB .dat/.hea files.

The CTU-UHB CTG dataset stores each record as a pair of files:

- 1121.hea: header file with metadata
- 1121.dat: raw signal values

The wfdb library reads both files together when we pass the record path
without the file extension.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import wfdb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"


def find_record_path(record_id, data_dir=DEFAULT_RAW_DIR):
    """Find the WFDB record path for a record ID.

    wfdb.rdrecord() expects a path without the .hea or .dat extension.
    For example, if the files are data/raw/1121.hea and data/raw/1121.dat,
    we pass data/raw/1121.

    The function also checks one folder below data/raw because downloaded
    PhysioNet datasets are often unzipped into a named subfolder.
    """
    data_dir = Path(data_dir)
    record_id = str(record_id)

    direct_header = data_dir / f"{record_id}.hea"
    if direct_header.exists():
        return data_dir / record_id

    nested_headers = sorted(data_dir.glob(f"*/{record_id}.hea"))
    if nested_headers:
        return nested_headers[0].with_suffix("")

    raise FileNotFoundError(
        f"Could not find {record_id}.hea in {data_dir}. "
        "Place the matching .hea and .dat files in data/raw."
    )


def load_record(record_id, data_dir=DEFAULT_RAW_DIR):
    """Load one CTG record using wfdb.rdrecord()."""
    record_path = find_record_path(record_id, data_dir)
    return wfdb.rdrecord(str(record_path))


def print_record_metadata(record):
    """Print beginner-friendly metadata for a loaded WFDB record."""
    print("\nRecord metadata")
    print("----------------")
    print(f"Record name: {record.record_name}")
    print(f"Signal names: {record.sig_name}")
    print(f"Sampling frequency: {record.fs} Hz")
    print(f"Signal length: {record.sig_len} samples")
    print(f"Number of signals: {record.n_sig}")
    print(f"Signal shape: {record.p_signal.shape}")


def get_signal(record, signal_name):
    """Return one signal by name, or None if that signal is unavailable."""
    if signal_name not in record.sig_name:
        return None

    signal_index = record.sig_name.index(signal_name)
    return record.p_signal[:, signal_index]


def get_fhr_uc_signals(record):
    """Return the FHR and UC signals from a loaded CTG record.

    Returns:
        tuple: (fhr, uc), where each item is a NumPy array or None.
    """
    fhr = get_signal(record, "FHR")
    uc = get_signal(record, "UC")
    return fhr, uc


def plot_record(record):
    """Plot FHR and UC signals for one CTG record."""
    fhr, uc = get_fhr_uc_signals(record)

    if fhr is None and uc is None:
        print("No FHR or UC signal was found, so there is nothing to plot.")
        return

    time_minutes = [sample / record.fs / 60 for sample in range(record.sig_len)]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle(f"CTG Record {record.record_name}", fontsize=14)

    if fhr is not None:
        axes[0].plot(time_minutes, fhr, color="tab:blue", linewidth=1)
        axes[0].set_ylabel("FHR (bpm)")
        axes[0].set_title("Fetal Heart Rate")
    else:
        axes[0].text(0.5, 0.5, "FHR signal not available", ha="center")

    if uc is not None:
        axes[1].plot(time_minutes, uc, color="tab:orange", linewidth=1)
        axes[1].set_ylabel("UC")
        axes[1].set_title("Uterine Contractions")
    else:
        axes[1].text(0.5, 0.5, "UC signal not available", ha="center")

    axes[1].set_xlabel("Time (minutes)")

    for axis in axes:
        axis.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
