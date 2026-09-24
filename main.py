"""Explore one CTG record from the CTU-UHB dataset.

Run from the project root:

    python main.py
"""

from src.data.load_ctg import (
    get_fhr_uc_signals,
    load_record,
    plot_record,
    print_record_metadata,
)

SAMPLE_RECORD_ID = "1121"


def print_first_values(fhr, uc, number_of_values=10):
    """Print the first few FHR and UC values so we can inspect the data."""
    print(f"\nFirst {number_of_values} values")
    print("----------------")

    if fhr is not None:
        print(f"FHR: {fhr[:number_of_values]}")
    else:
        print("FHR: not available")

    if uc is not None:
        print(f"UC: {uc[:number_of_values]}")
    else:
        print("UC: not available")


def main():
    """Load, print, and plot one sample CTG record."""
    print(f"Loading CTG record {SAMPLE_RECORD_ID}...")

    record = load_record(SAMPLE_RECORD_ID)
    fhr, uc = get_fhr_uc_signals(record)

    print_record_metadata(record)
    print_first_values(fhr, uc)
    plot_record(record)


if __name__ == "__main__":
    main()
