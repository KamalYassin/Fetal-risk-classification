"""Generate simple CTG risk labels from CTU-UHB header metadata.

This script reads every .hea file under data/raw, extracts outcome fields from
the commented metadata section, applies a simple rule-based labeling function,
and saves the result to data/processed/labels.csv.

Run from the project root:

    python src/data/generate_labels.py
"""

import csv
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "labels.csv"

FIELDS_TO_EXTRACT = ["pH", "BE", "BDecf", "Apgar5"]
LABELS = ["Normal", "Suspicious", "Pathological"]


def parse_number(value):
    """Convert a header value to a float.

    The CTU-UHB headers sometimes use "NaN" for missing numeric values.
    Python's float("nan") lets us keep that value while still detecting it
    later with math.isnan().
    """
    try:
        return float(value)
    except ValueError:
        return math.nan


def is_missing(value):
    """Return True when a parsed value is missing."""
    return value is None or math.isnan(value)


def label_ctg(ph, be, bdecf, apgar5):
    severe_metabolic = be <= -12 or bdecf >= 12
    moderate_metabolic = be <= -8 or bdecf >= 8

    if (ph < 7.05 and severe_metabolic) or apgar5 <= 3:
        return "Pathological"
    elif ph < 7.20 or moderate_metabolic or apgar5 <= 6:
        return "Suspicious"
    else:
        return "Normal"


def safe_label_ctg(ph, be, bdecf, apgar5):
    return label_ctg(ph, be, bdecf, apgar5)


def extract_header_fields(header_path):
    """Extract record_id, pH, BE, BDecf, and Apgar5 from one .hea file."""
    values = {field: math.nan for field in FIELDS_TO_EXTRACT}

    with header_path.open("r", encoding="utf-8") as header_file:
        first_line = header_file.readline().strip()
        record_id = first_line.split()[0]

        for line in header_file:
            line = line.strip()

            # Metadata lines look like "#pH           7.14".
            if not line.startswith("#"):
                continue

            clean_line = line.lstrip("#").strip()
            parts = clean_line.split()
            if len(parts) < 2:
                continue

            field_name = parts[0]
            field_value = parts[-1]

            if field_name in values:
                values[field_name] = parse_number(field_value)

    return {
        "record_id": record_id,
        "pH": values["pH"],
        "BE": values["BE"],
        "BDecf": values["BDecf"],
        "Apgar5": values["Apgar5"],
    }


def format_csv_value(value):
    """Write missing values as NaN so they are easy to spot in the CSV."""
    if is_missing(value):
        return "NaN"
    return value


def find_header_files(raw_dir):
    """Find all .hea files under data/raw, including nested dataset folders."""
    direct_headers = list(raw_dir.glob("*.hea"))
    nested_headers = list(raw_dir.glob("*/*.hea"))
    return sorted(direct_headers + nested_headers)


def main():
    """Generate labels.csv and print a small summary."""
    header_files = find_header_files(RAW_DIR)

    if not header_files:
        raise FileNotFoundError(f"No .hea files were found under {RAW_DIR}")

    rows = []
    missing_counts = {field: 0 for field in FIELDS_TO_EXTRACT}
    class_counts = {label: 0 for label in LABELS}

    for header_path in header_files:
        row = extract_header_fields(header_path)

        for field in FIELDS_TO_EXTRACT:
            if is_missing(row[field]):
                missing_counts[field] += 1

        label = safe_label_ctg(row["pH"], row["BE"], row["BDecf"], row["Apgar5"])
        row["label"] = label
        class_counts[label] += 1
        rows.append(row)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=["record_id", "pH", "BE", "BDecf", "Apgar5", "label"],
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    "record_id": row["record_id"],
                    "pH": format_csv_value(row["pH"]),
                    "BE": format_csv_value(row["BE"]),
                    "BDecf": format_csv_value(row["BDecf"]),
                    "Apgar5": format_csv_value(row["Apgar5"]),
                    "label": row["label"],
                }
            )

    print(f"Saved labels to: {OUTPUT_PATH}")
    print(f"Total records processed: {len(rows)}")

    print("\nMissing values")
    print("--------------")
    for field in FIELDS_TO_EXTRACT:
        print(f"{field}: {missing_counts[field]}")

    print("\nClass distribution")
    print("------------------")
    for label in LABELS:
        print(f"{label}: {class_counts[label]}")


if __name__ == "__main__":
    main()
