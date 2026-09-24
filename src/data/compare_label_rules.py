"""Compare several rule-based CTG labeling strategies.

This script reads all CTU-UHB .hea files under data/raw, extracts the same
outcome fields used by generate_labels.py, applies multiple labeling rules,
prints class counts and percentages, and saves a comparison table.

Run from the project root:

    python src/data/compare_label_rules.py
"""

import csv
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.generate_labels import (
    FIELDS_TO_EXTRACT,
    LABELS,
    RAW_DIR,
    extract_header_fields,
    find_header_files,
    parse_number,
)
from src.data.generate_labels import (
    OUTPUT_PATH as GENERATED_LABELS_PATH,
)

OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "label_rule_comparison.csv"


def label_v1(ph, be, bdecf, apgar5):
    """Current project rule.

    This is the existing label rule from generate_labels.py.
    """
    severe_metabolic = be <= -12 or bdecf >= 12
    moderate_metabolic = be <= -8 or bdecf >= 8

    if (ph < 7.05 and severe_metabolic) or apgar5 <= 3:
        return "Pathological"
    elif ph < 7.20 or moderate_metabolic or apgar5 <= 6:
        return "Suspicious"
    else:
        return "Normal"


def label_v2_stricter_pathological(ph, be, bdecf, apgar5):
    """Alternative 1: stricter Pathological rule.

    Pathological requires stronger evidence: very low pH together with severe
    metabolic acidosis, or a very low Apgar5. This should usually produce fewer
    Pathological labels than V1.
    """
    severe_metabolic = be <= -12 or bdecf >= 12
    moderate_metabolic = be <= -8 or bdecf >= 8

    if (ph < 7.00 and severe_metabolic) or apgar5 <= 2:
        return "Pathological"
    elif ph < 7.20 or moderate_metabolic or apgar5 <= 6:
        return "Suspicious"
    else:
        return "Normal"


def label_v3_more_sensitive_pathological(ph, be, bdecf, apgar5):
    """Alternative 2: more sensitive Pathological rule.

    Pathological is assigned when either pH is very low or severe metabolic
    acidosis is present. This should usually catch more possible high-risk
    cases, at the cost of a larger Pathological class.
    """
    severe_metabolic = be <= -12 or bdecf >= 12
    moderate_metabolic = be <= -8 or bdecf >= 8

    if ph < 7.05 or severe_metabolic or apgar5 <= 3:
        return "Pathological"
    elif ph < 7.20 or moderate_metabolic or apgar5 <= 6:
        return "Suspicious"
    else:
        return "Normal"


def label_v4_ph_apgar_only(ph, be, bdecf, apgar5):
    """Alternative 3: pH + Apgar5 only.

    This ignores BE and BDecf entirely, which is useful for understanding how
    much the metabolic acidosis fields affect class balance.
    """
    if ph < 7.05 or apgar5 <= 3:
        return "Pathological"
    elif ph < 7.20 or apgar5 <= 6:
        return "Suspicious"
    else:
        return "Normal"


RULES = [
    ("V1_current_rule", label_v1),
    ("V2_stricter_pathological", label_v2_stricter_pathological),
    ("V3_more_sensitive_pathological", label_v3_more_sensitive_pathological),
    ("V4_ph_apgar_only", label_v4_ph_apgar_only),
]


def load_header_rows():
    """Load all records and extracted metadata.

    If data/processed/labels.csv already exists, we use it because it contains
    the extracted pH, BE, BDecf, and Apgar5 values. If it does not exist yet,
    we scan the .hea files directly.
    """
    if GENERATED_LABELS_PATH.exists():
        return load_rows_from_generated_labels(GENERATED_LABELS_PATH)

    header_files = find_header_files(RAW_DIR)

    if not header_files:
        raise FileNotFoundError(f"No .hea files were found under {RAW_DIR}")

    rows = []
    for header_path in header_files:
        rows.append(extract_header_fields(header_path))

    return rows


def load_rows_from_generated_labels(labels_path):
    """Load metadata columns from data/processed/labels.csv."""
    rows = []

    with labels_path.open("r", newline="", encoding="utf-8") as labels_file:
        reader = csv.DictReader(labels_file)

        for row in reader:
            rows.append(
                {
                    "record_id": row["record_id"],
                    "pH": parse_number(row["pH"]),
                    "BE": parse_number(row["BE"]),
                    "BDecf": parse_number(row["BDecf"]),
                    "Apgar5": parse_number(row["Apgar5"]),
                }
            )

    return rows


def summarize_rule(rows, rule_function):
    """Return class counts for one labeling rule."""
    counts = Counter()

    for row in rows:
        label = rule_function(row["pH"], row["BE"], row["BDecf"], row["Apgar5"])
        counts[label] += 1

    return counts


def print_rule_summary(rule_name, counts, total_records):
    """Print count and percentage for each class."""
    print(f"\n{rule_name}")
    print("-" * len(rule_name))

    for label in LABELS:
        count = counts[label]
        percentage = (count / total_records) * 100
        print(f"{label}: {count} ({percentage:.1f}%)")


def save_comparison(comparison_rows):
    """Save the comparison table to data/processed/label_rule_comparison.csv."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=["rule", "class", "count", "percentage"],
        )
        writer.writeheader()
        writer.writerows(comparison_rows)


def main():
    """Compare all label rules and save the result."""
    rows = load_header_rows()
    total_records = len(rows)
    comparison_rows = []

    print(f"Total records processed: {total_records}")

    print("\nFields used")
    print("-----------")
    print(", ".join(FIELDS_TO_EXTRACT))

    for rule_name, rule_function in RULES:
        counts = summarize_rule(rows, rule_function)
        print_rule_summary(rule_name, counts, total_records)

        for label in LABELS:
            count = counts[label]
            percentage = (count / total_records) * 100
            comparison_rows.append(
                {
                    "rule": rule_name,
                    "class": label,
                    "count": count,
                    "percentage": f"{percentage:.2f}",
                }
            )

    save_comparison(comparison_rows)
    print(f"\nSaved comparison to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
