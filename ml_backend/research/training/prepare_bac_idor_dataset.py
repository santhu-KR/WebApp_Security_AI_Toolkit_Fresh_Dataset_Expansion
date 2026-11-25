
import os
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# BAC-ML-1M: IDOR Dataset Preparation
# --------------------------------------------------

SOURCE_FILE = Path.home() / "Downloads" / "enhanced_bac_results.csv"

OUTPUT_DIR = Path(__file__).parent / "dataset" / "external_sources" / "BAC_ML_1M"

OUTPUT_FILE = OUTPUT_DIR / "bac_idor_prepared.csv"
SUMMARY_FILE = OUTPUT_DIR / "bac_idor_summary.txt"

# Features selected for HTTP access-control classification
FEATURE_COLUMNS = [
    "request_method",
    "resource_type",
    "resource_classification",
    "expected_access",
    "actual_access_granted",
    "vulnerability_type"
]

def main():

    print("=" * 60)
    print("BAC-ML-1M IDOR DATASET PREPARATION")
    print("=" * 60)

    # Check source file
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {SOURCE_FILE}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nReading original dataset...")

    # Read only required columns
    df = pd.read_csv(
        SOURCE_FILE,
        usecols=FEATURE_COLUMNS,
        low_memory=False
    )

    print("Total original records:", len(df))

    # Remove records with missing selected features
    before = len(df)
    df = df.dropna(subset=FEATURE_COLUMNS).copy()
    removed = before - len(df)

    # Normalize text
    for column in FEATURE_COLUMNS:
        df[column] = df[column].astype(str).str.strip()

    # Preserve original labels
    valid_labels = ["No", "Yes (IDOR)"]

    df = df[df["vulnerability_type"].isin(valid_labels)].copy()

    # Create binary target label
    df["idor_label"] = df["vulnerability_type"].map({
        "No": 0,
        "Yes (IDOR)": 1
    })

    # Rename label for clarity
    df = df.rename(
        columns={"vulnerability_type": "original_label"}
    )

    # Save prepared dataset
    df.to_csv(OUTPUT_FILE, index=False)

    # Calculate statistics
    label_counts = df["original_label"].value_counts()

    total = len(df)
    idor_count = int(df["idor_label"].sum())
    non_idor_count = total - idor_count

    # Generate summary
    summary = f"""
BAC-ML-1M DATASET PREPARATION REPORT
===================================

Source: {SOURCE_FILE}
Output: {OUTPUT_FILE}

Total original records: {before}
Records removed for missing values: {removed}
Prepared records: {total}

IDOR records: {idor_count}
Non-IDOR records: {non_idor_count}

IDOR percentage: {(idor_count / total * 100) if total else 0:.2f}%
Non-IDOR percentage: {(non_idor_count / total * 100) if total else 0:.2f}%

Features:
{', '.join(FEATURE_COLUMNS)}

Target:
idor_label
0 = No
1 = Yes (IDOR)

IMPORTANT:
This dataset represents HTTP access-control records.
It must be used in a separate HTTP access-control ML pipeline.
The labels are inherited from the original dataset and have not
been independently verified.
"""

    SUMMARY_FILE.write_text(summary, encoding="utf-8")

    print("\nDATASET PREPARATION COMPLETED")
    print("-" * 60)
    print("Prepared records:", total)
    print("IDOR records:", idor_count)
    print("Non-IDOR records:", non_idor_count)
    print("Removed missing records:", removed)

    print("\nSaved files:")
    print(OUTPUT_FILE)
    print(SUMMARY_FILE)

    print("\nOriginal dataset remains unchanged.")
    print("Existing binary model remains unchanged.")

    print("=" * 60)


if __name__ == "__main__":
    main()
