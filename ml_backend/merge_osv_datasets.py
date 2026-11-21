import pandas as pd
from pathlib import Path
import json

# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).parent

OSV_DIR = BASE_DIR / "dataset" / "external_sources" / "osv_collector"
ANALYSIS_DIR = OSV_DIR / "analysis"

ORIGINAL_FILE = ANALYSIS_DIR / "osv_normalized_dataset.csv"

CLASSIFIED_FILE = OSV_DIR / "osv_uncategorized_classified.csv"

OUTPUT_FILE = ANALYSIS_DIR / "osv_merged_dataset.csv"

REPORT_FILE = ANALYSIS_DIR / "osv_merge_report.json"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_text(value):
    if pd.isna(value):
        return ""

    return str(value).strip()


def is_uncategorized(value):
    value = normalize_text(value).lower()

    return value in [
        "",
        "uncategorized",
        "unknown",
        "none",
        "nan",
        "null"
    ]


# ============================================================
# MAIN PROCESS
# ============================================================

print("=" * 70)
print("OSV DATASET MERGING AND VALIDATION")
print("=" * 70)


# ------------------------------------------------------------
# 1. CHECK FILES
# ------------------------------------------------------------

if not ORIGINAL_FILE.exists():
    print(f"\nERROR: Original dataset not found:")
    print(ORIGINAL_FILE)
    exit()

if not CLASSIFIED_FILE.exists():
    print(f"\nERROR: Classified dataset not found:")
    print(CLASSIFIED_FILE)
    exit()


# ------------------------------------------------------------
# 2. LOAD DATASETS
# ------------------------------------------------------------

print("\nLoading datasets...")

original_df = pd.read_csv(ORIGINAL_FILE)

classified_df = pd.read_csv(CLASSIFIED_FILE)

print(f"Original dataset records: {len(original_df)}")
print(f"Classified dataset records: {len(classified_df)}")


# ------------------------------------------------------------
# 3. VALIDATE REQUIRED COLUMNS
# ------------------------------------------------------------

required_original_columns = [
    "osv_id",
    "normalized_category"
]

required_classified_columns = [
    "osv_id",
    "suggested_category"
]

for column in required_original_columns:

    if column not in original_df.columns:

        print(f"\nERROR: Missing column in original dataset: {column}")

        print("\nAvailable original columns:")
        print(list(original_df.columns))

        exit()


for column in required_classified_columns:

    if column not in classified_df.columns:

        print(f"\nERROR: Missing column in classified dataset: {column}")

        print("\nAvailable classified columns:")
        print(list(classified_df.columns))

        exit()


# ------------------------------------------------------------
# 4. CLEAN ID VALUES
# ------------------------------------------------------------

original_df["osv_id"] = original_df["osv_id"].astype(str).str.strip()

classified_df["osv_id"] = classified_df["osv_id"].astype(str).str.strip()


# ------------------------------------------------------------
# 5. REMOVE DUPLICATES FROM CLASSIFIED DATASET
# ------------------------------------------------------------

classified_duplicates = classified_df.duplicated(
    subset=["osv_id"]
).sum()

classified_df = classified_df.drop_duplicates(
    subset=["osv_id"],
    keep="first"
)

print(f"\nDuplicates in classified dataset removed: {classified_duplicates}")


# ------------------------------------------------------------
# 6. CREATE CLASSIFICATION LOOKUP
# ------------------------------------------------------------

classification_lookup = {}

for _, row in classified_df.iterrows():

    osv_id = normalize_text(row["osv_id"])

    suggested_category = normalize_text(
        row["suggested_category"]
    )

    if not osv_id:
        continue

    if is_uncategorized(suggested_category):
        continue

    classification_lookup[osv_id] = {
        "category": suggested_category,
        "reason": normalize_text(
            row.get("classification_reason", "")
        ),
        "confidence": normalize_text(
            row.get("suggested_confidence", "")
        ),
        "manual_review": normalize_text(
            row.get("manual_review_required", "")
        )
    }


print(f"Valid classifications available: {len(classification_lookup)}")


# ------------------------------------------------------------
# 7. UPDATE ORIGINAL DATASET
# ------------------------------------------------------------

updated_records = 0

updated_ids = []

for index, row in original_df.iterrows():

    osv_id = normalize_text(row["osv_id"])

    current_category = normalize_text(
        row["normalized_category"]
    )

    # Only update records that are currently uncategorized

    if is_uncategorized(current_category):

        if osv_id in classification_lookup:

            new_category = classification_lookup[osv_id]["category"]

            original_df.at[index, "normalized_category"] = new_category

            updated_records += 1

            updated_ids.append(osv_id)


print(f"\nRecords successfully updated: {updated_records}")


# ------------------------------------------------------------
# 8. REMOVE DUPLICATES FROM ORIGINAL DATASET
# ------------------------------------------------------------

original_duplicates = original_df.duplicated(
    subset=["osv_id"]
).sum()

original_df = original_df.drop_duplicates(
    subset=["osv_id"],
    keep="first"
)

print(f"Duplicates removed from original dataset: {original_duplicates}")


# ------------------------------------------------------------
# 9. CALCULATE CATEGORY COVERAGE
# ------------------------------------------------------------

total_records = len(original_df)

categorized_records = original_df[
    ~original_df["normalized_category"].apply(is_uncategorized)
].shape[0]

uncategorized_records = total_records - categorized_records

coverage_percentage = (
    categorized_records / total_records * 100
    if total_records > 0
    else 0
)


# ------------------------------------------------------------
# 10. CATEGORY DISTRIBUTION
# ------------------------------------------------------------

category_distribution = (
    original_df["normalized_category"]
    .fillna("Uncategorized")
    .value_counts()
    .to_dict()
)


# ------------------------------------------------------------
# 11. CLASSIFICATION CONFIDENCE SUMMARY
# ------------------------------------------------------------

confidence_summary = {}

if "suggested_confidence" in classified_df.columns:

    confidence_summary = (
        classified_df["suggested_confidence"]
        .fillna("Unknown")
        .value_counts()
        .to_dict()
    )


# ------------------------------------------------------------
# 12. MANUAL REVIEW SUMMARY
# ------------------------------------------------------------

manual_review_summary = {}

if "manual_review_required" in classified_df.columns:

    manual_review_summary = (
        classified_df["manual_review_required"]
        .fillna("Unknown")
        .value_counts()
        .to_dict()
    )


# ------------------------------------------------------------
# 13. SAVE MERGED DATASET
# ------------------------------------------------------------

original_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)


# ------------------------------------------------------------
# 14. GENERATE REPORT
# ------------------------------------------------------------

report = {

    "dataset_name": "OSV Merged Vulnerability Dataset",

    "original_records": 893,

    "classified_records_available": len(classified_df),

    "records_updated": updated_records,

    "duplicates_removed_from_original": int(original_duplicates),

    "duplicates_removed_from_classified": int(classified_duplicates),

    "total_records_after_merge": int(total_records),

    "categorized_records": int(categorized_records),

    "uncategorized_records": int(uncategorized_records),

    "category_coverage_percentage": round(
        coverage_percentage,
        2
    ),

    "category_distribution": category_distribution,

    "classification_confidence_distribution": confidence_summary,

    "manual_review_distribution": manual_review_summary,

    "output_file": str(OUTPUT_FILE)

}


with open(REPORT_FILE, "w", encoding="utf-8") as file:

    json.dump(
        report,
        file,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("MERGE COMPLETED SUCCESSFULLY")
print("=" * 70)

print(f"\nOriginal records: {len(original_df) + original_duplicates}")

print(f"Classified records: {len(classified_df)}")

print(f"Records updated: {updated_records}")

print(f"Duplicates removed: {original_duplicates}")

print(f"\nTotal records: {total_records}")

print(f"Categorized records: {categorized_records}")

print(f"Uncategorized records: {uncategorized_records}")

print(f"Category coverage: {coverage_percentage:.2f}%")

print("\n" + "=" * 70)

print("CATEGORY DISTRIBUTION")

print("=" * 70)

for category, count in category_distribution.items():

    print(f"{category}: {count}")


print("\n" + "=" * 70)

print("OUTPUT FILES")

print("=" * 70)

print(f"\nMerged dataset:")
print(OUTPUT_FILE)

print(f"\nMerge report:")
print(REPORT_FILE)

print("\nDataset merge and validation completed!")