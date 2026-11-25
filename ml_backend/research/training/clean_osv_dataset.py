import pandas as pd
import re
from pathlib import Path
import json

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).parent

ANALYSIS_DIR = (
    BASE_DIR
    / "dataset"
    / "external_sources"
    / "osv_collector"
    / "analysis"
)

INPUT_FILE = ANALYSIS_DIR / "osv_merged_dataset.csv"

OUTPUT_FILE = ANALYSIS_DIR / "osv_ml_cleaned_dataset.csv"

REPORT_FILE = ANALYSIS_DIR / "osv_ml_cleaning_report.json"

UNCATEGORIZED_FILE = ANALYSIS_DIR / "osv_uncategorized_for_review.csv"


# ============================================================
# CATEGORY NORMALIZATION
# ============================================================

CATEGORY_MAPPING = {

    "Denial of Service (DoS)": "DoS",
    "Denial of Service": "DoS",
    "DOS": "DoS",
    "DoS": "DoS",

    "Cross-Site Scripting": "XSS",
    "Cross Site Scripting": "XSS",

    "SQL Injection": "SQL Injection",

    "Insecure Deserialization": "Insecure Deserialization",

    "Memory Safety": "Memory Safety",

    "Path Traversal / File Handling":
        "Path Traversal / File Handling",

    "Input Validation / Data Handling":
        "Input Validation / Data Handling",

    "Information Disclosure":
        "Information Disclosure",

    "Code / Template Injection":
        "Code / Template Injection",

    "Authentication / Session":
        "Authentication / Session",

    "Broken Access Control":
        "Broken Access Control",

    "SSRF / External Resource Access":
        "SSRF / External Resource Access",

    "Prototype Pollution":
        "Prototype Pollution",

    "HTTP / Protocol Handling":
        "HTTP / Protocol Handling",

    "CSRF / Request Forgery":
        "CSRF / Request Forgery",

    "Security Misconfiguration":
        "Security Misconfiguration",

    "Resource / Logic Errors":
        "Resource / Logic Errors",

    "Open Redirect":
        "Open Redirect",

    "Command Injection":
        "Command Injection",

    "Cryptographic Issues":
        "Cryptographic Issues",

    "XML / XPath / LDAP Injection":
        "XML / XPath / LDAP Injection",

    "Race Condition":
        "Race Condition",

    "Other":
        "Other"
}


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove unwanted special characters
    text = re.sub(
        r"[^a-z0-9\s\-_]",
        " ",
        text
    )

    # Remove repeated spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# MAIN PROCESS
# ============================================================

print("=" * 70)
print("OSV MACHINE LEARNING DATASET CLEANING")
print("=" * 70)


# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

if not INPUT_FILE.exists():

    print("\nERROR: Dataset not found!")

    print(INPUT_FILE)

    exit()


df = pd.read_csv(INPUT_FILE)

print("\nDataset loaded successfully")

print("Original records:", len(df))


# ------------------------------------------------------------
# 2. CHECK REQUIRED COLUMNS
# ------------------------------------------------------------

required_columns = [
    "osv_id",
    "summary",
    "normalized_category"
]

for column in required_columns:

    if column not in df.columns:

        print(f"\nERROR: Missing column: {column}")

        print("Available columns:")

        print(list(df.columns))

        exit()


# ------------------------------------------------------------
# 3. REMOVE DUPLICATE VULNERABILITY RECORDS
# ------------------------------------------------------------

duplicate_count = df.duplicated(
    subset=["osv_id"]
).sum()

df = df.drop_duplicates(
    subset=["osv_id"],
    keep="first"
)

print("\nDuplicate records removed:", duplicate_count)


# ------------------------------------------------------------
# 4. STANDARDIZE CATEGORY LABELS
# ------------------------------------------------------------

df["normalized_category"] = (
    df["normalized_category"]
    .fillna("Uncategorized")
    .astype(str)
    .str.strip()
)

df["normalized_category"] = df[
    "normalized_category"
].replace(
    CATEGORY_MAPPING
)

print("\nCategory labels standardized")


# ------------------------------------------------------------
# 5. COMBINE SUMMARY AND DESCRIPTION
# ------------------------------------------------------------

summary_column = "summary"

description_column = "recovered_description"

if description_column not in df.columns:

    description_column = "description"

    if description_column not in df.columns:

        description_column = None


if description_column:

    df["raw_text"] = (
        df[summary_column].fillna("").astype(str)
        + " "
        + df[description_column].fillna("").astype(str)
    )

else:

    df["raw_text"] = (
        df[summary_column].fillna("").astype(str)
    )


# ------------------------------------------------------------
# 6. CLEAN TEXT
# ------------------------------------------------------------

df["cleaned_text"] = df["raw_text"].apply(clean_text)

print("Text cleaning completed")


# ------------------------------------------------------------
# 7. REMOVE EMPTY TEXT RECORDS
# ------------------------------------------------------------

empty_text_count = (
    df["cleaned_text"]
    .str.strip()
    .eq("")
    .sum()
)

df = df[
    df["cleaned_text"].str.strip() != ""
].copy()

print("Empty text records removed:", empty_text_count)


# ------------------------------------------------------------
# 8. SEPARATE UNCATEGORIZED RECORDS
# ------------------------------------------------------------

uncategorized_mask = (
    df["normalized_category"]
    .str.lower()
    .isin([
        "uncategorized",
        "unknown",
        "none",
        "",
        "nan"
    ])
)

uncategorized_df = df[
    uncategorized_mask
].copy()

ml_df = df[
    ~uncategorized_mask
].copy()


# Save uncategorized records separately

uncategorized_df.to_csv(
    UNCATEGORIZED_FILE,
    index=False
)


print("\nUncategorized records:", len(uncategorized_df))


# ------------------------------------------------------------
# 9. REMOVE VERY SMALL TEXT RECORDS
# ------------------------------------------------------------

minimum_words = 3

short_text_mask = (
    ml_df["cleaned_text"]
    .str.split()
    .str.len()
    < minimum_words
)

short_text_count = short_text_mask.sum()

ml_df = ml_df[
    ~short_text_mask
].copy()

print("Very short text records removed:", short_text_count)


# ------------------------------------------------------------
# 10. CHECK CATEGORY DISTRIBUTION
# ------------------------------------------------------------

category_distribution = (
    ml_df["normalized_category"]
    .value_counts()
    .to_dict()
)


# ------------------------------------------------------------
# 11. SAVE CLEANED ML DATASET
# ------------------------------------------------------------

ml_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)


# ------------------------------------------------------------
# 12. GENERATE REPORT
# ------------------------------------------------------------

report = {

    "dataset": "OSV Machine Learning Dataset",

    "original_records": int(
        len(pd.read_csv(INPUT_FILE))
    ),

    "duplicate_records_removed": int(
        duplicate_count
    ),

    "empty_text_records_removed": int(
        empty_text_count
    ),

    "short_text_records_removed": int(
        short_text_count
    ),

    "uncategorized_records_separated": int(
        len(uncategorized_df)
    ),

    "final_ml_records": int(
        len(ml_df)
    ),

    "number_of_categories": int(
        ml_df["normalized_category"].nunique()
    ),

    "category_distribution": category_distribution,

    "output_dataset": str(
        OUTPUT_FILE
    ),

    "uncategorized_dataset": str(
        UNCATEGORIZED_FILE
    )

}


with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        report,
        file,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("DATASET CLEANING COMPLETED")
print("=" * 70)

print("\nOriginal records:", report["original_records"])

print("Duplicates removed:", duplicate_count)

print("Empty text removed:", empty_text_count)

print("Short text removed:", short_text_count)

print("Uncategorized separated:", len(uncategorized_df))

print("\nFinal ML dataset:", len(ml_df))

print("Number of categories:", ml_df["normalized_category"].nunique())


print("\n" + "=" * 70)
print("FINAL CATEGORY DISTRIBUTION")
print("=" * 70)


for category, count in category_distribution.items():

    print(f"{category}: {count}")


print("\n" + "=" * 70)
print("FILES CREATED")
print("=" * 70)

print("\nCleaned ML Dataset:")
print(OUTPUT_FILE)

print("\nUncategorized Records:")
print(UNCATEGORIZED_FILE)

print("\nCleaning Report:")
print(REPORT_FILE)


print("\nDataset is ready for ML feature extraction!")