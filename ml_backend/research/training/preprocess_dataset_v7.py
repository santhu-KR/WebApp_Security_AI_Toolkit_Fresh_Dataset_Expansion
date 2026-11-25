import os
import re
import hashlib
import pandas as pd

# ============================================================
# V7 SAFE DATASET PREPROCESSING
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "processed",
    "security_snippets_clean.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "v7"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "security_dataset_v7.csv"
)

REPORT_FILE = os.path.join(
    OUTPUT_DIR,
    "v7_preprocessing_report.txt"
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("=" * 70)
print("V7 DATASET PREPROCESSING")
print("=" * 70)

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Dataset not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print("\nOriginal Dataset Loaded")
print(f"Total Samples: {len(df)}")
print(f"Columns: {list(df.columns)}")


# ============================================================
# 2. REQUIRED COLUMN CHECK
# ============================================================

required_columns = [
    "source",
    "label",
    "category_original",
    "category_normalized",
    "source_id",
    "cwe",
    "code"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# 3. PRESERVE ORIGINAL CODE
# ============================================================

df["code_original"] = df["code"].astype(str)


# ============================================================
# 4. SAFE CODE NORMALIZATION
# ============================================================

def normalize_code(code):

    if pd.isna(code):
        return ""

    code = str(code)

    # Normalize line endings
    code = code.replace("\r\n", "\n")
    code = code.replace("\r", "\n")

    # Remove trailing spaces from lines
    lines = [
        line.rstrip()
        for line in code.split("\n")
    ]

    code = "\n".join(lines)

    # Remove excessive blank lines
    code = re.sub(
        r"\n{3,}",
        "\n\n",
        code
    )

    # Remove leading and trailing whitespace
    code = code.strip()

    return code


df["code_normalized"] = df["code_original"].apply(
    normalize_code
)


# ============================================================
# 5. BASIC DATA QUALITY ANALYSIS
# ============================================================

df["code_length"] = df["code_normalized"].str.len()

df["word_count"] = df["code_normalized"].apply(
    lambda x: len(x.split())
)

df["line_count"] = df["code_normalized"].apply(
    lambda x: len(x.splitlines())
)


# ============================================================
# 6. IDENTIFY EMPTY SAMPLES
# ============================================================

df["is_empty"] = (
    df["code_normalized"].str.strip() == ""
)

empty_samples = df[df["is_empty"]].copy()


# ============================================================
# 7. IDENTIFY VERY SHORT SAMPLES
# ============================================================

df["is_short_sample"] = (
    df["code_length"] < 30
)

short_samples = df[
    df["is_short_sample"]
].copy()


# ============================================================
# 8. IDENTIFY EXTREMELY SHORT SAMPLES
# ============================================================

df["is_extremely_short"] = (
    df["code_length"] < 10
)

extremely_short_samples = df[
    df["is_extremely_short"]
].copy()


# IMPORTANT:
# Short samples are NOT automatically deleted.
# They are preserved for manual inspection.


# ============================================================
# 9. DUPLICATE ANALYSIS
# ============================================================

def calculate_fingerprint(code):

    normalized = normalize_code(code)

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


df["normalized_fingerprint"] = df[
    "code_normalized"
].apply(calculate_fingerprint)


# Find normalized duplicates
normalized_duplicates = df[
    df.duplicated(
        subset=["normalized_fingerprint"],
        keep=False
    )
].copy()


# ============================================================
# 10. CHECK DUPLICATES WITH CONFLICTING LABELS
# ============================================================

duplicate_label_conflicts = []

for fingerprint, group in df.groupby(
    "normalized_fingerprint"
):

    if len(group) > 1:

        unique_labels = group["label"].unique()

        if len(unique_labels) > 1:

            duplicate_label_conflicts.append(
                {
                    "fingerprint": fingerprint,
                    "labels": list(unique_labels),
                    "count": len(group)
                }
            )


conflicting_duplicates = pd.DataFrame(
    duplicate_label_conflicts
)


# ============================================================
# 11. SOURCE INFORMATION
# ============================================================

source_distribution = df[
    "source"
].value_counts().reset_index()

source_distribution.columns = [
    "source",
    "sample_count"
]


# ============================================================
# 12. LABEL DISTRIBUTION
# ============================================================

label_distribution = df[
    "label"
].value_counts().reset_index()

label_distribution.columns = [
    "label",
    "sample_count"
]


# ============================================================
# 13. SOURCE AND LABEL DISTRIBUTION
# ============================================================

source_label_distribution = pd.crosstab(
    df["source"],
    df["label"]
)


# ============================================================
# 14. SOURCE AND CODE LENGTH ANALYSIS
# ============================================================

source_length_analysis = df.groupby(
    "source"
).agg(
    sample_count=("code", "count"),
    average_code_length=("code_length", "mean"),
    median_code_length=("code_length", "median"),
    average_word_count=("word_count", "mean"),
    average_line_count=("line_count", "mean"),
    short_samples=("is_short_sample", "sum"),
    extremely_short_samples=("is_extremely_short", "sum")
).reset_index()


# ============================================================
# 15. CREATE QUALITY FLAGS
# ============================================================

df["quality_flag"] = "NORMAL"

df.loc[
    df["is_short_sample"],
    "quality_flag"
] = "SHORT_SAMPLE"

df.loc[
    df["is_extremely_short"],
    "quality_flag"
] = "EXTREMELY_SHORT_SAMPLE"

df.loc[
    df["is_empty"],
    "quality_flag"
] = "EMPTY_SAMPLE"


# ============================================================
# 16. FINAL V7 DATASET
# ============================================================

# Preserve all samples.
# Do not remove samples automatically.

final_columns = [
    "source",
    "source_row",
    "label",
    "category_original",
    "category_normalized",
    "source_id",
    "cwe",
    "code_original",
    "code_normalized",
    "code_length",
    "word_count",
    "line_count",
    "quality_flag",
    "normalized_fingerprint"
]

available_columns = [
    column
    for column in final_columns
    if column in df.columns
]

v7_dataset = df[
    available_columns
].copy()


# ============================================================
# 17. SAVE V7 DATASET
# ============================================================

v7_dataset.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 18. SAVE ANALYSIS FILES
# ============================================================

source_distribution.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "source_distribution_v7.csv"
    ),
    index=False
)

label_distribution.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "label_distribution_v7.csv"
    ),
    index=False
)

source_label_distribution.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "source_label_distribution_v7.csv"
    )
)

source_length_analysis.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "source_code_length_analysis_v7.csv"
    ),
    index=False
)

short_samples.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "short_samples_v7.csv"
    ),
    index=False
)

extremely_short_samples.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "extremely_short_samples_v7.csv"
    ),
    index=False
)

empty_samples.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "empty_samples_v7.csv"
    ),
    index=False
)

normalized_duplicates.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "normalized_duplicates_v7.csv"
    ),
    index=False
)

conflicting_duplicates.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "duplicate_label_conflicts_v7.csv"
    ),
    index=False
)


# ============================================================
# 19. GENERATE REPORT
# ============================================================

report = []

report.append("=" * 70)
report.append("V7 DATASET PREPROCESSING REPORT")
report.append("=" * 70)

report.append(
    f"\nOriginal Samples: {len(df)}"
)

report.append(
    f"Final V7 Samples: {len(v7_dataset)}"
)

report.append(
    f"Empty Samples: {len(empty_samples)}"
)

report.append(
    f"Short Samples (<30 characters): {len(short_samples)}"
)

report.append(
    f"Extremely Short Samples (<10 characters): "
    f"{len(extremely_short_samples)}"
)

report.append(
    f"Normalized Duplicate Rows: {len(normalized_duplicates)}"
)

report.append(
    f"Conflicting Duplicate Groups: "
    f"{len(conflicting_duplicates)}"
)


report.append("\nSOURCE DISTRIBUTION")
report.append("-" * 50)

for _, row in source_distribution.iterrows():

    percentage = (
        row["sample_count"] / len(df)
    ) * 100

    report.append(
        f"{row['source']}: "
        f"{row['sample_count']} "
        f"({percentage:.2f}%)"
    )


report.append("\nLABEL DISTRIBUTION")
report.append("-" * 50)

for _, row in label_distribution.iterrows():

    percentage = (
        row["sample_count"] / len(df)
    ) * 100

    report.append(
        f"{row['label']}: "
        f"{row['sample_count']} "
        f"({percentage:.2f}%)"
    )


report.append("\nSOURCE CODE LENGTH")
report.append("-" * 50)

for _, row in source_length_analysis.iterrows():

    report.append(
        f"\nSource: {row['source']}"
    )

    report.append(
        f"Samples: {row['sample_count']}"
    )

    report.append(
        f"Average Code Length: "
        f"{row['average_code_length']:.2f}"
    )

    report.append(
        f"Median Code Length: "
        f"{row['median_code_length']:.2f}"
    )

    report.append(
        f"Average Words: "
        f"{row['average_word_count']:.2f}"
    )

    report.append(
        f"Average Lines: "
        f"{row['average_line_count']:.2f}"
    )


report.append("\nPREPROCESSING ACTIONS")
report.append("-" * 50)

report.append(
    "1. Original code was preserved."
)

report.append(
    "2. Normalized code was created separately."
)

report.append(
    "3. Line endings were standardized."
)

report.append(
    "4. Trailing whitespace was removed."
)

report.append(
    "5. Excessive blank lines were reduced."
)

report.append(
    "6. Empty samples were identified."
)

report.append(
    "7. Short samples were flagged, not deleted."
)

report.append(
    "8. Extremely short samples were flagged."
)

report.append(
    "9. Normalized duplicates were identified."
)

report.append(
    "10. Conflicting duplicate labels were checked."
)

report.append(
    "11. Source and CWE information were preserved."
)

report.append(
    "12. No samples were automatically deleted."
)


report.append("\nIMPORTANT")
report.append("-" * 50)

report.append(
    "This preprocessing stage does not train a machine-learning model."
)

report.append(
    "The purpose is to prepare a quality-controlled V7 dataset "
    "for later model training and evaluation."
)

report.append(
    "Short code samples are preserved because some security "
    "patterns may be represented in very small code snippets."
)


with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "\n".join(report)
    )


# ============================================================
# 20. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("V7 PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 70)

print(f"\nFinal Dataset:")
print(OUTPUT_FILE)

print(f"\nReport:")
print(REPORT_FILE)

print("\nSamples:")
print(f"Original: {len(df)}")
print(f"Final V7: {len(v7_dataset)}")

print("\nQuality Analysis:")
print(f"Empty: {len(empty_samples)}")
print(f"Short (<30 chars): {len(short_samples)}")
print(f"Extremely short (<10 chars): {len(extremely_short_samples)}")
print(f"Normalized duplicate rows: {len(normalized_duplicates)}")
print(f"Conflicting duplicate groups: {len(conflicting_duplicates)}")

print("\nV7 dataset is ready for the next stage.")