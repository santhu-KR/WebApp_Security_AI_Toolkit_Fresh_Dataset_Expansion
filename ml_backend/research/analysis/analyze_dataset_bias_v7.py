import pandas as pd
import numpy as np
from pathlib import Path
import hashlib
import re

# ============================================================
# V7 DATASET BIAS ANALYSIS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = (
    BASE_DIR
    / "dataset"
    / "processed"
    / "security_snippets_clean.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "v7"
    / "dataset_bias_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("V7 DATASET BIAS ANALYSIS")
print("=" * 80)

# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv(DATASET_PATH)

print("\nDataset loaded successfully!")

print("Total samples:", len(df))
print("Columns:", list(df.columns))

df["code"] = df["code"].fillna("").astype(str)
df["label"] = df["label"].astype(str).str.lower().str.strip()
df["source"] = df["source"].fillna("UNKNOWN")
df["cwe"] = df["cwe"].fillna("UNKNOWN")

# ============================================================
# 2. DATASET SOURCE DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("2. SOURCE DISTRIBUTION")
print("=" * 80)

source_distribution = df.groupby("source").agg(
    Total_Samples=("label", "count"),
    Safe=("label", lambda x: (x == "safe").sum()),
    Vulnerable=("label", lambda x: (x == "vulnerable").sum())
)

source_distribution["Safe_Percentage"] = (
    source_distribution["Safe"] /
    source_distribution["Total_Samples"] * 100
)

source_distribution["Vulnerable_Percentage"] = (
    source_distribution["Vulnerable"] /
    source_distribution["Total_Samples"] * 100
)

print(source_distribution.round(2))

source_distribution.to_csv(
    OUTPUT_DIR / "source_distribution.csv"
)

# ============================================================
# 3. SOURCE AND LABEL BIAS
# ============================================================

print("\n" + "=" * 80)
print("3. SOURCE AND LABEL BIAS")
print("=" * 80)

source_label = pd.crosstab(
    df["source"],
    df["label"]
)

source_label_percentage = pd.crosstab(
    df["source"],
    df["label"],
    normalize="index"
) * 100

print("\nSample Counts:")
print(source_label)

print("\nPercentages:")
print(source_label_percentage.round(2))

source_label.to_csv(
    OUTPUT_DIR / "source_label_counts.csv"
)

source_label_percentage.to_csv(
    OUTPUT_DIR / "source_label_percentages.csv"
)

# ============================================================
# 4. CODE LENGTH ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("4. CODE LENGTH ANALYSIS")
print("=" * 80)

df["code_length"] = df["code"].str.len()

df["line_count"] = df["code"].apply(
    lambda x: len(x.splitlines())
)

df["word_count"] = df["code"].apply(
    lambda x: len(x.split())
)

length_analysis = df.groupby("source").agg(
    Samples=("code", "count"),
    Mean_Length=("code_length", "mean"),
    Median_Length=("code_length", "median"),
    Min_Length=("code_length", "min"),
    Max_Length=("code_length", "max"),
    Mean_Lines=("line_count", "mean"),
    Median_Lines=("line_count", "median"),
    Mean_Words=("word_count", "mean")
)

print(length_analysis.round(2))

length_analysis.to_csv(
    OUTPUT_DIR / "source_code_length_analysis.csv"
)

# ============================================================
# 5. CODE LENGTH BY LABEL AND SOURCE
# ============================================================

print("\n" + "=" * 80)
print("5. CODE LENGTH BY LABEL")
print("=" * 80)

label_length = df.groupby(
    ["source", "label"]
).agg(
    Samples=("code", "count"),
    Average_Length=("code_length", "mean"),
    Median_Length=("code_length", "median"),
    Average_Lines=("line_count", "mean"),
    Average_Words=("word_count", "mean")
)

print(label_length.round(2))

label_length.to_csv(
    OUTPUT_DIR / "source_label_length_analysis.csv"
)

# ============================================================
# 6. EMPTY AND VERY SHORT CODE SAMPLES
# ============================================================

print("\n" + "=" * 80)
print("6. EMPTY AND VERY SHORT SAMPLES")
print("=" * 80)

empty_samples = df[df["code_length"] == 0]

short_samples = df[df["code_length"] < 30]

very_short_samples = df[df["code_length"] < 100]

short_analysis = df.groupby("source").agg(
    Empty_Samples=("code_length", lambda x: (x == 0).sum()),
    Under_30_Characters=("code_length", lambda x: (x < 30).sum()),
    Under_100_Characters=("code_length", lambda x: (x < 100).sum()),
    Under_500_Characters=("code_length", lambda x: (x < 500).sum())
)

print(short_analysis)

short_analysis.to_csv(
    OUTPUT_DIR / "short_sample_analysis.csv"
)

empty_samples.to_csv(
    OUTPUT_DIR / "empty_samples.csv",
    index=False
)

short_samples.to_csv(
    OUTPUT_DIR / "samples_under_30_characters.csv",
    index=False
)

# ============================================================
# 7. EXACT DUPLICATE CODE
# ============================================================

print("\n" + "=" * 80)
print("7. EXACT DUPLICATE CODE ANALYSIS")
print("=" * 80)

df["code_hash"] = df["code"].apply(
    lambda x: hashlib.sha256(
        x.encode("utf-8")
    ).hexdigest()
)

duplicate_code = df[
    df.duplicated(
        subset=["code_hash"],
        keep=False
    )
]

print("Total duplicate code rows:", len(duplicate_code))

print("\nDuplicates by source:")

if len(duplicate_code) > 0:

    duplicate_source_summary = (
        duplicate_code.groupby("source")
        .size()
    )

    print(duplicate_source_summary)

    duplicate_code.to_csv(
        OUTPUT_DIR / "exact_duplicate_code.csv",
        index=False
    )

else:

    print("No exact duplicate code detected.")

# ============================================================
# 8. SAME CODE WITH DIFFERENT LABELS
# ============================================================

print("\n" + "=" * 80)
print("8. SAME CODE WITH DIFFERENT LABELS")
print("=" * 80)

conflicting_hashes = (
    df.groupby("code_hash")["label"]
    .nunique()
)

conflicting_hashes = conflicting_hashes[
    conflicting_hashes > 1
]

conflicting_codes = df[
    df["code_hash"].isin(
        conflicting_hashes.index
    )
]

print(
    "Code hashes with conflicting labels:",
    len(conflicting_hashes)
)

print(
    "Rows affected:",
    len(conflicting_codes)
)

if len(conflicting_codes) > 0:

    conflicting_codes.to_csv(
        OUTPUT_DIR / "conflicting_labels.csv",
        index=False
    )

# ============================================================
# 9. CROSS-SOURCE EXACT DUPLICATES
# ============================================================

print("\n" + "=" * 80)
print("9. CROSS-SOURCE DUPLICATES")
print("=" * 80)

cross_source_duplicates = (
    df.groupby("code_hash")["source"]
    .nunique()
)

cross_source_duplicates = cross_source_duplicates[
    cross_source_duplicates > 1
]

cross_source_rows = df[
    df["code_hash"].isin(
        cross_source_duplicates.index
    )
]

print(
    "Code appearing across multiple sources:",
    len(cross_source_duplicates)
)

print(
    "Rows involved:",
    len(cross_source_rows)
)

cross_source_rows.to_csv(
    OUTPUT_DIR / "cross_source_duplicates.csv",
    index=False
)

# ============================================================
# 10. NORMALIZED CODE DUPLICATES
# ============================================================

print("\n" + "=" * 80)
print("10. NORMALIZED CODE DUPLICATES")
print("=" * 80)

def normalize_code(code):

    code = str(code).lower()

    # Remove comments
    code = re.sub(
        r"//.*?$",
        "",
        code,
        flags=re.MULTILINE
    )

    code = re.sub(
        r"/\*.*?\*/",
        "",
        code,
        flags=re.DOTALL
    )

    # Remove extra whitespace
    code = re.sub(
        r"\s+",
        "",
        code
    )

    return code


df["normalized_code"] = df["code"].apply(
    normalize_code
)

df["normalized_hash"] = df["normalized_code"].apply(
    lambda x: hashlib.sha256(
        x.encode("utf-8")
    ).hexdigest()
)

normalized_duplicates = df[
    df.duplicated(
        subset=["normalized_hash"],
        keep=False
    )
]

print(
    "Normalized duplicate rows:",
    len(normalized_duplicates)
)

normalized_duplicates.to_csv(
    OUTPUT_DIR / "normalized_duplicates.csv",
    index=False
)

# ============================================================
# 11. CWE DISTRIBUTION BY SOURCE
# ============================================================

print("\n" + "=" * 80)
print("11. CWE DISTRIBUTION")
print("=" * 80)

cwe_counts = pd.crosstab(
    df["source"],
    df["cwe"]
)

cwe_percentage = pd.crosstab(
    df["source"],
    df["cwe"],
    normalize="index"
) * 100

print("\nCWE Counts:")
print(cwe_counts)

print("\nCWE Percentages:")
print(cwe_percentage.round(2))

cwe_counts.to_csv(
    OUTPUT_DIR / "source_cwe_counts.csv"
)

cwe_percentage.to_csv(
    OUTPUT_DIR / "source_cwe_percentages.csv"
)

# ============================================================
# 12. SECURITY KEYWORD ANALYSIS
# ============================================================

keywords = [
    "sql",
    "query",
    "execute",
    "statement",
    "preparedstatement",
    "password",
    "secret",
    "token",
    "cookie",
    "session",
    "request",
    "response",
    "input",
    "output",
    "user",
    "file",
    "path",
    "shell",
    "runtime",
    "process",
    "script",
    "javascript",
    "html",
    "encode",
    "decode",
    "escape",
    "sanitize",
    "validate",
    "redirect",
    "url",
    "xml",
    "xpath",
    "ldap",
    "crypto",
    "cipher",
    "random",
    "hash",
    "socket",
    "deserialize",
    "objectinputstream",
    "access",
    "authorization",
    "authentication",
    "permission"
]

keyword_results = []

for source in df["source"].unique():

    source_df = df[
        df["source"] == source
    ]

    result = {
        "source": source,
        "total_samples": len(source_df)
    }

    for keyword in keywords:

        occurrence_count = source_df[
            "code"
        ].str.lower().str.count(
            re.escape(keyword)
        ).sum()

        sample_count = source_df[
            "code"
        ].str.lower().str.contains(
            re.escape(keyword),
            regex=True,
            na=False
        ).sum()

        result[keyword + "_occurrences"] = occurrence_count

        result[keyword + "_samples"] = sample_count

        result[keyword + "_percentage"] = (
            sample_count / len(source_df) * 100
        )

    keyword_results.append(result)

keyword_df = pd.DataFrame(
    keyword_results
)

keyword_df.to_csv(
    OUTPUT_DIR / "keyword_source_analysis.csv",
    index=False
)

# ============================================================
# 13. KEYWORD DISTRIBUTION BY LABEL
# ============================================================

print("\n" + "=" * 80)
print("13. KEYWORD DISTRIBUTION BY LABEL")
print("=" * 80)

keyword_label_results = []

for source in df["source"].unique():

    for label in ["safe", "vulnerable"]:

        subset = df[
            (df["source"] == source) &
            (df["label"] == label)
        ]

        if len(subset) == 0:
            continue

        result = {
            "source": source,
            "label": label,
            "samples": len(subset)
        }

        for keyword in keywords:

            count = subset[
                "code"
            ].str.lower().str.contains(
                re.escape(keyword),
                regex=True,
                na=False
            ).sum()

            result[keyword] = count

            result[keyword + "_percentage"] = (
                count / len(subset) * 100
            )

        keyword_label_results.append(result)

keyword_label_df = pd.DataFrame(
    keyword_label_results
)

keyword_label_df.to_csv(
    OUTPUT_DIR / "keyword_by_source_and_label.csv",
    index=False
)

# ============================================================
# 14. SAMPLE STRUCTURE ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("14. SAMPLE STRUCTURE ANALYSIS")
print("=" * 80)

structure_features = [
    "class ",
    "public ",
    "private ",
    "static ",
    "void ",
    "return ",
    "if ",
    "for ",
    "while ",
    "try ",
    "catch ",
    "new ",
    "import ",
    "package ",
    "System.out",
    "System.err"
]

structure_results = []

for source in df["source"].unique():

    subset = df[
        df["source"] == source
    ]

    result = {
        "source": source,
        "samples": len(subset)
    }

    for feature in structure_features:

        count = subset[
            "code"
        ].str.contains(
            re.escape(feature),
            regex=True,
            na=False
        ).sum()

        result[feature.strip().replace(" ", "_")] = count

    structure_results.append(result)

structure_df = pd.DataFrame(
    structure_results
)

structure_df.to_csv(
    OUTPUT_DIR / "source_code_structure_analysis.csv",
    index=False
)

print(structure_df.to_string(index=False))

# ============================================================
# 15. SOURCE AND LABEL CODE STRUCTURE
# ============================================================

print("\n" + "=" * 80)
print("15. SOURCE AND LABEL STRUCTURE")
print("=" * 80)

structure_label_results = []

for source in df["source"].unique():

    for label in ["safe", "vulnerable"]:

        subset = df[
            (df["source"] == source) &
            (df["label"] == label)
        ]

        if len(subset) == 0:
            continue

        result = {
            "source": source,
            "label": label,
            "samples": len(subset),
            "mean_length": subset["code_length"].mean(),
            "median_length": subset["code_length"].median(),
            "mean_lines": subset["line_count"].mean(),
            "mean_words": subset["word_count"].mean()
        }

        structure_label_results.append(result)

structure_label_df = pd.DataFrame(
    structure_label_results
)

print(structure_label_df.round(2))

structure_label_df.to_csv(
    OUTPUT_DIR / "source_label_structure.csv",
    index=False
)

# ============================================================
# 16. SOURCE LEAKAGE CHECK
# ============================================================

print("\n" + "=" * 80)
print("16. SOURCE LEAKAGE CHECK")
print("=" * 80)

source_leakage_results = []

for source in df["source"].unique():

    subset = df[
        df["source"] == source
    ]

    source_results = {
        "source": source,
        "samples": len(subset),
        "safe": (subset["label"] == "safe").sum(),
        "vulnerable": (
            subset["label"] == "vulnerable"
        ).sum(),
        "average_length": subset["code_length"].mean(),
        "median_length": subset["code_length"].median(),
        "average_lines": subset["line_count"].mean(),
        "unique_cwe": subset["cwe"].nunique(),
        "unique_category": (
            subset["category_normalized"].nunique()
        )
    }

    source_leakage_results.append(
        source_results
    )

source_leakage_df = pd.DataFrame(
    source_leakage_results
)

print(source_leakage_df.round(2))

source_leakage_df.to_csv(
    OUTPUT_DIR / "source_leakage_analysis.csv",
    index=False
)

# ============================================================
# 17. AUTOMATIC FINDINGS
# ============================================================

print("\n" + "=" * 80)
print("17. AUTOMATIC FINDINGS")
print("=" * 80)

findings = []

# Source size imbalance
max_samples = source_distribution["Total_Samples"].max()
min_samples = source_distribution["Total_Samples"].min()

if min_samples > 0:

    ratio = max_samples / min_samples

    if ratio > 3:

        findings.append(
            f"Source imbalance detected. Largest source is "
            f"{ratio:.2f} times larger than smallest source."
        )

# Code length differences
max_length = length_analysis["Mean_Length"].max()
min_length = length_analysis["Mean_Length"].min()

if min_length > 0:

    length_ratio = max_length / min_length

    if length_ratio > 5:

        findings.append(
            f"Major code length difference detected. "
            f"Average code length ratio is {length_ratio:.2f}."
        )

# Short snippets
if len(short_samples) > 0:

    findings.append(
        f"{len(short_samples)} samples contain fewer than "
        f"30 characters."
    )

# Conflicting labels
if len(conflicting_hashes) > 0:

    findings.append(
        f"{len(conflicting_hashes)} identical code samples "
        f"have conflicting labels."
    )

# Cross-source duplication
if len(cross_source_duplicates) > 0:

    findings.append(
        f"{len(cross_source_duplicates)} code samples appear "
        f"in multiple sources."
    )

# Label imbalance
for source, row in source_distribution.iterrows():

    difference = abs(
        row["Safe_Percentage"] -
        row["Vulnerable_Percentage"]
    )

    if difference > 15:

        findings.append(
            f"Label imbalance detected in source: {source}"
        )

if len(findings) == 0:

    findings.append(
        "No major automatic bias indicators detected."
    )

for finding in findings:

    print("-", finding)

with open(
    OUTPUT_DIR / "automatic_findings.txt",
    "w",
    encoding="utf-8"
) as f:

    for finding in findings:

        f.write(f"- {finding}\n")

# ============================================================
# 18. FINAL REPORT
# ============================================================

report_path = (
    OUTPUT_DIR /
    "v7_dataset_bias_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as report:

    report.write("=" * 80 + "\n")
    report.write("V7 DATASET BIAS ANALYSIS REPORT\n")
    report.write("=" * 80 + "\n\n")

    report.write(
        f"Total Dataset Samples: {len(df)}\n\n"
    )

    report.write("SOURCE DISTRIBUTION\n")
    report.write("-" * 80 + "\n")

    report.write(
        source_distribution.round(2).to_string()
    )

    report.write("\n\nSOURCE LABEL DISTRIBUTION\n")
    report.write("-" * 80 + "\n")

    report.write(
        source_label_percentage.round(2).to_string()
    )

    report.write("\n\nCODE LENGTH ANALYSIS\n")
    report.write("-" * 80 + "\n")

    report.write(
        length_analysis.round(2).to_string()
    )

    report.write("\n\nLABEL CODE LENGTH\n")
    report.write("-" * 80 + "\n")

    report.write(
        label_length.round(2).to_string()
    )

    report.write("\n\nSHORT SAMPLE ANALYSIS\n")
    report.write("-" * 80 + "\n")

    report.write(
        short_analysis.to_string()
    )

    report.write("\n\nCWE DISTRIBUTION\n")
    report.write("-" * 80 + "\n")

    report.write(
        cwe_counts.to_string()
    )

    report.write("\n\nAUTOMATIC FINDINGS\n")
    report.write("-" * 80 + "\n")

    for finding in findings:

        report.write(f"- {finding}\n")

# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 80)
print("V7 DATASET BIAS ANALYSIS COMPLETED")
print("=" * 80)

print("\nOutput folder:")

print(OUTPUT_DIR)

print("\nGenerated files:")

for file in sorted(OUTPUT_DIR.iterdir()):

    print("-", file.name)

print("\nNext step:")
print("Review the generated results before creating V7 preprocessing.")