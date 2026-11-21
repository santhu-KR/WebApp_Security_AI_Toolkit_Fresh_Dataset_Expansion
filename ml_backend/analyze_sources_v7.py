import pandas as pd
import numpy as np
from pathlib import Path
import re

# =========================================================
# V7 SOURCE ANALYSIS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "dataset" / "processed" / "security_snippets_clean.csv"

OUTPUT_DIR = BASE_DIR / "results" / "v7" / "source_analysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("V7 DATASET SOURCE ANALYSIS")
print("=" * 70)

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(DATASET_PATH)

print("\nDataset loaded successfully!")

print("Total samples:", len(df))
print("Columns:", list(df.columns))

# =========================================================
# 1. SOURCE DISTRIBUTION
# =========================================================

print("\n" + "=" * 70)
print("1. SOURCE DISTRIBUTION")
print("=" * 70)

source_summary = (
    df.groupby("source")
    .agg(
        Total_Samples=("label", "count"),
        Safe_Samples=("label", lambda x: (x == "Safe").sum()),
        Vulnerable_Samples=("label", lambda x: (x == "Vulnerable").sum())
    )
)

source_summary["Safe_Percentage"] = (
    source_summary["Safe_Samples"] /
    source_summary["Total_Samples"] * 100
)

source_summary["Vulnerable_Percentage"] = (
    source_summary["Vulnerable_Samples"] /
    source_summary["Total_Samples"] * 100
)

print(source_summary)

source_summary.to_csv(
    OUTPUT_DIR / "source_summary.csv"
)

# =========================================================
# 2. SOURCE AND LABEL DISTRIBUTION
# =========================================================

print("\n" + "=" * 70)
print("2. SOURCE VS LABEL DISTRIBUTION")
print("=" * 70)

source_label_table = pd.crosstab(
    df["source"],
    df["label"],
    normalize="index"
) * 100

print(source_label_table.round(2))

source_label_table.to_csv(
    OUTPUT_DIR / "source_label_distribution.csv"
)

# =========================================================
# 3. CWE DISTRIBUTION PER SOURCE
# =========================================================

print("\n" + "=" * 70)
print("3. CWE DISTRIBUTION PER SOURCE")
print("=" * 70)

cwe_distribution = pd.crosstab(
    df["source"],
    df["cwe"]
)

print(cwe_distribution)

cwe_distribution.to_csv(
    OUTPUT_DIR / "source_cwe_distribution.csv"
)

# =========================================================
# 4. SOURCE-SPECIFIC CWE PERCENTAGE
# =========================================================

cwe_percentage = pd.crosstab(
    df["source"],
    df["cwe"],
    normalize="index"
) * 100

cwe_percentage = cwe_percentage.round(2)

cwe_percentage.to_csv(
    OUTPUT_DIR / "source_cwe_percentage.csv"
)

# =========================================================
# 5. CODE LENGTH ANALYSIS
# =========================================================

print("\n" + "=" * 70)
print("5. CODE LENGTH ANALYSIS")
print("=" * 70)

df["code"] = df["code"].fillna("").astype(str)

df["code_length"] = df["code"].str.len()

df["line_count"] = df["code"].apply(
    lambda x: len(x.splitlines())
)

code_length_summary = (
    df.groupby("source")
    .agg(
        Average_Code_Length=("code_length", "mean"),
        Median_Code_Length=("code_length", "median"),
        Minimum_Code_Length=("code_length", "min"),
        Maximum_Code_Length=("code_length", "max"),
        Average_Line_Count=("line_count", "mean"),
        Median_Line_Count=("line_count", "median")
    )
)

print(code_length_summary.round(2))

code_length_summary.to_csv(
    OUTPUT_DIR / "code_length_summary.csv"
)

# =========================================================
# 6. SECURITY KEYWORD ANALYSIS
# =========================================================

keywords = [
    "sql",
    "execute",
    "query",
    "statement",
    "preparedstatement",
    "password",
    "secret",
    "token",
    "cookie",
    "session",
    "http",
    "https",
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

print("\n" + "=" * 70)
print("6. SECURITY KEYWORD ANALYSIS")
print("=" * 70)

keyword_results = []

for source in df["source"].unique():

    source_df = df[df["source"] == source]

    source_text = " ".join(
        source_df["code"].astype(str).tolist()
    ).lower()

    result = {
        "source": source,
        "total_samples": len(source_df)
    }

    for keyword in keywords:

        count = source_text.count(keyword)

        result[keyword] = count

    keyword_results.append(result)

keyword_df = pd.DataFrame(keyword_results)

print(keyword_df.to_string(index=False))

keyword_df.to_csv(
    OUTPUT_DIR / "source_keyword_analysis.csv",
    index=False
)

# =========================================================
# 7. KEYWORD FREQUENCY PER SAMPLE
# =========================================================

print("\n" + "=" * 70)
print("7. KEYWORD FREQUENCY PER SAMPLE")
print("=" * 70)

for keyword in keywords:

    pattern = re.escape(keyword)

    df[f"contains_{keyword}"] = df["code"].str.lower().str.contains(
        pattern,
        regex=True,
        na=False
    )

keyword_sample_summary = []

for source in df["source"].unique():

    source_df = df[df["source"] == source]

    result = {
        "source": source,
        "total_samples": len(source_df)
    }

    for keyword in keywords:

        column = f"contains_{keyword}"

        result[keyword] = (
            source_df[column].sum()
        )

        result[f"{keyword}_percentage"] = (
            source_df[column].mean() * 100
        )

    keyword_sample_summary.append(result)

keyword_sample_df = pd.DataFrame(
    keyword_sample_summary
)

keyword_sample_df.to_csv(
    OUTPUT_DIR / "source_keyword_sample_frequency.csv",
    index=False
)

# =========================================================
# 8. SAFE/VULNERABLE CODE LENGTH
# =========================================================

print("\n" + "=" * 70)
print("8. SAFE VS VULNERABLE CODE LENGTH")
print("=" * 70)

label_length_summary = (
    df.groupby(["source", "label"])
    .agg(
        Samples=("label", "count"),
        Average_Length=("code_length", "mean"),
        Median_Length=("code_length", "median"),
        Average_Lines=("line_count", "mean")
    )
)

print(label_length_summary.round(2))

label_length_summary.to_csv(
    OUTPUT_DIR / "source_label_code_length.csv"
)

# =========================================================
# 9. SOURCE-SPECIFIC SAMPLE EXAMPLES
# =========================================================

print("\n" + "=" * 70)
print("9. SOURCE SAMPLE EXAMPLES")
print("=" * 70)

for source in df["source"].unique():

    print("\nSOURCE:", source)

    source_df = df[df["source"] == source]

    print("\nSAFE EXAMPLE:")

    safe_examples = source_df[
        source_df["label"] == "Safe"
    ]

    if len(safe_examples) > 0:

        print(
            safe_examples.iloc[0]["code"][:500]
        )

    print("\nVULNERABLE EXAMPLE:")

    vulnerable_examples = source_df[
        source_df["label"] == "Vulnerable"
    ]

    if len(vulnerable_examples) > 0:

        print(
            vulnerable_examples.iloc[0]["code"][:500]
        )

# =========================================================
# 10. SOURCE ANALYSIS REPORT
# =========================================================

report_path = OUTPUT_DIR / "v7_source_analysis_report.txt"

with open(report_path, "w", encoding="utf-8") as report:

    report.write("=" * 70 + "\n")
    report.write("V7 DATASET SOURCE ANALYSIS REPORT\n")
    report.write("=" * 70 + "\n\n")

    report.write(f"Total Samples: {len(df)}\n\n")

    report.write("SOURCE SUMMARY\n")
    report.write("-" * 70 + "\n")

    report.write(
        source_summary.to_string()
    )

    report.write("\n\nSOURCE LABEL DISTRIBUTION\n")
    report.write("-" * 70 + "\n")

    report.write(
        source_label_table.round(2).to_string()
    )

    report.write("\n\nCODE LENGTH SUMMARY\n")
    report.write("-" * 70 + "\n")

    report.write(
        code_length_summary.round(2).to_string()
    )

    report.write("\n\nLABEL CODE LENGTH SUMMARY\n")
    report.write("-" * 70 + "\n")

    report.write(
        label_length_summary.round(2).to_string()
    )

    report.write("\n\nCWE DISTRIBUTION\n")
    report.write("-" * 70 + "\n")

    report.write(
        cwe_distribution.to_string()
    )

print("\n" + "=" * 70)
print("ANALYSIS COMPLETED")
print("=" * 70)

print("\nOutput files saved in:")

print(OUTPUT_DIR)

print("\nGenerated files:")

for file in OUTPUT_DIR.iterdir():

    print("-", file.name)

print("\nNext step:")
print("Analyze the source differences and design V7.")