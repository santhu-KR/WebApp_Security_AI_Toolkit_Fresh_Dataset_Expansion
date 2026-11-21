import json
from datetime import datetime
from pathlib import Path

import pandas as pd


# ============================================================
# 1. PATH CONFIGURATION
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
    / "evaluation"
    / "multiclass_audit"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. TARGET VULNERABILITIES (24 CATEGORIES)
# ============================================================

TARGET_CATEGORIES = [
    # Original 18 categories
    "SQL Injection",
    "Cross-Site Scripting (XSS)",
    "Cross-Site Request Forgery (CSRF)",
    "Broken Authentication",
    "Broken Access Control",
    "Path Traversal",
    "Command Injection",
    "Insecure Deserialization",
    "Sensitive Data Exposure",
    "Security Misconfiguration",
    "IDOR",
    "Server-Side Request Forgery (SSRF)",
    "Open Redirect",
    "Weak Password Storage",
    "Insecure TLS Configuration",
    "Insecure Cookie Configuration",
    "XML External Entity (XXE)",
    "LDAP Injection",

    # Six additional categories
    "Server-Side Template Injection (SSTI)",
    "XPath Injection",
    "Weak Cryptography",
    "Weak Hashing",
    "Weak Randomness",
    "Trust Boundary Violation",
]


# ============================================================
# 3. CATEGORY MAPPING
# ============================================================
# Only map categories whose source labels have a clear meaning.
# Do not map broad or ambiguous labels to a specific vulnerability.

CATEGORY_MAP = {
    # Original categories
    "SQL Injection": "SQL Injection",
    "Cross-Site Scripting (XSS)": "Cross-Site Scripting (XSS)",
    "Cross-Site Request Forgery (CSRF)":
        "Cross-Site Request Forgery (CSRF)",
    "Path Traversal": "Path Traversal",
    "Command Injection": "Command Injection",
    "Insecure Deserialization": "Insecure Deserialization",
    "Hardcoded Secrets": "Sensitive Data Exposure",
    "Security Misconfiguration / Debug": "Security Misconfiguration",
    "Server-Side Request Forgery (SSRF)":
        "Server-Side Request Forgery (SSRF)",
    "Open Redirect": "Open Redirect",
    "Weak Password Storage": "Weak Password Storage",
    "Insecure TLS Configuration": "Insecure TLS Configuration",
    "Insecure Cookie": "Insecure Cookie Configuration",
    "XML External Entity (XXE)": "XML External Entity (XXE)",
    "LDAP Injection": "LDAP Injection",

    # Additional categories
    "Server-Side Template Injection":
        "Server-Side Template Injection (SSTI)",
    "Server-Side Template Injection (SSTI)":
        "Server-Side Template Injection (SSTI)",
    "XPath Injection": "XPath Injection",
    "Weak Cryptography": "Weak Cryptography",
    "Weak Hashing": "Weak Hashing",
    "Weak Randomness": "Weak Randomness",
    "Trust Boundary Violation": "Trust Boundary Violation",
}


# Ambiguous categories are kept separate for manual review.
MANUAL_REVIEW_CATEGORIES = [
    "Broken Access Control / IDOR",
    "JWT Misconfiguration",
    "CORS Misconfiguration",
    "Uncategorized",
]


# Generic labels are not treated as vulnerability classes.
GENERIC_CATEGORIES = [
    "",
    "Uncategorized",
    "Unknown",
    "Generic",
]


# Minimum count used only to describe dataset coverage.
# It does not determine whether a sample is truly valid.
LOW_SAMPLE_THRESHOLD = 20


# ============================================================
# 4. LOAD DATASET
# ============================================================

print("\n======================================")
print(" 24-CLASS MULTICLASS DATASET AUDIT")
print("======================================")

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_PATH}\n"
        "Check that this script is located in the ml_backend folder "
        "and that the processed CSV exists."
    )

df = pd.read_csv(DATASET_PATH)

required_columns = [
    "source",
    "label",
    "category_original",
    "category_normalized",
    "code",
    "fingerprint",
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required dataset columns: {missing_columns}"
    )

print("\nDataset loaded successfully.")
print("Dataset file:", DATASET_PATH)


# ============================================================
# 5. DATASET STATISTICS
# ============================================================

# Normalize labels for reliable comparisons.
df["label"] = (
    df["label"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.lower()
)

total_records = len(df)
safe_records = int((df["label"] == "safe").sum())
vulnerable_records = int((df["label"] == "vulnerable").sum())
invalid_records = int(
    (~df["label"].isin(["safe", "vulnerable"])).sum()
)

print("\nDATASET STATISTICS")
print("--------------------------------")
print("Total records:", total_records)
print("Safe records:", safe_records)
print("Vulnerable records:", vulnerable_records)
print("Invalid labels:", invalid_records)


# ============================================================
# 6. SOURCE DISTRIBUTION
# ============================================================

print("\nDATASET SOURCE DISTRIBUTION")
print("--------------------------------")

source_counts = (
    df["source"]
    .fillna("Unknown")
    .astype(str)
    .value_counts()
)

print(source_counts.to_string())


# ============================================================
# 7. CATEGORY NORMALIZATION AND MAPPING
# ============================================================

df["category_normalized"] = (
    df["category_normalized"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df["target_category"] = df["category_normalized"].map(CATEGORY_MAP)

df["mapping_status"] = "unmapped"

df.loc[
    df["target_category"].notna(),
    "mapping_status"
] = "mapped"

df.loc[
    df["category_normalized"].isin(MANUAL_REVIEW_CATEGORIES),
    "mapping_status"
] = "manual_review"

df.loc[
    df["category_normalized"].isin(GENERIC_CATEGORIES),
    "mapping_status"
] = "generic"

# Ensure generic/manual-review records cannot accidentally remain mapped.
df.loc[
    df["mapping_status"].isin(["manual_review", "generic"]),
    "target_category"
] = pd.NA


# ============================================================
# 8. VULNERABLE RECORDS ONLY
# ============================================================

vulnerable = df[df["label"] == "vulnerable"].copy()

mapped = vulnerable[
    vulnerable["mapping_status"] == "mapped"
].copy()

unmapped = vulnerable[
    vulnerable["mapping_status"] == "unmapped"
].copy()

manual_review = vulnerable[
    vulnerable["mapping_status"] == "manual_review"
].copy()

generic = vulnerable[
    vulnerable["mapping_status"] == "generic"
].copy()


# ============================================================
# 9. 24-CATEGORY SAMPLE COUNTS
# ============================================================

category_counts = mapped["target_category"].value_counts()

print("\n24 VULNERABILITY CATEGORY COUNTS")
print("--------------------------------")

category_report = []
missing_categories = []
low_sample_categories = []

for category in TARGET_CATEGORIES:
    count = int(category_counts.get(category, 0))

    if count == 0:
        status = "MISSING"
        missing_categories.append(category)
    elif count < LOW_SAMPLE_THRESHOLD:
        status = "LOW SAMPLES"
        low_sample_categories.append(category)
    else:
        status = "AVAILABLE"

    print(f"{category:48} {count:6} {status}")

    category_report.append({
        "category": category,
        "vulnerable_samples": count,
        "status": status,
    })


# ============================================================
# 10. UNMAPPED CATEGORY ANALYSIS
# ============================================================

print("\nUNMAPPED CATEGORY DISTRIBUTION")
print("--------------------------------")

if not unmapped.empty:
    unmapped_counts = (
        unmapped["category_normalized"]
        .value_counts(dropna=False)
    )
    print(unmapped_counts.to_string())
else:
    print("No unmapped samples.")


# ============================================================
# 11. DUPLICATE AND LABEL-CONFLICT DETECTION
# ============================================================

# Missing fingerprints should not all be treated as duplicates.
fingerprints = df["fingerprint"].fillna("").astype(str).str.strip()
valid_fingerprint_mask = fingerprints.ne("")

duplicate_source = df.loc[valid_fingerprint_mask].copy()

duplicates = duplicate_source[
    duplicate_source.duplicated(
        subset=["fingerprint"],
        keep=False,
    )
].copy()

conflict_records = []

for fingerprint, group in duplicates.groupby("fingerprint"):
    labels = group["label"].dropna().unique()

    if len(labels) > 1:
        conflict_records.append({
            "fingerprint": fingerprint,
            "labels": list(labels),
            "sources": list(group["source"].fillna("").astype(str)),
            "categories": list(
                group["category_normalized"].fillna("").astype(str)
            ),
        })

conflicts = pd.DataFrame(
    conflict_records,
    columns=[
        "fingerprint",
        "labels",
        "sources",
        "categories",
    ],
)

print("\nDUPLICATE ANALYSIS")
print("--------------------------------")
print("Duplicate records:", len(duplicates))
print("Conflicting code groups:", len(conflicts))


# ============================================================
# 12. SAVE CSV REPORTS
# ============================================================

pd.DataFrame(category_report).to_csv(
    OUTPUT_DIR / "category_report.csv",
    index=False,
)

mapped.to_csv(
    OUTPUT_DIR / "mapped_vulnerable_samples.csv",
    index=False,
)

unmapped.to_csv(
    OUTPUT_DIR / "unmapped_vulnerable_samples.csv",
    index=False,
)

manual_review.to_csv(
    OUTPUT_DIR / "manual_review_samples.csv",
    index=False,
)

generic.to_csv(
    OUTPUT_DIR / "generic_vulnerable_samples.csv",
    index=False,
)

duplicates.to_csv(
    OUTPUT_DIR / "duplicate_samples.csv",
    index=False,
)

conflicts.to_csv(
    OUTPUT_DIR / "conflicting_samples.csv",
    index=False,
)


# ============================================================
# 13. GENERATE JSON SUMMARY
# ============================================================

summary = {
    "audit_date": datetime.now().isoformat(),
    "target_category_count": len(TARGET_CATEGORIES),
    "low_sample_threshold": LOW_SAMPLE_THRESHOLD,
    "total_records": total_records,
    "safe_records": safe_records,
    "vulnerable_records": vulnerable_records,
    "invalid_records": invalid_records,
    "mapped_vulnerable_records": len(mapped),
    "unmapped_vulnerable_records": len(unmapped),
    "manual_review_records": len(manual_review),
    "generic_vulnerable_records": len(generic),
    "duplicate_records": len(duplicates),
    "conflicting_code_groups": len(conflicts),
    "source_distribution": {
        str(key): int(value)
        for key, value in source_counts.items()
    },
    "category_counts": {
        category: int(category_counts.get(category, 0))
        for category in TARGET_CATEGORIES
    },
    "missing_categories": missing_categories,
    "low_sample_categories": low_sample_categories,
}

with open(
    OUTPUT_DIR / "audit_summary.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(summary, file, indent=4, ensure_ascii=False)


# ============================================================
# 14. COMPLETION
# ============================================================

print("\n======================================")
print(" DATASET AUDIT COMPLETED")
print("======================================")

print("Total records:", total_records)
print("Mapped vulnerable samples:", len(mapped))
print("Unmapped vulnerable samples:", len(unmapped))
print("Manual review samples:", len(manual_review))
print("Generic vulnerable samples:", len(generic))
print("Duplicate records:", len(duplicates))
print("Conflicting groups:", len(conflicts))
print("Missing categories:", len(missing_categories))
print("Low-sample categories:", len(low_sample_categories))

print("\nREPORT LOCATION:")
print(OUTPUT_DIR)

print("\nAudit completed successfully.")
print("NOTE: This script audits and maps labels. It does NOT train an ML model.")
