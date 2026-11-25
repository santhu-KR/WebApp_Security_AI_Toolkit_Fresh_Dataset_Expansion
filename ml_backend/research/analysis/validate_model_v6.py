import os
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from sklearn.model_selection import train_test_split


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "processed",
    "security_snippets_clean.csv"
)

TRAIN_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "v6",
    "train_v6.csv"
)

TEST_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "v6",
    "test_v6.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "v6",
    "validation"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# LOAD DATA
# =========================================================

print("=" * 60)
print("V6 MODEL VALIDATION")
print("=" * 60)

original_df = pd.read_csv(DATASET_PATH)

train_df = pd.read_csv(TRAIN_PATH)

test_df = pd.read_csv(TEST_PATH)


# =========================================================
# 1. DUPLICATE CHECK
# =========================================================

print("\n1. DUPLICATE FINGERPRINT CHECK")

train_fingerprints = set(train_df["fingerprint"])

test_fingerprints = set(test_df["fingerprint"])

overlap = train_fingerprints.intersection(
    test_fingerprints
)

print("Train fingerprints:", len(train_fingerprints))

print("Test fingerprints:", len(test_fingerprints))

print("Shared fingerprints:", len(overlap))

if len(overlap) == 0:

    print("RESULT: No exact fingerprint leakage detected.")

else:

    print("WARNING: Exact duplicates exist between train and test.")


# =========================================================
# 2. SOURCE LEAKAGE CHECK
# =========================================================

print("\n2. SOURCE DISTRIBUTION CHECK")

if "source" in train_df.columns and "source" in test_df.columns:

    train_sources = set(train_df["source"].dropna())

    test_sources = set(test_df["source"].dropna())

    shared_sources = train_sources.intersection(
        test_sources
    )

    print("Training sources:", len(train_sources))

    print("Testing sources:", len(test_sources))

    print("Sources shared between train/test:",
          len(shared_sources))

    print("Sources appearing in both sets:",
          len(shared_sources))

    print("\nShared source names:")

    for source in list(shared_sources)[:20]:

        print(source)

else:

    print("Source column unavailable.")


# =========================================================
# 3. CWE DISTRIBUTION CHECK
# =========================================================

print("\n3. CWE DISTRIBUTION")

if "cwe" in train_df.columns:

    train_cwe = train_df["cwe"].value_counts(
        normalize=True
    )

    test_cwe = test_df["cwe"].value_counts(
        normalize=True
    )

    cwe_comparison = pd.DataFrame({

        "Train_Percentage": train_cwe * 100,

        "Test_Percentage": test_cwe * 100

    }).fillna(0)

    print(cwe_comparison.sort_values(
        "Train_Percentage",
        ascending=False
    ).to_string())


    cwe_comparison.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "cwe_distribution_comparison.csv"
        )
    )


# =========================================================
# 4. LABEL DISTRIBUTION
# =========================================================

print("\n4. LABEL DISTRIBUTION")

print("\nTraining:")

print(train_df["label"].value_counts())

print("\nTesting:")

print(test_df["label"].value_counts())


# =========================================================
# 5. SAMPLE SIMILARITY CHECK
# =========================================================

print("\n5. TEXT SIMILARITY CHECK")

train_texts = set(
    train_df["code"].fillna("").astype(str)
)

test_texts = set(
    test_df["code"].fillna("").astype(str)
)

exact_code_overlap = train_texts.intersection(
    test_texts
)

print("Exact code matches:", len(exact_code_overlap))


# =========================================================
# 6. FINAL SUMMARY
# =========================================================

print("\n" + "=" * 60)

print("VALIDATION SUMMARY")

print("=" * 60)

print("Training samples:", len(train_df))

print("Testing samples:", len(test_df))

print("Exact fingerprint overlap:", len(overlap))

print("Exact code overlap:", len(exact_code_overlap))

print("\nValidation completed.")

print("Results saved to:")

print(OUTPUT_DIR)