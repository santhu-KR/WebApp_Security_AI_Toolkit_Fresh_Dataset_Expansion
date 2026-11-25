import os
import pandas as pd

from sklearn.model_selection import train_test_split

# ============================================================
# V7 SOURCE-AWARE DATASET BALANCING
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "v7",
    "security_dataset_v7.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "v7",
    "balanced"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# OUTPUT FILES
# ============================================================

TRAIN_FILE = os.path.join(
    OUTPUT_DIR,
    "train_v7_balanced.csv"
)

TEST_FILE = os.path.join(
    OUTPUT_DIR,
    "test_v7.csv"
)

REPORT_FILE = os.path.join(
    OUTPUT_DIR,
    "v7_balancing_report.txt"
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("=" * 70)
print("V7 SOURCE-AWARE DATASET BALANCING")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print(f"\nDataset Loaded: {len(df)} samples")


# ============================================================
# 2. LABEL NORMALIZATION
# ============================================================

df["label"] = (
    df["label"]
    .astype(str)
    .str.strip()
    .str.lower()
)


# ============================================================
# 3. CHECK LABELS
# ============================================================

valid_labels = [
    "safe",
    "vulnerable"
]

invalid_labels = df[
    ~df["label"].isin(valid_labels)
]

if len(invalid_labels) > 0:

    print("\nWARNING: Invalid labels detected")

    print(
        invalid_labels["label"].value_counts()
    )

    raise ValueError(
        "Dataset contains unexpected labels."
    )


# ============================================================
# 4. REMOVE EMPTY CODE ONLY
# ============================================================

df["code_normalized"] = (
    df["code_normalized"]
    .fillna("")
    .astype(str)
)

df = df[
    df["code_normalized"].str.strip() != ""
].copy()

print(f"Samples after empty-code filtering: {len(df)}")


# ============================================================
# 5. SOURCE DISTRIBUTION
# ============================================================

print("\nSOURCE DISTRIBUTION")

source_counts = df["source"].value_counts()

print(source_counts)


# ============================================================
# 6. SOURCE AND LABEL DISTRIBUTION
# ============================================================

source_label_counts = pd.crosstab(
    df["source"],
    df["label"]
)

print("\nSOURCE-LABEL DISTRIBUTION")

print(source_label_counts)


# ============================================================
# 7. CREATE TRAIN/TEST SPLIT
# ============================================================

# IMPORTANT:
# We first split the dataset into training and test data.
# The test data is preserved for evaluation.
#
# Random stratified split:
# 80% training
# 20% testing

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)


print("\nTRAIN/TEST SPLIT")

print(f"Training Samples: {len(train_df)}")
print(f"Testing Samples: {len(test_df)}")


# ============================================================
# 8. SOURCE-LABEL BALANCING
# ============================================================

# Balance source contribution within the TRAINING dataset.
#
# We calculate the smallest source-label group.
#
# Example:
#
# Juliet Safe
# Juliet Vulnerable
# OWASP Safe
# OWASP Vulnerable
# Curated Safe
# Curated Vulnerable
#
# Each group receives equal samples.
#
# TEST DATA IS NOT BALANCED OR MODIFIED.


train_groups = train_df.groupby(
    ["source", "label"]
)

group_sizes = train_groups.size()

print("\nTRAINING SOURCE-LABEL GROUP SIZES")

print(group_sizes)


# ============================================================
# 9. FIND MINIMUM GROUP SIZE
# ============================================================

minimum_group_size = group_sizes.min()

print(
    f"\nMinimum group size: {minimum_group_size}"
)


# ============================================================
# 10. BALANCE TRAINING DATA
# ============================================================

balanced_groups = []

for (source, label), group in train_groups:

    sampled_group = group.sample(
        n=minimum_group_size,
        random_state=42
    )

    balanced_groups.append(
        sampled_group
    )


balanced_train_df = pd.concat(
    balanced_groups,
    ignore_index=True
)


# ============================================================
# 11. SHUFFLE BALANCED DATA
# ============================================================

balanced_train_df = balanced_train_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# ============================================================
# 12. SAVE DATASETS
# ============================================================

balanced_train_df.to_csv(
    TRAIN_FILE,
    index=False
)

test_df.to_csv(
    TEST_FILE,
    index=False
)


# ============================================================
# 13. FINAL DISTRIBUTION
# ============================================================

balanced_source_counts = (
    balanced_train_df["source"]
    .value_counts()
)

balanced_label_counts = (
    balanced_train_df["label"]
    .value_counts()
)

balanced_source_label_counts = pd.crosstab(
    balanced_train_df["source"],
    balanced_train_df["label"]
)


print("\n" + "=" * 70)

print("BALANCED TRAINING SOURCE DISTRIBUTION")

print("=" * 70)

print(balanced_source_counts)


print("\nBALANCED TRAINING LABEL DISTRIBUTION")

print(balanced_label_counts)


print("\nBALANCED SOURCE-LABEL DISTRIBUTION")

print(balanced_source_label_counts)


print("\nTEST SOURCE DISTRIBUTION")

print(test_df["source"].value_counts())


print("\nTEST LABEL DISTRIBUTION")

print(test_df["label"].value_counts())


# ============================================================
# 14. GENERATE REPORT
# ============================================================

report = []

report.append("=" * 70)
report.append("V7 SOURCE-AWARE BALANCING REPORT")
report.append("=" * 70)

report.append(
    f"\nOriginal dataset: {len(df)}"
)

report.append(
    f"Training dataset before balancing: {len(train_df)}"
)

report.append(
    f"Testing dataset: {len(test_df)}"
)

report.append(
    f"Balanced training dataset: {len(balanced_train_df)}"
)

report.append(
    f"Minimum source-label group: {minimum_group_size}"
)


report.append("\nORIGINAL SOURCE DISTRIBUTION")
report.append("-" * 50)

for source, count in source_counts.items():

    percentage = (
        count / len(df)
    ) * 100

    report.append(
        f"{source}: {count} ({percentage:.2f}%)"
    )


report.append("\nBALANCED TRAINING SOURCE DISTRIBUTION")
report.append("-" * 50)

for source, count in balanced_source_counts.items():

    percentage = (
        count / len(balanced_train_df)
    ) * 100

    report.append(
        f"{source}: {count} ({percentage:.2f}%)"
    )


report.append("\nBALANCED TRAINING LABEL DISTRIBUTION")
report.append("-" * 50)

for label, count in balanced_label_counts.items():

    percentage = (
        count / len(balanced_train_df)
    ) * 100

    report.append(
        f"{label}: {count} ({percentage:.2f}%)"
    )


report.append("\nSOURCE-LABEL GROUP DISTRIBUTION")
report.append("-" * 50)

report.append(
    balanced_source_label_counts.to_string()
)


report.append("\nTEST DATA SOURCE DISTRIBUTION")
report.append("-" * 50)

report.append(
    test_df["source"].value_counts().to_string()
)


report.append("\nIMPORTANT METHODOLOGY")
report.append("-" * 50)

report.append(
    "1. Original V7 dataset was preserved."
)

report.append(
    "2. Dataset was split into training and testing."
)

report.append(
    "3. Only the training dataset was balanced."
)

report.append(
    "4. Test dataset was not balanced."
)

report.append(
    "5. Source-label groups were balanced using undersampling."
)

report.append(
    "6. Random state 42 was used for reproducibility."
)

report.append(
    "7. Source information was preserved."
)

report.append(
    "8. This experiment reduces source imbalance "
    "but also reduces the number of training samples."
)

report.append(
    "9. Model evaluation must also include source-specific "
    "and cross-source experiments."
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
# 15. COMPLETION
# ============================================================

print("\n" + "=" * 70)

print("V7 SOURCE-AWARE BALANCING COMPLETED")

print("=" * 70)

print("\nTraining Dataset:")

print(TRAIN_FILE)

print("\nTesting Dataset:")

print(TEST_FILE)

print("\nReport:")

print(REPORT_FILE)