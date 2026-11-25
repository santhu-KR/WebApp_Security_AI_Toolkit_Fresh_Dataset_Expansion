import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# ============================================================
# V6 DATASET PREPARATION
# ============================================================

DATASET_PATH = "dataset/processed/security_snippets_clean.csv"

OUTPUT_DIR = Path("dataset/v6")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("V6 DATASET PREPARATION")
print("=" * 65)

# Load dataset
df = pd.read_csv(DATASET_PATH)

print("\nOriginal dataset:")
print("Total samples:", len(df))

# Normalize labels
label_mapping = {
    "safe": 0,
    "benign": 0,
    "non-vulnerable": 0,
    "non_vulnerable": 0,
    "0": 0,

    "vulnerable": 1,
    "unsafe": 1,
    "malicious": 1,
    "1": 1
}

df["label_normalized"] = (
    df["label"]
    .astype(str)
    .str.lower()
    .str.strip()
    .map(label_mapping)
)

# Validate labels
if df["label_normalized"].isna().any():

    print("\nERROR: Unknown labels detected!")

    print(
        df[df["label_normalized"].isna()]["label"]
        .value_counts()
    )

    raise ValueError("Label normalization failed.")

df["label_normalized"] = df["label_normalized"].astype(int)

# Remove missing code samples
df = df.dropna(subset=["code"])

df["code"] = df["code"].astype(str)

# Remove empty code samples
df = df[df["code"].str.strip() != ""]

print("\nClean dataset:")
print("Total samples:", len(df))

print("\nLabel distribution:")
print(df["label_normalized"].value_counts())

# Split dataset
X = df["code"]

y = df["label_normalized"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Save train/test datasets
train_df = df.loc[X_train.index].copy()

test_df = df.loc[X_test.index].copy()

train_df["label"] = y_train

test_df["label"] = y_test

train_path = OUTPUT_DIR / "train_v6.csv"

test_path = OUTPUT_DIR / "test_v6.csv"

train_df.to_csv(train_path, index=False)

test_df.to_csv(test_path, index=False)

# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 65)

print("V6 DATASET SPLIT RESULTS")

print("=" * 65)

print("\nTraining samples:", len(train_df))

print("Testing samples:", len(test_df))

print("\nTraining label distribution:")

print(train_df["label"].value_counts())

print("\nTesting label distribution:")

print(test_df["label"].value_counts())

print("\nTraining percentages:")

print(
    train_df["label"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nTesting percentages:")

print(
    test_df["label"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nSaved files:")

print(train_path)

print(test_path)

print("\nV6 DATASET PREPARATION COMPLETED")