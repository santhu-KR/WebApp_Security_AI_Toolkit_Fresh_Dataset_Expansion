import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from feature_model_v5 import build_model


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "dataset/processed/security_snippets_clean.csv"

OUTPUT_DIR = "results/v5"
MODEL_DIR = "models/v5"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("WEB APPLICATION SECURITY AI - V5 TRAINING")
print("=" * 70)

df = pd.read_csv(DATASET_PATH)

print("\nDataset loaded successfully.")
print("Total rows:", len(df))
print("Columns:", list(df.columns))


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = ["code", "label", "cwe"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Missing required column: {column}")


df = df.dropna(subset=["code", "label"]).copy()

df["code"] = df["code"].astype(str)
df["label"] = df["label"].astype(str).str.lower().str.strip()


# ============================================================
# LABEL MAPPING
# ============================================================

label_mapping = {
    "safe": 0,
    "benign": 0,
    "non-vulnerable": 0,
    "non_vulnerable": 0,
    "vulnerable": 1,
    "unsafe": 1,
    "malicious": 1,
    "0": 0,
    "1": 1
}

df["label"] = df["label"].map(label_mapping)

# Remove invalid labels
df = df.dropna(subset=["label"])

df["label"] = df["label"].astype(int)

print("\nLabel distribution:")
print(df["label"].value_counts())


# ============================================================
# CWE GROUPING
# ============================================================

df["cwe"] = df["cwe"].fillna("UNKNOWN").astype(str)

groups = df["cwe"].values
X = df["code"].values
y = df["label"].values


print("\nNumber of CWE groups:", df["cwe"].nunique())
print("Dataset size:", len(df))


# ============================================================
# 5-FOLD GROUPED CROSS VALIDATION
# ============================================================

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

fold_results = []

all_predictions = []
all_actual = []

print("\nStarting 5-fold CWE-grouped cross-validation...\n")


for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), start=1):

    print("=" * 70)
    print(f"FOLD {fold}/5")
    print("=" * 70)

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    model = build_model()

    print("\nTraining model...")

    model.fit(X_train, y_train)

    print("Training completed.")

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    print("\nFold Results:")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    fold_results.append({
        "fold": fold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "train_samples": len(X_train),
        "test_samples": len(X_test)
    })

    all_predictions.extend(predictions)
    all_actual.extend(y_test)


# ============================================================
# OVERALL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("V5 OVERALL CROSS-VALIDATION RESULTS")
print("=" * 70)

overall_accuracy = accuracy_score(all_actual, all_predictions)

overall_precision = precision_score(
    all_actual,
    all_predictions,
    zero_division=0
)

overall_recall = recall_score(
    all_actual,
    all_predictions,
    zero_division=0
)

overall_f1 = f1_score(
    all_actual,
    all_predictions,
    zero_division=0
)

cm = confusion_matrix(all_actual, all_predictions)

print(f"\nAccuracy : {overall_accuracy:.4f}")
print(f"Precision: {overall_precision:.4f}")
print(f"Recall   : {overall_recall:.4f}")
print(f"F1 Score : {overall_f1:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        all_actual,
        all_predictions,
        target_names=["Safe", "Vulnerable"],
        zero_division=0
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

results = {
    "model_version": "V5",
    "dataset_size": len(df),
    "cwe_groups": int(df["cwe"].nunique()),
    "cross_validation": "5-Fold Stratified Group CV",
    "grouping": "CWE",
    "accuracy": float(overall_accuracy),
    "precision": float(overall_precision),
    "recall": float(overall_recall),
    "f1_score": float(overall_f1),
    "confusion_matrix": cm.tolist(),
    "fold_results": fold_results
}


results_path = os.path.join(
    OUTPUT_DIR,
    "v5_evaluation_results.json"
)

with open(results_path, "w") as f:
    json.dump(results, f, indent=4)


fold_df = pd.DataFrame(fold_results)

fold_csv_path = os.path.join(
    OUTPUT_DIR,
    "v5_fold_results.csv"
)

fold_df.to_csv(fold_csv_path, index=False)


# ============================================================
# TRAIN FINAL MODEL ON COMPLETE DATASET
# ============================================================

print("\n" + "=" * 70)
print("TRAINING FINAL V5 MODEL ON COMPLETE DATASET")
print("=" * 70)

final_model = build_model()

final_model.fit(X, y)

model_path = os.path.join(
    MODEL_DIR,
    "security_model_v5.pkl"
)

joblib.dump(final_model, model_path)


print("\nFinal model saved successfully:")
print(model_path)

print("\nEvaluation results saved:")
print(results_path)

print("\nFold results saved:")
print(fold_csv_path)


# ============================================================
# COMPLETED
# ============================================================

print("\n" + "=" * 70)
print("V5 TRAINING COMPLETED")
print("=" * 70)