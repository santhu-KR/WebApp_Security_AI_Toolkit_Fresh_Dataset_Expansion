# ============================================================
# V4 MACHINE LEARNING TRAINING & EVALUATION
# CWE-GROUPED 5-FOLD CROSS VALIDATION
# ============================================================

import os
import json
import warnings

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

from feature_model_v4 import build_model

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "processed",
    "security_snippets_clean.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "v4"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

PREDICTIONS_FILE = os.path.join(
    OUTPUT_DIR,
    "v4_grouped_predictions.csv"
)

METRICS_FILE = os.path.join(
    OUTPUT_DIR,
    "v4_grouped_metrics.json"
)

N_SPLITS = 5
RANDOM_STATE = 42


# ============================================================
# LABEL MAPPING
# ============================================================

LABEL_MAPPING = {
    "safe": 0,
    "benign": 0,
    "non-vulnerable": 0,
    "non_vulnerable": 0,
    "normal": 0,
    "clean": 0,

    "vulnerable": 1,
    "unsafe": 1,
    "malicious": 1,

    "0": 0,
    "1": 1
}


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("V4 SECURITY MACHINE LEARNING TRAINING")
print("=" * 70)

print("\nLoading dataset...")

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset not found:\n{DATASET_PATH}"
    )

df = pd.read_csv(DATASET_PATH)

print("\nDataset loaded successfully.")

print("Dataset shape:", df.shape)

print("\nAvailable columns:")
print(df.columns.tolist())


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = [
    "label",
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
# CLEAN DATA
# ============================================================

print("\nCleaning dataset...")

df = df.copy()

# Preserve original labels for validation
original_labels = (
    df["label"]
    .astype(str)
    .str.strip()
    .str.lower()
)

# Convert labels to numeric
mapped_labels = original_labels.map(LABEL_MAPPING)

# Check unknown labels
unknown_mask = mapped_labels.isna()

if unknown_mask.any():

    unknown_values = (
        original_labels[unknown_mask]
        .value_counts()
    )

    print("\nERROR: Unknown labels detected")

    print(unknown_values)

    raise ValueError(
        "Dataset contains unknown labels. "
        "Please check the label column."
    )


df["label"] = mapped_labels.astype(int)


# Clean code column

df["code"] = (
    df["code"]
    .fillna("")
    .astype(str)
)


# Clean CWE column

df["cwe"] = (
    df["cwe"]
    .fillna("UNKNOWN")
    .astype(str)
    .str.strip()
)


# Remove empty code samples

df = df[
    df["code"].str.strip() != ""
].copy()


# ============================================================
# DATASET INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print("Total samples:", len(df))

print("\nLabel distribution:")

print(
    df["label"]
    .value_counts()
    .sort_index()
)

print("\nLabel names:")

print("0 = SAFE")
print("1 = VULNERABLE")

print("\nCWE groups:", df["cwe"].nunique())

print("\nCWE distribution:")

print(
    df["cwe"]
    .value_counts()
)


# ============================================================
# PREPARE FEATURES AND LABELS
# ============================================================

X = df["code"]

y = df["label"]

groups = df["cwe"]


print("\nPreparing grouped cross-validation...")

print("Samples:", len(X))

print("Safe samples:", int((y == 0).sum()))

print("Vulnerable samples:", int((y == 1).sum()))

print("CWE groups:", groups.nunique())


# ============================================================
# STRATIFIED GROUP K-FOLD
# ============================================================

cv = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)


# ============================================================
# STORAGE FOR RESULTS
# ============================================================

all_actual = []

all_predictions = []

all_prediction_indices = []

fold_results = []


# ============================================================
# TRAINING LOOP
# ============================================================

print("\n" + "=" * 70)
print("STARTING 5-FOLD GROUPED CROSS VALIDATION")
print("=" * 70)


for fold_number, (train_index, test_index) in enumerate(
    cv.split(X, y, groups),
    start=1
):

    print("\n")
    print("=" * 70)

    print(f"FOLD {fold_number}/{N_SPLITS}")

    print("=" * 70)


    # --------------------------------------------------------
    # SPLIT DATA
    # --------------------------------------------------------

    X_train = X.iloc[train_index]

    X_test = X.iloc[test_index]

    y_train = y.iloc[train_index]

    y_test = y.iloc[test_index]

    train_groups = groups.iloc[train_index]

    test_groups = groups.iloc[test_index]


    # --------------------------------------------------------
    # GROUP LEAKAGE CHECK
    # --------------------------------------------------------

    overlapping_groups = set(
        train_groups
    ).intersection(
        set(test_groups)
    )


    if overlapping_groups:

        raise RuntimeError(
            f"Group leakage detected in Fold {fold_number}"
        )


    print("\nTraining samples:", len(X_train))

    print("Testing samples:", len(X_test))

    print("Training CWE groups:", train_groups.nunique())

    print("Testing CWE groups:", test_groups.nunique())


    # --------------------------------------------------------
    # CREATE MODEL
    # --------------------------------------------------------

    print("\nCreating V4 model...")

    model = build_model()


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print("Training model...")

    model.fit(
        X_train,
        y_train
    )


    print("Training completed.")


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    print("Predicting test data...")

    predictions = model.predict(
        X_test
    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    fold_accuracy = accuracy_score(
        y_test,
        predictions
    )

    fold_precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    fold_recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    fold_f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )


    fold_cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print("\nFold Results")

    print("-" * 50)

    print(
        f"Accuracy  : {fold_accuracy:.4f}"
    )

    print(
        f"Precision : {fold_precision:.4f}"
    )

    print(
        f"Recall    : {fold_recall:.4f}"
    )

    print(
        f"F1 Score  : {fold_f1:.4f}"
    )


    print("\nConfusion Matrix:")

    print(
        "                Predicted SAFE  Predicted VULNERABLE"
    )

    print(
        f"Actual SAFE        {fold_cm[0][0]:>8} "
        f"{fold_cm[0][1]:>12}"
    )

    print(
        f"Actual VULNERABLE  {fold_cm[1][0]:>8} "
        f"{fold_cm[1][1]:>12}"
    )


    # --------------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------------

    fold_results.append({

        "fold": fold_number,

        "accuracy": float(fold_accuracy),

        "precision": float(fold_precision),

        "recall": float(fold_recall),

        "f1_score": float(fold_f1),

        "confusion_matrix": fold_cm.tolist(),

        "training_samples": int(len(X_train)),

        "testing_samples": int(len(X_test)),

        "training_cwe_groups": int(train_groups.nunique()),

        "testing_cwe_groups": int(test_groups.nunique())

    })


    # --------------------------------------------------------
    # STORE PREDICTIONS
    # --------------------------------------------------------

    all_actual.extend(
        y_test.tolist()
    )

    all_predictions.extend(
        predictions.tolist()
    )

    all_prediction_indices.extend(
        test_index.tolist()
    )


# ============================================================
# OVERALL CROSS VALIDATION RESULTS
# ============================================================

print("\n")
print("=" * 70)

print("OVERALL V4 CROSS-VALIDATION RESULTS")

print("=" * 70)


overall_accuracy = accuracy_score(
    all_actual,
    all_predictions
)

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


overall_cm = confusion_matrix(
    all_actual,
    all_predictions,
    labels=[0, 1]
)


print("\nOverall Metrics")

print("-" * 50)

print(
    f"Accuracy  : {overall_accuracy:.4f}"
)

print(
    f"Precision : {overall_precision:.4f}"
)

print(
    f"Recall    : {overall_recall:.4f}"
)

print(
    f"F1 Score  : {overall_f1:.4f}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\nOverall Confusion Matrix")

print("-" * 50)

print(
    "                Predicted SAFE  Predicted VULNERABLE"
)

print(
    f"Actual SAFE        {overall_cm[0][0]:>8} "
    f"{overall_cm[0][1]:>12}"
)

print(
    f"Actual VULNERABLE  {overall_cm[1][0]:>8} "
    f"{overall_cm[1][1]:>12}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report")

print("-" * 50)

print(
    classification_report(
        all_actual,
        all_predictions,
        labels=[0, 1],
        target_names=[
            "SAFE",
            "VULNERABLE"
        ],
        zero_division=0
    )
)


# ============================================================
# FOLD-WISE AVERAGE
# ============================================================

fold_accuracy_mean = np.mean([
    result["accuracy"]
    for result in fold_results
])

fold_precision_mean = np.mean([
    result["precision"]
    for result in fold_results
])

fold_recall_mean = np.mean([
    result["recall"]
    for result in fold_results
])

fold_f1_mean = np.mean([
    result["f1_score"]
    for result in fold_results
])


print("\nFold-Wise Average")

print("-" * 50)

print(
    f"Average Accuracy  : {fold_accuracy_mean:.4f}"
)

print(
    f"Average Precision : {fold_precision_mean:.4f}"
)

print(
    f"Average Recall    : {fold_recall_mean:.4f}"
)

print(
    f"Average F1 Score  : {fold_f1_mean:.4f}"
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

print("\nSaving predictions...")


prediction_df = df.copy()


# Create prediction columns

prediction_df["predicted_label"] = np.nan

prediction_df["prediction_probability"] = np.nan


# Correctly align predictions to original dataframe indices

for position, prediction in zip(
    all_prediction_indices,
    all_predictions
):

    prediction_df.iloc[
        position,
        prediction_df.columns.get_loc(
            "predicted_label"
        )
    ] = prediction


# Save predictions

prediction_df["predicted_label"] = (
    prediction_df["predicted_label"]
    .astype("Int64")
)


prediction_df["actual_label_name"] = (
    prediction_df["label"]
    .map({
        0: "SAFE",
        1: "VULNERABLE"
    })
)


prediction_df["predicted_label_name"] = (
    prediction_df["predicted_label"]
    .map({
        0: "SAFE",
        1: "VULNERABLE"
    })
)


prediction_df["correct_prediction"] = (
    prediction_df["label"]
    == prediction_df["predicted_label"]
)


prediction_df.to_csv(
    PREDICTIONS_FILE,
    index=False
)


print(
    "\nPredictions saved to:"
)

print(
    PREDICTIONS_FILE
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics_data = {

    "model_version": "V4",

    "evaluation_method": "StratifiedGroupKFold",

    "number_of_folds": N_SPLITS,

    "grouping_column": "cwe",

    "random_state": RANDOM_STATE,

    "dataset_path": DATASET_PATH,

    "total_samples": int(len(df)),

    "safe_samples": int(
        (df["label"] == 0).sum()
    ),

    "vulnerable_samples": int(
        (df["label"] == 1).sum()
    ),

    "number_of_cwe_groups": int(
        df["cwe"].nunique()
    ),

    "overall_metrics": {

        "accuracy": float(
            overall_accuracy
        ),

        "precision": float(
            overall_precision
        ),

        "recall": float(
            overall_recall
        ),

        "f1_score": float(
            overall_f1
        )

    },

    "fold_average_metrics": {

        "accuracy": float(
            fold_accuracy_mean
        ),

        "precision": float(
            fold_precision_mean
        ),

        "recall": float(
            fold_recall_mean
        ),

        "f1_score": float(
            fold_f1_mean
        )

    },

    "confusion_matrix": overall_cm.tolist(),

    "fold_results": fold_results

}


with open(
    METRICS_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metrics_data,
        file,
        indent=4
    )


print(
    "\nMetrics saved to:"
)

print(
    METRICS_FILE
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")

print("=" * 70)

print("V4 TRAINING AND EVALUATION COMPLETED")

print("=" * 70)


print("\nFinal Overall Results:")

print(
    f"Accuracy  : {overall_accuracy * 100:.2f}%"
)

print(
    f"Precision : {overall_precision * 100:.2f}%"
)

print(
    f"Recall    : {overall_recall * 100:.2f}%"
)

print(
    f"F1 Score  : {overall_f1 * 100:.2f}%"
)


print("\nGenerated Files:")

print(
    "1. V4 Grouped Predictions CSV"
)

print(
    PREDICTIONS_FILE
)


print(
    "\n2. V4 Metrics JSON"
)

print(
    METRICS_FILE
)


print("\nDone.")