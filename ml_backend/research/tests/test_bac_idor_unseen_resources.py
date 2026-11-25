
import json
import joblib
import pandas as pd

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

BASE_DIR = Path(__file__).parent

DATASET = (
    BASE_DIR / "dataset" / "external_sources"
    / "BAC_ML_1M" / "bac_idor_prepared.csv"
)

OUTPUT_DIR = BASE_DIR / "models" / "bac_idor" / "unseen_resources"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = OUTPUT_DIR / "bac_idor_unseen_resources.joblib"
METRICS_FILE = OUTPUT_DIR / "evaluation_metrics.json"
REPORT_FILE = OUTPUT_DIR / "classification_report.txt"

FEATURES = [
    "request_method",
    "resource_type",
    "resource_classification"
]

TARGET = "idor_label"

# Resource types deliberately excluded from training
UNSEEN_RESOURCES = ["payment", "transaction"]


def main():

    print("=" * 60)
    print("IDOR UNSEEN RESOURCE GENERALIZATION TEST")
    print("=" * 60)

    df = pd.read_csv(DATASET)

    print("\nOriginal records:", len(df))

    # Separate training and testing resource types
    train_df = df[
        ~df["resource_type"].isin(UNSEEN_RESOURCES)
    ].copy()

    test_df = df[
        df["resource_type"].isin(UNSEEN_RESOURCES)
    ].copy()

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    print("Training records:", len(train_df))
    print("Testing records:", len(test_df))
    print("Unseen resource types:", UNSEEN_RESOURCES)

    if train_df.empty or test_df.empty:
        raise ValueError("Training or testing dataset is empty.")

    print("\nTraining label distribution:")
    print(y_train.value_counts().to_string())

    print("\nTesting label distribution:")
    print(y_test.value_counts().to_string())

    preprocessor = ColumnTransformer([
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            FEATURES
        )
    ])

    classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    print("\nTraining model on seen resource types...")

    model.fit(X_train, y_train)

    print("Training completed.")

    print("\nEvaluating on unseen resource types...")

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    report = classification_report(
        y_test,
        predictions,
        target_names=["No IDOR", "IDOR"],
        zero_division=0
    )

    matrix = confusion_matrix(y_test, predictions)

    metrics = {
        "experiment": "Unseen resource types",
        "unseen_resources": UNSEEN_RESOURCES,
        "training_records": int(len(train_df)),
        "testing_records": int(len(test_df)),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": matrix.tolist()
    }

    joblib.dump(model, MODEL_FILE)

    METRICS_FILE.write_text(
        json.dumps(metrics, indent=4),
        encoding="utf-8"
    )

    REPORT_FILE.write_text(
        report + "\nConfusion Matrix:\n" + str(matrix),
        encoding="utf-8"
    )

    print("\n" + "=" * 60)
    print("UNSEEN RESOURCE TEST RESULTS")
    print("=" * 60)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\nClassification Report:")
    print(report)

    print("Confusion Matrix:")
    print(matrix)

    print("\nSaved model:", MODEL_FILE)
    print("Saved metrics:", METRICS_FILE)
    print("Saved report:", REPORT_FILE)

    print("\nExisting models remain unchanged.")
    print("=" * 60)


if __name__ == "__main__":
    main()
