
import json
import joblib
import pandas as pd

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Project paths
BASE_DIR = Path(__file__).parent

DATASET = (
    BASE_DIR / "dataset" / "external_sources"
    / "BAC_ML_1M" / "bac_idor_prepared.csv"
)

OUTPUT_DIR = (
    BASE_DIR / "models" / "bac_idor"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = OUTPUT_DIR / "bac_idor_random_forest.joblib"
METRICS_FILE = OUTPUT_DIR / "evaluation_metrics.json"
REPORT_FILE = OUTPUT_DIR / "classification_report.txt"

# Features and target
FEATURES = [
    "request_method",
    "resource_type",
    "resource_classification",
    "expected_access",
    "actual_access_granted"
]

TARGET = "idor_label"


def main():

    print("=" * 60)
    print("BAC-ML-1M IDOR MODEL TRAINING")
    print("=" * 60)

    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET}")

    print("\nLoading dataset...")

    df = pd.read_csv(DATASET)

    X = df[FEATURES]
    y = df[TARGET]

    print("Total records:", len(df))
    print("\nClass distribution:")
    print(y.value_counts().to_string())

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining records:", len(X_train))
    print("Testing records:", len(X_test))

    # Categorical feature encoding
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                FEATURES
            )
        ]
    )

    # Random Forest classifier
    classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    # Complete ML pipeline
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    print("\nTraining model...")

    model.fit(X_train, y_train)

    print("Training completed.")

    # Predictions
    print("\nEvaluating model...")

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
        "model": "Random Forest",
        "dataset": "BAC-ML-1M",
        "total_records": int(len(df)),
        "training_records": int(len(X_train)),
        "testing_records": int(len(X_test)),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": matrix.tolist(),
        "features": FEATURES
    }

    # Save model and evaluation
    joblib.dump(model, MODEL_FILE)

    METRICS_FILE.write_text(
        json.dumps(metrics, indent=4),
        encoding="utf-8"
    )

    REPORT_FILE.write_text(
        report + "\nConfusion Matrix:\n" + str(matrix),
        encoding="utf-8"
    )

    # Display results
    print("\n" + "=" * 60)
    print("IDOR MODEL EVALUATION RESULTS")
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

    print("\nTraining and evaluation completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
