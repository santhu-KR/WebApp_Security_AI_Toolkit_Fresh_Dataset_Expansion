"""
Compare Feature Model V1 and V2.

Uses the same stratified train/test split for both models.
This is an initial row-level comparison, not proof of generalization
to unseen applications.
"""

from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from feature_model import build_model as build_model_v1
from feature_model_v2 import build_model as build_model_v2


HERE = Path(__file__).resolve().parent
DATASET_PATH = HERE / "dataset" / "processed" / "security_snippets_clean.csv"
METRICS_PATH = HERE / "compare_feature_models_v1_v2_metrics.json"
PREDICTIONS_PATH = HERE / "compare_feature_models_v1_v2_predictions.csv"


def convert_label(value):
    value = str(value).strip().lower()

    if value in {"0", "safe", "benign"}:
        return 0
    if value in {"1", "vulnerable", "vulnerability"}:
        return 1

    return None


def load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)

    required_columns = {"code", "label"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required dataset columns: {missing}")

    df["label_numeric"] = df["label"].apply(convert_label)
    df["code"] = df["code"].fillna("").astype(str)

    df = df[
        (df["code"].str.strip() != "")
        & df["label_numeric"].notna()
    ].copy()

    df["label_numeric"] = df["label_numeric"].astype(int)

    if df["label_numeric"].nunique() != 2:
        raise ValueError("Dataset must contain both safe and vulnerable examples.")

    return df


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["safe", "vulnerable"],
            output_dict=True,
            zero_division=0,
        ),
    }

    print(f"\n{model_name} results")
    print("-" * 40)
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1_score']:.4f}")
    print("Confusion matrix [safe, vulnerable]:")
    print(metrics["confusion_matrix"])

    return metrics, predictions


def main():
    print("=" * 55)
    print("FEATURE MODEL V1 vs V2 COMPARISON")
    print("=" * 55)

    df = load_dataset()

    X = df["code"].tolist()
    y = df["label_numeric"].tolist()
    row_indices = df.index.tolist()

    print("\nDataset summary")
    print("-" * 40)
    print("Total examples:", len(df))
    print("Safe examples:", y.count(0))
    print("Vulnerable examples:", y.count(1))

    (
        X_train,
        X_test,
        y_train,
        y_test,
        train_indices,
        test_indices,
    ) = train_test_split(
        X,
        y,
        row_indices,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("\nSplit summary")
    print("-" * 40)
    print("Training examples:", len(X_train))
    print("Testing examples:", len(X_test))
    print("Split: 80% training / 20% testing")
    print("Random state: 42")
    print("Both models use exactly the same split.")

    model_v1 = build_model_v1()
    model_v2 = build_model_v2()

    metrics_v1, predictions_v1 = evaluate_model(
        model_v1, X_train, y_train, X_test, y_test, "Feature Model V1"
    )
    metrics_v2, predictions_v2 = evaluate_model(
        model_v2, X_train, y_train, X_test, y_test, "Feature Model V2"
    )

    comparison = pd.DataFrame(
        [
            {"Metric": "Accuracy", "V1": metrics_v1["accuracy"], "V2": metrics_v2["accuracy"]},
            {"Metric": "Precision", "V1": metrics_v1["precision"], "V2": metrics_v2["precision"]},
            {"Metric": "Recall", "V1": metrics_v1["recall"], "V2": metrics_v2["recall"]},
            {"Metric": "F1-score", "V1": metrics_v1["f1_score"], "V2": metrics_v2["f1_score"]},
        ]
    )
    comparison["Difference (V2 - V1)"] = (
        comparison["V2"] - comparison["V1"]
    ).round(4)

    print("\n" + "=" * 55)
    print("FINAL MODEL COMPARISON")
    print("=" * 55)
    print(comparison.to_string(index=False))

    prediction_df = df.loc[test_indices].copy()
    prediction_df["actual_label"] = y_test
    prediction_df["v1_prediction"] = predictions_v1
    prediction_df["v2_prediction"] = predictions_v2
    prediction_df["v1_correct"] = (
        prediction_df["actual_label"] == prediction_df["v1_prediction"]
    )
    prediction_df["v2_correct"] = (
        prediction_df["actual_label"] == prediction_df["v2_prediction"]
    )
    prediction_df.to_csv(PREDICTIONS_PATH, index=False)

    results = {
        "dataset": str(DATASET_PATH),
        "total_examples": len(df),
        "training_examples": len(X_train),
        "testing_examples": len(X_test),
        "split": "80/20 stratified row-level split",
        "random_state": 42,
        "models": {"V1": metrics_v1, "V2": metrics_v2},
        "comparison": comparison.to_dict(orient="records"),
        "evaluation_warning": (
            "This is a random row-level holdout evaluation. Related code "
            "templates may occur in both train and test sets. It does not "
            "establish performance on independent projects or production apps."
        ),
    }

    METRICS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\nCreated output files:")
    print(METRICS_PATH.name)
    print(PREDICTIONS_PATH.name)
    print("\nComparison completed.")


if __name__ == "__main__":
    main()
