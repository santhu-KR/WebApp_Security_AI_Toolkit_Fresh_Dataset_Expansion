"""
Improved evaluation script for the Web Application Security AI Toolkit.

Keeps the previous grouped-training outputs unchanged.
Creates:
  security_model_grouped_v2.joblib
  training_metrics_grouped_v2.json
  evaluation_predictions_grouped_v2.csv

Evaluations:
1. OWASP cross-source: train on Original + all Juliet; test on 20% OWASP.
2. OWASP within-source: train on Original + Juliet + 80% OWASP; test on 20% OWASP.
3. Juliet cross-source: train on Original + all OWASP; test on one grouped Juliet fold.
4. Juliet within-source: train on Original + OWASP + the other Juliet groups;
   test on the same grouped Juliet fold.

The cross-source evaluations measure transfer between corpora.
The within-source evaluations measure held-out examples from the same corpus.
Neither is a production-security guarantee.
"""

from pathlib import Path
import json
import sys

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from sklearn.model_selection import train_test_split, StratifiedGroupKFold

from feature_model import build_model

HERE = Path(__file__).resolve().parent
DATASET = HERE / "dataset" / "processed" / "security_snippets_clean.csv"
MODEL_OUT = HERE / "security_model_grouped_v2.joblib"
METRICS_OUT = HERE / "training_metrics_grouped_v2.json"
PREDICTIONS_OUT = HERE / "evaluation_predictions_grouped_v2.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_dataset():
    if not DATASET.exists():
        raise SystemExit(f"Dataset not found: {DATASET}")

    df = pd.read_csv(DATASET)
    required = {
        "source", "source_id", "label", "category_normalized",
        "code", "fingerprint",
    }
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns: {sorted(missing)}")

    df = df.copy()
    df["source"] = df["source"].fillna("").astype(str)
    df["source_id"] = df["source_id"].fillna("").astype(str)
    df["code"] = df["code"].fillna("").astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["category_normalized"] = (
        df["category_normalized"].fillna("Uncategorized").astype(str)
    )

    label_map = {"safe": 0, "vulnerable": 1}
    invalid = sorted(set(df["label"]) - set(label_map))
    if invalid:
        raise SystemExit(f"Unexpected labels: {invalid}")
    df["target"] = df["label"].map(label_map)
    df = df[df["code"].ne("")].copy()

    if df["fingerprint"].duplicated().any():
        n = int(df["fingerprint"].duplicated(keep=False).sum())
        raise SystemExit(f"Duplicate fingerprints remain ({n} rows).")

    src = df["source"].str.lower()
    original = df[src.str.contains("original_curated", regex=False)].copy()
    owasp = df[src.str.contains("owasp", regex=False)].copy()
    juliet = df[src.str.contains("juliet", regex=False)].copy()

    for name, part in (
        ("Original", original), ("OWASP", owasp), ("Juliet", juliet)
    ):
        if part.empty or part["target"].nunique() != 2:
            raise SystemExit(f"{name} must contain examples from both labels.")

    if owasp["source_id"].eq("").any() or owasp["source_id"].duplicated().any():
        raise SystemExit("OWASP source_id values must be present and unique.")
    if juliet["source_id"].eq("").any():
        raise SystemExit("Juliet source_id values are missing.")

    # Importer records each Juliet sample as <test-case-id>::<method-id>.
    juliet["group_id"] = (
        juliet["source_id"].str.split("::", n=1).str[0]
    )
    if juliet["group_id"].eq("").any():
        raise SystemExit("Empty Juliet group IDs found.")

    return df, original, owasp, juliet


def metric_record(name, y_true, y_pred, test_df, train_count, protocol):
    return {
        "evaluation_name": name,
        "evaluation_protocol": protocol,
        "train_rows": int(train_count),
        "test_rows": int(len(test_df)),
        "test_class_counts": {
            "safe": int((pd.Series(y_true) == 0).sum()),
            "vulnerable": int((pd.Series(y_true) == 1).sum()),
        },
        "predicted_class_counts": {
            "safe": int((pd.Series(y_pred) == 0).sum()),
            "vulnerable": int((pd.Series(y_pred) == 1).sum()),
        },
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "confusion_matrix_labels": ["safe", "vulnerable"],
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=[0, 1],
            target_names=["safe", "vulnerable"],
            output_dict=True, zero_division=0,
        ),
        "test_category_counts": {
            str(k): int(v)
            for k, v in test_df["category_normalized"].value_counts().items()
        },
    }


def evaluate(name, train_df, test_df, protocol, prediction_rows):
    print(f"\nStarting: {name}", flush=True)
    model = build_model()
    X_train = train_df["code"].tolist()
    y_train = train_df["target"].astype(int).tolist()
    X_test = test_df["code"].tolist()
    y_test = test_df["target"].astype(int).tolist()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test).astype(int)

    for idx, actual, predicted in zip(test_df.index, y_test, y_pred):
        row = test_df.loc[idx]
        prediction_rows.append({
            "evaluation": name,
            "dataset_index": int(idx),
            "source": str(row["source"]),
            "source_id": str(row["source_id"]),
            "category": str(row["category_normalized"]),
            "actual": "vulnerable" if actual else "safe",
            "predicted": "vulnerable" if predicted else "safe",
            "correct": bool(actual == predicted),
        })

    result = metric_record(
        name, y_test, y_pred, test_df, len(train_df), protocol
    )
    print(
        f"Finished {name}: accuracy={result['accuracy']:.4f}, "
        f"F1={result['f1']:.4f}, "
        f"predicted={result['predicted_class_counts']}",
        flush=True,
    )
    return result


def main():
    df, original, owasp, juliet = load_dataset()

    print("DATASET SUMMARY", flush=True)
    print(f"Total rows: {len(df)}", flush=True)
    print(f"Original curated: {len(original)}", flush=True)
    print(f"OWASP: {len(owasp)}", flush=True)
    print(f"Juliet: {len(juliet)}", flush=True)
    print(f"Juliet test-case groups: {juliet['group_id'].nunique()}", flush=True)

    predictions = []
    results = {}

    # OWASP split is stratified by label. Its source IDs are unique per row.
    owasp_train, owasp_test = train_test_split(
        owasp, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=owasp["target"],
    )

    # Cross-source: no OWASP examples are used for fitting.
    results["owasp_cross_source"] = evaluate(
        "OWASP cross-source",
        pd.concat([original, juliet], axis=0),
        owasp_test,
        "Train on Original + all Juliet; test on held-out OWASP examples. "
        "Measures transfer to another corpus.",
        predictions,
    )

    # Within-source: add the OWASP training partition.
    results["owasp_within_source"] = evaluate(
        "OWASP within-source",
        pd.concat([original, juliet, owasp_train], axis=0),
        owasp_test,
        "Train on Original + all Juliet + 80% OWASP; test on the remaining "
        "stratified 20% OWASP examples.",
        predictions,
    )

    # Split Juliet by test-case identity so related methods cannot cross folds.
    splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE
    )
    juliet_train_pos, juliet_test_pos = next(
        splitter.split(
            X=juliet["code"],
            y=juliet["target"],
            groups=juliet["group_id"],
        )
    )
    juliet_train = juliet.iloc[juliet_train_pos].copy()
    juliet_test = juliet.iloc[juliet_test_pos].copy()

    overlap = set(juliet_train["group_id"]) & set(juliet_test["group_id"])
    if overlap:
        raise RuntimeError(f"Juliet group leakage: {len(overlap)} overlapping groups.")

    # Cross-source: no Juliet examples are used for fitting.
    results["juliet_cross_source"] = evaluate(
        "Juliet cross-source",
        pd.concat([original, owasp], axis=0),
        juliet_test,
        "Train on Original + all OWASP; test on one grouped Juliet fold. "
        "Measures transfer to another corpus.",
        predictions,
    )

    # Within-source: train on the other Juliet groups as well.
    results["juliet_within_source"] = evaluate(
        "Juliet within-source",
        pd.concat([original, owasp, juliet_train], axis=0),
        juliet_test,
        "Train on Original + all OWASP + 80% of Juliet groups; test on the "
        "remaining grouped Juliet fold.",
        predictions,
    )
    results["juliet_within_source"]["juliet_train_groups"] = int(
        juliet_train["group_id"].nunique()
    )
    results["juliet_within_source"]["juliet_test_groups"] = int(
        juliet_test["group_id"].nunique()
    )
    results["juliet_within_source"]["group_overlap"] = int(len(overlap))

    # Final deployment fit is separate from evaluation scores.
    print("\nFitting final deployment model on all clean rows...", flush=True)
    deployment_model = build_model()
    deployment_model.fit(
        df["code"].tolist(), df["target"].astype(int).tolist()
    )
    joblib.dump(deployment_model, MODEL_OUT)

    pd.DataFrame(predictions).to_csv(PREDICTIONS_OUT, index=False)

    summary = {
        "model": "Character TF-IDF + engineered security indicators + Logistic Regression",
        "dataset_file": DATASET.name,
        "total_rows_used_for_final_model": int(len(df)),
        "source_counts": {
            str(k): int(v) for k, v in df["source"].value_counts().items()
        },
        "overall_label_counts": {
            "safe": int((df["target"] == 0).sum()),
            "vulnerable": int((df["target"] == 1).sum()),
        },
        "evaluation_results": results,
        "juliet_group_overlap": int(len(overlap)),
        "final_model_file": MODEL_OUT.name,
        "predictions_file": PREDICTIONS_OUT.name,
        "warnings": [
            "Dataset labels and CWE/category mappings are provisional and require manual review.",
            "Cross-source results measure corpus transfer, not within-corpus performance.",
            "Within-source results use held-out examples; Juliet is split by test-case group.",
            "The final deployment model is trained on all rows; its training fit is not an unbiased test score.",
            "These benchmark results do not establish production detection accuracy.",
        ],
    }
    METRICS_OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nEVALUATION SUMMARY", flush=True)
    for result in results.values():
        print(f"\n{result['evaluation_name']}", flush=True)
        print(f"Accuracy : {result['accuracy']:.4f}", flush=True)
        print(f"Precision: {result['precision']:.4f}", flush=True)
        print(f"Recall   : {result['recall']:.4f}", flush=True)
        print(f"F1-score : {result['f1']:.4f}", flush=True)
        print("Confusion matrix [safe, vulnerable]:", flush=True)
        print(result["confusion_matrix"], flush=True)
        print("Predicted classes:", result["predicted_class_counts"], flush=True)

    print(f"\nJuliet group overlap: {len(overlap)}", flush=True)
    print(f"New model saved: {MODEL_OUT.name}", flush=True)
    print(f"Metrics saved: {METRICS_OUT.name}", flush=True)
    print(f"Predictions saved: {PREDICTIONS_OUT.name}", flush=True)
    print("Previous model and metrics files were not overwritten.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except ImportError as exc:
        print(
            "Required package/API unavailable. Check scikit-learn version.\n"
            f"Details: {exc}",
            file=sys.stderr,
        )
        raise
