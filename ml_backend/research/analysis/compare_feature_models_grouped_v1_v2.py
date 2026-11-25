"""Grouped cross-validation comparison for Feature Model V1 and V2.

Juliet rows are grouped by source_id prefix before "::".
Other rows are grouped by source + source_id (with source_row/index fallback).
Both models use identical StratifiedGroupKFold splits.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

from feature_model import build_model as build_model_v1
from feature_model_v2 import build_model as build_model_v2

HERE = Path(__file__).resolve().parent
DATASET_PATH = HERE / "dataset" / "processed" / "security_snippets_clean.csv"
METRICS_PATH = HERE / "compare_feature_models_grouped_v1_v2_metrics.json"
PREDICTIONS_PATH = HERE / "compare_feature_models_grouped_v1_v2_predictions.csv"
FOLD_METRICS_PATH = HERE / "compare_feature_models_grouped_v1_v2_fold_metrics.csv"

N_SPLITS = 5
RANDOM_STATE = 42


def convert_label(value):
    value = str(value).strip().lower()
    if value in {"0", "safe", "benign"}:
        return 0
    if value in {"1", "vulnerable", "vulnerability"}:
        return 1
    return None


def make_group_id(row, fallback_index):
    source = str(row.get("source", "unknown")).strip()
    source_id = row.get("source_id")

    if pd.notna(source_id) and str(source_id).strip():
        source_id = str(source_id).strip()
        if "juliet" in source.lower():
            source_id = source_id.split("::", 1)[0]
        return f"{source}::{source_id}"

    source_row = row.get("source_row")
    if pd.notna(source_row):
        return f"{source}::row::{source_row}"

    return f"{source}::fallback::{fallback_index}"


def load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    missing = {"code", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["label_numeric"] = df["label"].apply(convert_label)
    df["code"] = df["code"].fillna("").astype(str)
    df = df[
    (df["code"].str.strip() != "")
    & (df["label_numeric"].notna())
].copy()
    df["label_numeric"] = df["label_numeric"].astype(int)

    if df["label_numeric"].nunique() != 2:
        raise ValueError("Dataset must contain both safe and vulnerable examples.")

    df["group_id"] = [
        make_group_id(row, idx) for idx, (_, row) in enumerate(df.iterrows())
    ]
    return df


def get_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=[0, 1],
            target_names=["safe", "vulnerable"],
            output_dict=True, zero_division=0,
        ),
    }


def run_cv(df, builder, model_name, splits):
    X = df["code"].tolist()
    y = df["label_numeric"].to_numpy()
    groups = df["group_id"].to_numpy()
    oof = np.full(len(df), -1, dtype=int)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        overlap = train_groups.intersection(test_groups)
        if overlap:
            raise RuntimeError(f"{model_name} fold {fold}: group leakage detected.")

        if len(np.unique(y[train_idx])) < 2:
            raise RuntimeError(f"{model_name} fold {fold}: training set has one class.")

        print(
            f"\n{model_name} fold {fold}/{N_SPLITS}: "
            f"train={len(train_idx)}, test={len(test_idx)}, "
            f"held-out groups={len(test_groups)}, overlap={len(overlap)}"
        )

        model = builder()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit([X[i] for i in train_idx], y[train_idx])

        for item in caught:
            print(f"WARNING ({model_name}, fold {fold}): {item.message}")

        pred = model.predict([X[i] for i in test_idx])
        oof[test_idx] = pred
        m = get_metrics(y[test_idx], pred)

        fold_results.append({
            "model": model_name,
            "fold": fold,
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "train_groups": int(len(train_groups)),
            "test_groups": int(len(test_groups)),
            "group_overlap": int(len(overlap)),
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1_score": m["f1_score"],
            "confusion_matrix": json.dumps(m["confusion_matrix"]),
        })

        print(
            f"accuracy={m['accuracy']:.4f}, precision={m['precision']:.4f}, "
            f"recall={m['recall']:.4f}, F1={m['f1_score']:.4f}"
        )

    if np.any(oof < 0):
        raise RuntimeError(f"{model_name}: missing out-of-fold predictions.")

    return get_metrics(y, oof), oof, fold_results


def main():
    print("=" * 65)
    print("GROUPED CROSS-VALIDATION: FEATURE MODEL V1 vs V2")
    print("=" * 65)

    df = load_dataset()
    y = df["label_numeric"].to_numpy()
    groups = df["group_id"].to_numpy()

    print("\nDATASET SUMMARY")
    print("Rows:", len(df))
    print("Safe:", int((y == 0).sum()))
    print("Vulnerable:", int((y == 1).sum()))
    print("Groups:", int(df["group_id"].nunique()))
    print("Folds:", N_SPLITS)

    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    splits = list(splitter.split(np.zeros(len(df)), y, groups))

    test_counts = np.zeros(len(df), dtype=int)
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        test_counts[test_idx] += 1
        overlap = set(groups[train_idx]).intersection(set(groups[test_idx]))
        if overlap:
            raise RuntimeError(f"Fold {fold}: {len(overlap)} groups overlap.")
        print(
            f"Fold {fold}: train rows={len(train_idx)}, "
            f"test rows={len(test_idx)}, group overlap=0"
        )

    if not np.all(test_counts == 1):
        raise RuntimeError("Every row must appear in exactly one test fold.")

    print("\nRunning V1...")
    v1_metrics, v1_pred, v1_folds = run_cv(df, build_model_v1, "V1", splits)

    print("\nRunning V2...")
    v2_metrics, v2_pred, v2_folds = run_cv(df, build_model_v2, "V2", splits)

    comparison = pd.DataFrame([
        {"Metric": "Accuracy", "V1": v1_metrics["accuracy"], "V2": v2_metrics["accuracy"]},
        {"Metric": "Precision", "V1": v1_metrics["precision"], "V2": v2_metrics["precision"]},
        {"Metric": "Recall", "V1": v1_metrics["recall"], "V2": v2_metrics["recall"]},
        {"Metric": "F1-score", "V1": v1_metrics["f1_score"], "V2": v2_metrics["f1_score"]},
    ])
    comparison["Difference (V2 - V1)"] = comparison["V2"] - comparison["V1"]

    print("\n" + "=" * 65)
    print("OVERALL OUT-OF-FOLD COMPARISON")
    print("=" * 65)
    print(comparison.round(4).to_string(index=False))
    print("\nV1 confusion matrix [safe, vulnerable]:")
    print(v1_metrics["confusion_matrix"])
    print("\nV2 confusion matrix [safe, vulnerable]:")
    print(v2_metrics["confusion_matrix"])

    pred_df = df.copy()
    pred_df["v1_oof_prediction"] = v1_pred
    pred_df["v2_oof_prediction"] = v2_pred
    pred_df["v1_correct"] = pred_df["label_numeric"] == v1_pred
    pred_df["v2_correct"] = pred_df["label_numeric"] == v2_pred
    pred_df.to_csv(PREDICTIONS_PATH, index=False)

    pd.DataFrame(v1_folds + v2_folds).to_csv(FOLD_METRICS_PATH, index=False)

    results = {
        "dataset": str(DATASET_PATH),
        "rows": int(len(df)),
        "safe_rows": int((y == 0).sum()),
        "vulnerable_rows": int((y == 1).sum()),
        "groups": int(df["group_id"].nunique()),
        "cross_validation": "StratifiedGroupKFold",
        "n_splits": N_SPLITS,
        "random_state": RANDOM_STATE,
        "grouping_strategy": (
            "Juliet: source_id prefix before '::'; other sources: "
            "source + source_id, with source_row/index fallback."
        ),
        "models": {"V1": v1_metrics, "V2": v2_metrics},
        "comparison": comparison.round(6).to_dict(orient="records"),
        "limitations": [
            "Rows sharing the derived group_id are kept in one fold.",
            "This does not guarantee separation of similar templates across different IDs.",
            "CWE categories are not used as groups in this experiment.",
            "Benchmark performance does not establish production-app performance.",
        ],
    }
    METRICS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\nCreated output files:")
    print(METRICS_PATH.name)
    print(PREDICTIONS_PATH.name)
    print(FOLD_METRICS_PATH.name)
    print("\nGrouped comparison completed.")


if __name__ == "__main__":
    main()
