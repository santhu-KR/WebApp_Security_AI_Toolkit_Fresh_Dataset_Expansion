"""
V3: CWE-family grouped evaluation for the Web Application Security AI Toolkit.

Purpose:
- Test whether the classifier generalizes to CWE identifiers excluded from training.
- Keep each CWE group entirely within one fold.
- Evaluate all sources together, while reporting per-source metrics.
- Do NOT train or overwrite a deployment model.

Outputs:
- cwe_grouped_v3_metrics.json
- cwe_grouped_v3_predictions.csv

This is a stricter CWE-held-out benchmark, not a production-security guarantee.
"""

from pathlib import Path
import json
import sys

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import StratifiedGroupKFold

from feature_model import build_model

HERE = Path(__file__).resolve().parent
DATASET = HERE / "dataset" / "processed" / "security_snippets_clean.csv"
METRICS_OUT = HERE / "cwe_grouped_v3_metrics.json"
PREDICTIONS_OUT = HERE / "cwe_grouped_v3_predictions.csv"

RANDOM_STATE = 42
N_SPLITS = 5


def normalize_cwe(value):
    """Return a stable CWE group key, or None when unavailable."""
    text = str(value).strip().upper()
    if not text or text in {"NAN", "NONE", "NULL", "UNKNOWN", "N/A"}:
        return None

    # Keep identifiers such as CWE-89. If multiple CWEs are listed,
    # use the first identifier so each row receives one deterministic group.
    import re
    match = re.search(r"CWE[\s_-]*0*(\d+)", text)
    if not match:
        return None
    return f"CWE-{int(match.group(1))}"


def metric_block(y_true, y_pred):
    return {
        "rows": int(len(y_true)),
        "safe": int((pd.Series(y_true) == 0).sum()),
        "vulnerable": int((pd.Series(y_true) == 1).sum()),
        "predicted_safe": int((pd.Series(y_pred) == 0).sum()),
        "predicted_vulnerable": int((pd.Series(y_pred) == 1).sum()),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "confusion_matrix_labels": ["safe", "vulnerable"],
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["safe", "vulnerable"],
            output_dict=True,
            zero_division=0,
        ),
    }


def main():
    if not DATASET.exists():
        raise SystemExit(f"Dataset not found: {DATASET}")

    df = pd.read_csv(DATASET)
    required = {"source", "source_id", "label", "cwe", "code"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["source"] = df["source"].fillna("unknown").astype(str)
    df["source_id"] = df["source_id"].fillna("").astype(str)
    df["code"] = df["code"].fillna("").astype(str).str.strip()
    df["label"] = df["label"].fillna("").astype(str).str.strip().str.lower()

    label_map = {"safe": 0, "vulnerable": 1}
    invalid_labels = sorted(set(df["label"]) - set(label_map))
    if invalid_labels:
        raise SystemExit(f"Unexpected labels: {invalid_labels}")

    df["target"] = df["label"].map(label_map).astype(int)
    df["cwe_group"] = df["cwe"].map(normalize_cwe)

    before = len(df)
    df = df[df["code"].ne("") & df["cwe_group"].notna()].copy()
    dropped = before - len(df)

    if df.empty:
        raise SystemExit("No rows remain after excluding blank code/missing CWE.")
    if df["target"].nunique() != 2:
        raise SystemExit("The retained dataset must contain both safe and vulnerable labels.")

    group_counts = df.groupby("cwe_group")["target"].nunique()
    if len(group_counts) < N_SPLITS:
        raise SystemExit(
            f"Need at least {N_SPLITS} distinct CWE groups; found {len(group_counts)}."
        )

    print("CWE-GROUPED V3 EVALUATION", flush=True)
    print(f"Rows retained: {len(df)}", flush=True)
    print(f"Rows excluded (blank code or missing/unrecognized CWE): {dropped}", flush=True)
    print(f"Distinct CWE groups: {df['cwe_group'].nunique()}", flush=True)
    print("Each CWE identifier will be assigned wholly to one fold.", flush=True)

    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_results = []
    prediction_frames = []
    all_group_tested = set()

    for fold_number, (train_pos, test_pos) in enumerate(
        splitter.split(
            X=df["code"],
            y=df["target"],
            groups=df["cwe_group"],
        ),
        start=1,
    ):
        train_df = df.iloc[train_pos].copy()
        test_df = df.iloc[test_pos].copy()

        train_groups = set(train_df["cwe_group"])
        test_groups = set(test_df["cwe_group"])
        overlap = train_groups & test_groups
        if overlap:
            raise RuntimeError(
                f"Fold {fold_number} has CWE leakage: {sorted(overlap)[:10]}"
            )

        if train_df["target"].nunique() != 2 or test_df["target"].nunique() != 2:
            raise RuntimeError(
                f"Fold {fold_number} does not contain both labels in train and test. "
                "Do not report incomplete folds as a valid binary evaluation."
            )

        print(
            f"\nFold {fold_number}/{N_SPLITS}: "
            f"train={len(train_df)}, test={len(test_df)}, "
            f"held-out CWE groups={len(test_groups)}",
            flush=True,
        )
        print(f"Training on {len(train_groups)} CWE groups; testing on held-out groups.",
              flush=True)

        model = build_model()
        model.fit(train_df["code"].tolist(), train_df["target"].astype(int).tolist())
        y_true = test_df["target"].astype(int).tolist()
        y_pred = model.predict(test_df["code"].tolist()).astype(int).tolist()

        overall = metric_block(y_true, y_pred)

        per_source = {}
        for source_name, source_test in test_df.groupby("source"):
            positions = source_test.index
            # Preserve the exact test order through index lookup.
            pred_by_index = dict(zip(test_df.index.tolist(), y_pred))
            source_pred = [pred_by_index[i] for i in positions]
            per_source[str(source_name)] = metric_block(
                source_test["target"].astype(int).tolist(), source_pred
            )

        fold_results.append({
            "fold": fold_number,
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "train_cwe_groups": int(len(train_groups)),
            "test_cwe_groups": int(len(test_groups)),
            "cwe_group_overlap": int(len(overlap)),
            "held_out_cwe_groups": sorted(test_groups),
            "overall": overall,
            "per_source": per_source,
        })

        fold_predictions = test_df[
            ["source", "source_id", "cwe", "cwe_group", "label"]
        ].copy()
        fold_predictions["fold"] = fold_number
        fold_predictions["predicted"] = [
            "vulnerable" if value else "safe" for value in y_pred
        ]
        fold_predictions["correct"] = (
            fold_predictions["label"] == fold_predictions["predicted"]
        )
        prediction_frames.append(fold_predictions)
        all_group_tested.update(test_groups)

        print(
            f"Fold {fold_number} result: accuracy={overall['accuracy']:.4f}, "
            f"F1={overall['f1']:.4f}, group_overlap={len(overlap)}",
            flush=True,
        )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(PREDICTIONS_OUT, index=False)

    # Every retained row should be tested exactly once by the 5-fold OOF process.
    if len(predictions) != len(df):
        raise RuntimeError(
            f"Out-of-fold prediction count mismatch: {len(predictions)} vs {len(df)}"
        )
    if predictions["source_id"].duplicated().all():
        print("Note: source_id values are not globally unique; row counts are used for coverage.",
              flush=True)

    y_true_all = predictions["label"].map(label_map).astype(int).tolist()
    y_pred_all = predictions["predicted"].map(label_map).astype(int).tolist()

    summary = {
        "evaluation": "5-fold StratifiedGroupKFold grouped by normalized CWE identifier",
        "purpose": "Measure performance on CWE identifiers absent from the corresponding training fold",
        "model": "Character TF-IDF + engineered security indicators + Logistic Regression",
        "dataset_file": DATASET.name,
        "random_state": RANDOM_STATE,
        "folds": N_SPLITS,
        "rows_before_filtering": int(before),
        "rows_used": int(len(df)),
        "rows_excluded_missing_cwe_or_blank_code": int(dropped),
        "distinct_cwe_groups": int(df["cwe_group"].nunique()),
        "source_counts": {
            str(k): int(v) for k, v in df["source"].value_counts().items()
        },
        "overall_out_of_fold": metric_block(y_true_all, y_pred_all),
        "fold_results": fold_results,
        "total_cwe_group_overlap": int(
            sum(item["cwe_group_overlap"] for item in fold_results)
        ),
        "predictions_file": PREDICTIONS_OUT.name,
        "warnings": [
            "CWE identifiers are used as groups; all rows sharing a normalized CWE stay in one fold.",
            "This is stricter than a random row split but does not guarantee independence from shared code templates or related CWE families.",
            "Rows without a recognizable CWE or with blank code are excluded and counted.",
            "The evaluation does not train or save a deployment model.",
            "Results are benchmark estimates, not production detection guarantees.",
        ],
    }

    METRICS_OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nOVERALL OUT-OF-FOLD RESULTS", flush=True)
    for key in ("accuracy", "precision", "recall", "f1"):
        print(f"{key.capitalize():10s}: {summary['overall_out_of_fold'][key]:.4f}", flush=True)
    print("Confusion matrix [safe, vulnerable]:",
          summary["overall_out_of_fold"]["confusion_matrix"], flush=True)
    print("Total CWE group overlap:", summary["total_cwe_group_overlap"], flush=True)
    print("Metrics saved:", METRICS_OUT.name, flush=True)
    print("Predictions saved:", PREDICTIONS_OUT.name, flush=True)
    print("No model or dataset files were modified.", flush=True)


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
