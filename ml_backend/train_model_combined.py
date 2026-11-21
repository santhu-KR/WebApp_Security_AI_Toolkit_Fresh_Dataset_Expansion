"""Train the binary vulnerability classifier using the original corpus plus OWASP Benchmark.

Evaluation protocol:
- Keep the original curated snippets in training.
- Hold out 20% of OWASP Benchmark Java test cases (stratified by label) for evaluation.
- Train on the remaining OWASP cases plus the original corpus.
- After reporting holdout metrics, refit the deployment model on all deduplicated examples.

This is test-case/source-corpus holdout, NOT project-separated evaluation: the imported
OWASP BenchmarkJava corpus is one project. Do not describe these metrics as production accuracy.
"""
from pathlib import Path
import csv
import json
import hashlib
from collections import Counter

import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from feature_model import build_model

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE / "dataset" / "security_snippets.csv"
OWASP = HERE / "dataset" / "external" / "owasp_benchmark_java.csv"
MODEL = HERE / "security_model.joblib"
METRICS = HERE / "training_metrics.json"


def read_csv(path, source_name):
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for i, row in enumerate(reader):
            code = (row.get("code") or "").strip()
            label = str(row.get("label", "")).strip()
            if not code or label not in {"0", "1"}:
                continue
            rows.append({
                "code": code,
                "label": int(label),
                "category": (row.get("category") or "unspecified").strip(),
                "source": (row.get("source") or source_name).strip(),
                "source_id": (row.get("source_id") or f"{source_name}-{i}").strip(),
                "cwe": (row.get("cwe") or "").strip(),
            })
        return rows


def fingerprint(code):
    normalized = " ".join(code.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def main():
    original = read_csv(ORIGINAL, "original_curated")
    benchmark = read_csv(OWASP, "OWASP-Benchmark-Java")
    if not original:
        raise SystemExit(f"No valid rows found in {ORIGINAL}")
    if not benchmark:
        raise SystemExit(
            f"No valid OWASP rows found in {OWASP}. Run import_owasp_benchmark.py first."
        )

    # Remove exact duplicates within each source and remove OWASP examples that
    # duplicate an original snippet, so identical code cannot cross the split.
    original_seen = set()
    clean_original = []
    for row in original:
        key = fingerprint(row["code"])
        if key not in original_seen:
            original_seen.add(key)
            clean_original.append(row)

    benchmark_seen = set()
    clean_benchmark = []
    conflicting = 0
    for row in benchmark:
        key = fingerprint(row["code"])
        if key in original_seen or key in benchmark_seen:
            continue
        benchmark_seen.add(key)
        clean_benchmark.append(row)

    # The holdout is composed only of OWASP test cases, not random rows from
    # the original synthetic/curated corpus. OWASP's source_id is the testcase ID.
    labels = [r["label"] for r in clean_benchmark]
    if len(clean_benchmark) < 10 or len(set(labels)) < 2:
        raise SystemExit("Not enough OWASP examples from both labels for a meaningful holdout.")

    benchmark_train, benchmark_test = train_test_split(
        clean_benchmark,
        test_size=0.20,
        random_state=42,
        stratify=labels,
    )

    train_rows = clean_original + benchmark_train
    X_train = [r["code"] for r in train_rows]
    y_train = [r["label"] for r in train_rows]
    X_test = [r["code"] for r in benchmark_test]
    y_test = [r["label"] for r in benchmark_test]

    evaluation_model = build_model()
    evaluation_model.fit(X_train, y_train)
    predictions = evaluation_model.predict(X_test)

    all_rows = clean_original + clean_benchmark
    all_labels = [r["label"] for r in all_rows]
    category_counts = Counter(r["category"] for r in all_rows)
    source_counts = Counter(r["source"] for r in all_rows)
    test_category_counts = Counter(r["category"] for r in benchmark_test)

    metrics = {
        "model": "Character TF-IDF + engineered security indicators + Logistic Regression",
        "original_examples_valid": len(original),
        "original_examples_after_deduplication": len(clean_original),
        "owasp_examples_imported": len(benchmark),
        "owasp_examples_after_deduplication": len(clean_benchmark),
        "combined_examples_after_deduplication": len(all_rows),
        "duplicate_owasp_examples_removed": len(benchmark) - len(clean_benchmark),
        "class_counts_all": {"safe": all_labels.count(0), "vulnerable": all_labels.count(1)},
        "source_counts": dict(source_counts),
        "category_counts": dict(category_counts),
        "evaluation_protocol": "Train on all original curated examples plus 80% of OWASP Benchmark cases; test on the held-out 20% of OWASP Benchmark cases.",
        "evaluation_split": "stratified OWASP test-case holdout (random_state=42); not project-separated because only one OWASP Java project corpus was imported",
        "train_rows": len(train_rows),
        "test_rows": len(benchmark_test),
        "test_class_counts": {"safe": y_test.count(0), "vulnerable": y_test.count(1)},
        "test_category_counts": dict(test_category_counts),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "confusion_matrix_labels": ["safe", "vulnerable"],
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_test, predictions, labels=[0, 1],
            target_names=["safe", "vulnerable"], output_dict=True, zero_division=0
        ),
        "evaluation_warning": "These metrics measure performance on held-out OWASP Benchmark Java test cases, not unseen independent projects or production applications. The original corpus may be curated/synthetic. OWASP categories do not cover all 18 toolkit categories. Review per-category performance and false negatives.",
    }

    # Refit the model for local inference using every deduplicated labelled example.
    deployment_model = build_model()
    deployment_model.fit([r["code"] for r in all_rows], all_labels)
    joblib.dump(deployment_model, MODEL)
    METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("DATASET SUMMARY")
    print(f"Original valid examples: {len(original)}")
    print(f"Original after deduplication: {len(clean_original)}")
    print(f"OWASP imported examples: {len(benchmark)}")
    print(f"OWASP after deduplication: {len(clean_benchmark)}")
    print(f"Combined unique examples: {len(all_rows)}")
    print("\nOWASP labels:", dict(Counter(r["label"] for r in clean_benchmark)))
    print("\nCATEGORY COUNTS:")
    for category, count in sorted(category_counts.items()):
        print(f"{category}: {count}")
    print("\nHOLDOUT EVALUATION (OWASP test cases only)")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1']:.4f}")
    print("Confusion matrix [safe, vulnerable]:")
    print(metrics["confusion_matrix"])
    print("\nSaved model:", MODEL)
    print("Saved metrics:", METRICS)
    print("\nWARNING:", metrics["evaluation_warning"])


if __name__ == "__main__":
    main()
