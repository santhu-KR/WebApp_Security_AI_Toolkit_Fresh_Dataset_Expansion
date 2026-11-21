"""Train the binary vulnerability classifier on curated + OWASP Benchmark data.

Evaluation: 20% stratified holdout of OWASP Benchmark test cases. This is a
held-out test-case evaluation within one benchmark project, NOT project-separated
or production evaluation. After evaluation, refit a deployment model on all
unique examples.
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
JULIET = HERE / "dataset" / "external" / "nist_juliet_java.csv"
MODEL = HERE / "security_model.joblib"
METRICS = HERE / "training_metrics.json"


def normalize_label(value):
    value = str(value or "").strip().lower()
    if value in {"1", "vulnerable", "true", "unsafe", "positive"}:
        return 1
    if value in {"0", "safe", "false", "not_vulnerable", "negative"}:
        return 0
    return None


def read_csv(path, source_name):
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            code = (row.get("code") or row.get("source_code") or "").strip()
            label = normalize_label(row.get("label"))
            if not code or label is None:
                continue
            rows.append({
                "code": code,
                "label": label,
                "category": (row.get("category") or "unspecified").strip(),
                "source": (row.get("source") or source_name).strip(),
                "source_id": (row.get("source_id") or f"{source_name}-{i}").strip(),
                "cwe": (row.get("cwe") or "").strip(),
            })
    return rows


def fingerprint(code):
    normalized = " ".join(code.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def deduplicate(rows, seen):
    clean = []
    for row in rows:
        key = fingerprint(row["code"])
        if key in seen:
            continue
        seen.add(key)
        clean.append(row)
    return clean


def main():
    original_raw = read_csv(ORIGINAL, "original_curated")
    benchmark_raw = read_csv(OWASP, "OWASP-Benchmark-Java")
    juliet_raw = read_csv(JULIET, "NIST-SARD-Juliet-Java-1.3")
    if not original_raw:
        raise SystemExit(f"No valid original rows found in {ORIGINAL}")
    if not benchmark_raw:
        raise SystemExit(f"No valid OWASP rows found in {OWASP}. Run the importer first.")

    seen = set()
    original = deduplicate(original_raw, seen)
    juliet = deduplicate(juliet_raw, seen)
    benchmark = deduplicate(benchmark_raw, seen)
    if len({r["label"] for r in benchmark}) < 2:
        raise SystemExit("OWASP data must contain both safe and vulnerable labels.")

    benchmark_train, benchmark_test = train_test_split(
        benchmark,
        test_size=0.20,
        random_state=42,
        stratify=[r["label"] for r in benchmark],
    )
    train_rows = original + juliet + benchmark_train
    X_train = [r["code"] for r in train_rows]
    y_train = [r["label"] for r in train_rows]
    X_test = [r["code"] for r in benchmark_test]
    y_test = [r["label"] for r in benchmark_test]

    evaluation_model = build_model()
    evaluation_model.fit(X_train, y_train)
    predictions = evaluation_model.predict(X_test)

    all_rows = original + juliet + benchmark
    all_labels = [r["label"] for r in all_rows]
    category_counts = Counter(r["category"] for r in all_rows)
    source_counts = Counter(r["source"] for r in all_rows)
    report = classification_report(
        y_test, predictions, labels=[0, 1],
        target_names=["safe", "vulnerable"], output_dict=True, zero_division=0,
    )
    metrics = {
        "model": "Character TF-IDF + engineered security indicators + Logistic Regression",
        "original_examples_valid": len(original_raw),
        "original_examples_after_deduplication": len(original),
        "nist_juliet_examples_imported": len(juliet_raw),
        "nist_juliet_examples_after_deduplication": len(juliet),
        "owasp_examples_imported": len(benchmark_raw),
        "owasp_examples_after_deduplication": len(benchmark),
        "combined_examples_after_deduplication": len(all_rows),
        "duplicates_removed_total": len(original_raw) + len(benchmark_raw) - len(all_rows),
        "class_counts_all": {"safe": all_labels.count(0), "vulnerable": all_labels.count(1)},
        "source_counts": dict(source_counts),
        "category_counts": dict(category_counts),
        "evaluation_protocol": "Train on all unique original examples plus 80% of OWASP Benchmark test cases; evaluate on held-out 20% of OWASP Benchmark test cases.",
        "evaluation_split": "stratified OWASP test-case holdout; one benchmark project only; not project-separated",
        "train_rows": len(train_rows),
        "test_rows": len(benchmark_test),
        "test_class_counts": {"safe": y_test.count(0), "vulnerable": y_test.count(1)},
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "confusion_matrix_labels": ["safe", "vulnerable"],
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        "classification_report": report,
        "evaluation_warning": "These metrics measure held-out OWASP Benchmark Java test cases, not unseen independent projects or production applications. The original corpus may be curated/synthetic. OWASP categories do not cover all 18 toolkit categories.",
    }

    # Refit for local inference on all unique labelled data after recording holdout metrics.
    deployment_model = build_model()
    deployment_model.fit([r["code"] for r in all_rows], all_labels)
    joblib.dump(deployment_model, MODEL)
    METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("DATASET SUMMARY")
    print(f"Original valid examples: {len(original_raw)}")
    print(f"Original after deduplication: {len(original)}")
    print(f"NIST Juliet imported examples: {len(juliet_raw)}")
    print(f"NIST Juliet after deduplication: {len(juliet)}")
    print(f"OWASP imported examples: {len(benchmark_raw)}")
    print(f"OWASP after deduplication: {len(benchmark)}")
    print(f"Combined unique examples: {len(all_rows)}")
    print("Combined labels:", {"safe": all_labels.count(0), "vulnerable": all_labels.count(1)})
    print("\nCATEGORY COUNTS:")
    for category, count in sorted(category_counts.items()):
        print(f"{category}: {count}")
    print("\nHOLDOUT EVALUATION (OWASP test cases only)")
    print(f"Test examples: {len(benchmark_test)}")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1']:.4f}")
    print("Confusion matrix [safe, vulnerable]:", metrics["confusion_matrix"])
    print("\nSaved model:", MODEL)
    print("Saved metrics:", METRICS)
    print("\nWARNING:", metrics["evaluation_warning"])


if __name__ == "__main__":
    main()
