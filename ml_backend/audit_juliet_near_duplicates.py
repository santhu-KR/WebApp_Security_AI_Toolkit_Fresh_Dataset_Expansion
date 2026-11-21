"""
Near-duplicate similarity audit for the NIST Juliet grouped holdout.
Read-only: does not modify the dataset or any trained model.
"""

from pathlib import Path
import json

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).resolve().parent
DATASET = HERE / "dataset" / "processed" / "security_snippets_clean.csv"
REPORT_JSON = HERE / "juliet_near_duplicate_audit.json"
REPORT_CSV = HERE / "juliet_near_duplicate_audit.csv"

RANDOM_STATE = 42
N_SPLITS = 5
TOP_N = 25


def main():
    if not DATASET.exists():
        raise SystemExit(f"Dataset not found: {DATASET}")

    df = pd.read_csv(DATASET)
    required = {"source", "source_id", "label", "category_normalized", "code"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing dataset columns: {sorted(missing)}")

    juliet = df[
        df["source"].astype(str).str.contains("juliet", case=False, na=False)
    ].copy().reset_index(drop=True)

    juliet["code"] = juliet["code"].fillna("").astype(str)
    juliet["source_id"] = juliet["source_id"].fillna("").astype(str)
    juliet["group_id"] = juliet["source_id"].str.split("::", n=1).str[0]
    juliet["label"] = juliet["label"].astype(str).str.strip().str.lower()
    label_map = {"safe": 0, "vulnerable": 1}
    if not set(juliet["label"]).issubset(label_map):
        raise SystemExit("Unexpected labels in Juliet dataset.")
    juliet["target"] = juliet["label"].map(label_map)

    if juliet.empty or juliet["group_id"].eq("").any():
        raise SystemExit("Juliet data is empty or contains missing group IDs.")

    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    train_pos, test_pos = next(
        splitter.split(
            X=juliet["code"],
            y=juliet["target"],
            groups=juliet["group_id"],
        )
    )
    train = juliet.iloc[train_pos].copy()
    test = juliet.iloc[test_pos].copy()

    overlap = set(train["group_id"]) & set(test["group_id"])
    if overlap:
        raise RuntimeError(f"Group leakage detected: {len(overlap)} groups.")

    print("Juliet near-duplicate audit", flush=True)
    print(f"Training rows: {len(train)}", flush=True)
    print(f"Testing rows: {len(test)}", flush=True)
    print("Building character n-gram TF-IDF vectors...", flush=True)

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_features=50000,
        sublinear_tf=True,
        dtype="float32",
    )
    X_train = vectorizer.fit_transform(train["code"].tolist())
    X_test = vectorizer.transform(test["code"].tolist())

    print("Finding each test sample's most similar training sample...", flush=True)
    nn = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute", n_jobs=-1)
    nn.fit(X_train)
    distances, indices = nn.kneighbors(X_test, return_distance=True)
    similarities = 1.0 - distances.ravel()
    nearest_rows = train.iloc[indices.ravel()].reset_index(drop=True)

    audit = pd.DataFrame({
        "test_source_id": test["source_id"].values,
        "test_group_id": test["group_id"].values,
        "test_category": test["category_normalized"].values,
        "test_actual_label": test["label"].values,
        "nearest_train_source_id": nearest_rows["source_id"].values,
        "nearest_train_group_id": nearest_rows["group_id"].values,
        "nearest_train_category": nearest_rows["category_normalized"].values,
        "nearest_train_label": nearest_rows["label"].values,
        "nearest_similarity": similarities,
    }).sort_values("nearest_similarity", ascending=False)

    thresholds = [0.80, 0.90, 0.95, 0.99]
    threshold_counts = {
        str(t): int((similarities >= t).sum()) for t in thresholds
    }
    summary = {
        "method": "Character TF-IDF 3-5 grams; maximum cosine similarity to any Juliet training row",
        "split": "First StratifiedGroupKFold fold, n_splits=5, shuffle=True, random_state=42",
        "training_rows": int(len(train)),
        "testing_rows": int(len(test)),
        "group_overlap": int(len(overlap)),
        "vocabulary_features": int(X_train.shape[1]),
        "similarity_mean": round(float(similarities.mean()), 4),
        "similarity_median": round(float(pd.Series(similarities).median()), 4),
        "similarity_p90": round(float(pd.Series(similarities).quantile(0.90)), 4),
        "similarity_p95": round(float(pd.Series(similarities).quantile(0.95)), 4),
        "similarity_max": round(float(similarities.max()), 4),
        "test_samples_at_or_above_threshold": threshold_counts,
        "interpretation_note": (
            "Cosine similarity is a screening signal, not proof of leakage. "
            "High similarity can arise from shared Juliet templates or common syntax. "
            "Inspect the highest-scoring pairs before drawing conclusions."
        ),
    }

    audit.to_csv(REPORT_CSV, index=False)
    REPORT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nSIMILARITY SUMMARY")
    print(json.dumps(summary, indent=2))
    print(f"\nTop {min(TOP_N, len(audit))} most similar test/train pairs:")
    print(audit.head(TOP_N).to_string(index=False))
    print(f"\nDetailed CSV saved: {REPORT_CSV.name}")
    print(f"Summary JSON saved: {REPORT_JSON.name}")
    print("No dataset or model files were modified.")


if __name__ == "__main__":
    main()
