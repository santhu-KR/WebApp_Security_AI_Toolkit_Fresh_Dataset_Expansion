from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import pandas as pd
import joblib

from feature_model import build_model


# --------------------------------------------------
# PATH CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = (
    BASE_DIR
    / "dataset"
    / "processed"
    / "security_snippets_clean.csv"
)

MODEL_DIR = BASE_DIR / "models"
EVALUATION_DIR = BASE_DIR / "evaluation"

MODEL_PATH = MODEL_DIR / "security_model_v1.joblib"

METADATA_PATH = (
    EVALUATION_DIR
    / "final_v1_training_metadata.json"
)


# --------------------------------------------------
# LABEL NORMALIZATION
# --------------------------------------------------

def normalize_label(value):

    value = str(value).strip().lower()

    if value in {"1", "vulnerable", "unsafe", "true"}:
        return 1

    if value in {"0", "safe", "false", "not_vulnerable"}:
        return 0

    return None


# --------------------------------------------------
# CODE FINGERPRINT
# --------------------------------------------------

def create_fingerprint(code):

    normalized = " ".join(str(code).split())

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


# --------------------------------------------------
# LOAD AND VALIDATE DATASET
# --------------------------------------------------

def load_dataset():

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    required_columns = {
        "code",
        "label",
        "source",
        "category_normalized"
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing dataset columns: {missing}"
        )

    print("\nINITIAL DATASET")
    print("------------------------------")
    print("Total records:", len(df))

    # Remove empty code
    df["code"] = df["code"].fillna("").astype(str)

    df = df[df["code"].str.strip() != ""].copy()

    # Normalize labels
    df["label_numeric"] = df["label"].apply(
        normalize_label
    )

    df = df[df["label_numeric"].notna()].copy()

    df["label_numeric"] = df["label_numeric"].astype(int)

    # Generate fingerprints from actual code
    df["calculated_fingerprint"] = df["code"].apply(
        create_fingerprint
    )

    # Detect duplicate code with conflicting labels
    label_counts = df.groupby(
        "calculated_fingerprint"
    )["label_numeric"].nunique()

    conflicting_hashes = set(
        label_counts[label_counts > 1].index
    )

    conflicting_count = len(conflicting_hashes)

    if conflicting_hashes:

        print(
            "Conflicting duplicate code groups excluded:",
            conflicting_count
        )

        df = df[
            ~df["calculated_fingerprint"].isin(
                conflicting_hashes
            )
        ].copy()

    # Remove exact duplicates
    before_deduplication = len(df)

    df = df.drop_duplicates(
        subset="calculated_fingerprint",
        keep="first"
    ).copy()

    duplicates_removed = before_deduplication - len(df)

    if df["label_numeric"].nunique() != 2:
        raise ValueError(
            "Dataset must contain both safe and vulnerable examples."
        )

    print("Empty/invalid records removed:", 14496 - before_deduplication)
    print("Duplicate records removed:", duplicates_removed)
    print("Final unique records:", len(df))

    print("\nCLASS DISTRIBUTION")
    print("------------------------------")

    print(
        df["label_numeric"]
        .value_counts()
        .rename(index={
            0: "Safe",
            1: "Vulnerable"
        })
    )

    print("\nSOURCE DISTRIBUTION")
    print("------------------------------")

    print(df["source"].value_counts())

    return df


# --------------------------------------------------
# FINAL V1 TRAINING
# --------------------------------------------------

def main():

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    EVALUATION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = load_dataset()

    X = df["code"].tolist()

    y = df["label_numeric"].tolist()

    print("\nBUILDING V1 MODEL")
    print("------------------------------")

    model = build_model()

    print("Training started...")

    model.fit(X, y)

    print("Training completed.")

    # Save trained model
    joblib.dump(model, MODEL_PATH)

    print("\nMODEL SAVED")
    print("------------------------------")
    print(MODEL_PATH)

    # Save training metadata
    metadata = {

        "model_name": "Feature Model V1",

        "model_type": (
            "Character TF-IDF + engineered security "
            "features + Logistic Regression"
        ),

        "dataset": str(DATASET_PATH),

        "training_date_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "training_samples": len(df),

        "class_distribution": {

            "safe": int(
                (df["label_numeric"] == 0).sum()
            ),

            "vulnerable": int(
                (df["label_numeric"] == 1).sum()
            )
        },

        "source_distribution": {

            str(key): int(value)

            for key, value in
            df["source"].value_counts().items()
        },

        "category_distribution": {

            str(key): int(value)

            for key, value in
            df["category_normalized"]
            .fillna("Uncategorized")
            .value_counts()
            .items()
        },

        "evaluation_reference": {

            "method": "Grouped 5-fold cross-validation",

            "accuracy": 0.9639,

            "precision": 0.9567,

            "recall": 0.9688,

            "f1_score": 0.9627,

            "note": (
                "These are previously measured out-of-fold "
                "metrics, not training-set performance."
            )
        },

        "model_file": str(MODEL_PATH)

    }

    METADATA_PATH.write_text(
        json.dumps(metadata, indent=4),
        encoding="utf-8"
    )

    print("\nMETADATA SAVED")
    print("------------------------------")
    print(METADATA_PATH)

    print("\nFINAL V1 TRAINING COMPLETED")


if __name__ == "__main__":
    main()