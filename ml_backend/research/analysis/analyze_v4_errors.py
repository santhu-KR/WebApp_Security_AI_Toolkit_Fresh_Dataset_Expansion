import os
import pandas as pd

# ==========================================================
# V4 ERROR ANALYSIS
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(
    BASE_DIR,
    "results",
    "v4",
    "v4_grouped_predictions.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "v4",
    "error_analysis"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================================
# LOAD DATA
# ==========================================================

print("=" * 70)
print("V4 ERROR ANALYSIS")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print("\nDataset loaded:", df.shape)


# ==========================================================
# VALIDATE COLUMNS
# ==========================================================

required_columns = [
    "label",
    "predicted_label",
    "cwe",
    "code"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )


# ==========================================================
# REMOVE ROWS WITHOUT PREDICTIONS
# ==========================================================

df = df.dropna(
    subset=[
        "label",
        "predicted_label"
    ]
).copy()

df["label"] = df["label"].astype(int)

df["predicted_label"] = (
    df["predicted_label"].astype(int)
)


# ==========================================================
# IDENTIFY ERRORS
# ==========================================================

df["prediction_correct"] = (
    df["label"] == df["predicted_label"]
)


# FALSE POSITIVE
# Actual SAFE but predicted VULNERABLE

false_positives = df[
    (df["label"] == 0) &
    (df["predicted_label"] == 1)
].copy()


# FALSE NEGATIVE
# Actual VULNERABLE but predicted SAFE

false_negatives = df[
    (df["label"] == 1) &
    (df["predicted_label"] == 0)
].copy()


# TRUE POSITIVE

true_positives = df[
    (df["label"] == 1) &
    (df["predicted_label"] == 1)
].copy()


# TRUE NEGATIVE

true_negatives = df[
    (df["label"] == 0) &
    (df["predicted_label"] == 0)
].copy()


# ==========================================================
# SUMMARY
# ==========================================================

print("\n" + "=" * 70)
print("ERROR SUMMARY")
print("=" * 70)

print("Total samples:", len(df))

print("Correct predictions:", int(df["prediction_correct"].sum()))

print("False Positives:", len(false_positives))

print("False Negatives:", len(false_negatives))

print("True Positives:", len(true_positives))

print("True Negatives:", len(true_negatives))


# ==========================================================
# CWE ERROR ANALYSIS
# ==========================================================

print("\n" + "=" * 70)
print("FALSE POSITIVES BY CWE")
print("=" * 70)

fp_by_cwe = (
    false_positives["cwe"]
    .value_counts()
)

print(fp_by_cwe)


print("\n" + "=" * 70)
print("FALSE NEGATIVES BY CWE")
print("=" * 70)

fn_by_cwe = (
    false_negatives["cwe"]
    .value_counts()
)

print(fn_by_cwe)


# ==========================================================
# CWE PERFORMANCE ANALYSIS
# ==========================================================

print("\n" + "=" * 70)
print("PERFORMANCE BY CWE")
print("=" * 70)

cwe_results = []

for cwe, group in df.groupby("cwe"):

    actual = group["label"]

    predicted = group["predicted_label"]

    tp = (
        (actual == 1) &
        (predicted == 1)
    ).sum()

    tn = (
        (actual == 0) &
        (predicted == 0)
    ).sum()

    fp = (
        (actual == 0) &
        (predicted == 1)
    ).sum()

    fn = (
        (actual == 1) &
        (predicted == 0)
    ).sum()

    total = len(group)

    accuracy = (
        (tp + tn) / total
        if total > 0
        else 0
    )

    vulnerable_count = (
        actual == 1
    ).sum()

    safe_count = (
        actual == 0
    ).sum()

    recall = (
        tp / vulnerable_count
        if vulnerable_count > 0
        else 0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    cwe_results.append({

        "CWE": cwe,

        "Total_Samples": total,

        "Safe_Samples": safe_count,

        "Vulnerable_Samples": vulnerable_count,

        "True_Positive": int(tp),

        "True_Negative": int(tn),

        "False_Positive": int(fp),

        "False_Negative": int(fn),

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall

    })


cwe_analysis = pd.DataFrame(
    cwe_results
)


cwe_analysis = cwe_analysis.sort_values(
    "Recall"
)


print(
    cwe_analysis.to_string(
        index=False
    )
)


# ==========================================================
# SAVE ERROR FILES
# ==========================================================

print("\nSaving error analysis files...")


false_positives.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "v4_false_positives.csv"
    ),
    index=False
)


false_negatives.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "v4_false_negatives.csv"
    ),
    index=False
)


true_positives.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "v4_true_positives.csv"
    ),
    index=False
)


true_negatives.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "v4_true_negatives.csv"
    ),
    index=False
)


cwe_analysis.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "v4_cwe_performance.csv"
    ),
    index=False
)


# ==========================================================
# CREATE TOP ERROR REPORT
# ==========================================================

top_fp = (
    false_positives
    .groupby("cwe")
    .size()
    .sort_values(ascending=False)
    .head(10)
)


top_fn = (
    false_negatives
    .groupby("cwe")
    .size()
    .sort_values(ascending=False)
    .head(10)
)


report_file = os.path.join(
    OUTPUT_DIR,
    "v4_error_summary.txt"
)


with open(
    report_file,
    "w",
    encoding="utf-8"
) as file:

    file.write("V4 ERROR ANALYSIS\n")

    file.write("=" * 60 + "\n\n")

    file.write(
        f"Total samples: {len(df)}\n"
    )

    file.write(
        f"False Positives: {len(false_positives)}\n"
    )

    file.write(
        f"False Negatives: {len(false_negatives)}\n"
    )

    file.write(
        f"True Positives: {len(true_positives)}\n"
    )

    file.write(
        f"True Negatives: {len(true_negatives)}\n\n"
    )


    file.write(
        "TOP FALSE POSITIVE CWEs\n"
    )

    file.write(
        "-" * 60 + "\n"
    )

    file.write(
        top_fp.to_string()
    )

    file.write("\n\n")


    file.write(
        "TOP FALSE NEGATIVE CWEs\n"
    )

    file.write(
        "-" * 60 + "\n"
    )

    file.write(
        top_fn.to_string()
    )

    file.write("\n\n")


    file.write(
        "CWE PERFORMANCE\n"
    )

    file.write(
        "-" * 60 + "\n"
    )

    file.write(
        cwe_analysis.to_string(
            index=False
        )
    )


# ==========================================================
# COMPLETION
# ==========================================================

print("\n" + "=" * 70)

print("ERROR ANALYSIS COMPLETED")

print("=" * 70)

print("\nOutput directory:")

print(OUTPUT_DIR)

print("\nGenerated files:")

print("1. v4_false_positives.csv")

print("2. v4_false_negatives.csv")

print("3. v4_true_positives.csv")

print("4. v4_true_negatives.csv")

print("5. v4_cwe_performance.csv")

print("6. v4_error_summary.txt")

print("\nDone.")