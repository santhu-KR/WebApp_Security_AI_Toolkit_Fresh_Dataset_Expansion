import os
import pandas as pd

# ============================================================
# V4 SAMPLE INSPECTOR
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ERROR_DIR = os.path.join(
    BASE_DIR,
    "results",
    "v4",
    "error_analysis"
)

OUTPUT_DIR = ERROR_DIR


# ============================================================
# LOAD ERROR DATA
# ============================================================

fp_file = os.path.join(
    ERROR_DIR,
    "v4_false_positives.csv"
)

fn_file = os.path.join(
    ERROR_DIR,
    "v4_false_negatives.csv"
)

fp = pd.read_csv(fp_file)

fn = pd.read_csv(fn_file)


# ============================================================
# DISPLAY FUNCTION
# ============================================================

def display_samples(data, title, count=20):

    print("\n")
    print("=" * 100)
    print(title)
    print("=" * 100)

    if data.empty:

        print("No samples found.")

        return

    samples = data.head(count)

    for index, row in samples.iterrows():

        print("\n" + "-" * 100)

        print("CWE:", row.get("cwe", "Unknown"))

        print("Actual label:", row.get("label"))

        print("Predicted label:", row.get("predicted_label"))

        print("Source:", row.get("source", "Unknown"))

        print("Source ID:", row.get("source_id", "Unknown"))

        print("\nCODE:")

        code = str(row.get("code", ""))

        print(code[:2000])

        print("-" * 100)


# ============================================================
# FALSE POSITIVE SQL SAMPLES
# ============================================================

fp_sql = fp[
    fp["cwe"].astype(str) == "CWE-89"
]

display_samples(
    fp_sql,
    "FALSE POSITIVE CWE-89 SQL SAMPLES",
    count=20
)


# ============================================================
# TRUE VULNERABLE SQL SAMPLES
# ============================================================

predictions_file = os.path.join(
    BASE_DIR,
    "results",
    "v4",
    "v4_grouped_predictions.csv"
)

all_data = pd.read_csv(predictions_file)

true_positive_sql = all_data[
    (all_data["cwe"].astype(str) == "CWE-89") &
    (all_data["label"] == 1) &
    (all_data["predicted_label"] == 1)
]


display_samples(
    true_positive_sql,
    "TRUE POSITIVE CWE-89 SQL SAMPLES",
    count=20
)


# ============================================================
# UNKNOWN FALSE NEGATIVE SAMPLES
# ============================================================

unknown_fn = fn[
    fn["cwe"].astype(str) == "UNKNOWN"
]


display_samples(
    unknown_fn,
    "UNKNOWN CWE FALSE NEGATIVE SAMPLES",
    count=20
)


# ============================================================
# EXPORT SAMPLES
# ============================================================

fp_sql.head(100).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "inspect_sql_false_positives.csv"
    ),
    index=False
)


true_positive_sql.head(100).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "inspect_sql_true_positives.csv"
    ),
    index=False
)


unknown_fn.head(100).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "inspect_unknown_false_negatives.csv"
    ),
    index=False
)


print("\n")
print("=" * 100)

print("SAMPLE INSPECTION COMPLETED")

print("=" * 100)

print("\nSaved files:")

print("inspect_sql_false_positives.csv")

print("inspect_sql_true_positives.csv")

print("inspect_unknown_false_negatives.csv")