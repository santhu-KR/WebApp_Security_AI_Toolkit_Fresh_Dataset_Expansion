import pandas as pd

DATASET_PATH = "dataset/processed/security_snippets_clean.csv"

df = pd.read_csv(DATASET_PATH)

print("=" * 60)
print("V6 DATASET ANALYSIS")
print("=" * 60)

print("\n1. LABEL DISTRIBUTION")
print(df["label"].value_counts(dropna=False))

print("\n2. CWE-89 LABEL DISTRIBUTION")
cwe89 = df[df["cwe"] == "CWE-89"]
print(cwe89["label"].value_counts(dropna=False))

print("\n3. CWE-89 SAMPLE ROWS")
print(
    cwe89[
        ["cwe", "label", "category_original", "category_normalized"]
    ].head(10).to_string()
)

print("\n4. ALL CWE LABEL DISTRIBUTIONS")

result = df.groupby(["cwe", "label"]).size().unstack(fill_value=0)

print(result.to_string())

print("\n5. DATASET SUMMARY")
print("Total samples:", len(df))
print("Unique fingerprints:", df["fingerprint"].nunique())
print("Duplicate fingerprints:", df["fingerprint"].duplicated().sum())

print("\n6. LABEL DATA TYPES")
print(df["label"].dtype)

print("\nV6 DATASET ANALYSIS COMPLETED")