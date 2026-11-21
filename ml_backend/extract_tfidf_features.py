import pandas as pd
import numpy as np
import json

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).parent

ANALYSIS_DIR = (
    BASE_DIR
    / "dataset"
    / "external_sources"
    / "osv_collector"
    / "analysis"
)

INPUT_FILE = ANALYSIS_DIR / "osv_ml_cleaned_dataset.csv"

FEATURES_FILE = ANALYSIS_DIR / "osv_tfidf_features.npz"

LABELS_FILE = ANALYSIS_DIR / "osv_ml_labels.csv"

VOCABULARY_FILE = ANALYSIS_DIR / "osv_tfidf_vocabulary.json"

REPORT_FILE = ANALYSIS_DIR / "osv_tfidf_report.json"


# ============================================================
# PARAMETERS
# ============================================================

MAX_FEATURES = 10000

NGRAM_RANGE = (1, 2)

MIN_DF = 2

MAX_DF = 0.95

TEST_SIZE = 0.20

RANDOM_STATE = 42


# ============================================================
# MAIN
# ============================================================

print("=" * 70)

print("OSV TF-IDF FEATURE EXTRACTION")

print("=" * 70)


# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

if not INPUT_FILE.exists():

    print("\nERROR: Cleaned dataset not found!")

    print(INPUT_FILE)

    exit()


df = pd.read_csv(INPUT_FILE)

print("\nDataset loaded")

print("Total records:", len(df))


# ------------------------------------------------------------
# 2. VALIDATE COLUMNS
# ------------------------------------------------------------

required_columns = [
    "cleaned_text",
    "normalized_category"
]

for column in required_columns:

    if column not in df.columns:

        print(f"\nERROR: Missing column: {column}")

        print("Available columns:")

        print(list(df.columns))

        exit()


# ------------------------------------------------------------
# 3. REMOVE EMPTY TEXT
# ------------------------------------------------------------

df["cleaned_text"] = (
    df["cleaned_text"]
    .fillna("")
    .astype(str)
)

df["normalized_category"] = (
    df["normalized_category"]
    .fillna("")
    .astype(str)
)

df = df[
    (df["cleaned_text"].str.strip() != "")
    &
    (df["normalized_category"].str.strip() != "")
].copy()


print("\nRecords after validation:", len(df))


# ------------------------------------------------------------
# 4. PREPARE TEXT AND LABELS
# ------------------------------------------------------------

X_text = df["cleaned_text"]

y = df["normalized_category"]


print("\nNumber of vulnerability categories:")

print(y.nunique())


print("\nCategory distribution:")

print(y.value_counts())


# ------------------------------------------------------------
# 5. TF-IDF VECTORISATION
# ------------------------------------------------------------

print("\nApplying TF-IDF...")


vectorizer = TfidfVectorizer(

    max_features=MAX_FEATURES,

    ngram_range=NGRAM_RANGE,

    min_df=MIN_DF,

    max_df=MAX_DF,

    sublinear_tf=True,

    strip_accents="unicode",

    dtype=np.float32

)


X = vectorizer.fit_transform(X_text)


# ------------------------------------------------------------
# 6. FEATURE INFORMATION
# ------------------------------------------------------------

feature_names = vectorizer.get_feature_names_out()

print("\nTF-IDF extraction completed!")

print("Number of records:", X.shape[0])

print("Number of features:", X.shape[1])

print("Matrix shape:", X.shape)

print("Matrix sparsity:", round(
    100 * (1 - X.nnz / (X.shape[0] * X.shape[1])),
    2
), "%")


# ------------------------------------------------------------
# 7. SAVE TF-IDF FEATURES
# ------------------------------------------------------------

print("\nSaving TF-IDF feature matrix...")


from scipy.sparse import save_npz

save_npz(

    FEATURES_FILE,

    X

)


# ------------------------------------------------------------
# 8. SAVE LABELS
# ------------------------------------------------------------

labels_df = pd.DataFrame({

    "text": X_text,

    "category": y

})


labels_df.to_csv(

    LABELS_FILE,

    index=False,

    encoding="utf-8"

)


# ------------------------------------------------------------
# 9. SAVE VOCABULARY
# ------------------------------------------------------------

vocabulary = {

    word: int(index)

    for word, index in vectorizer.vocabulary_.items()

}


with open(

    VOCABULARY_FILE,

    "w",

    encoding="utf-8"

) as file:

    json.dump(

        vocabulary,

        file,

        indent=4,

        ensure_ascii=False

    )


# ------------------------------------------------------------
# 10. TRAIN / TEST SPLIT INFORMATION
# ------------------------------------------------------------

print("\nPreparing train/test split...")


try:

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE,

        stratify=y

    )

    split_success = True

except ValueError as error:

    print("\nStratified split could not be performed.")

    print("Reason:", error)

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE

    )

    split_success = False


print("\nTraining records:", X_train.shape[0])

print("Testing records:", X_test.shape[0])


# ------------------------------------------------------------
# 11. SAVE TRAIN / TEST DATA
# ------------------------------------------------------------

from scipy.sparse import save_npz


save_npz(

    ANALYSIS_DIR / "osv_X_train.npz",

    X_train

)


save_npz(

    ANALYSIS_DIR / "osv_X_test.npz",

    X_test

)


pd.DataFrame({

    "category": y_train

}).to_csv(

    ANALYSIS_DIR / "osv_y_train.csv",

    index=False

)


pd.DataFrame({

    "category": y_test

}).to_csv(

    ANALYSIS_DIR / "osv_y_test.csv",

    index=False

)


# ------------------------------------------------------------
# 12. GENERATE REPORT
# ------------------------------------------------------------

report = {

    "dataset_name": "OSV Vulnerability TF-IDF Dataset",

    "total_records": int(X.shape[0]),

    "total_features": int(X.shape[1]),

    "number_of_categories": int(y.nunique()),

    "max_features": MAX_FEATURES,

    "ngram_range": list(NGRAM_RANGE),

    "min_df": MIN_DF,

    "max_df": MAX_DF,

    "test_size": TEST_SIZE,

    "random_state": RANDOM_STATE,

    "training_records": int(X_train.shape[0]),

    "testing_records": int(X_test.shape[0]),

    "stratified_split": split_success,

    "matrix_shape": list(X.shape),

    "matrix_sparsity_percentage": round(

        100 * (
            1 - X.nnz / (X.shape[0] * X.shape[1])
        ),

        2

    ),

    "category_distribution": {

        str(category): int(count)

        for category, count in y.value_counts().items()

    }

}


with open(

    REPORT_FILE,

    "w",

    encoding="utf-8"

) as file:

    json.dump(

        report,

        file,

        indent=4,

        ensure_ascii=False

    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)

print("TF-IDF FEATURE EXTRACTION COMPLETED")

print("=" * 70)


print("\nTotal samples:", X.shape[0])

print("Total TF-IDF features:", X.shape[1])

print("Categories:", y.nunique())

print("Training samples:", X_train.shape[0])

print("Testing samples:", X_test.shape[0])


print("\nGenerated files:")

print("\nTF-IDF matrix:")

print(FEATURES_FILE)


print("\nLabels:")

print(LABELS_FILE)


print("\nVocabulary:")

print(VOCABULARY_FILE)


print("\nTraining features:")

print(ANALYSIS_DIR / "osv_X_train.npz")


print("\nTesting features:")

print(ANALYSIS_DIR / "osv_X_test.npz")


print("\nTraining labels:")

print(ANALYSIS_DIR / "osv_y_train.csv")


print("\nTesting labels:")

print(ANALYSIS_DIR / "osv_y_test.csv")


print("\nReport:")

print(REPORT_FILE)


print("\nDataset is ready for machine learning model training!")

print("=" * 70)