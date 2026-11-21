# ============================================================
# WEB APPLICATION SECURITY AI TOOLKIT - V9
# Contextual Security Features + Threshold Optimization
# ============================================================

import os
import re
import time
import pickle
import warnings

import numpy as np
import pandas as pd

from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_DATA = os.path.join(
    BASE_DIR,
    "dataset",
    "v7",
    "balanced",
    "train_v7_balanced.csv"
)

TEST_DATA = os.path.join(
    BASE_DIR,
    "dataset",
    "v7",
    "balanced",
    "test_v7.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models", "v9")

RESULT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "v9",
    "model_training"
)

ERROR_DIR = os.path.join(
    BASE_DIR,
    "results",
    "v9",
    "error_analysis"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(ERROR_DIR, exist_ok=True)

RANDOM_STATE = 42

VALIDATION_SIZE = 0.20

MIN_PRECISION = 0.90

THRESHOLDS = np.arange(0.20, 0.81, 0.01)


# ============================================================
# LABEL CONVERSION
# ============================================================

def convert_label(value):

    value = str(value).strip().lower()

    if value in [
        "vulnerable",
        "1",
        "1.0",
        "true",
        "unsafe"
    ]:
        return 1

    elif value in [
        "safe",
        "0",
        "0.0",
        "false",
        "benign",
        "non-vulnerable",
        "non_vulnerable"
    ]:
        return 0

    else:
        raise ValueError(
            f"Unknown label found in dataset: {value}"
        )


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("WEB APPLICATION SECURITY AI TOOLKIT - V9")
print("=" * 70)

print("\nLoading datasets...")

train_df = pd.read_csv(TRAIN_DATA)
test_df = pd.read_csv(TEST_DATA)

print("Training dataset:", train_df.shape)
print("Testing dataset:", test_df.shape)


# ============================================================
# VALIDATE DATA
# ============================================================

required_columns = [
    "code_normalized",
    "label"
]

for column in required_columns:

    if column not in train_df.columns:

        raise ValueError(
            f"Missing column in training dataset: {column}"
        )

    if column not in test_df.columns:

        raise ValueError(
            f"Missing column in testing dataset: {column}"
        )


# Remove missing values

train_df = train_df.dropna(
    subset=["code_normalized", "label"]
).copy()

test_df = test_df.dropna(
    subset=["code_normalized", "label"]
).copy()


# Convert code to strings

train_df["code_normalized"] = (
    train_df["code_normalized"].astype(str)
)

test_df["code_normalized"] = (
    test_df["code_normalized"].astype(str)
)


# ============================================================
# FIXED LABEL CONVERSION
# ============================================================

print("\nConverting labels...")

train_df["label"] = train_df["label"].apply(
    convert_label
)

test_df["label"] = test_df["label"].apply(
    convert_label
)


print("\nTraining labels:")

print(train_df["label"].value_counts())


print("\nTesting labels:")

print(test_df["label"].value_counts())


# ============================================================
# DATA PREPARATION
# ============================================================

X = train_df["code_normalized"]

y = train_df["label"]

X_test = test_df["code_normalized"]

y_test = test_df["label"]


# ============================================================
# TRAIN VALIDATION SPLIT
# ============================================================

print("\nCreating train-validation split...")

X_train, X_validation, y_train, y_validation = train_test_split(
    X,
    y,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)


print("\nTraining samples:", len(X_train))

print("Validation samples:", len(X_validation))

print("Testing samples:", len(X_test))


# ============================================================
# SECURITY PATTERNS
# ============================================================

SECURITY_PATTERNS = {

    "sql_keywords": [
        r"\bSELECT\b",
        r"\bINSERT\b",
        r"\bUPDATE\b",
        r"\bDELETE\b",
        r"\bDROP\b",
        r"\bUNION\b",
        r"\bWHERE\b",
        r"\bEXEC\b"
    ],

    "sql_concatenation": [
        r"\+\s*.*sql",
        r"sql\s*\+\s*",
        r"query\s*\+\s*",
        r"execute\s*\(.*\+",
        r"prepareStatement\s*\(.*\+"
    ],

    "sql_parameters": [
        r"\?",
        r":\w+",
        r"%s",
        r"\$\d+"
    ],

    "xss_patterns": [
        r"innerHTML",
        r"outerHTML",
        r"document\.write",
        r"eval\s*\(",
        r"<script",
        r"javascript:",
        r"onerror\s*=",
        r"onclick\s*=",
        r"onload\s*="
    ],

    "html_output": [
        r"response\.write",
        r"print\s*\(",
        r"write\s*\(",
        r"render\s*\(",
        r"html",
        r"HTML"
    ],

    "encoding_patterns": [
        r"escape",
        r"encode",
        r"htmlspecialchars",
        r"sanitize",
        r"DOMPurify",
        r"Encoder"
    ],

    "command_patterns": [
        r"os\.system",
        r"Runtime\.getRuntime",
        r"ProcessBuilder",
        r"subprocess",
        r"exec\s*\(",
        r"shell\s*=\s*True",
        r"popen"
    ],

    "path_patterns": [
        r"\.\./",
        r"\.\.\\",
        r"PathTraversal",
        r"getCanonicalPath",
        r"getAbsolutePath",
        r"File\s*\(",
        r"open\s*\("
    ],

    "file_input": [
        r"filename",
        r"filepath",
        r"file_path",
        r"upload",
        r"multipart",
        r"file"
    ],

    "network_patterns": [
        r"URL\s*\(",
        r"HttpURLConnection",
        r"requests\.get",
        r"requests\.post",
        r"curl",
        r"fetch\s*\(",
        r"socket",
        r"connect\s*\("
    ],

    "validation_patterns": [
        r"validate",
        r"validation",
        r"isValid",
        r"checkInput",
        r"allowlist",
        r"whitelist",
        r"regex",
        r"matches\s*\("
    ],

    "sanitization_patterns": [
        r"sanitize",
        r"clean",
        r"escape",
        r"encode",
        r"filter",
        r"strip",
        r"remove"
    ],

    "authentication_patterns": [
        r"login",
        r"password",
        r"authenticate",
        r"authentication",
        r"credentials",
        r"session"
    ],

    "authorization_patterns": [
        r"authorize",
        r"authorization",
        r"permission",
        r"role",
        r"isAdmin",
        r"accessControl",
        r"checkAccess"
    ],

    "weak_crypto": [
        r"MD5",
        r"SHA1",
        r"DES",
        r"RC4",
        r"ECB",
        r"Random\s*\("
    ],

    "strong_crypto": [
        r"SHA256",
        r"SHA512",
        r"AES",
        r"RSA",
        r"SecureRandom",
        r"PBKDF2",
        r"bcrypt",
        r"scrypt"
    ],

    "error_patterns": [
        r"printStackTrace",
        r"stackTrace",
        r"Exception",
        r"SQLException",
        r"DEBUG",
        r"debug",
        r"errorMessage",
        r"traceback"
    ],

    "session_patterns": [
        r"session",
        r"cookie",
        r"sessionId",
        r"SessionID",
        r"token"
    ],

    "sensitive_data": [
        r"api_key",
        r"apikey",
        r"secret",
        r"private_key",
        r"password",
        r"access_token",
        r"authorization"
    ]
}


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_security_features(code):

    code = str(code)

    features = {}

    # Basic code characteristics

    features["code_length"] = len(code)

    features["line_count"] = len(code.splitlines())

    features["comment_count"] = len(
        re.findall(r"//|/\*|\*/|#", code)
    )

    features["string_count"] = len(
        re.findall(r'["\'].*?["\']', code)
    )

    features["function_count"] = len(
        re.findall(
            r"\bfunction\b|\bdef\b|\bmethod\b",
            code,
            flags=re.IGNORECASE
        )
    )

    features["variable_count"] = len(
        re.findall(
            r"\bvar\b|\blet\b|\bconst\b",
            code,
            flags=re.IGNORECASE
        )
    )


    # Security pattern features

    for category, patterns in SECURITY_PATTERNS.items():

        category_count = 0

        for pattern in patterns:

            try:

                category_count += len(
                    re.findall(
                        pattern,
                        code,
                        flags=re.IGNORECASE
                    )
                )

            except re.error:

                pass


        features[f"{category}_count"] = category_count

        features[f"{category}_present"] = int(
            category_count > 0
        )


    # User input detection

    user_input_patterns = [
        r"request",
        r"getParameter",
        r"query",
        r"input",
        r"userInput",
        r"request\.args",
        r"request\.form",
        r"request\.json",
        r"params",
        r"body"
    ]


    user_input_present = any(
        re.search(
            pattern,
            code,
            re.IGNORECASE
        )
        for pattern in user_input_patterns
    )


    # SQL detection

    sql_sink_present = any(
        re.search(
            pattern,
            code,
            re.IGNORECASE
        )
        for pattern in SECURITY_PATTERNS["sql_keywords"]
    )


    # HTML output detection

    html_sink_present = any(
        re.search(
            pattern,
            code,
            re.IGNORECASE
        )
        for pattern in SECURITY_PATTERNS["html_output"]
    )


    # Command detection

    command_sink_present = any(
        re.search(
            pattern,
            code,
            re.IGNORECASE
        )
        for pattern in SECURITY_PATTERNS["command_patterns"]
    )


    # Network detection

    network_sink_present = any(
        re.search(
            pattern,
            code,
            re.IGNORECASE
        )
        for pattern in SECURITY_PATTERNS["network_patterns"]
    )


    # File operation detection

    file_sink_present = any(
        re.search(
            pattern,
            code,
            re.IGNORECASE
        )
        for pattern in SECURITY_PATTERNS["path_patterns"]
    )


    # Validation detection

    validation_present = any(
        re.search(
            pattern,
            code,
            re.IGNORECASE
        )
        for pattern in SECURITY_PATTERNS["validation_patterns"]
    )


    # Sanitization detection

    sanitization_present = any(
        re.search(
            pattern,
            code,
            re.IGNORECASE
        )
        for pattern in SECURITY_PATTERNS["sanitization_patterns"]
    )


    # Parameterization detection

    parameterization_present = any(
        re.search(
            pattern,
            code,
            re.IGNORECASE
        )
        for pattern in SECURITY_PATTERNS["sql_parameters"]
    )


    # Encoding detection

    encoding_present = any(
        re.search(
            pattern,
            code,
            re.IGNORECASE
        )
        for pattern in SECURITY_PATTERNS["encoding_patterns"]
    )


    # Source-to-sink features

    features["input_to_sql"] = int(
        user_input_present and sql_sink_present
    )

    features["input_to_html"] = int(
        user_input_present and html_sink_present
    )

    features["input_to_command"] = int(
        user_input_present and command_sink_present
    )

    features["input_to_network"] = int(
        user_input_present and network_sink_present
    )

    features["input_to_file"] = int(
        user_input_present and file_sink_present
    )


    # Missing protection features

    features["sql_without_validation"] = int(
        sql_sink_present and not validation_present
    )

    features["sql_without_parameterization"] = int(
        sql_sink_present and not parameterization_present
    )

    features["html_without_encoding"] = int(
        html_sink_present and not encoding_present
    )

    features["command_without_validation"] = int(
        command_sink_present and not validation_present
    )

    features["file_without_validation"] = int(
        file_sink_present and not validation_present
    )

    features["input_without_validation"] = int(
        user_input_present and not validation_present
    )

    features["input_without_sanitization"] = int(
        user_input_present and not sanitization_present
    )


    # Combined risk features

    features["dangerous_source_sink_count"] = sum([
        features["input_to_sql"],
        features["input_to_html"],
        features["input_to_command"],
        features["input_to_network"],
        features["input_to_file"]
    ])

    features["missing_protection_count"] = sum([
        features["sql_without_validation"],
        features["sql_without_parameterization"],
        features["html_without_encoding"],
        features["command_without_validation"],
        features["file_without_validation"],
        features["input_without_validation"],
        features["input_without_sanitization"]
    ])


    return features


# ============================================================
# EXTRACT ENGINEERED FEATURES
# ============================================================

print("\nExtracting security features...")

train_features = pd.DataFrame(
    [
        extract_security_features(code)
        for code in X_train
    ]
)

validation_features = pd.DataFrame(
    [
        extract_security_features(code)
        for code in X_validation
    ]
)

test_features = pd.DataFrame(
    [
        extract_security_features(code)
        for code in X_test
    ]
)

print(
    "Engineered feature count:",
    train_features.shape[1]
)


# ============================================================
# TF-IDF
# ============================================================

print("\nBuilding TF-IDF features...")


char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5),
    max_features=15000,
    min_df=2,
    max_df=0.98,
    sublinear_tf=True
)


word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    max_features=10000,
    min_df=2,
    max_df=0.98,
    sublinear_tf=True
)


# Character TF-IDF

char_train = char_vectorizer.fit_transform(
    X_train
)

char_validation = char_vectorizer.transform(
    X_validation
)

char_test = char_vectorizer.transform(
    X_test
)


# Word TF-IDF

word_train = word_vectorizer.fit_transform(
    X_train
)

word_validation = word_vectorizer.transform(
    X_validation
)

word_test = word_vectorizer.transform(
    X_test
)


# ============================================================
# COMBINE FEATURES
# ============================================================

print("\nCombining features...")


train_matrix = hstack([
    char_train,
    word_train,
    csr_matrix(train_features.values)
]).tocsr()


validation_matrix = hstack([
    char_validation,
    word_validation,
    csr_matrix(validation_features.values)
]).tocsr()


test_matrix = hstack([
    char_test,
    word_test,
    csr_matrix(test_features.values)
]).tocsr()


print(
    "Training feature matrix:",
    train_matrix.shape
)

print(
    "Validation feature matrix:",
    validation_matrix.shape
)

print(
    "Test feature matrix:",
    test_matrix.shape
)


# ============================================================
# TRAIN MODEL
# ============================================================

print("\nTraining Extra Trees V9 model...")


model = ExtraTreesClassifier(
    n_estimators=500,
    max_features="sqrt",
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)


start_time = time.time()

model.fit(
    train_matrix,
    y_train
)

training_time = time.time() - start_time


print(
    f"\nTraining completed in {training_time:.2f} seconds"
)


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

print("\nEvaluating validation data...")


validation_probabilities = model.predict_proba(
    validation_matrix
)[:, 1]


# ============================================================
# THRESHOLD OPTIMIZATION
# ============================================================

print("\nOptimizing threshold using validation dataset...")


threshold_results = []

best_threshold = 0.50

best_recall = -1

best_f1 = -1

best_precision = 0


for threshold in THRESHOLDS:

    validation_predictions = (
        validation_probabilities >= threshold
    ).astype(int)


    precision_value = precision_score(
        y_validation,
        validation_predictions,
        zero_division=0
    )


    recall_value = recall_score(
        y_validation,
        validation_predictions,
        zero_division=0
    )


    f1_value = f1_score(
        y_validation,
        validation_predictions,
        zero_division=0
    )


    accuracy_value = accuracy_score(
        y_validation,
        validation_predictions
    )


    threshold_results.append({

        "threshold": threshold,

        "accuracy": accuracy_value,

        "precision": precision_value,

        "recall": recall_value,

        "f1": f1_value

    })


    # Select maximum recall while meeting precision constraint

    if precision_value >= MIN_PRECISION:

        if recall_value > best_recall:

            best_recall = recall_value

            best_f1 = f1_value

            best_precision = precision_value

            best_threshold = threshold


        elif recall_value == best_recall:

            if f1_value > best_f1:

                best_f1 = f1_value

                best_precision = precision_value

                best_threshold = threshold


threshold_results_df = pd.DataFrame(
    threshold_results
)


threshold_results_df.to_csv(
    os.path.join(
        RESULT_DIR,
        "v9_threshold_optimization.csv"
    ),
    index=False
)


# ============================================================
# FALLBACK
# ============================================================

if best_recall == -1:

    print(
        "\nNo threshold achieved minimum precision requirement."
    )

    print(
        "Using threshold with highest validation F1."
    )

    best_row = threshold_results_df.loc[
        threshold_results_df["f1"].idxmax()
    ]

    best_threshold = best_row["threshold"]

    best_precision = best_row["precision"]

    best_recall = best_row["recall"]

    best_f1 = best_row["f1"]


print("\nSelected threshold:", best_threshold)

print(
    "Validation precision:",
    round(best_precision * 100, 2),
    "%"
)

print(
    "Validation recall:",
    round(best_recall * 100, 2),
    "%"
)

print(
    "Validation F1:",
    round(best_f1 * 100, 2),
    "%"
)


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print("\n" + "=" * 70)

print("FINAL TEST EVALUATION")

print("=" * 70)


test_probabilities = model.predict_proba(
    test_matrix
)[:, 1]


# Default threshold

test_predictions_050 = (
    test_probabilities >= 0.50
).astype(int)


# Optimized threshold

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    test_predictions
)

precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0
)

auc = roc_auc_score(
    y_test,
    test_probabilities
)

cm = confusion_matrix(
    y_test,
    test_predictions
)


print("\nOptimized Threshold:", best_threshold)

print(
    "\nAccuracy:",
    round(accuracy * 100, 2),
    "%"
)

print(
    "Precision:",
    round(precision * 100, 2),
    "%"
)

print(
    "Recall:",
    round(recall * 100, 2),
    "%"
)

print(
    "F1 Score:",
    round(f1 * 100, 2),
    "%"
)

print(
    "ROC-AUC:",
    round(auc * 100, 2),
    "%"
)


print("\nConfusion Matrix:")

print(cm)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        test_predictions,
        zero_division=0
    )
)


# ============================================================
# THRESHOLD COMPARISON
# ============================================================

def calculate_metrics(y_true, predictions):

    return {

        "Accuracy": accuracy_score(
            y_true,
            predictions
        ),

        "Precision": precision_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "Recall": recall_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "F1": f1_score(
            y_true,
            predictions,
            zero_division=0
        )

    }


metrics_050 = calculate_metrics(
    y_test,
    test_predictions_050
)


metrics_optimized = calculate_metrics(
    y_test,
    test_predictions
)


comparison = pd.DataFrame([

    {
        "Threshold": 0.50,
        **metrics_050
    },

    {
        "Threshold": best_threshold,
        **metrics_optimized
    }

])


print("\nThreshold Comparison:")

print(comparison)


comparison.to_csv(
    os.path.join(
        RESULT_DIR,
        "v9_threshold_comparison.csv"
    ),
    index=False
)


# ============================================================
# ERROR ANALYSIS
# ============================================================

print("\nPerforming error analysis...")


error_records = []


for index in range(len(y_test)):

    actual = y_test.iloc[index]

    predicted = test_predictions[index]

    probability = test_probabilities[index]


    if actual != predicted:

        if actual == 1 and predicted == 0:

            error_type = "False Negative"

        else:

            error_type = "False Positive"


        row = test_df.iloc[index]


        source = (
            row["source"]
            if "source" in test_df.columns
            else "Unknown"
        )


        error_records.append({

            "actual": actual,

            "predicted": predicted,

            "probability": probability,

            "error_type": error_type,

            "source": source,

            "code": row["code_normalized"]

        })


error_df = pd.DataFrame(
    error_records
)


error_file = os.path.join(
    ERROR_DIR,
    "v9_test_errors.csv"
)


error_df.to_csv(
    error_file,
    index=False
)


false_negatives = error_df[
    error_df["error_type"] == "False Negative"
]


false_positives = error_df[
    error_df["error_type"] == "False Positive"
]


print(
    "\nFalse Negatives:",
    len(false_negatives)
)

print(
    "False Positives:",
    len(false_positives)
)


if len(false_negatives) > 0:

    print("\nFalse Negative Sources:")

    print(
        false_negatives["source"].value_counts()
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\nExtracting feature importance...")


engineered_feature_names = list(
    train_features.columns
)


char_feature_names = [
    "CHAR_" + feature
    for feature in char_vectorizer.get_feature_names_out()
]


word_feature_names = [
    "WORD_" + feature
    for feature in word_vectorizer.get_feature_names_out()
]


all_feature_names = (
    char_feature_names
    + word_feature_names
    + engineered_feature_names
)


feature_importances = model.feature_importances_


feature_importance_df = pd.DataFrame({

    "feature": all_feature_names,

    "importance": feature_importances

})


feature_importance_df = feature_importance_df.sort_values(
    by="importance",
    ascending=False
)


feature_importance_df.to_csv(
    os.path.join(
        RESULT_DIR,
        "v9_feature_importance.csv"
    ),
    index=False
)


print("\nTop 30 Features:")

print(
    feature_importance_df.head(30)
)


# ============================================================
# SAVE MODEL
# ============================================================

print("\nSaving V9 model...")


model_package = {

    "model": model,

    "char_vectorizer": char_vectorizer,

    "word_vectorizer": word_vectorizer,

    "feature_names": engineered_feature_names,

    "threshold": best_threshold,

    "version": "V9",

    "description": (
        "Contextual security feature model "
        "with validation-based threshold optimization"
    )

}


model_file = os.path.join(
    MODEL_DIR,
    "best_model_v9.pkl"
)


with open(
    model_file,
    "wb"
) as file:

    pickle.dump(
        model_package,
        file
    )


# ============================================================
# SAVE TRAINING REPORT
# ============================================================

report = f"""

============================================================
WEB APPLICATION SECURITY AI TOOLKIT - V9
TRAINING REPORT
============================================================

DATASET INFORMATION
------------------------------------------------------------

Training samples: {len(X_train)}

Validation samples: {len(X_validation)}

Final test samples: {len(X_test)}

Training features: {train_matrix.shape[1]}

Training time: {training_time:.2f} seconds


MODEL
------------------------------------------------------------

Extra Trees Classifier

Number of estimators: 500

Class weight: Balanced


THRESHOLD OPTIMIZATION
------------------------------------------------------------

Selected threshold: {best_threshold:.4f}

Validation precision: {best_precision:.4f}

Validation recall: {best_recall:.4f}

Validation F1: {best_f1:.4f}


FINAL TEST RESULTS
------------------------------------------------------------

Accuracy: {accuracy:.4f}

Precision: {precision:.4f}

Recall: {recall:.4f}

F1 Score: {f1:.4f}

ROC-AUC: {auc:.4f}


CONFUSION MATRIX
------------------------------------------------------------

{cm}


ERROR ANALYSIS
------------------------------------------------------------

False Negatives: {len(false_negatives)}

False Positives: {len(false_positives)}


IMPORTANT
------------------------------------------------------------

This model predicts patterns associated with security
vulnerabilities in source code.

Predictions are not proof of exploitable vulnerabilities.

Results should be validated through authorized security
testing and manual verification.

============================================================
"""


report_file = os.path.join(
    RESULT_DIR,
    "v9_training_report.txt"
)


with open(
    report_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(report)


# ============================================================
# SAVE RESULTS CSV
# ============================================================

results = pd.DataFrame({

    "model": ["Extra Trees V9"],

    "accuracy": [accuracy],

    "precision": [precision],

    "recall": [recall],

    "f1": [f1],

    "roc_auc": [auc],

    "threshold": [best_threshold],

    "training_time": [training_time],

    "false_negatives": [len(false_negatives)],

    "false_positives": [len(false_positives)]

})


results.to_csv(
    os.path.join(
        RESULT_DIR,
        "v9_model_results.csv"
    ),
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)

print("V9 TRAINING COMPLETED SUCCESSFULLY")

print("=" * 70)


print("\nModel saved:")

print(model_file)


print("\nResults saved:")

print(RESULT_DIR)


print("\nError analysis saved:")

print(ERROR_DIR)


print("\nSelected threshold:")

print(best_threshold)


print("\nFINAL TEST RESULTS")

print("-" * 50)


print(
    f"Accuracy:  {accuracy * 100:.2f}%"
)

print(
    f"Precision: {precision * 100:.2f}%"
)

print(
    f"Recall:    {recall * 100:.2f}%"
)

print(
    f"F1 Score:  {f1 * 100:.2f}%"
)

print(
    f"ROC-AUC:   {auc * 100:.2f}%"
)


print(
    f"False Negatives: {len(false_negatives)}"
)

print(
    f"False Positives: {len(false_positives)}"
)


print("\nV9 training process finished.")