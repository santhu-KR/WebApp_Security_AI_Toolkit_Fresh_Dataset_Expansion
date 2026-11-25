import os
import re
import time
import joblib
import numpy as np
import pandas as pd

from scipy.sparse import hstack, csr_matrix

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# V7 MACHINE LEARNING TRAINING
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "v7",
    "balanced"
)

TRAIN_FILE = os.path.join(
    DATA_DIR,
    "train_v7_balanced.csv"
)

TEST_FILE = os.path.join(
    DATA_DIR,
    "test_v7.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "v7"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "v7",
    "model_training"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 75)
print("V7 MACHINE LEARNING MODEL TRAINING")
print("=" * 75)

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print("\nDataset Loaded")

print(f"Training samples: {len(train_df)}")
print(f"Testing samples: {len(test_df)}")


# ============================================================
# 2. NORMALIZE LABELS
# ============================================================

train_df["label"] = (
    train_df["label"]
    .astype(str)
    .str.lower()
    .str.strip()
)

test_df["label"] = (
    test_df["label"]
    .astype(str)
    .str.lower()
    .str.strip()
)


# ============================================================
# 3. LABEL ENCODING
# ============================================================

label_mapping = {
    "safe": 0,
    "vulnerable": 1
}

y_train = train_df["label"].map(label_mapping)
y_test = test_df["label"].map(label_mapping)

if y_train.isna().any() or y_test.isna().any():

    raise ValueError(
        "Invalid labels detected."
    )


y_train = y_train.astype(int)
y_test = y_test.astype(int)


# ============================================================
# 4. PREPARE CODE
# ============================================================

X_train_text = (
    train_df["code_normalized"]
    .fillna("")
    .astype(str)
)

X_test_text = (
    test_df["code_normalized"]
    .fillna("")
    .astype(str)
)


# ============================================================
# 5. CHARACTER TF-IDF
# ============================================================

print("\nCreating Character TF-IDF Features...")

char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5),
    max_features=15000,
    min_df=2,
    max_df=0.98,
    sublinear_tf=True
)

X_train_char = char_vectorizer.fit_transform(
    X_train_text
)

X_test_char = char_vectorizer.transform(
    X_test_text
)

print("Character TF-IDF completed.")


# ============================================================
# 6. WORD TF-IDF
# ============================================================

print("\nCreating Word TF-IDF Features...")

word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    max_features=10000,
    min_df=2,
    max_df=0.98,
    sublinear_tf=True,
    token_pattern=r"(?u)\b\w+\b"
)

X_train_word = word_vectorizer.fit_transform(
    X_train_text
)

X_test_word = word_vectorizer.transform(
    X_test_text
)

print("Word TF-IDF completed.")


# ============================================================
# 7. ENGINEERED SECURITY FEATURES
# ============================================================

SECURITY_PATTERNS = {

    "sql_keywords": [
        "select",
        "insert",
        "update",
        "delete",
        "drop",
        "union",
        "where"
    ],

    "sql_injection": [
        " or 1=1",
        "' or '1'='1",
        "union select",
        "drop table",
        "sql injection"
    ],

    "command_execution": [
        "runtime.getruntime",
        "processbuilder",
        "exec(",
        "system("
    ],

    "xss": [
        "<script",
        "javascript:",
        "innerhtml",
        "document.write",
        "eval("
    ],

    "file_operations": [
        "fileinputstream",
        "fileoutputstream",
        "filewriter",
        "filereader",
        "filepath"
    ],

    "path_traversal": [
        "../",
        "..\\",
        "/etc/passwd",
        "path traversal"
    ],

    "authentication": [
        "password",
        "login",
        "authenticate",
        "authentication",
        "session"
    ],

    "cryptography": [
        "md5",
        "sha1",
        "sha256",
        "cipher",
        "encrypt",
        "decrypt"
    ],

    "network": [
        "http",
        "https",
        "socket",
        "urlconnection",
        "request"
    ],

    "input_handling": [
        "request.getparameter",
        "getparameter",
        "input",
        "userinput",
        "parameter"
    ]
}


def extract_features(code):

    code = str(code)

    code_lower = code.lower()

    features = []

    # --------------------------------------------------------
    # LENGTH FEATURES
    # --------------------------------------------------------

    features.append(len(code))

    features.append(len(code.split()))

    features.append(len(code.splitlines()))

    # --------------------------------------------------------
    # STRUCTURE FEATURES
    # --------------------------------------------------------

    features.append(code.count("{"))

    features.append(code.count("}"))

    features.append(code.count("("))

    features.append(code.count(")"))

    features.append(code.count(";"))

    features.append(code.count("="))

    features.append(code.count("."))

    features.append(code.count(","))

    features.append(code.count(":"))

    # --------------------------------------------------------
    # COMMENT FEATURES
    # --------------------------------------------------------

    features.append(code.count("//"))

    features.append(code.count("/*"))

    features.append(code.count("*/"))

    # --------------------------------------------------------
    # STRING FEATURES
    # --------------------------------------------------------

    features.append(code.count('"'))

    features.append(code.count("'"))

    # --------------------------------------------------------
    # NUMBER OF UNIQUE TOKENS
    # --------------------------------------------------------

    tokens = re.findall(
        r"\b\w+\b",
        code_lower
    )

    features.append(
        len(set(tokens))
    )

    # --------------------------------------------------------
    # SECURITY PATTERN FEATURES
    # --------------------------------------------------------

    for category, patterns in SECURITY_PATTERNS.items():

        category_count = 0

        for pattern in patterns:

            category_count += code_lower.count(
                pattern.lower()
            )

        features.append(
            category_count
        )

    # --------------------------------------------------------
    # INDIVIDUAL SECURITY INDICATORS
    # --------------------------------------------------------

    individual_patterns = [

        "preparedstatement",
        "statement.execute",
        "executequery",
        "executeupdate",

        "parameterized",
        "sanitize",
        "escape",
        "validation",

        "allowlist",
        "whitelist",
        "blacklist",

        "try",
        "catch",
        "finally",

        "throw",
        "throws",

        "return",

        "public",
        "private",
        "protected",

        "static",

        "new",

        "null",

        "true",
        "false",

        "getparameter",
        "setparameter",

        "request",
        "response",

        "cookie",
        "session",

        "authorization",

        "csrf",
        "cors",

        "ssl",
        "tls",

        "certificate",

        "deserialize",
        "serialization",

        "reflection",

        "ldap",

        "xpath",

        "xml",

        "json"
    ]

    for pattern in individual_patterns:

        features.append(
            code_lower.count(pattern)
        )

    # --------------------------------------------------------
    # DANGEROUS API FEATURES
    # --------------------------------------------------------

    dangerous_apis = [

        "runtime",
        "processbuilder",
        "exec",
        "eval",

        "system",

        "scriptengine",

        "class.forname",

        "reflect",

        "deserialize",

        "objectinputstream",

        "readobject",

        "writeobject",

        "random",

        "math.random",

        "secure random",

        "md5",

        "sha1",

        "hardcoded",

        "secret",

        "token",

        "api_key",

        "password",

        "passwd"
    ]

    for pattern in dangerous_apis:

        features.append(
            code_lower.count(pattern)
        )

    # --------------------------------------------------------
    # URL AND HTML FEATURES
    # --------------------------------------------------------

    features.append(
        code_lower.count("http://")
    )

    features.append(
        code_lower.count("https://")
    )

    features.append(
        code_lower.count("<")
    )

    features.append(
        code_lower.count(">")
    )

    features.append(
        code_lower.count("&")
    )

    # --------------------------------------------------------
    # WHITESPACE FEATURES
    # --------------------------------------------------------

    features.append(
        code.count(" ")
    )

    features.append(
        code.count("\t")
    )

    features.append(
        code.count("\n")
    )

    # --------------------------------------------------------
    # DIGIT FEATURES
    # --------------------------------------------------------

    features.append(
        sum(char.isdigit() for char in code)
    )

    # --------------------------------------------------------
    # SPECIAL CHARACTERS
    # --------------------------------------------------------

    special_chars = [
        "$",
        "%",
        "&",
        "*",
        "!",
        "?",
        "#",
        "@",
        "/",
        "\\",
        "<",
        ">",
        "|"
    ]

    for char in special_chars:

        features.append(
            code.count(char)
        )

    return features


print("\nExtracting Engineered Features...")

X_train_engineered = np.array(
    [
        extract_features(code)
        for code in X_train_text
    ]
)

X_test_engineered = np.array(
    [
        extract_features(code)
        for code in X_test_text
    ]
)


# ============================================================
# 8. SCALE ENGINEERED FEATURES
# ============================================================

scaler = StandardScaler()

X_train_engineered_scaled = scaler.fit_transform(
    X_train_engineered
)

X_test_engineered_scaled = scaler.transform(
    X_test_engineered
)


# ============================================================
# 9. COMBINE FEATURES
# ============================================================

print("\nCombining Features...")

X_train = hstack(
    [
        X_train_char,
        X_train_word,
        csr_matrix(
            X_train_engineered_scaled
        )
    ]
).tocsr()


X_test = hstack(
    [
        X_test_char,
        X_test_word,
        csr_matrix(
            X_test_engineered_scaled
        )
    ]
).tocsr()


print(f"Training Feature Shape: {X_train.shape}")
print(f"Testing Feature Shape: {X_test.shape}")


# ============================================================
# 10. DEFINE MODELS
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        C=1.0,
        class_weight=None,
        random_state=42
    ),

    "Linear SVM": LinearSVC(
        C=1.0,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
}


# ============================================================
# 11. TRAIN AND EVALUATE
# ============================================================

results = []

trained_models = {}

model_reports = {}

print("\n" + "=" * 75)
print("MODEL TRAINING")
print("=" * 75)


for model_name, model in models.items():

    print(f"\nTraining: {model_name}")

    start_time = time.time()

    model.fit(
        X_train,
        y_train
    )

    training_time = (
        time.time() - start_time
    )

    print("Training completed.")

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    y_pred = model.predict(
        X_test
    )

    # --------------------------------------------------------
    # PROBABILITY / SCORE
    # --------------------------------------------------------

    try:

        if hasattr(
            model,
            "predict_proba"
        ):

            y_score = model.predict_proba(
                X_test
            )[:, 1]

        elif hasattr(
            model,
            "decision_function"
        ):

            y_score = model.decision_function(
                X_test
            )

        else:

            y_score = y_pred

    except Exception:

        y_score = y_pred


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    try:

        auc = roc_auc_score(
            y_test,
            y_score
        )

    except Exception:

        auc = 0


    cm = confusion_matrix(
        y_test,
        y_pred
    )


    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    classification_report_text = (
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "Safe",
                "Vulnerable"
            ],
            zero_division=0
        )
    )


    # --------------------------------------------------------
    # SAVE RESULT
    # --------------------------------------------------------

    result = {

        "Model": model_name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1_Score": f1,

        "ROC_AUC": auc,

        "Training_Time_Seconds": training_time
    }


    results.append(
        result
    )


    trained_models[
        model_name
    ] = model


    model_reports[
        model_name
    ] = {

        "classification_report":
            classification_report_text,

        "confusion_matrix":
            cm,

        "accuracy":
            accuracy,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "auc":
            auc,

        "training_time":
            training_time
    }


    print("\nResults")

    print(f"Accuracy : {accuracy:.4f}")

    print(f"Precision: {precision:.4f}")

    print(f"Recall   : {recall:.4f}")

    print(f"F1 Score : {f1:.4f}")

    print(f"ROC-AUC  : {auc:.4f}")

    print(
        f"Training Time: {training_time:.2f} seconds"
    )


# ============================================================
# 12. COMPARISON TABLE
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="F1_Score",
    ascending=False
)


comparison_file = os.path.join(
    RESULT_DIR,
    "v7_balanced_model_comparison.csv"
)

results_df.to_csv(
    comparison_file,
    index=False
)


print("\n" + "=" * 75)

print("MODEL COMPARISON")

print("=" * 75)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 13. SELECT BEST MODEL
# ============================================================

best_model_name = results_df.iloc[0][
    "Model"
]

best_model = trained_models[
    best_model_name
]


print("\n" + "=" * 75)

print(f"BEST MODEL: {best_model_name}")

print("=" * 75)


# ============================================================
# 14. SAVE BEST MODEL
# ============================================================

best_model_package = {

    "model": best_model,

    "char_vectorizer": char_vectorizer,

    "word_vectorizer": word_vectorizer,

    "scaler": scaler,

    "label_mapping": label_mapping,

    "best_model_name": best_model_name,

    "feature_description": {

        "character_tfidf": True,

        "word_tfidf": True,

        "engineered_security_features": True
    }
}


BEST_MODEL_FILE = os.path.join(
    MODEL_DIR,
    "best_model_v7_balanced.pkl"
)


joblib.dump(
    best_model_package,
    BEST_MODEL_FILE
)


# ============================================================
# 15. SAVE ALL MODELS
# ============================================================

ALL_MODELS_FILE = os.path.join(
    MODEL_DIR,
    "all_models_v7_balanced.pkl"
)


joblib.dump(
    trained_models,
    ALL_MODELS_FILE
)


# ============================================================
# 16. SAVE DETAILED REPORT
# ============================================================

report_file = os.path.join(
    RESULT_DIR,
    "v7_balanced_training_report.txt"
)


with open(
    report_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "=" * 75
    )

    file.write(
        "\nV7 BALANCED MODEL TRAINING REPORT\n"
    )

    file.write(
        "=" * 75
    )

    file.write(
        f"\nTraining Samples: {len(train_df)}"
    )

    file.write(
        f"\nTesting Samples: {len(test_df)}"
    )

    file.write(
        f"\nFeature Count: {X_train.shape[1]}"
    )

    file.write(
        "\n\nMODEL COMPARISON\n"
    )

    file.write(
        results_df.to_string(
            index=False
        )
    )


    for model_name, report in model_reports.items():

        file.write(
            "\n\n"
        )

        file.write(
            "=" * 75
        )

        file.write(
            f"\n{model_name}\n"
        )

        file.write(
            "=" * 75
        )

        file.write(
            "\n\nClassification Report:\n"
        )

        file.write(
            report[
                "classification_report"
            ]
        )

        file.write(
            "\nConfusion Matrix:\n"
        )

        file.write(
            str(
                report[
                    "confusion_matrix"
                ]
            )
        )


# ============================================================
# 17. SAVE CONFUSION MATRICES
# ============================================================

for model_name, report in model_reports.items():

    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
    )

    cm_df = pd.DataFrame(
        report[
            "confusion_matrix"
        ],
        index=[
            "Actual Safe",
            "Actual Vulnerable"
        ],
        columns=[
            "Predicted Safe",
            "Predicted Vulnerable"
        ]
    )


    cm_file = os.path.join(
        RESULT_DIR,
        f"{safe_name}_confusion_matrix.csv"
    )


    cm_df.to_csv(
        cm_file
    )


# ============================================================
# 18. COMPLETION
# ============================================================

print("\n" + "=" * 75)

print("V7 BALANCED TRAINING COMPLETED")

print("=" * 75)


print("\nBest Model:")

print(best_model_name)


print("\nComparison File:")

print(comparison_file)


print("\nReport File:")

print(report_file)


print("\nSaved Model:")

print(BEST_MODEL_FILE)


print("\nAll Models:")

print(ALL_MODELS_FILE)


print("\nTraining completed successfully.")