# ============================================================
# WEB APPLICATION SECURITY AI TOOLKIT
# V8 - ADVANCED FEATURE ENGINEERING & MODEL TRAINING
# ============================================================

import os
import re
import time
import warnings
import joblib
import numpy as np
import pandas as pd

from scipy.sparse import hstack, csr_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

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

TRAIN_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "v7",
    "balanced",
    "train_v7_balanced.csv"
)

TEST_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "v7",
    "balanced",
    "test_v7.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models", "v8")

RESULT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "v8",
    "model_training"
)

ERROR_DIR = os.path.join(
    BASE_DIR,
    "results",
    "v8",
    "error_analysis"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(ERROR_DIR, exist_ok=True)


RANDOM_STATE = 42

# Classification threshold
DEFAULT_THRESHOLD = 0.50


# ============================================================
# SECURITY FEATURE PATTERNS
# ============================================================

SECURITY_PATTERNS = {

    # INPUT SOURCES
    "input_sources": [
        r"request\.getParameter",
        r"request\.getParameterValues",
        r"request\.getParameterMap",
        r"request\.getHeader",
        r"request\.getCookies",
        r"request\.getQueryString",
        r"request\.getInputStream",
        r"request\.getReader",
        r"request\.args",
        r"request\.form",
        r"request\.json",
        r"request\.values",
        r"request\.headers",
        r"request\.cookies",
        r"req\.body",
        r"req\.query",
        r"req\.params",
        r"req\.headers",
        r"req\.cookies",
        r"input\s*\(",
        r"Scanner\s*\("
    ],

    # SQL
    "sql_patterns": [
        r"execute\s*\(",
        r"executeQuery\s*\(",
        r"executeUpdate\s*\(",
        r"executemany\s*\(",
        r"cursor\.execute",
        r"\.query\s*\(",
        r"SELECT\s+",
        r"INSERT\s+INTO",
        r"UPDATE\s+",
        r"DELETE\s+FROM",
        r"Statement\s+",
        r"PreparedStatement",
        r"createQuery",
        r"createNativeQuery"
    ],

    # XSS
    "xss_patterns": [
        r"innerHTML",
        r"outerHTML",
        r"document\.write",
        r"document\.writeln",
        r"response\.write",
        r"response\.getWriter",
        r"\.printf\s*\(",
        r"print\s*\(",
        r"render_template_string",
        r"Markup\s*\(",
        r"dangerouslySetInnerHTML",
        r"eval\s*\("
    ],

    # COMMAND INJECTION
    "command_patterns": [
        r"os\.system",
        r"os\.popen",
        r"subprocess\.run",
        r"subprocess\.call",
        r"subprocess\.Popen",
        r"subprocess\.check_output",
        r"Runtime\.getRuntime",
        r"\.exec\s*\(",
        r"ProcessBuilder",
        r"shell\s*=\s*True"
    ],

    # SSRF / NETWORK
    "network_patterns": [
        r"requests\.get",
        r"requests\.post",
        r"requests\.put",
        r"requests\.delete",
        r"urllib\.request",
        r"urlopen\s*\(",
        r"HttpURLConnection",
        r"URL\s*\(",
        r"socket\.connect",
        r"socket\.create_connection",
        r"fetch\s*\(",
        r"axios\.",
        r"httpClient",
        r"WebClient"
    ],

    # CRYPTOGRAPHY
    "crypto_patterns": [
        r"DES",
        r"DESede",
        r"MD5",
        r"SHA1",
        r"SHA-1",
        r"RC4",
        r"Blowfish",
        r"AES/ECB",
        r"ECB",
        r"Cipher\.getInstance",
        r"MessageDigest",
        r"password",
        r"secret",
        r"privateKey",
        r"Random\s*\(",
        r"Math\.random"
    ],

    # AUTHENTICATION
    "auth_patterns": [
        r"login",
        r"logout",
        r"authenticate",
        r"authentication",
        r"password",
        r"passwd",
        r"credential",
        r"verifyPassword",
        r"checkPassword",
        r"comparePassword",
        r"bcrypt",
        r"hashPassword"
    ],

    # SESSION
    "session_patterns": [
        r"getSession",
        r"session\s*\[",
        r"session\.",
        r"putValue",
        r"setAttribute",
        r"sessionId",
        r"session_id",
        r"SESSION"
    ],

    # ACCESS CONTROL
    "access_control_patterns": [
        r"authorize",
        r"authorization",
        r"isAdmin",
        r"admin",
        r"role",
        r"permission",
        r"accessControl",
        r"checkAccess",
        r"ownership",
        r"userId",
        r"ownerId"
    ],

    # FILE OPERATIONS
    "file_patterns": [
        r"open\s*\(",
        r"File\s*\(",
        r"FileInputStream",
        r"FileOutputStream",
        r"Files\.read",
        r"Files\.write",
        r"readFile",
        r"writeFile",
        r"pathlib",
        r"os\.path",
        r"Path\s*\("
    ],

    # DESERIALIZATION
    "deserialization_patterns": [
        r"ObjectInputStream",
        r"readObject",
        r"pickle\.loads",
        r"pickle\.load",
        r"yaml\.load",
        r"unpickle",
        r"deserialize",
        r"JSONDecoder"
    ],

    # TLS / SSL
    "tls_patterns": [
        r"verify\s*=\s*False",
        r"verify=False",
        r"ssl_verify\s*=\s*False",
        r"CERT_NONE",
        r"check_hostname\s*=\s*False",
        r"TrustManager",
        r"HostnameVerifier",
        r"disable.*certificate",
        r"SSLContext",
        r"ignore.*certificate"
    ],

    # VALIDATION / SANITIZATION
    "validation_patterns": [
        r"validate",
        r"validator",
        r"sanitize",
        r"sanitizer",
        r"escape",
        r"encode",
        r"decode",
        r"strip\s*\(",
        r"replace\s*\(",
        r"regex",
        r"re\.match",
        r"re\.search",
        r"isinstance",
        r"allowlist",
        r"whitelist"
    ],

    # ERROR HANDLING
    "error_patterns": [
        r"try\s*:",
        r"catch\s*\(",
        r"except\s*:",
        r"except\s+Exception",
        r"printStackTrace",
        r"stacktrace",
        r"debug",
        r"traceback"
    ],

    # SECURITY CONFIGURATION
    "configuration_patterns": [
        r"DEBUG\s*=\s*True",
        r"debug\s*=\s*True",
        r"allow_all",
        r"cors",
        r"Access-Control-Allow-Origin",
        r"\*\s*",
        r"permitAll",
        r"allowAny"
    ]
}


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_security_features(code):

    if pd.isna(code):
        code = ""

    code = str(code)

    features = {}

    # --------------------------------------------------------
    # BASIC CODE FEATURES
    # --------------------------------------------------------

    features["code_length"] = len(code)

    features["line_count"] = len(code.splitlines())

    features["word_count"] = len(code.split())

    features["character_count"] = len(code)

    features["uppercase_count"] = sum(
        1 for c in code if c.isupper()
    )

    features["lowercase_count"] = sum(
        1 for c in code if c.islower()
    )

    features["digit_count"] = sum(
        1 for c in code if c.isdigit()
    )

    features["special_character_count"] = sum(
        1 for c in code if not c.isalnum()
    )

    features["comment_count"] = len(
        re.findall(r"//|#|/\*|\*/", code)
    )

    features["string_count"] = len(
        re.findall(r"['\"].*?['\"]", code)
    )

    features["function_count"] = len(
        re.findall(
            r"\bdef\b|\bfunction\b|\bvoid\b|\bpublic\b|\bprivate\b",
            code
        )
    )

    features["class_count"] = len(
        re.findall(r"\bclass\b", code)
    )

    features["loop_count"] = len(
        re.findall(r"\bfor\b|\bwhile\b", code)
    )

    features["conditional_count"] = len(
        re.findall(r"\bif\b|\belse\b|\belif\b|\bswitch\b", code)
    )

    # --------------------------------------------------------
    # SECURITY PATTERN FEATURES
    # --------------------------------------------------------

    for category, patterns in SECURITY_PATTERNS.items():

        count = 0

        for pattern in patterns:

            try:
                matches = re.findall(
                    pattern,
                    code,
                    flags=re.IGNORECASE
                )

                count += len(matches)

            except re.error:
                continue

        features[f"{category}_count"] = count

        features[f"{category}_present"] = (
            1 if count > 0 else 0
        )

    # --------------------------------------------------------
    # SOURCE / SINK COMBINATIONS
    # --------------------------------------------------------

    input_count = features["input_sources_count"]

    sql_count = features["sql_patterns_count"]

    xss_count = features["xss_patterns_count"]

    command_count = features["command_patterns_count"]

    network_count = features["network_patterns_count"]

    file_count = features["file_patterns_count"]

    features["input_sql_combination"] = (
        input_count > 0 and sql_count > 0
    )

    features["input_xss_combination"] = (
        input_count > 0 and xss_count > 0
    )

    features["input_command_combination"] = (
        input_count > 0 and command_count > 0
    )

    features["input_network_combination"] = (
        input_count > 0 and network_count > 0
    )

    features["input_file_combination"] = (
        input_count > 0 and file_count > 0
    )

    # --------------------------------------------------------
    # SECURITY INDICATORS
    # --------------------------------------------------------

    features["has_input_and_sink"] = (
        input_count > 0 and (
            sql_count > 0 or
            xss_count > 0 or
            command_count > 0 or
            network_count > 0 or
            file_count > 0
        )
    )

    features["has_weak_crypto"] = (
        features["crypto_patterns_count"] > 0
    )

    features["has_tls_issue_indicator"] = (
        features["tls_patterns_count"] > 0
    )

    features["has_authentication_indicator"] = (
        features["auth_patterns_count"] > 0
    )

    features["has_session_indicator"] = (
        features["session_patterns_count"] > 0
    )

    features["has_access_control_indicator"] = (
        features["access_control_patterns_count"] > 0
    )

    features["has_validation_indicator"] = (
        features["validation_patterns_count"] > 0
    )

    # --------------------------------------------------------
    # TRANSFORMATION FEATURES
    # --------------------------------------------------------

    transformation_patterns = [
        r"encode",
        r"decode",
        r"replace",
        r"substring",
        r"split",
        r"join",
        r"trim",
        r"strip",
        r"base64",
        r"URLEncoder",
        r"URLDecoder"
    ]

    features["transformation_count"] = sum(
        len(re.findall(pattern, code, re.IGNORECASE))
        for pattern in transformation_patterns
    )

    # --------------------------------------------------------
    # BOOLEAN FEATURES -> INTEGER
    # --------------------------------------------------------

    for key, value in list(features.items()):

        if isinstance(value, (bool, np.bool_)):
            features[key] = int(value)

    return features


# ============================================================
# LOAD DATASET
# ============================================================

def load_datasets():

    print("\n" + "=" * 70)
    print("LOADING V7 DATASETS FOR V8 TRAINING")
    print("=" * 70)

    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(
            f"Training dataset not found:\n{TRAIN_PATH}"
        )

    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(
            f"Test dataset not found:\n{TEST_PATH}"
        )

    train_df = pd.read_csv(TRAIN_PATH)

    test_df = pd.read_csv(TEST_PATH)

    print("\nTraining dataset:")
    print(f"Rows: {len(train_df)}")

    print("\nTest dataset:")
    print(f"Rows: {len(test_df)}")

    print("\nTraining columns:")
    print(train_df.columns.tolist())

    print("\nTest columns:")
    print(test_df.columns.tolist())

    return train_df, test_df


# ============================================================
# DETERMINE CODE COLUMN
# ============================================================

def find_code_column(df):

    possible_columns = [
        "code_normalized",
        "code",
        "source_code",
        "snippet",
        "content",
        "text"
    ]

    for column in possible_columns:

        if column in df.columns:
            return column

    raise ValueError(
        "Could not find code column.\n"
        f"Available columns: {df.columns.tolist()}"
    )


# ============================================================
# DETERMINE LABEL COLUMN
# ============================================================

def find_label_column(df):

    possible_columns = [
        "label",
        "actual",
        "target",
        "class",
        "vulnerability"
    ]

    for column in possible_columns:

        if column in df.columns:
            return column

    raise ValueError(
        "Could not find label column.\n"
        f"Available columns: {df.columns.tolist()}"
    )


# ============================================================
# NORMALIZE LABELS
# ============================================================

def normalize_labels(series):

    normalized = []

    for value in series:

        value = str(value).strip().lower()

        if value in [
            "1",
            "vulnerable",
            "vulnerability",
            "unsafe",
            "malicious",
            "true",
            "yes"
        ]:

            normalized.append(1)

        elif value in [
            "0",
            "safe",
            "secure",
            "benign",
            "false",
            "no"
        ]:

            normalized.append(0)

        else:

            try:
                normalized.append(int(float(value)))

            except Exception:

                raise ValueError(
                    f"Unknown label encountered: {value}"
                )

    return np.array(normalized)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_features(train_codes, test_codes):

    print("\n" + "=" * 70)
    print("V8 FEATURE ENGINEERING")
    print("=" * 70)

    # --------------------------------------------------------
    # CHARACTER TF-IDF
    # --------------------------------------------------------

    print("\nCreating Character TF-IDF features...")

    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        max_features=15000,
        min_df=2,
        max_df=0.98,
        sublinear_tf=True
    )

    X_train_char = char_vectorizer.fit_transform(
        train_codes
    )

    X_test_char = char_vectorizer.transform(
        test_codes
    )

    print(
        "Character features:",
        X_train_char.shape[1]
    )

    # --------------------------------------------------------
    # WORD TF-IDF
    # --------------------------------------------------------

    print("\nCreating Word TF-IDF features...")

    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=10000,
        min_df=2,
        max_df=0.98,
        sublinear_tf=True
    )

    X_train_word = word_vectorizer.fit_transform(
        train_codes
    )

    X_test_word = word_vectorizer.transform(
        test_codes
    )

    print(
        "Word features:",
        X_train_word.shape[1]
    )

    # --------------------------------------------------------
    # SECURITY FEATURES
    # --------------------------------------------------------

    print("\nExtracting security features...")

    train_features = []

    for i, code in enumerate(train_codes):

        if i % 500 == 0:

            print(
                f"Training feature extraction: "
                f"{i}/{len(train_codes)}"
            )

        train_features.append(
            extract_security_features(code)
        )

    test_features = []

    for i, code in enumerate(test_codes):

        if i % 500 == 0:

            print(
                f"Test feature extraction: "
                f"{i}/{len(test_codes)}"
            )

        test_features.append(
            extract_security_features(code)
        )

    train_feature_df = pd.DataFrame(
        train_features
    )

    test_feature_df = pd.DataFrame(
        test_features
    )

    train_feature_df = train_feature_df.fillna(0)

    test_feature_df = test_feature_df.fillna(0)

    # Ensure feature columns match

    test_feature_df = test_feature_df.reindex(
        columns=train_feature_df.columns,
        fill_value=0
    )

    X_train_engineered = csr_matrix(
        train_feature_df.values
    )

    X_test_engineered = csr_matrix(
        test_feature_df.values
    )

    print(
        "Engineered security features:",
        X_train_engineered.shape[1]
    )

    # --------------------------------------------------------
    # COMBINE FEATURES
    # --------------------------------------------------------

    print("\nCombining all V8 features...")

    X_train = hstack(
        [
            X_train_char,
            X_train_word,
            X_train_engineered
        ],
        format="csr"
    )

    X_test = hstack(
        [
            X_test_char,
            X_test_word,
            X_test_engineered
        ],
        format="csr"
    )

    print("\nFinal feature dimensions:")

    print("Training:", X_train.shape)

    print("Testing:", X_test.shape)

    feature_artifacts = {

        "char_vectorizer": char_vectorizer,

        "word_vectorizer": word_vectorizer,

        "engineered_feature_columns":
            train_feature_df.columns.tolist()
    }

    return (
        X_train,
        X_test,
        feature_artifacts,
        train_feature_df,
        test_feature_df
    )


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def create_models():

    models = {

        "Logistic Regression":

            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                C=1.0
            ),

        "Linear SVM":

            LinearSVC(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                C=1.0,
                max_iter=5000
            ),

        "Random Forest":

            RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),

        "Extra Trees":

            ExtraTreesClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
    }

    return models


# ============================================================
# PREDICTION SCORES
# ============================================================

def get_prediction_scores(model, X):

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(X)

        return probabilities[:, 1]

    elif hasattr(model, "decision_function"):

        scores = model.decision_function(X)

        # Convert decision scores to approximate probabilities

        scores = np.clip(scores, -20, 20)

        probabilities = 1 / (1 + np.exp(-scores))

        return probabilities

    else:

        predictions = model.predict(X)

        return predictions.astype(float)


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test,
    threshold=DEFAULT_THRESHOLD
):

    scores = get_prediction_scores(
        model,
        X_test
    )

    predictions = (
        scores >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    try:

        auc = roc_auc_score(
            y_test,
            scores
        )

    except Exception:

        auc = 0

    cm = confusion_matrix(
        y_test,
        predictions
    )

    tn, fp, fn, tp = cm.ravel()

    return {

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "roc_auc": auc,

        "tn": tn,

        "fp": fp,

        "fn": fn,

        "tp": tp,

        "predictions": predictions,

        "scores": scores,

        "confusion_matrix": cm
    }


# ============================================================
# TRAINING
# ============================================================

def train_models(
    X_train,
    X_test,
    y_train,
    y_test,
    test_df
):

    print("\n" + "=" * 70)
    print("V8 MODEL TRAINING")
    print("=" * 70)

    models = create_models()

    results = []

    trained_models = {}

    best_model_name = None

    best_model = None

    best_f1 = -1

    error_records = []

    for model_name, model in models.items():

        print("\n" + "-" * 70)

        print(f"Training: {model_name}")

        start_time = time.time()

        model.fit(
            X_train,
            y_train
        )

        training_time = time.time() - start_time

        print(
            f"Training completed in "
            f"{training_time:.2f} seconds"
        )

        # ----------------------------------------------------
        # EVALUATION
        # ----------------------------------------------------

        evaluation = evaluate_model(
            model,
            X_test,
            y_test
        )

        print("\nResults:")

        print(
            f"Accuracy:  "
            f"{evaluation['accuracy'] * 100:.2f}%"
        )

        print(
            f"Precision: "
            f"{evaluation['precision'] * 100:.2f}%"
        )

        print(
            f"Recall:    "
            f"{evaluation['recall'] * 100:.2f}%"
        )

        print(
            f"F1 Score:  "
            f"{evaluation['f1'] * 100:.2f}%"
        )

        print(
            f"ROC-AUC:   "
            f"{evaluation['roc_auc'] * 100:.2f}%"
        )

        print("\nConfusion Matrix:")

        print(evaluation["confusion_matrix"])

        print("\nFalse Negatives:", evaluation["fn"])

        print("False Positives:", evaluation["fp"])

        # ----------------------------------------------------
        # SAVE RESULTS
        # ----------------------------------------------------

        result = {

            "model": model_name,

            "accuracy": evaluation["accuracy"],

            "precision": evaluation["precision"],

            "recall": evaluation["recall"],

            "f1": evaluation["f1"],

            "roc_auc": evaluation["roc_auc"],

            "training_time_seconds": training_time,

            "true_negative": evaluation["tn"],

            "false_positive": evaluation["fp"],

            "false_negative": evaluation["fn"],

            "true_positive": evaluation["tp"]
        }

        results.append(result)

        trained_models[model_name] = model

        # ----------------------------------------------------
        # ERROR ANALYSIS
        # ----------------------------------------------------

        predictions = evaluation["predictions"]

        scores = evaluation["scores"]

        for index in range(len(y_test)):

            actual = y_test[index]

            predicted = predictions[index]

            if actual != predicted:

                row = {}

                if index < len(test_df):

                    row = test_df.iloc[index].to_dict()

                row["model"] = model_name

                row["actual"] = int(actual)

                row["predicted"] = int(predicted)

                row["score_vulnerable"] = float(
                    scores[index]
                )

                row["error_type"] = (
                    "False Negative"
                    if actual == 1 and predicted == 0
                    else "False Positive"
                )

                error_records.append(row)

        # ----------------------------------------------------
        # BEST MODEL SELECTION
        # ----------------------------------------------------

        if evaluation["f1"] > best_f1:

            best_f1 = evaluation["f1"]

            best_model_name = model_name

            best_model = model

    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="f1",
        ascending=False
    )

    comparison_path = os.path.join(
        RESULT_DIR,
        "v8_model_comparison.csv"
    )

    results_df.to_csv(
        comparison_path,
        index=False
    )

    print("\n" + "=" * 70)

    print("V8 MODEL COMPARISON")

    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )

    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    print("\nBest model:", best_model_name)

    best_model_path = os.path.join(
        MODEL_DIR,
        "best_model_v8.pkl"
    )

    joblib.dump(
        best_model,
        best_model_path
    )

    # Save all models

    all_models_path = os.path.join(
        MODEL_DIR,
        "all_models_v8.pkl"
    )

    joblib.dump(
        trained_models,
        all_models_path
    )

    # ========================================================
    # SAVE ERROR ANALYSIS
    # ========================================================

    errors_df = pd.DataFrame(
        error_records
    )

    if not errors_df.empty:

        errors_path = os.path.join(
            ERROR_DIR,
            "v8_model_errors.csv"
        )

        errors_df.to_csv(
            errors_path,
            index=False
        )

        # Best model errors

        best_errors = errors_df[
            errors_df["model"] == best_model_name
        ]

        best_errors_path = os.path.join(
            ERROR_DIR,
            "v8_best_model_errors.csv"
        )

        best_errors.to_csv(
            best_errors_path,
            index=False
        )

    # ========================================================
    # BEST MODEL REPORT
    # ========================================================

    best_evaluation = evaluate_model(
        best_model,
        X_test,
        y_test
    )

    report_path = os.path.join(
        RESULT_DIR,
        "v8_training_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write("=" * 70 + "\n")

        file.write("V8 MODEL TRAINING REPORT\n")

        file.write("=" * 70 + "\n\n")

        file.write(
            f"Best model: {best_model_name}\n\n"
        )

        file.write(
            f"Accuracy: "
            f"{best_evaluation['accuracy'] * 100:.2f}%\n"
        )

        file.write(
            f"Precision: "
            f"{best_evaluation['precision'] * 100:.2f}%\n"
        )

        file.write(
            f"Recall: "
            f"{best_evaluation['recall'] * 100:.2f}%\n"
        )

        file.write(
            f"F1 Score: "
            f"{best_evaluation['f1'] * 100:.2f}%\n"
        )

        file.write(
            f"ROC-AUC: "
            f"{best_evaluation['roc_auc'] * 100:.2f}%\n\n"
        )

        file.write(
            "CONFUSION MATRIX\n"
        )

        file.write(
            str(best_evaluation["confusion_matrix"])
            + "\n\n"
        )

        file.write(
            f"True Negative: "
            f"{best_evaluation['tn']}\n"
        )

        file.write(
            f"False Positive: "
            f"{best_evaluation['fp']}\n"
        )

        file.write(
            f"False Negative: "
            f"{best_evaluation['fn']}\n"
        )

        file.write(
            f"True Positive: "
            f"{best_evaluation['tp']}\n\n"
        )

        file.write(
            "MODEL COMPARISON\n\n"
        )

        file.write(
            results_df.to_string(
                index=False
            )
        )

    print("\nSaved model:")
    print(best_model_path)

    print("\nSaved comparison:")
    print(comparison_path)

    print("\nSaved report:")
    print(report_path)

    print("\nSaved error analysis:")
    print(ERROR_DIR)

    return (
        best_model,
        best_model_name,
        results_df,
        best_evaluation
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")

    print("=" * 70)

    print("WEB APPLICATION SECURITY AI TOOLKIT")

    print("V8 ADVANCED ML TRAINING")

    print("=" * 70)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    train_df, test_df = load_datasets()

    # --------------------------------------------------------
    # IDENTIFY COLUMNS
    # --------------------------------------------------------

    code_column = find_code_column(
        train_df
    )

    label_column = find_label_column(
        train_df
    )

    test_code_column = find_code_column(
        test_df
    )

    test_label_column = find_label_column(
        test_df
    )

    print("\nDetected training code column:")
    print(code_column)

    print("\nDetected training label column:")
    print(label_column)

    # --------------------------------------------------------
    # EXTRACT CODE
    # --------------------------------------------------------

    train_codes = (
        train_df[code_column]
        .fillna("")
        .astype(str)
        .tolist()
    )

    test_codes = (
        test_df[test_code_column]
        .fillna("")
        .astype(str)
        .tolist()
    )

    # --------------------------------------------------------
    # LABELS
    # --------------------------------------------------------

    y_train = normalize_labels(
        train_df[label_column]
    )

    y_test = normalize_labels(
        test_df[test_label_column]
    )

    print("\nTraining class distribution:")

    print(
        pd.Series(y_train)
        .value_counts()
        .sort_index()
    )

    print("\nTesting class distribution:")

    print(
        pd.Series(y_test)
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # FEATURE ENGINEERING
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        feature_artifacts,
        train_engineered,
        test_engineered
    ) = build_features(
        train_codes,
        test_codes
    )

    # --------------------------------------------------------
    # SAVE FEATURE INFORMATION
    # --------------------------------------------------------

    feature_info_path = os.path.join(
        RESULT_DIR,
        "v8_feature_information.csv"
    )

    feature_information = pd.DataFrame({

        "feature_name":
            feature_artifacts[
                "engineered_feature_columns"
            ]

    })

    feature_information.to_csv(
        feature_info_path,
        index=False
    )

    # --------------------------------------------------------
    # TRAIN MODELS
    # --------------------------------------------------------

    (
        best_model,
        best_model_name,
        results_df,
        best_evaluation
    ) = train_models(
        X_train,
        X_test,
        y_train,
        y_test,
        test_df
    )

    # --------------------------------------------------------
    # SAVE COMPLETE ARTIFACTS
    # --------------------------------------------------------

    complete_artifacts = {

        "model": best_model,

        "model_name": best_model_name,

        "char_vectorizer":
            feature_artifacts["char_vectorizer"],

        "word_vectorizer":
            feature_artifacts["word_vectorizer"],

        "engineered_feature_columns":
            feature_artifacts[
                "engineered_feature_columns"
            ],

        "feature_version": "V8",

        "threshold": DEFAULT_THRESHOLD

    }

    complete_artifact_path = os.path.join(
        MODEL_DIR,
        "v8_complete_pipeline.pkl"
    )

    joblib.dump(
        complete_artifacts,
        complete_artifact_path
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n")

    print("=" * 70)

    print("V8 TRAINING COMPLETED SUCCESSFULLY")

    print("=" * 70)

    print("\nBest model:", best_model_name)

    print(
        "Accuracy:",
        f"{best_evaluation['accuracy'] * 100:.2f}%"
    )

    print(
        "Precision:",
        f"{best_evaluation['precision'] * 100:.2f}%"
    )

    print(
        "Recall:",
        f"{best_evaluation['recall'] * 100:.2f}%"
    )

    print(
        "F1 Score:",
        f"{best_evaluation['f1'] * 100:.2f}%"
    )

    print(
        "ROC-AUC:",
        f"{best_evaluation['roc_auc'] * 100:.2f}%"
    )

    print("\nFalse Negatives:")

    print(
        best_evaluation["fn"]
    )

    print("\nFalse Positives:")

    print(
        best_evaluation["fp"]
    )

    print("\nModel saved to:")

    print(MODEL_DIR)

    print("\nReports saved to:")

    print(RESULT_DIR)

    print("\nError analysis saved to:")

    print(ERROR_DIR)

    print("\nComplete pipeline saved to:")

    print(complete_artifact_path)

    print("\n")


# ============================================================
# EXECUTION ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()