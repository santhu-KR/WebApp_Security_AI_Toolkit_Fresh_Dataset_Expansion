import os
import time
import json
import warnings

import numpy as np
import pandas as pd

from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler

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


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "processed",
    "security_snippets_clean.csv"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "v6",
    "unseen_source_validation"
)

os.makedirs(RESULT_DIR, exist_ok=True)

TEXT_COLUMN = "code"
LABEL_COLUMN = "label"
SOURCE_COLUMN = "source"

RANDOM_STATE = 42


# =========================================================
# FEATURE EXTRACTION
# =========================================================

def extract_security_features(code):

    code = str(code).lower()

    features = {}

    keywords = [
        "eval", "exec", "system", "subprocess",
        "os.system", "shell", "sql", "select",
        "insert", "update", "delete", "execute",
        "query", "cursor", "input", "request",
        "getparameter", "password", "secret",
        "token", "key", "cookie", "session",
        "authenticate", "authorization", "permission",
        "admin", "escape", "sanitize", "validate",
        "encode", "decode", "html", "script",
        "javascript", "innerhtml", "document.write",
        "redirect", "url", "path", "file", "open",
        "read", "write", "upload", "download",
        "chmod", "crypto", "random", "hash",
        "md5", "sha1", "ssl", "tls",
        "certificate", "debug", "trace",
        "exception", "error", "stack",
        "cors", "csrf", "xml", "xpath",
        "ldap", "deserialize", "pickle",
        "yaml", "json"
    ]

    for keyword in keywords:

        features[f"keyword_{keyword}"] = code.count(keyword)


    features["code_length"] = len(code)

    features["line_count"] = len(code.splitlines())

    features["semicolon_count"] = code.count(";")

    features["quote_count"] = (
        code.count("'") + code.count('"')
    )

    features["parenthesis_count"] = (
        code.count("(") + code.count(")")
    )

    features["brace_count"] = (
        code.count("{") + code.count("}")
    )

    features["bracket_count"] = (
        code.count("[") + code.count("]")
    )


    sql_keywords = [
        "select", "from", "where", "union",
        "insert", "update", "delete", "drop",
        "create", "alter", "or 1=1",
        "or '1'='1", "--", "/*",
        "order by", "group by"
    ]

    for keyword in sql_keywords:

        features[f"sql_{keyword}"] = code.count(keyword)


    xss_keywords = [
        "<script", "</script>", "javascript:",
        "onerror", "onload", "onclick",
        "onmouseover", "innerhtml",
        "document.write", "alert(",
        "prompt(", "confirm("
    ]

    for keyword in xss_keywords:

        features[f"xss_{keyword}"] = code.count(keyword)


    command_keywords = [
        "os.system", "subprocess",
        "shell=true", "exec(",
        "eval(", "popen", "system(",
        "bash", "cmd.exe", "/bin/sh"
    ]

    for keyword in command_keywords:

        features[f"command_{keyword}"] = code.count(keyword)


    path_keywords = [
        "../", "..\\", "/etc/",
        "/windows/", "file://",
        "filepath", "filename",
        "path.join", "open("
    ]

    for keyword in path_keywords:

        features[f"path_{keyword}"] = code.count(keyword)


    auth_keywords = [
        "password", "passwd", "login",
        "authenticate", "authorization",
        "admin", "session", "token",
        "jwt", "cookie"
    ]

    for keyword in auth_keywords:

        features[f"auth_{keyword}"] = code.count(keyword)


    crypto_keywords = [
        "md5", "sha1", "sha256",
        "sha512", "hash", "encrypt",
        "decrypt", "random", "rand",
        "crypto", "cipher", "secret"
    ]

    for keyword in crypto_keywords:

        features[f"crypto_{keyword}"] = code.count(keyword)


    return features


# =========================================================
# FEATURE BUILDING
# =========================================================

def build_features(train_text, test_text):

    print("Generating character TF-IDF...")

    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        max_features=15000,
        min_df=2,
        max_df=0.98,
        sublinear_tf=True
    )

    X_train_char = char_vectorizer.fit_transform(
        train_text
    )

    X_test_char = char_vectorizer.transform(
        test_text
    )


    print("Generating word TF-IDF...")

    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=10000,
        min_df=2,
        max_df=0.98,
        sublinear_tf=True
    )

    X_train_word = word_vectorizer.fit_transform(
        train_text
    )

    X_test_word = word_vectorizer.transform(
        test_text
    )


    print("Generating security features...")

    train_features = [
        extract_security_features(code)
        for code in train_text
    ]

    test_features = [
        extract_security_features(code)
        for code in test_text
    ]


    dict_vectorizer = DictVectorizer()

    X_train_security = dict_vectorizer.fit_transform(
        train_features
    )

    X_test_security = dict_vectorizer.transform(
        test_features
    )


    scaler = StandardScaler(with_mean=False)

    X_train_security = scaler.fit_transform(
        X_train_security
    )

    X_test_security = scaler.transform(
        X_test_security
    )


    X_train = hstack(
        [
            X_train_char,
            X_train_word,
            X_train_security
        ]
    ).tocsr()


    X_test = hstack(
        [
            X_test_char,
            X_test_word,
            X_test_security
        ]
    ).tocsr()


    print("Train features:", X_train.shape)

    print("Test features:", X_test.shape)


    return X_train, X_test


# =========================================================
# MODEL EVALUATION
# =========================================================

def evaluate_experiment(
    experiment_name,
    train_df,
    test_df
):

    print("\n")

    print("=" * 70)

    print(experiment_name)

    print("=" * 70)


    X_train_text = (
        train_df[TEXT_COLUMN]
        .fillna("")
        .astype(str)
    )

    X_test_text = (
        test_df[TEXT_COLUMN]
        .fillna("")
        .astype(str)
    )


    y_train = train_df[LABEL_COLUMN].astype(int)

    y_test = test_df[LABEL_COLUMN].astype(int)


    print("\nTraining samples:", len(train_df))

    print("Testing samples:", len(test_df))


    print("\nTraining label distribution:")

    print(y_train.value_counts())


    print("\nTesting label distribution:")

    print(y_test.value_counts()


    )


    # Feature extraction

    X_train, X_test = build_features(
        X_train_text,
        X_test_text
    )


    # Model

    model = ExtraTreesClassifier(
        n_estimators=100,
        max_depth=30,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )


    # Training

    print("\nTraining Extra Trees...")

    start_time = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start_time


    print(
        f"Training time: {training_time:.2f} seconds"
    )


    # Prediction

    y_pred = model.predict(X_test)

    y_scores = model.predict_proba(X_test)[:, 1]


    # Metrics

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

    roc_auc = roc_auc_score(
        y_test,
        y_scores
    )


    matrix = confusion_matrix(
        y_test,
        y_pred
    )


    print("\nRESULTS")

    print("Accuracy :", round(accuracy, 4))

    print("Precision:", round(precision, 4))

    print("Recall   :", round(recall, 4))

    print("F1 Score :", round(f1, 4))

    print("ROC-AUC  :", round(roc_auc, 4))


    print("\nConfusion Matrix")

    print(matrix)


    print("\nClassification Report")

    print(
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


    # Save confusion matrix

    safe_name = experiment_name.replace(
        " ",
        "_"
    ).lower()


    matrix_df = pd.DataFrame(
        matrix,
        index=[
            "Actual Safe",
            "Actual Vulnerable"
        ],
        columns=[
            "Predicted Safe",
            "Predicted Vulnerable"
        ]
    )


    matrix_df.to_csv(
        os.path.join(
            RESULT_DIR,
            safe_name + "_confusion_matrix.csv"
        )
    )


    # Save results

    return {

        "Experiment": experiment_name,

        "Training_Samples": len(train_df),

        "Testing_Samples": len(test_df),

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1_Score": f1,

        "ROC_AUC": roc_auc,

        "Training_Time_Seconds": training_time,

        "TN": int(matrix[0][0]),

        "FP": int(matrix[0][1]),

        "FN": int(matrix[1][0]),

        "TP": int(matrix[1][1])

    }


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)

    print("V6 UNSEEN-SOURCE VALIDATION")

    print("=" * 70)


    # Load dataset

    df = pd.read_csv(DATASET_PATH)


    print("\nDataset size:", len(df))


    print("\nSources:")

    print(df[SOURCE_COLUMN].value_counts())


    # Normalize labels

    df[LABEL_COLUMN] = df[LABEL_COLUMN].map({

        "safe": 0,

        "vulnerable": 1,

        0: 0,

        1: 1

    })


    df = df.dropna(
        subset=[
            SOURCE_COLUMN,
            TEXT_COLUMN,
            LABEL_COLUMN
        ]
    )


    sources = df[SOURCE_COLUMN].unique()


    if len(sources) != 3:

        print("\nWARNING:")

        print(
            "Expected 3 sources, but found:",
            len(sources)
        )


    print("\nAvailable sources:")

    for source in sources:

        print(source)


    results = []


    # =====================================================
    # EXPERIMENT 1
    # =====================================================

    source_a = "OWASP-Benchmark-Java"

    source_b = "original_curated"

    source_c = "NIST-SARD-Juliet-Java-1.3"


    train_df = df[
        df[SOURCE_COLUMN].isin([
            source_a,
            source_b
        ])
    ]

    test_df = df[
        df[SOURCE_COLUMN] == source_c
    ]


    results.append(
        evaluate_experiment(
            "Train OWASP + Curated Test Juliet",
            train_df,
            test_df
        )
    )


    # =====================================================
    # EXPERIMENT 2
    # =====================================================

    train_df = df[
        df[SOURCE_COLUMN].isin([
            source_a,
            source_c
        ])
    ]

    test_df = df[
        df[SOURCE_COLUMN] == source_b
    ]


    results.append(
        evaluate_experiment(
            "Train OWASP + Juliet Test Curated",
            train_df,
            test_df
        )
    )


    # =====================================================
    # EXPERIMENT 3
    # =====================================================

    train_df = df[
        df[SOURCE_COLUMN].isin([
            source_b,
            source_c
        ])
    ]

    test_df = df[
        df[SOURCE_COLUMN] == source_a
    ]


    results.append(
        evaluate_experiment(
            "Train Curated + Juliet Test OWASP",
            train_df,
            test_df
        )
    )


    # =====================================================
    # FINAL COMPARISON
    # =====================================================

    results_df = pd.DataFrame(
        results
    )


    results_df.to_csv(
        os.path.join(
            RESULT_DIR,
            "unseen_source_comparison.csv"
        ),
        index=False
    )


    print("\n")

    print("=" * 70)

    print("UNSEEN SOURCE VALIDATION SUMMARY")

    print("=" * 70)


    print(
        results_df[
            [
                "Experiment",
                "Accuracy",
                "Precision",
                "Recall",
                "F1_Score",
                "ROC_AUC"
            ]
        ].to_string(
            index=False
        )
    )


    print("\nResults saved to:")

    print(RESULT_DIR)


    print("\nVALIDATION COMPLETED!")


if __name__ == "__main__":

    main()