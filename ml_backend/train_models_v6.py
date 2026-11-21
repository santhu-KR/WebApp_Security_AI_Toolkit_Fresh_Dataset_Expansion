import os
import time
import json
import pickle
import warnings

import numpy as np
import pandas as pd

from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
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

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "v6",
    "train_v6.csv"
)

TEST_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "v6",
    "test_v6.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models", "v6")
RESULT_DIR = os.path.join(BASE_DIR, "results", "v6")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

RANDOM_STATE = 42

TEXT_COLUMN = "code"
LABEL_COLUMN = "label"


# ============================================================
# SECURITY FEATURE EXTRACTION
# ============================================================

def extract_security_features(code):

    code = str(code).lower()

    features = {}

    keywords = [
        "eval",
        "exec",
        "system",
        "subprocess",
        "os.system",
        "shell",
        "sql",
        "select",
        "insert",
        "update",
        "delete",
        "execute",
        "query",
        "cursor",
        "input",
        "request",
        "getparameter",
        "getattribute",
        "password",
        "secret",
        "token",
        "key",
        "cookie",
        "session",
        "authenticate",
        "authorization",
        "permission",
        "admin",
        "escape",
        "sanitize",
        "validate",
        "encode",
        "decode",
        "html",
        "script",
        "javascript",
        "innerhtml",
        "outerhtml",
        "document.write",
        "redirect",
        "url",
        "path",
        "file",
        "open",
        "read",
        "write",
        "upload",
        "download",
        "chmod",
        "crypto",
        "random",
        "hash",
        "md5",
        "sha1",
        "ssl",
        "tls",
        "certificate",
        "debug",
        "trace",
        "exception",
        "error",
        "stack",
        "cors",
        "csrf",
        "xml",
        "xpath",
        "ldap",
        "deserialize",
        "pickle",
        "yaml",
        "json"
    ]

    for keyword in keywords:
        features[f"keyword_{keyword}"] = code.count(keyword)

    # ========================================================
    # General code characteristics
    # ========================================================

    features["code_length"] = len(code)

    features["line_count"] = len(code.splitlines())

    features["semicolon_count"] = code.count(";")

    features["quote_count"] = (
        code.count("'") +
        code.count('"')
    )

    features["parenthesis_count"] = (
        code.count("(") +
        code.count(")")
    )

    features["brace_count"] = (
        code.count("{") +
        code.count("}")
    )

    features["bracket_count"] = (
        code.count("[") +
        code.count("]")
    )

    features["comment_count"] = (
        code.count("#") +
        code.count("//") +
        code.count("/*")
    )

    # ========================================================
    # SQL-related features
    # ========================================================

    sql_keywords = [
        "select",
        "from",
        "where",
        "union",
        "insert",
        "update",
        "delete",
        "drop",
        "create",
        "alter",
        "or 1=1",
        "or '1'='1",
        "--",
        "/*",
        "order by",
        "group by"
    ]

    for keyword in sql_keywords:
        features[f"sql_{keyword}"] = code.count(keyword)

    # ========================================================
    # XSS-related features
    # ========================================================

    xss_keywords = [
        "<script",
        "</script>",
        "javascript:",
        "onerror",
        "onload",
        "onclick",
        "onmouseover",
        "innerhtml",
        "document.write",
        "alert(",
        "prompt(",
        "confirm("
    ]

    for keyword in xss_keywords:
        features[f"xss_{keyword}"] = code.count(keyword)

    # ========================================================
    # Command injection features
    # ========================================================

    command_keywords = [
        "os.system",
        "subprocess",
        "shell=true",
        "exec(",
        "eval(",
        "popen",
        "system(",
        "bash",
        "cmd.exe",
        "/bin/sh"
    ]

    for keyword in command_keywords:
        features[f"command_{keyword}"] = code.count(keyword)

    # ========================================================
    # Path traversal features
    # ========================================================

    path_keywords = [
        "../",
        "..\\",
        "/etc/",
        "/windows/",
        "file://",
        "filepath",
        "filename",
        "path.join",
        "open("
    ]

    for keyword in path_keywords:
        features[f"path_{keyword}"] = code.count(keyword)

    # ========================================================
    # Authentication features
    # ========================================================

    auth_keywords = [
        "password",
        "passwd",
        "login",
        "authenticate",
        "authorization",
        "admin",
        "session",
        "token",
        "jwt",
        "cookie"
    ]

    for keyword in auth_keywords:
        features[f"auth_{keyword}"] = code.count(keyword)

    # ========================================================
    # Cryptographic features
    # ========================================================

    crypto_keywords = [
        "md5",
        "sha1",
        "sha256",
        "sha512",
        "hash",
        "encrypt",
        "decrypt",
        "random",
        "rand",
        "crypto",
        "cipher",
        "secret"
    ]

    for keyword in crypto_keywords:
        features[f"crypto_{keyword}"] = code.count(keyword)

    return features


# ============================================================
# FEATURE GENERATION
# ============================================================

def build_features(train_text, test_text):

    print("\nGenerating TF-IDF features...")

    # Character-level features

    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        max_features=15000,
        min_df=2,
        max_df=0.98,
        sublinear_tf=True
    )

    X_train_char = char_vectorizer.fit_transform(train_text)

    X_test_char = char_vectorizer.transform(test_text)

    print("Character features:", X_train_char.shape)


    # Word-level features

    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=10000,
        min_df=2,
        max_df=0.98,
        sublinear_tf=True
    )

    X_train_word = word_vectorizer.fit_transform(train_text)

    X_test_word = word_vectorizer.transform(test_text)

    print("Word features:", X_train_word.shape)


    # Engineered security features

    print("Extracting engineered security features...")

    train_security_features = [
        extract_security_features(code)
        for code in train_text
    ]

    test_security_features = [
        extract_security_features(code)
        for code in test_text
    ]

    dict_vectorizer = DictVectorizer()

    X_train_security = dict_vectorizer.fit_transform(
        train_security_features
    )

    X_test_security = dict_vectorizer.transform(
        test_security_features
    )

    scaler = StandardScaler(with_mean=False)

    X_train_security = scaler.fit_transform(
        X_train_security
    )

    X_test_security = scaler.transform(
        X_test_security
    )

    print("Security features:", X_train_security.shape)


    # Combine all feature representations

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

    print("\nCombined feature shape:")
    print("Train:", X_train.shape)
    print("Test:", X_test.shape)

    feature_objects = {
        "char_vectorizer": char_vectorizer,
        "word_vectorizer": word_vectorizer,
        "dict_vectorizer": dict_vectorizer,
        "scaler": scaler
    }

    return X_train, X_test, feature_objects


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():

    print("=" * 60)
    print("LOADING V6 DATASET")
    print("=" * 60)

    train_df = pd.read_csv(TRAIN_FILE)

    test_df = pd.read_csv(TEST_FILE)

    print("\nTrain dataset:", train_df.shape)

    print("Test dataset:", test_df.shape)

    print("\nTrain labels:")

    print(train_df[LABEL_COLUMN].value_counts())

    print("\nTest labels:")

    print(test_df[LABEL_COLUMN].value_counts())

    X_train_text = train_df[TEXT_COLUMN].fillna("").astype(str)

    X_test_text = test_df[TEXT_COLUMN].fillna("").astype(str)

    y_train = train_df[LABEL_COLUMN].astype(int)

    y_test = test_df[LABEL_COLUMN].astype(int)

    return (
        train_df,
        test_df,
        X_train_text,
        X_test_text,
        y_train,
        y_test
    )


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def get_models():

    models = {

        "Logistic Regression": LogisticRegression(
            C=0.5,
            class_weight="balanced",
            max_iter=3000,
            solver="liblinear",
            random_state=RANDOM_STATE
        ),

        "Linear SVM": LinearSVC(
            C=0.5,
            class_weight="balanced",
            max_iter=5000,
            random_state=RANDOM_STATE
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=30,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "Extra Trees": ExtraTreesClassifier(
            n_estimators=100,
            max_depth=30,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    }

    return models


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model_name,
    model,
    X_train,
    X_test,
    y_train,
    y_test
):

    print("\n" + "=" * 60)

    print("MODEL:", model_name)

    print("=" * 60)

    # Training

    start_time = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    print(f"Training time: {training_time:.2f} seconds")


    # Predictions

    y_pred = model.predict(X_test)


    # Probability / decision scores

    if hasattr(model, "predict_proba"):

        y_scores = model.predict_proba(X_test)[:, 1]

    elif hasattr(model, "decision_function"):

        y_scores = model.decision_function(X_test)

    else:

        y_scores = y_pred


    # Metrics

    accuracy = accuracy_score(y_test, y_pred)

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

        roc_auc = roc_auc_score(
            y_test,
            y_scores
        )

    except Exception:

        roc_auc = 0.0


    matrix = confusion_matrix(
        y_test,
        y_pred
    )


    print("\nMETRICS")

    print(f"Accuracy : {accuracy:.4f}")

    print(f"Precision: {precision:.4f}")

    print(f"Recall   : {recall:.4f}")

    print(f"F1 Score : {f1:.4f}")

    print(f"ROC-AUC  : {roc_auc:.4f}")


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

    matrix_file = os.path.join(
        RESULT_DIR,
        model_name.replace(" ", "_").lower()
        + "_confusion_matrix.csv"
    )

    pd.DataFrame(
        matrix,
        index=["Actual Safe", "Actual Vulnerable"],
        columns=["Predicted Safe", "Predicted Vulnerable"]
    ).to_csv(matrix_file)


    # Save model

    model_file = os.path.join(
        MODEL_DIR,
        model_name.replace(" ", "_").lower()
        + ".pkl"
    )

    with open(model_file, "wb") as file:

        pickle.dump(model, file)


    results = {

        "Model": model_name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1_Score": f1,

        "ROC_AUC": roc_auc,

        "Training_Time_Seconds": training_time,

        "True_Negatives": int(matrix[0][0]),

        "False_Positives": int(matrix[0][1]),

        "False_Negatives": int(matrix[1][0]),

        "True_Positives": int(matrix[1][1]),

        "Model_Path": model_file

    }

    return results, model, y_pred, y_scores


# ============================================================
# MAIN TRAINING
# ============================================================

def main():

    print("\n")
    print("#" * 70)

    print("WEB APPLICATION SECURITY AI TOOLKIT")

    print("V6 MODEL COMPARISON")

    print("#" * 70)


    # Load data

    (
        train_df,
        test_df,
        X_train_text,
        X_test_text,
        y_train,
        y_test
    ) = load_dataset()


    # Build features

    X_train, X_test, feature_objects = build_features(
        X_train_text,
        X_test_text
    )


    # Save feature extraction objects

    feature_file = os.path.join(
        MODEL_DIR,
        "v6_feature_extractors.pkl"
    )

    with open(feature_file, "wb") as file:

        pickle.dump(
            feature_objects,
            file
        )


    # Model comparison

    models = get_models()

    all_results = []

    trained_models = {}

    predictions = {}

    scores = {}


    # Train all models

    for model_name, model in models.items():

        result, trained_model, y_pred, y_score = evaluate_model(
            model_name,
            model,
            X_train,
            X_test,
            y_train,
            y_test
        )

        all_results.append(result)

        trained_models[model_name] = trained_model

        predictions[model_name] = y_pred

        scores[model_name] = y_score


    # ========================================================
    # COMPARISON TABLE
    # ========================================================

    results_df = pd.DataFrame(all_results)

    results_df = results_df.sort_values(
        by="F1_Score",
        ascending=False
    )


    comparison_file = os.path.join(
        RESULT_DIR,
        "v6_model_comparison.csv"
    )

    results_df.to_csv(
        comparison_file,
        index=False
    )


    print("\n")

    print("=" * 70)

    print("V6 MODEL COMPARISON")

    print("=" * 70)

    print(
        results_df[
            [
                "Model",
                "Accuracy",
                "Precision",
                "Recall",
                "F1_Score",
                "ROC_AUC",
                "Training_Time_Seconds"
            ]
        ].to_string(index=False)
    )


    # ========================================================
    # SELECT BEST MODEL
    # ========================================================

    best_model_name = results_df.iloc[0]["Model"]

    best_model = trained_models[best_model_name]

    best_result = results_df.iloc[0].to_dict()


    print("\n")

    print("=" * 70)

    print("BEST MODEL")

    print("=" * 70)

    print("Model:", best_model_name)

    print("F1 Score:", best_result["F1_Score"])

    print("Accuracy:", best_result["Accuracy"])

    print("Precision:", best_result["Precision"])

    print("Recall:", best_result["Recall"])


    # Save best model

    best_model_file = os.path.join(
        MODEL_DIR,
        "best_model_v6.pkl"
    )

    with open(best_model_file, "wb") as file:

        pickle.dump(
            best_model,
            file
        )


    # Save best model metadata

    best_model_metadata = {

        "version": "V6",

        "best_model": best_model_name,

        "selection_metric": "F1_Score",

        "accuracy": float(best_result["Accuracy"]),

        "precision": float(best_result["Precision"]),

        "recall": float(best_result["Recall"]),

        "f1_score": float(best_result["F1_Score"]),

        "roc_auc": float(best_result["ROC_AUC"]),

        "training_time_seconds": float(
            best_result["Training_Time_Seconds"]
        ),

        "dataset_train_size": len(train_df),

        "dataset_test_size": len(test_df),

        "random_state": RANDOM_STATE

    }


    metadata_file = os.path.join(
        RESULT_DIR,
        "best_model_v6_metadata.json"
    )

    with open(metadata_file, "w") as file:

        json.dump(
            best_model_metadata,
            file,
            indent=4
        )


    # ========================================================
    # SAVE FINAL SUMMARY
    # ========================================================

    print("\n")

    print("=" * 70)

    print("V6 TRAINING COMPLETED")

    print("=" * 70)


    print("\nGenerated files:")

    print("\nModels:")

    print(MODEL_DIR)

    print("\nResults:")

    print(RESULT_DIR)


    print("\nComparison CSV:")

    print(comparison_file)


    print("\nBest model:")

    print(best_model_file)


    print("\nMetadata:")

    print(metadata_file)


    print("\nFinal selected model:")

    print(best_model_name)


    print("\nDONE!")


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":

    main()