import os
import re
import time
import joblib
import numpy as np
import pandas as pd

from scipy.sparse import hstack, csr_matrix

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEST_DATA_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "v7",
    "balanced",
    "test_v7.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "v7",
    "best_model_v7_balanced.pkl"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "v7",
    "sourcewise_evaluation"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# SECURITY FEATURE PATTERNS
# ============================================================

SECURITY_PATTERNS = {
    "sql_keywords": [
        "select", "insert", "update", "delete",
        "drop", "union", "where"
    ],

    "sqli_patterns": [
        " or 1=1",
        "' or '1'='1",
        "union select",
        "drop table",
        "sql injection"
    ],

    "command_exec": [
        "runtime.getruntime",
        "processbuilder",
        "exec(",
        "system("
    ],

    "xss_patterns": [
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

    "crypto": [
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


INDIVIDUAL_PATTERNS = [
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


DANGEROUS_APIS = [
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


# ============================================================
# FEATURE EXTRACTION
# ============================================================

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


# ============================================================
# FEATURE MATRIX GENERATION
# ============================================================

def generate_features(code_data, package):

    char_vectorizer = package["char_vectorizer"]

    word_vectorizer = package["word_vectorizer"]

    scaler = package["scaler"]

    # Character TF-IDF
    char_features = char_vectorizer.transform(code_data)

    # Word TF-IDF
    word_features = word_vectorizer.transform(code_data)

    # Engineered features
    engineered_features = np.array([
        extract_features(code)
        for code in code_data
    ])

    # Apply trained scaler
    engineered_features = scaler.transform(
        engineered_features
    )

    engineered_features = csr_matrix(
        engineered_features
    )

    # Combine features
    combined_features = hstack(
        [
            char_features,
            word_features,
            engineered_features
        ],
        format="csr"
    )

    return combined_features


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, X, y):

    predictions = model.predict(X)

    accuracy = accuracy_score(y, predictions)

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )

    roc_auc = None

    try:

        probabilities = model.predict_proba(X)

        # Locate the vulnerable class (label 1) safely
        class_labels = list(model.classes_)
        vulnerable_index = class_labels.index(1)
        vulnerable_probability = probabilities[:, vulnerable_index]

        roc_auc = roc_auc_score(
            y,
            vulnerable_probability
        )

    except Exception:

        try:

            scores = model.decision_function(X)

            roc_auc = roc_auc_score(
                y,
                scores
            )

        except Exception:

            roc_auc = None

    cm = confusion_matrix(
        y,
        predictions,
        labels=[0, 1]
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "predictions": predictions
    }


# ============================================================
# CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(cm, source_name):

    safe_name = re.sub(
        r"[^a-zA-Z0-9_-]",
        "_",
        source_name
    )

    plt.figure(figsize=(7, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Safe", "Vulnerable"],
        yticklabels=["Safe", "Vulnerable"]
    )

    plt.title(
        f"V7 Confusion Matrix - {source_name}"
    )

    plt.xlabel("Predicted Label")

    plt.ylabel("Actual Label")

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        f"confusion_matrix_{safe_name}.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(f"Confusion matrix saved: {output_path}")


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("=" * 70)

    print("V7 SOURCE-WISE MODEL EVALUATION")

    print("=" * 70)

    start_time = time.time()

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("\nLoading test dataset...")

    test_df = pd.read_csv(TEST_DATA_PATH)

    print(f"Test samples: {len(test_df)}")

    # --------------------------------------------------------
    # Load trained model
    # --------------------------------------------------------

    print("\nLoading trained V7 model...")

    package = joblib.load(MODEL_PATH)

    model = package["model"]

    print(
        f"Model loaded: "
        f"{package.get('best_model_name', 'Extra Trees')}"
    )

    # --------------------------------------------------------
    # Prepare test data
    # --------------------------------------------------------

    code_column = "code_normalized"

    if code_column not in test_df.columns:

        raise ValueError(
            f"Missing column: {code_column}"
        )

    test_df[code_column] = (
        test_df[code_column]
        .fillna("")
        .astype(str)
    )

    # --------------------------------------------------------
    # Label conversion
    # --------------------------------------------------------

    label_mapping = package.get(
        "label_mapping",
        {
            "safe": 0,
            "vulnerable": 1
        }
    )

    print("\nLabel mapping:")

    print(label_mapping)

    if "label" not in test_df.columns:

        raise ValueError(
            "Test dataset does not contain label column"
        )

    test_df["label"] = (
        test_df["label"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    test_df["label_numeric"] = test_df["label"].map(
        label_mapping
    )

    if test_df["label_numeric"].isna().any():

        raise ValueError(
            "Unknown labels found in test dataset"
        )

    y = test_df["label_numeric"].astype(int).values

    # --------------------------------------------------------
    # Generate features
    # --------------------------------------------------------

    print("\nGenerating features...")

    X = generate_features(
        test_df[code_column].tolist(),
        package
    )

    print(
        f"Feature matrix shape: {X.shape}"
    )

    # --------------------------------------------------------
    # Overall evaluation
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print("OVERALL TEST PERFORMANCE")

    print("=" * 70)

    overall_results = evaluate_model(
        model,
        X,
        y
    )

    print(
        f"Accuracy : "
        f"{overall_results['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{overall_results['precision']:.4f}"
    )

    print(
        f"Recall   : "
        f"{overall_results['recall']:.4f}"
    )

    print(
        f"F1 Score : "
        f"{overall_results['f1']:.4f}"
    )

    if overall_results["roc_auc"] is not None:

        print(
            f"ROC-AUC  : "
            f"{overall_results['roc_auc']:.4f}"
        )

    print("\nOverall Confusion Matrix:")

    print(
        overall_results["confusion_matrix"]
    )

    # --------------------------------------------------------
    # Source-wise evaluation
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print("SOURCE-WISE EVALUATION")

    print("=" * 70)

    source_column = "source"

    if source_column not in test_df.columns:

        raise ValueError(
            "Test dataset does not contain source column"
        )

    results_list = []

    report_lines = []

    report_lines.append(
        "V7 SOURCE-WISE EVALUATION REPORT"
    )

    report_lines.append("=" * 70)

    report_lines.append(
        f"Total test samples: {len(test_df)}"
    )

    report_lines.append("")

    report_lines.append(
        "OVERALL PERFORMANCE"
    )

    report_lines.append("-" * 70)

    report_lines.append(
        f"Accuracy : {overall_results['accuracy']:.4f}"
    )

    report_lines.append(
        f"Precision: {overall_results['precision']:.4f}"
    )

    report_lines.append(
        f"Recall   : {overall_results['recall']:.4f}"
    )

    report_lines.append(
        f"F1 Score : {overall_results['f1']:.4f}"
    )

    if overall_results["roc_auc"] is not None:

        report_lines.append(
            f"ROC-AUC  : {overall_results['roc_auc']:.4f}"
        )

    report_lines.append("")

    # --------------------------------------------------------
    # Evaluate every source
    # --------------------------------------------------------

    for source_name in test_df[source_column].unique():

        print("\n" + "-" * 70)

        print(f"SOURCE: {source_name}")

        print("-" * 70)

        source_mask = (
            test_df[source_column] == source_name
        )

        source_df = test_df[source_mask]

        source_X = X[source_mask.values]

        source_y = y[source_mask.values]

        print(
            f"Samples: {len(source_df)}"
        )

        print(
            f"Safe: {(source_y == 0).sum()}"
        )

        print(
            f"Vulnerable: {(source_y == 1).sum()}"
        )

        # Evaluate source
        source_results = evaluate_model(
            model,
            source_X,
            source_y
        )

        print(
            f"Accuracy : "
            f"{source_results['accuracy']:.4f}"
        )

        print(
            f"Precision: "
            f"{source_results['precision']:.4f}"
        )

        print(
            f"Recall   : "
            f"{source_results['recall']:.4f}"
        )

        print(
            f"F1 Score : "
            f"{source_results['f1']:.4f}"
        )

        if source_results["roc_auc"] is not None:

            print(
                f"ROC-AUC  : "
                f"{source_results['roc_auc']:.4f}"
            )

        print("\nConfusion Matrix:")

        print(
            source_results["confusion_matrix"]
        )

        print("\nClassification Report:")

        print(
            classification_report(
                source_y,
                source_results["predictions"],
                labels=[0, 1],
                target_names=[
                    "Safe",
                    "Vulnerable"
                ],
                zero_division=0
            )
        )

        # Save confusion matrix
        save_confusion_matrix(
            source_results["confusion_matrix"],
            source_name
        )

        # ----------------------------------------------------
        # Store metrics
        # ----------------------------------------------------

        results_list.append({
            "source": source_name,
            "samples": len(source_df),
            "safe_samples": int(
                (source_y == 0).sum()
            ),
            "vulnerable_samples": int(
                (source_y == 1).sum()
            ),
            "accuracy": source_results["accuracy"],
            "precision": source_results["precision"],
            "recall": source_results["recall"],
            "f1_score": source_results["f1"],
            "roc_auc": source_results["roc_auc"]
        })

        # ----------------------------------------------------
        # Report
        # ----------------------------------------------------

        report_lines.append(
            f"SOURCE: {source_name}"
        )

        report_lines.append("-" * 70)

        report_lines.append(
            f"Samples: {len(source_df)}"
        )

        report_lines.append(
            f"Safe samples: {(source_y == 0).sum()}"
        )

        report_lines.append(
            f"Vulnerable samples: {(source_y == 1).sum()}"
        )

        report_lines.append(
            f"Accuracy : {source_results['accuracy']:.4f}"
        )

        report_lines.append(
            f"Precision: {source_results['precision']:.4f}"
        )

        report_lines.append(
            f"Recall   : {source_results['recall']:.4f}"
        )

        report_lines.append(
            f"F1 Score : {source_results['f1']:.4f}"
        )

        if source_results["roc_auc"] is not None:

            report_lines.append(
                f"ROC-AUC  : {source_results['roc_auc']:.4f}"
            )

        report_lines.append("")

        report_lines.append(
            "Confusion Matrix:"
        )

        report_lines.append(
            str(source_results["confusion_matrix"])
        )

        report_lines.append("")

        report_lines.append(
            "Classification Report:"
        )

        report_lines.append(
            classification_report(
                source_y,
                source_results["predictions"],
                labels=[0, 1],
                target_names=[
                    "Safe",
                    "Vulnerable"
                ],
                zero_division=0
            )
        )

        report_lines.append("=" * 70)

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results_list
    )

    csv_path = os.path.join(
        OUTPUT_DIR,
        "sourcewise_metrics.csv"
    )

    results_df.to_csv(
        csv_path,
        index=False
    )

    print("\nCSV saved:")

    print(csv_path)

    # --------------------------------------------------------
    # Save TXT report
    # --------------------------------------------------------

    report_lines.append("")

    report_lines.append(
        f"Execution time: "
        f"{time.time() - start_time:.2f} seconds"
    )

    report_path = os.path.join(
        OUTPUT_DIR,
        "sourcewise_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "\n".join(report_lines)
        )

    print("\nReport saved:")

    print(report_path)

    # --------------------------------------------------------
    # Final table
    # --------------------------------------------------------

    print("\n" + "=" * 100)

    print("FINAL SOURCE-WISE COMPARISON")

    print("=" * 100)

    print(
        results_df.to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)

    print("EVALUATION COMPLETED SUCCESSFULLY")

    print("=" * 70)

    print(
        f"Total execution time: "
        f"{time.time() - start_time:.2f} seconds"
    )


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":

    main()