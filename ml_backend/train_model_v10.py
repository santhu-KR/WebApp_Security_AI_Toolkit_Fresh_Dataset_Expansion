"""
Web Application Security AI Toolkit - V10 Training Pipeline
Full replacement script.

Dataset inputs:
  dataset/v7/balanced/train_v7_balanced.csv
  dataset/v7/balanced/test_v7.csv

Outputs:
  models/v10/
  results/v10/model_training/
  results/v10/error_analysis/

This script trains multiple classifiers, selects a threshold on a validation split,
evaluates once on the untouched test set, performs error analysis, and saves artifacts.
"""

import os
import re
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")

SEED = 42
PRECISION_FLOOR = 0.90

TRAIN_PATH = Path("dataset/v7/balanced/train_v7_balanced.csv")
TEST_PATH = Path("dataset/v7/balanced/test_v7.csv")

MODEL_DIR = Path("models/v10")
RESULTS_DIR = Path("results/v10/model_training")
ERROR_DIR = Path("results/v10/error_analysis")

for folder in (MODEL_DIR, RESULTS_DIR, ERROR_DIR):
    folder.mkdir(parents=True, exist_ok=True)


def print_header(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def find_column(df, candidates):
    lower_map = {str(c).lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def normalize_labels(series):
    """Normalize common dataset label representations to 0=safe, 1=vulnerable."""
    if pd.api.types.is_numeric_dtype(series):
        vals = pd.to_numeric(series, errors="coerce")
        unique = set(vals.dropna().unique())
        if unique.issubset({0, 1}):
            return vals.astype(int)
        if unique.issubset({-1, 1}):
            return vals.map({-1: 0, 1: 1}).astype(int)

    mapping = {
        "safe": 0, "normal": 0, "benign": 0, "non-vulnerable": 0,
        "non_vulnerable": 0, "clean": 0, "0": 0, "false": 0,
        "vulnerable": 1, "vuln": 1, "unsafe": 1, "malicious": 1,
        "1": 1, "true": 1
    }
    normalized = series.astype(str).str.strip().str.lower().map(mapping)
    if normalized.isna().any():
        bad = series[normalized.isna()].value_counts().head(10).to_dict()
        raise ValueError(f"Unrecognized label values: {bad}")
    return normalized.astype(int)


def load_dataset(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Check that you are running this script from the ml_backend folder."
        )
    df = pd.read_csv(path)
    code_col = find_column(df, [
        "code_original", "code", "code_normalized", "text", "source_code",
        "payload", "snippet", "content"
    ])
    label_col = find_column(df, ["label", "label_normalized", "target", "class", "is_vulnerable"])
    if code_col is None or label_col is None:
        raise ValueError(f"Could not detect code/label columns. Columns: {list(df.columns)}")
    df = df.copy()
    df[code_col] = df[code_col].fillna("").astype(str)
    df[label_col] = normalize_labels(df[label_col])
    df = df[df[code_col].str.strip().ne("")].reset_index(drop=True)
    return df, code_col, label_col


# 49 engineered/contextual features (regex/keyword indicators and numeric signals)
FEATURE_PATTERNS = {
    "sql_select": r"\bselect\b",
    "sql_insert": r"\binsert\b",
    "sql_update": r"\bupdate\b",
    "sql_delete": r"\bdelete\b",
    "sql_union": r"\bunion\b",
    "sql_where": r"\bwhere\b",
    "sql_concat": r"(?:\+\s*\w+|%\s*\w+|f[\"'].*\{)",
    "sql_execute": r"\b(?:execute|exec|cursor\.execute|query)\b",
    "sql_parameterized": r"\b(?:preparedstatement|parameterized|execute\s*\([^)]*,|%\s*s|:\w+|\?\s*\))",
    "xss_innerhtml": r"\.innerhtml\s*=",
    "xss_outerhtml": r"\.outerhtml\s*=",
    "xss_document_write": r"\bdocument\.write\s*\(",
    "xss_dangerously": r"\bdangerouslysetinnerhtml\b",
    "xss_eval": r"\beval\s*\(",
    "xss_location": r"\b(?:location\.hash|location\.search|location\.href)\b",
    "xss_html_escape": r"\b(?:escape|htmlspecialchars|htmlencode|sanitizehtml)\b",
    "command_os_system": r"\bos\.system\s*\(",
    "command_shell_true": r"\bshell\s*=\s*true",
    "command_subprocess": r"\bsubprocess\.(?:run|popen|call|check_output)\s*\(",
    "command_exec": r"\b(?:exec|system|popen)\s*\(",
    "code_eval": r"\b(?:eval|exec)\s*\(",
    "path_traversal": r"\.\./|\.\\\.\\",
    "file_open": r"\b(?:open|file|readfile|writefile)\s*\(",
    "file_upload": r"\b(?:upload|multipart|filename|file_name)\b",
    "deserialization": r"\b(?:pickle\.loads|yaml\.load|unserialize|deserialize)\s*\(",
    "weak_crypto_md5": r"\bmd5\s*\(",
    "weak_crypto_sha1": r"\bsha1\s*\(",
    "weak_crypto_des": r"\bdes\b",
    "crypto_random": r"\b(?:random\.random|math\.random)\s*\(",
    "strong_crypto": r"\b(?:bcrypt|scrypt|argon2|pbkdf2|sha256|sha512)\b",
    "auth_password": r"\b(?:password|passwd|pwd)\b",
    "auth_login": r"\b(?:login|signin|authenticate)\b",
    "auth_session": r"\b(?:session|jwt|token)\b",
    "auth_cookie": r"\b(?:cookie|setcookie)\b",
    "cookie_secure": r"\bsecure\s*=\s*true|secure:\s*true",
    "cookie_httponly": r"\bhttponly\s*=\s*true|httponly:\s*true",
    "cookie_samesite": r"\bsamesite\b",
    "csrf": r"\b(?:csrf|xsrf|csrf_token|xsrf_token)\b",
    "validation": r"\b(?:validate|validation|validator|checkinput)\b",
    "sanitization": r"\b(?:sanitize|sanitise|strip_tags|clean_input)\b",
    "input_request": r"\b(?:request\.(?:get|post|args|data|form)|req\.(?:body|query|params))\b",
    "network_url": r"\b(?:https?://|urlopen|requests\.get|fetch\s*\()\b",
    "debug": r"\b(?:debug\s*=\s*true|debugger|traceback|stacktrace)\b",
    "error_handling": r"\b(?:try\s*:|catch\s*\(|except\s*:|finally\s*:)\b",
    "hardcoded_secret": r"\b(?:api[_-]?key|secret|password)\s*[:=]\s*[\"'][^\"']+[\"']",
    "permissions": r"\b(?:chmod|permission|authorize|authorization|access_control)\b",
    "redirect": r"\b(?:redirect|location\s*=|window\.location)\b",
    "ldap": r"\b(?:ldap|directorysearch|searchfilter)\b",
    "xml": r"\b(?:xml|etree|expat|documentbuilder)\b",
    "reflection": r"\b(?:reflect|reflection|getattr|setattr)\s*\(",
    "length": r"."
}


def extract_features(code):
    s = str(code)
    low = s.lower()
    feats = {}
    for name, pattern in FEATURE_PATTERNS.items():
        if name == "length":
            continue
        try:
            feats[name] = int(bool(re.search(pattern, low, re.IGNORECASE)))
        except re.error:
            feats[name] = 0

    # Numeric/context features
    feats["code_length"] = min(len(s), 100000)
    feats["line_count"] = min(s.count("\n") + 1, 5000)
    feats["word_count"] = min(len(re.findall(r"\b\w+\b", s)), 20000)
    feats["quote_count"] = s.count("'") + s.count('"')
    feats["semicolon_count"] = s.count(";")
    feats["bracket_count"] = sum(s.count(x) for x in "()[]{}")
    feats["comment_count"] = len(re.findall(r"(?m)^\s*(#|//|/\*|\*)", s))
    feats["uppercase_ratio"] = sum(c.isupper() for c in s) / max(len(s), 1)
    feats["digit_ratio"] = sum(c.isdigit() for c in s) / max(len(s), 1)
    feats["special_ratio"] = sum(not c.isalnum() and not c.isspace() for c in s) / max(len(s), 1)
    feats["has_user_input"] = int(bool(re.search(
        r"user_input|userinput|request|req\.|params|query|form|input|argv", low
    )))
    feats["has_sink"] = int(bool(re.search(
        r"execute|eval|exec|innerhtml|system|popen|write|open|query", low
    )))
    feats["input_sink_pair"] = int(feats["has_user_input"] and feats["has_sink"])
    feats["has_concat"] = int(bool(re.search(r"\+\s*\w+|\w+\s*\+", s)))
    feats["has_interpolation"] = int(bool(re.search(r"\{[^}]+\}|%\s*[sdif]", s)))
    feats["has_exception"] = int(bool(re.search(r"\btry\b|\bcatch\b|\bexcept\b", low)))
    feats["has_function"] = int(bool(re.search(r"\bdef\b|\bfunction\b|=>", low)))
    feats["has_variable"] = int(bool(re.search(r"\b(?:var|let|const)\s+\w+|\b\w+\s*=", s)))
    feats["has_comment"] = int(bool(re.search(r"#|//|/\*|\*/", s)))
    feats["has_http"] = int(bool(re.search(r"http://|https://", low)))
    feats["has_database"] = int(bool(re.search(r"database|db\.|cursor|sql|query", low)))
    feats["has_security_keyword"] = int(bool(re.search(
        r"security|secure|sanitize|escape|validate|encrypt|hash|auth", low
    )))
    return feats


def build_engineered_matrix(codes):
    rows = [extract_features(c) for c in codes]
    frame = pd.DataFrame(rows).fillna(0)
    return csr_matrix(frame.values.astype(float)), list(frame.columns)


def get_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    raw = model.decision_function(X)
    # Convert decision scores to a bounded score for threshold search/display.
    raw = np.clip(raw, -30, 30)
    return 1.0 / (1.0 + np.exp(-raw))


def choose_threshold(y_true, scores, precision_floor=PRECISION_FLOOR):
    thresholds = np.linspace(0.05, 0.95, 181)
    candidates = []
    for threshold in thresholds:
        pred = (scores >= threshold).astype(int)
        p = precision_score(y_true, pred, zero_division=0)
        r = recall_score(y_true, pred, zero_division=0)
        f = f1_score(y_true, pred, zero_division=0)
        if p >= precision_floor:
            candidates.append((f, r, p, threshold))
    if candidates:
        # Max F1, then recall, then precision.
        candidates.sort(reverse=True)
        return float(candidates[0][3])
    # If precision floor cannot be met, maximize F1.
    best = max(
        ((f1_score(y_true, (scores >= t).astype(int), zero_division=0), t)
         for t in thresholds),
        key=lambda x: x[0]
    )
    return float(best[1])


def calculate_metrics(y_true, pred, scores):
    metrics = {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, scores)
    except Exception:
        metrics["roc_auc"] = float("nan")
    return metrics


def main():
    print_header("WEB APPLICATION SECURITY AI TOOLKIT - V10")

    print("\n[1] Loading datasets...")
    train_df, code_col, label_col = load_dataset(TRAIN_PATH)
    test_df, test_code_col, test_label_col = load_dataset(TEST_PATH)

    print(f"Training dataset: {train_df.shape}")
    print(f"Testing dataset:  {test_df.shape}")
    print(f"Detected code column: {code_col}")
    print(f"Detected label column: {label_col}")
    print("\nTraining labels:")
    print(train_df[label_col].value_counts().sort_index())
    print("\nTesting labels:")
    print(test_df[test_label_col].value_counts().sort_index())

    X_train_text = train_df[code_col].astype(str).tolist()
    y_train_all = train_df[label_col].values.astype(int)
    X_test_text = test_df[test_code_col].astype(str).tolist()
    y_test = test_df[test_label_col].values.astype(int)

    print("\n[2] Extracting engineered security features...")
    X_train_engineered, feature_columns = build_engineered_matrix(X_train_text)
    X_test_engineered, _ = build_engineered_matrix(X_test_text)
    print(f"Engineered feature count: {len(feature_columns)}")

    print("\n[3] Creating TF-IDF features...")
    char_vectorizer = TfidfVectorizer(
        analyzer="char", ngram_range=(2, 5), max_features=15000,
        min_df=1, sublinear_tf=True
    )
    word_vectorizer = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), max_features=10000,
        min_df=1, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b"
    )

    X_train_char_all = char_vectorizer.fit_transform(X_train_text)
    X_test_char = char_vectorizer.transform(X_test_text)
    X_train_word_all = word_vectorizer.fit_transform(X_train_text)
    X_test_word = word_vectorizer.transform(X_test_text)

    X_train_all = hstack(
        [X_train_char_all, X_train_word_all, X_train_engineered],
        format="csr"
    )
    X_test = hstack(
        [X_test_char, X_test_word, X_test_engineered],
        format="csr"
    )

    print(f"Combined training matrix: {X_train_all.shape}")
    print(f"Combined testing matrix:  {X_test.shape}")

    print("\n[4] Creating validation split...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_all, y_train_all, test_size=0.20,
        stratify=y_train_all, random_state=SEED
    )
    print(f"Train subset: {X_train.shape[0]}")
    print(f"Validation:   {X_val.shape[0]}")
    print(f"Final test:  {X_test.shape[0]} (untouched)")

    models = {
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=500, max_features="sqrt", min_samples_leaf=1,
            class_weight="balanced", random_state=SEED, n_jobs=-1
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=400, max_features="sqrt", min_samples_leaf=1,
            class_weight="balanced", random_state=SEED, n_jobs=-1
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=3000, class_weight="balanced", C=2.0,
            solver="liblinear", random_state=SEED
        ),
        "LinearSVM": LinearSVC(
            C=1.0, class_weight="balanced", max_iter=10000,
            random_state=SEED
        )
    }

    validation_results = []
    trained_models = {}

    print("\n[5] Training and validation model comparison...")
    for name, model in models.items():
        print_header(f"Training: {name}")
        start = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start
        scores = get_scores(model, X_val)
        threshold = choose_threshold(y_val, scores)
        pred = (scores >= threshold).astype(int)
        metrics = calculate_metrics(y_val, pred, scores)

        row = {
            "Model": name,
            "Validation_Accuracy": metrics["accuracy"],
            "Validation_Precision": metrics["precision"],
            "Validation_Recall": metrics["recall"],
            "Validation_F1": metrics["f1"],
            "Validation_ROC_AUC": metrics["roc_auc"],
            "Threshold": threshold,
            "Training_Time_Seconds": elapsed
        }
        validation_results.append(row)
        trained_models[name] = model

        print(f"Validation Accuracy:  {metrics['accuracy'] * 100:.2f}%")
        print(f"Validation Precision: {metrics['precision'] * 100:.2f}%")
        print(f"Validation Recall:    {metrics['recall'] * 100:.2f}%")
        print(f"Validation F1:        {metrics['f1'] * 100:.2f}%")
        print(f"Validation ROC-AUC:   {metrics['roc_auc'] * 100:.2f}%")
        print(f"Selected Threshold:   {threshold:.3f}")
        print(f"Training Time:        {elapsed:.2f}s")

    comparison = pd.DataFrame(validation_results).sort_values(
        "Validation_F1", ascending=False
    )
    comparison.to_csv(RESULTS_DIR / "v10_validation_model_comparison.csv", index=False)
    best_name = comparison.iloc[0]["Model"]
    best_threshold = float(comparison.iloc[0]["Threshold"])
    best_model = trained_models[best_name]

    print_header("[6] Selected model")
    print(f"Best model based on validation F1: {best_name}")
    print(f"Threshold selected on validation data: {best_threshold:.3f}")

    print("\n[7] Final evaluation on untouched test dataset...")
    test_scores = get_scores(best_model, X_test)
    y_test_pred = (test_scores >= best_threshold).astype(int)
    test_metrics = calculate_metrics(y_test, y_test_pred, test_scores)
    cm = confusion_matrix(y_test, y_test_pred, labels=[0, 1])

    print_header("V10 FINAL TEST RESULTS")
    print(f"Model:      {best_name}")
    print(f"Threshold:  {best_threshold:.4f}")
    print(f"Accuracy:   {test_metrics['accuracy'] * 100:.2f}%")
    print(f"Precision:  {test_metrics['precision'] * 100:.2f}%")
    print(f"Recall:     {test_metrics['recall'] * 100:.2f}%")
    print(f"F1 Score:   {test_metrics['f1'] * 100:.2f}%")
    print(f"ROC-AUC:    {test_metrics['roc_auc'] * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_test_pred, labels=[0, 1],
        target_names=["Safe", "Vulnerable"], zero_division=0
    ))

    print("\n[8] Error analysis...")
    result = pd.DataFrame({
        "actual": y_test,
        "predicted": y_test_pred,
        "probability": test_scores,
        "error_type": "Correct",
        "source": test_df["source"].values if "source" in test_df.columns else "Unknown",
        "cwe": test_df["cwe"].values if "cwe" in test_df.columns else "Unknown",
        "code": X_test_text
    })
    result.loc[(result.actual == 0) & (result.predicted == 1), "error_type"] = "False Positive"
    result.loc[(result.actual == 1) & (result.predicted == 0), "error_type"] = "False Negative"

    # Parenthesized masks avoid pandas operator-precedence errors.
    fp = result[(result["error_type"] == "False Positive") & (result["actual"] == 0)]
    fn = result[(result["error_type"] == "False Negative") & (result["actual"] == 1)]
    tp = result[(result["error_type"] == "Correct") & (result["actual"] == 1)]
    tn = result[(result["error_type"] == "Correct") & (result["actual"] == 0)]

    result.to_csv(ERROR_DIR / "v10_test_predictions.csv", index=False)
    fp.to_csv(ERROR_DIR / "v10_false_positives.csv", index=False)
    fn.to_csv(ERROR_DIR / "v10_false_negatives.csv", index=False)
    tp.to_csv(ERROR_DIR / "v10_true_positives.csv", index=False)
    tn.to_csv(ERROR_DIR / "v10_true_negatives.csv", index=False)

    print(f"Correct predictions: {len(tp) + len(tn)}")
    print(f"True positives:      {len(tp)}")
    print(f"True negatives:      {len(tn)}")
    print(f"False positives:     {len(fp)}")
    print(f"False negatives:     {len(fn)}")

    print("\nFalse-negative source distribution:")
    if len(fn):
        print(fn["source"].value_counts().to_string())
        print("\nSample false negatives:")
        for _, row in fn.head(10).iterrows():
            print(f"\nProbability: {row['probability']:.4f} | Source: {row['source']} | CWE: {row['cwe']}")
            print(str(row["code"])[:400])
    else:
        print("No false negatives.")

    print("\nFalse-positive source distribution:")
    if len(fp):
        print(fp["source"].value_counts().to_string())
        print("\nSample false positives:")
        for _, row in fp.head(10).iterrows():
            print(f"\nProbability: {row['probability']:.4f} | Source: {row['source']} | CWE: {row['cwe']}")
            print(str(row["code"])[:400])
    else:
        print("No false positives.")

    print("\n[9] Saving model and artifacts...")
    joblib.dump(best_model, MODEL_DIR / "v10_best_model.pkl")
    joblib.dump(char_vectorizer, MODEL_DIR / "v10_char_vectorizer.pkl")
    joblib.dump(word_vectorizer, MODEL_DIR / "v10_word_vectorizer.pkl")
    joblib.dump(feature_columns, MODEL_DIR / "v10_feature_columns.pkl")
    joblib.dump(best_threshold, MODEL_DIR / "v10_threshold.pkl")

    metadata = {
        "version": "V10",
        "model_name": best_name,
        "threshold": best_threshold,
        "accuracy": float(test_metrics["accuracy"]),
        "precision": float(test_metrics["precision"]),
        "recall": float(test_metrics["recall"]),
        "f1_score": float(test_metrics["f1"]),
        "roc_auc": float(test_metrics["roc_auc"]),
        "training_samples": int(len(y_train_all)),
        "testing_samples": int(len(y_test)),
        "engineered_features": int(len(feature_columns)),
        "combined_features": int(X_train_all.shape[1]),
        "false_positives": int(len(fp)),
        "false_negatives": int(len(fn))
    }
    with open(MODEL_DIR / "v10_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    metrics_df = pd.DataFrame([{
        "Version": "V10",
        "Model": best_name,
        "Threshold": best_threshold,
        "Accuracy": test_metrics["accuracy"],
        "Precision": test_metrics["precision"],
        "Recall": test_metrics["recall"],
        "F1_Score": test_metrics["f1"],
        "ROC_AUC": test_metrics["roc_auc"],
        "False_Positives": len(fp),
        "False_Negatives": len(fn)
    }])
    metrics_df.to_csv(RESULTS_DIR / "v10_final_metrics.csv", index=False)

    pd.DataFrame(
        cm,
        index=["Actual Safe", "Actual Vulnerable"],
        columns=["Predicted Safe", "Predicted Vulnerable"]
    ).to_csv(RESULTS_DIR / "v10_confusion_matrix.csv")

    report = classification_report(
        y_test, y_test_pred, labels=[0, 1],
        target_names=["Safe", "Vulnerable"], output_dict=True,
        zero_division=0
    )
    pd.DataFrame(report).transpose().to_csv(
        RESULTS_DIR / "v10_classification_report.csv"
    )

    print_header("V10 COMPLETED SUCCESSFULLY")
    print(f"Best model: {best_name}")
    print(f"Accuracy:   {test_metrics['accuracy'] * 100:.2f}%")
    print(f"Precision:  {test_metrics['precision'] * 100:.2f}%")
    print(f"Recall:     {test_metrics['recall'] * 100:.2f}%")
    print(f"F1 Score:   {test_metrics['f1'] * 100:.2f}%")
    print(f"ROC-AUC:    {test_metrics['roc_auc'] * 100:.2f}%")
    print(f"\nSaved model artifacts to: {MODEL_DIR}")
    print(f"Saved metrics to: {RESULTS_DIR}")
    print(f"Saved error analysis to: {ERROR_DIR}")
    print("\nDone.")


if __name__ == "__main__":
    main()
