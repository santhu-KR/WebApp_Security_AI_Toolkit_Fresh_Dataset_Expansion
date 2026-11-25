"""Feature extraction and binary vulnerability classifier.

The engineered numeric features are scaled without centering so that the
combined feature matrix remains sparse. Logistic Regression uses LIBLINEAR's
dual L2 solver, which is suitable for this high-dimensional sparse feature
space when the number of features is larger than the number of samples.
"""

import math
import re

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression


PATTERNS = {
    "sql_dynamic": r"(select|insert|update|delete).*(\+|%s|\$\{|\.format\s*\()",
    "xss_sink": r"(innerhtml|dangerouslysetinnerhtml|document\.write|render_template_string)",
    "command_exec": r"(os\.system|subprocess|child_process\.exec|shell\s*=\s*true)",
    "dynamic_eval": r"(\beval\s*\(|new\s+function\s*\()",
    "unsafe_deser": r"(pickle\.loads|yaml\.load\s*\()",
    "weak_hash": r"\b(md5|sha1)\s*\(",
    "path_risk": r"(\.\./|\.\.\\|os\.path\.join|safe_join)",
    "secret_assignment": r"(api[_-]?key|secret|password|token)\s*[:=]",
    "tls_disabled": r"verify\s*=\s*false",
    "debug_enabled": r"debug\s*=\s*true",
    "redirect_input": r"redirect\s*\([^)]*(request|query|params)",
}


def shannon_entropy(text):
    if not text:
        return 0.0
    return -sum(
        (text.count(ch) / len(text)) * math.log2(text.count(ch) / len(text))
        for ch in set(text)
    )


def extract_feature_dicts(texts):
    output = []
    for text in texts:
        low = text.lower()
        compact = re.sub(r"\s+", "", text)
        row = {
            "length_log": math.log1p(len(text)),
            "entropy": shannon_entropy(compact),
            "line_count": text.count("\n") + 1,
            "quote_count": text.count("'") + text.count('"'),
            "operator_count": sum(
                text.count(x) for x in ["+", "%", "${", ".format", "&&"]
            ),
            "input_reference": int(
                bool(re.search(r"(request\.|req\.|params|user_?input|query\.)", low))
            ),
            "parameterized_query": int(
                bool(re.search(r"execute\s*\([^)]*,\s*\(", low))
            ),
        }
        for name, pattern in PATTERNS.items():
            row["signal_" + name] = int(bool(re.search(pattern, low, re.I | re.S)))
        output.append(row)
    return output


def build_model():
    features = FeatureUnion(
        [
            (
                "char_ngrams",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(2, 5),
                    min_df=1,
                    max_features=30000,
                    sublinear_tf=True,
                ),
            ),
            (
                "engineered",
                Pipeline(
                    [
                        (
                            "extract",
                            FunctionTransformer(extract_feature_dicts, validate=False),
                        ),
                        ("vectorize", DictVectorizer(sparse=True)),
                        # Do not center sparse matrices; preserve sparse format.
                        ("scale", StandardScaler(with_mean=False)),
                    ]
                ),
            ),
        ]
    )

    return Pipeline(
        [
            ("features", features),
            (
                "classifier",
                LogisticRegression(
                    solver="liblinear",
                    dual=True,
                    max_iter=5000,
                    tol=1e-4,
                    class_weight="balanced",
                    C=1.0,
                    random_state=42,
                ),
            ),
        ]
    )
