"""Feature extraction and binary vulnerability classifier - V2.

Adds lexical heuristic features for SQL construction, stored procedure calls,
request/cookie/custom input sources, JDBC APIs, and arithmetic comparisons.

IMPORTANT: These are lexical heuristics only. They do not perform Java
control-flow, constant-propagation, or data-flow analysis and cannot prove
that a value reaches a SQL sink or that a condition is always true/false.
"""

import math
import re

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression


PATTERNS = {
    # SQL syntax and construction
    "sql_keyword": r"\b(select|insert|update|delete|merge|create|alter|drop)\b",
    "sql_dynamic": (
        r"\b(select|insert|update|delete|merge)\b"
        r"[\s\S]{0,1200}(\+|%s|\$\{|\.format\s*\(|String\.format\s*\()"
    ),
    "sql_string_concatenation": (
        r"(?:[\"'][^\"']*\b(select|insert|update|delete)\b[^\"']*[\"']\s*\+"
        r"|\+\s*[\"'][^\"']*\b(where|values|set)\b)"
    ),
    "stored_procedure_call": r"\{\s*call\s+[^}]*\}",

    # Database APIs
    "jdbc_statement_api": (
        r"\b(createStatement|prepareCall|executeQuery|executeUpdate|"
        r"executeBatch|addBatch)\s*\("
    ),
    "prepared_statement_api": r"\b(prepareStatement|PreparedStatement)\b",

    # Potential untrusted input sources
    "request_parameter_source": (
        r"\b(getParameter|getQueryString|getHeader|getPathInfo|getRemoteUser)\s*\("
    ),
    "request_object_source": r"\b(request|httpRequest)\s*\.",
    "cookie_input_source": r"\b(getCookies|getCookie|getValue)\s*\(",
    "url_decode_source": r"\b(URLDecoder|decode)\s*\(",
    "generic_input_source": r"\b(req|request|params|userInput|user_input|query)\b",
    # Project/dataset-specific accessor observed in OWASP Benchmark examples.
    "custom_input_accessor": r"\b(getTheValue|getUserInput|getInputValue)\s*\(",

    # Other vulnerability-related lexical signals
    "xss_sink": (
        r"(innerHTML|dangerouslySetInnerHTML|document\.write|"
        r"render_template_string)"
    ),
    "command_exec": (
        r"(os\.system|subprocess|child_process\.exec|shell\s*=\s*true)"
    ),
    "dynamic_eval": r"(\beval\s*\(|new\s+function\s*\()",
    "unsafe_deser": r"(pickle\.loads|yaml\.load\s*\()",
    "weak_hash": r"\b(md5|sha1)\s*\(",
    "path_risk": r"(\.\./|\.\.\\|os\.path\.join|safe_join)",
    "secret_assignment": r"(api[_-]?key|secret|password|token)\s*[:=]",
    "tls_disabled": r"verify\s*=\s*false",
    "debug_enabled": r"debug\s*=\s*true",
    "redirect_input": r"redirect\s*\([^)]*(request|query|params)",
}


# Detects arithmetic expressions containing numeric operands and optionally
# variables, followed by a comparison with a numeric literal. Parentheses and
# whitespace are allowed. This is NOT a constant-expression evaluator.
ARITHMETIC_COMPARISON = re.compile(
    r"\(?\s*\d+\s*(?:\*|/|%|\+|-)\s*"
    r"(?:\d+|[A-Za-z_$][\w$]*)\s*\)?\s*"
    r"(?:[+\-*/%]\s*(?:\d+|[A-Za-z_$][\w$]*)\s*)*"
    r"(?:>=|<=|==|!=|>|<)\s*-?\d+",
    re.I,
)


def shannon_entropy(text):
    """Return Shannon character entropy for a string."""
    if not text:
        return 0.0
    return -sum(
        (text.count(ch) / len(text)) * math.log2(text.count(ch) / len(text))
        for ch in set(text)
    )


def extract_feature_dicts_v2(texts):
    """Convert source-code snippets into numeric and lexical feature dictionaries."""
    output = []

    for value in texts:
        text = "" if value is None else str(value)
        low = text.lower()
        compact = re.sub(r"\s+", "", text)

        matches = {
            name: bool(re.search(pattern, text, re.I | re.S))
            for name, pattern in PATTERNS.items()
        }

        sql_context = any(
            matches[name]
            for name in (
                "sql_keyword",
                "stored_procedure_call",
                "jdbc_statement_api",
            )
        )

        input_source = any(
            matches[name]
            for name in (
                "request_parameter_source",
                "request_object_source",
                "cookie_input_source",
                "url_decode_source",
                "generic_input_source",
                "custom_input_accessor",
            )
        )

        arithmetic_comparison = bool(ARITHMETIC_COMPARISON.search(text))

        row = {
            "length_log": math.log1p(len(text)),
            "entropy": shannon_entropy(compact),
            "line_count": text.count("\n") + 1,
            "quote_count": text.count("'") + text.count('"'),
            "operator_count": sum(
                text.count(token) for token in ("+", "%", "${", ".format", "&&")
            ),
            "input_reference": int(
                bool(re.search(r"(request\.|req\.|params|user_?input|query\.)", low))
            ),
            "sql_context_present": int(sql_context),
            "input_source_present": int(input_source),
            "potential_input_and_sql_cooccurrence": int(
                sql_context and input_source
            ),
            # Retain the old key for compatibility with existing test scripts.
            # It means a pattern was found, not that the condition is proven constant.
            "constant_arithmetic_condition": int(arithmetic_comparison),
            "arithmetic_comparison_present": int(arithmetic_comparison),
            "prepared_statement_api_present": int(matches["prepared_statement_api"]),
            "jdbc_statement_api_present": int(matches["jdbc_statement_api"]),
        }

        for name, matched in matches.items():
            row["signal_" + name] = int(matched)

        output.append(row)

    return output


def build_model():
    """Build the V2 text + engineered-feature classification pipeline."""
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
                            FunctionTransformer(
                                extract_feature_dicts_v2, validate=False
                            ),
                        ),
                        ("vectorize", DictVectorizer(sparse=True)),
                        ("scale", StandardScaler(with_mean=False)),
                    ]
                ),
            ),
        ]
    )

    # Keep V1 classifier settings initially so the first comparison isolates
    # feature changes. Solver changes should be evaluated separately.
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
