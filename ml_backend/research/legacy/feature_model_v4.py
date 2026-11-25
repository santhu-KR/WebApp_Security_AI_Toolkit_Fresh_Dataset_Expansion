"""
Feature Model V4
Web Application Security AI Toolkit

Improved feature extraction:
- Character TF-IDF
- Word TF-IDF
- Input/source detection
- SQL sink detection
- SQL injection pattern detection
- Parameterized query detection
- Sanitization indicators
- Command execution indicators
- XSS indicators
- Deserialization indicators
- Security configuration indicators

This is an ML classification model, not a standalone vulnerability proof.
"""

import math
import re

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression


# ============================================================
# SECURITY PATTERNS
# ============================================================

PATTERNS = {

    # SQL Injection indicators
    "sql_keyword": r"\b(select|insert|update|delete|drop|alter|union|exec|execute)\b",

    "sql_dynamic_concat": (
        r"(select|insert|update|delete|union|exec|execute)"
        r".{0,300}(\+|%s|\$\{|\.format\s*\(|f['\"]|concat\s*\()"
    ),

    "sql_string_concat": (
        r"(['\"].{0,200}(select|insert|update|delete)"
        r".{0,200}['\"]\s*\+|\+\s*['\"]\s*(select|insert|update|delete))"
    ),

    "sql_raw_query": (
        r"(execute\s*\(|executeraw\s*\(|"
        r"raw\s*\(|preparestatement\s*\(|preparecall\s*\()"
    ),

    "sql_parameterized": (
        r"(preparedstatement|parameterized|"
        r"execute\s*\([^)]*,|"
        r"execute\s*\([^)]*\?\s*,|"
        r"executequery\s*\([^)]*\?)"
    ),

    # Input sources
    "http_input": (
        r"(request\.|request\[|req\.|"
        r"getparameter\s*\(|getheader\s*\(|"
        r"getquerystring\s*\(|"
        r"requestargs|requestform|requestjson|"
        r"queryparams|searchparams|"
        r"httprequest|httpcontext)"
    ),

    "user_input": (
        r"(user_?input|input_?value|"
        r"user_?data|form_?data|query_?data|"
        r"external_?input)"
    ),

    # Sanitization / validation
    "sanitization": (
        r"(sanitize|sanitise|escape|escaped|"
        r"htmlspecialchars|htmlentities|"
        r"encodeforhtml|encodeforjavascript|"
        r"validator|validate|validation|"
        r"allowlist|whitelist|regex)"
    ),

    # XSS
    "xss_sink": (
        r"(innerhtml|outerhtml|dangerouslysetinnerhtml|"
        r"document\.write|document\.writeln|"
        r"eval\s*\(|render_template_string)"
    ),

    "xss_input_output": (
        r"(request|params|input|query).{0,200}"
        r"(innerhtml|document\.write|html)"
    ),

    # Command injection
    "command_exec": (
        r"(os\.system|subprocess|"
        r"child_process\.exec|child_process\.spawn|"
        r"runtime\.getruntime|processbuilder|"
        r"shell\s*=\s*true)"
    ),

    # Deserialization
    "unsafe_deserialization": (
        r"(pickle\.loads|pickle\.load|"
        r"yaml\.load\s*\(|"
        r"objectinputstream|readobject\s*\(|"
        r"deserialize\s*\(|unserialize\s*\()"
    ),

    # Path traversal
    "path_traversal": (
        r"(\.\./|\.\.\\|"
        r"file\.open|open\s*\(|"
        r"filesystem|readfile|writefile)"
    ),

    # SSRF
    "ssrf": (
        r"(requests\.get|requests\.post|"
        r"urllib\.request|httpclient|"
        r"urlopen|fetch\s*\(|axios|"
        r"curl|webclient)"
    ),

    # Authentication
    "weak_auth": (
        r"(password|passwd|credential|"
        r"authentication|auth_token|"
        r"sessionid|session_token)"
    ),

    # Weak cryptography
    "weak_crypto": (
        r"(md5\s*\(|sha1\s*\(|"
        r"des\s*\(|rc4|"
        r"math\.random|random\.random)"
    ),

    # Hardcoded secrets
    "hardcoded_secret": (
        r"(api[_-]?key|secret|password|token)"
        r"\s*[:=]\s*['\"]"
    ),

    # TLS/security configuration
    "tls_disabled": (
        r"(verify\s*=\s*false|"
        r"sslverify\s*=\s*false|"
        r"trustall|"
        r"check_hostname\s*=\s*false)"
    ),

    # Debug mode
    "debug_enabled": (
        r"(debug\s*=\s*true|"
        r"debug\s*:\s*true|"
        r"app\.debug\s*=\s*true)"
    ),

    # Open redirect
    "redirect_input": (
        r"(redirect\s*\([^)]*(request|query|params|input)|"
        r"sendredirect\s*\([^)]*(request|input))"
    ),

    # Insecure cookies
    "cookie_security": (
        r"(httponly|secure|samesite)"
    ),

    # Dangerous dynamic execution
    "dynamic_eval": (
        r"(\beval\s*\(|"
        r"new\s+function\s*\(|"
        r"exec\s*\(|compile\s*\()"
    ),

}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def shannon_entropy(text):
    """Calculate Shannon entropy."""

    if not text:
        return 0.0

    entropy = 0.0

    for character in set(text):
        probability = text.count(character) / len(text)

        entropy -= probability * math.log2(probability)

    return entropy


def count_matches(pattern, text):
    """Count regex matches safely."""

    try:
        return len(re.findall(pattern, text, re.IGNORECASE | re.DOTALL))
    except re.error:
        return 0


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_feature_dicts(texts):

    output = []

    for text in texts:

        text = str(text)

        low = text.lower()

        compact = re.sub(r"\s+", "", text)

        features = {

            # General code features
            "length_log": math.log1p(len(text)),

            "entropy": shannon_entropy(compact),

            "line_count": text.count("\n") + 1,

            "quote_count": (
                text.count("'") +
                text.count('"')
            ),

            "operator_count": sum(
                text.count(operator)
                for operator in [
                    "+",
                    "%",
                    "${",
                    ".format",
                    "&&",
                    "||",
                ]
            ),

            "parentheses_count": (
                text.count("(") +
                text.count(")")
            ),

            "semicolon_count": text.count(";"),

            "brace_count": (
                text.count("{") +
                text.count("}")
            ),

            # Input features
            "input_reference": int(
                bool(
                    re.search(
                        PATTERNS["http_input"],
                        low,
                        re.IGNORECASE
                    )
                )
            ),

            "user_input_reference": int(
                bool(
                    re.search(
                        PATTERNS["user_input"],
                        low,
                        re.IGNORECASE
                    )
                )
            ),

            # SQL features
            "sql_keyword_present": int(
                bool(
                    re.search(
                        PATTERNS["sql_keyword"],
                        low,
                        re.IGNORECASE
                    )
                )
            ),

            "sql_dynamic_concat": int(
                bool(
                    re.search(
                        PATTERNS["sql_dynamic_concat"],
                        low,
                        re.IGNORECASE | re.DOTALL
                    )
                )
            ),

            "sql_string_concat": int(
                bool(
                    re.search(
                        PATTERNS["sql_string_concat"],
                        low,
                        re.IGNORECASE | re.DOTALL
                    )
                )
            ),

            "sql_raw_query": int(
                bool(
                    re.search(
                        PATTERNS["sql_raw_query"],
                        low,
                        re.IGNORECASE
                    )
                )
            ),

            "sql_parameterized": int(
                bool(
                    re.search(
                        PATTERNS["sql_parameterized"],
                        low,
                        re.IGNORECASE
                    )
                )
            ),

            # Sanitization
            "sanitization_present": int(
                bool(
                    re.search(
                        PATTERNS["sanitization"],
                        low,
                        re.IGNORECASE
                    )
                )
            ),

            # Security sinks
            "xss_sink": int(
                bool(
                    re.search(
                        PATTERNS["xss_sink"],
                        low,
                        re.IGNORECASE
                    )
                )
            ),

            "command_exec": int(
                bool(
                    re.search(
                        PATTERNS["command_exec"],
                        low,
                        re.IGNORECASE
                    )
                )
            ),

            "unsafe_deserialization": int(
                bool(
                    re.search(
                        PATTERNS["unsafe_deserialization"],
                        low,
                        re.IGNORECASE
                    )
                )
            ),

            "path_traversal": int(
                bool(
                    re.search(
                        PATTERNS["path_traversal"],
                        low,
                        re.IGNORECASE
                    )
                )
            ),

            "ssrf": int(
                bool(
                    re.search(
                        PATTERNS["ssrf"],
                        low,
                        re.IGNORECASE
                    )
                )
            ),

            "weak_auth": int(
                bool(
                    re.search(
                        PATTERNS["weak_auth"],
                        low,
                        re.IGNORECASE
                    )
                )
            ),

            "weak_crypto": int(
                bool(
                    re.search(
                        PATTERNS["weak_crypto"],
                        low,
                        re.IGNORECASE
                    )
                )
            ),

            "hardcoded_secret": int(
                bool(
                    re.search(
                        PATTERNS["hardcoded_secret"],
                        low,
                        re.IGNORECASE
                    )
                )
            ),

            "tls_disabled": int(
                bool(
                    re.search(
                        PATTERNS["tls_disabled"],
                        low,
                        re.IGNORECASE
                    )
                )
            ),

            "debug_enabled": int(
                bool(
                    re.search(
                        PATTERNS["debug_enabled"],
                        low,
                        re.IGNORECASE
                    )
                )
            ),

            "redirect_input": int(
                bool(
                    re.search(
                        PATTERNS["redirect_input"],
                        low,
                        re.IGNORECASE
                    )
                )
            ),

            "cookie_security": int(
                bool(
                    re.search(
                        PATTERNS["cookie_security"],
                        low,
                        re.IGNORECASE
                    )
                )
            ),

            "dynamic_eval": int(
                bool(
                    re.search(
                        PATTERNS["dynamic_eval"],
                        low,
                        re.IGNORECASE
                    )
                )
            ),

        }

        # Pattern counts
        for name, pattern in PATTERNS.items():

            features["count_" + name] = count_matches(
                pattern,
                low
            )

        output.append(features)

    return output


# ============================================================
# MODEL BUILDER
# ============================================================

def build_model():

    features = FeatureUnion(

        [

            # Character TF-IDF
            (
                "char_tfidf",

                TfidfVectorizer(

                    analyzer="char",

                    ngram_range=(2, 5),

                    min_df=2,

                    max_features=30000,

                    sublinear_tf=True,

                ),
            ),

            # Word TF-IDF
            (
                "word_tfidf",

                TfidfVectorizer(

                    analyzer="word",

                    token_pattern=r"(?u)\b\w+\b",

                    ngram_range=(1, 2),

                    min_df=2,

                    max_features=20000,

                    sublinear_tf=True,

                ),
            ),

            # Engineered security features
            (
                "engineered",

                Pipeline(

                    [

                        (
                            "extract",

                            FunctionTransformer(
                                extract_feature_dicts,
                                validate=False
                            ),
                        ),

                        (
                            "vectorize",

                            DictVectorizer(
                                sparse=True
                            ),
                        ),

                        (
                            "scale",

                            StandardScaler(
                                with_mean=False
                            ),
                        ),

                    ]

                ),
            ),

        ]

    )


    classifier = LogisticRegression(

        solver="liblinear",

        dual=False,

        max_iter=5000,

        tol=1e-4,

        class_weight="balanced",

        C=1.0,

        random_state=42,

    )


    model = Pipeline(

        [

            (
                "features",
                features
            ),

            (
                "classifier",
                classifier
            ),

        ]

    )

    return model


# ============================================================
# FEATURE TEST
# ============================================================

if __name__ == "__main__":

    samples = [

        # Vulnerable-style SQL example
        """
        String user = request.getParameter("username");
        String sql = "SELECT * FROM users WHERE name = '" + user + "'";
        statement.executeQuery(sql);
        """,

        # Parameterized SQL example
        """
        String sql = "SELECT * FROM users WHERE name = ?";
        PreparedStatement stmt = connection.prepareStatement(sql);
        stmt.setString(1, username);
        """,

        # Command execution example
        """
        subprocess.run(user_input, shell=True)
        """,

        # XSS example
        """
        element.innerHTML = request.getParameter("name");
        """,

    ]


    extracted = extract_feature_dicts(samples)

    print("\nFEATURE EXTRACTION TEST\n")

    for index, feature in enumerate(extracted, start=1):

        print(f"\nSample {index}")

        for key, value in feature.items():

            if value:

                print(f"{key}: {value}")