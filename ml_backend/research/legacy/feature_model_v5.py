import re
import math
from collections import Counter

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# ============================================================
# V5 SECURITY FEATURE EXTRACTION
# ============================================================

def extract_security_features(code):

    code = str(code)
    text = code.lower()

    features = {}

    # --------------------------------------------------------
    # General code characteristics
    # --------------------------------------------------------

    features["code_length"] = len(code)
    features["line_count"] = len(code.splitlines())
    features["quote_count"] = code.count("'") + code.count('"')
    features["semicolon_count"] = code.count(";")
    features["parenthesis_count"] = code.count("(") + code.count(")")
    features["brace_count"] = code.count("{") + code.count("}")

    # --------------------------------------------------------
    # SQL indicators
    # --------------------------------------------------------

    sql_keywords = [
        "select",
        "insert",
        "update",
        "delete",
        "drop",
        "union",
        "where",
        "from",
        "execute",
        "exec",
        "query"
    ]

    for keyword in sql_keywords:
        features[f"sql_keyword_{keyword}"] = text.count(keyword)

    # --------------------------------------------------------
    # SQL Injection - unsafe construction
    # --------------------------------------------------------

    sql_concat_patterns = [
        r"""["']\s*\+\s*[a-zA-Z_]\w*""",
        r"""[a-zA-Z_]\w*\s*\+\s*["']""",
        r"""f["'].*?\{.*?\}.*?(select|insert|update|delete)""",
        r"""(select|insert|update|delete).*?\+""",
        r"""(select|insert|update|delete).*?%s""",
        r"""(select|insert|update|delete).*?\.format\s*\(""",
        r"""(select|insert|update|delete).*?format\s*\(""",
        r"""(select|insert|update|delete).*?\$\{"""
    ]

    for i, pattern in enumerate(sql_concat_patterns):

        features[f"sql_unsafe_pattern_{i}"] = len(
            re.findall(
                pattern,
                text,
                re.IGNORECASE | re.DOTALL
            )
        )

    # --------------------------------------------------------
    # SQL safe parameterization indicators
    # --------------------------------------------------------

    safe_sql_patterns = [
        r"preparedstatement",
        r"preparestatement",
        r"parameterized",
        r"cursor\.execute\s*\([^,]+,",
        r"execute\s*\([^,]+,\s*\(",
        r"\?\s*,",
        r":\w+",
        r"\$\d+"
    ]

    for i, pattern in enumerate(safe_sql_patterns):

        features[f"sql_safe_pattern_{i}"] = len(
            re.findall(
                pattern,
                text,
                re.IGNORECASE
            )
        )

    # --------------------------------------------------------
    # SQL injection contextual indicators
    # --------------------------------------------------------

    features["sql_string_concat"] = int(
        bool(
            re.search(
                r"(select|insert|update|delete).*?(\+|%|\{)",
                text,
                re.IGNORECASE | re.DOTALL
            )
        )
    )

    features["sql_parameterized"] = int(
        bool(
            re.search(
                r"execute\s*\([^)]*,",
                text,
                re.IGNORECASE
            )
        )
    )

    # --------------------------------------------------------
    # User input indicators
    # --------------------------------------------------------

    user_input_patterns = [
        r"request\.getparameter",
        r"request\.args",
        r"request\.form",
        r"request\.data",
        r"input\s*\(",
        r"user_input",
        r"userinput",
        r"username",
        r"password",
        r"query_param",
        r"params",
        r"location\.hash",
        r"location\.search",
        r"request\.query",
        r"request\.json",
        r"getparameter",
        r"getparam"
    ]

    for i, pattern in enumerate(user_input_patterns):

        features[f"user_input_{i}"] = len(
            re.findall(
                pattern,
                text,
                re.IGNORECASE
            )
        )

    # --------------------------------------------------------
    # Cross-site scripting (XSS) indicators
    # --------------------------------------------------------

    xss_patterns = [
        r"innerhtml",
        r"outerhtml",
        r"document\.write",
        r"document\.writeln",
        r"dangerouslysetinnerhtml",
        r"response\.write",
        r"\.html\s*\(",
        r"\.append\s*\(",
        r"eval\s*\(",
        r"javascript:",
        r"onerror\s*=",
        r"onload\s*=",
        r"insertadjacenthtml",
        r"createelement",
        r"setattribute"
    ]

    for i, pattern in enumerate(xss_patterns):

        features[f"xss_pattern_{i}"] = len(
            re.findall(
                pattern,
                text,
                re.IGNORECASE
            )
        )

    # --------------------------------------------------------
    # Command injection indicators
    # --------------------------------------------------------

    command_patterns = [
        r"os\.system",
        r"subprocess\.popen",
        r"subprocess\.run",
        r"subprocess\.call",
        r"subprocess\.check_output",
        r"shell\s*=\s*true",
        r"exec\s*\(",
        r"popen\s*\(",
        r"command\s*\+",
        r"cmd\s*\+",
        r"runtime\.getruntime",
        r"processbuilder",
        r"child_process",
        r"execsync",
        r"spawn\s*\("
    ]

    for i, pattern in enumerate(command_patterns):

        features[f"command_pattern_{i}"] = len(
            re.findall(
                pattern,
                text,
                re.IGNORECASE
            )
        )

    # --------------------------------------------------------
    # Path traversal indicators
    # --------------------------------------------------------

    path_patterns = [
        r"\.\./",
        r"\.\.\\",
        r"path\.join",
        r"os\.path\.join",
        r"open\s*\(",
        r"file\s*\(",
        r"filename",
        r"filepath",
        r"file_path",
        r"realpath",
        r"normalize",
        r"getpath",
        r"readfile",
        r"writefile"
    ]

    for i, pattern in enumerate(path_patterns):

        features[f"path_pattern_{i}"] = len(
            re.findall(
                pattern,
                text,
                re.IGNORECASE
            )
        )

    # --------------------------------------------------------
    # SSRF indicators
    # --------------------------------------------------------

    ssrf_patterns = [
        r"requests\.get",
        r"requests\.post",
        r"urllib",
        r"httpclient",
        r"urlopen",
        r"fetch\s*\(",
        r"axios",
        r"http://",
        r"https://",
        r"curl",
        r"httprequest",
        r"webclient",
        r"resttemplate",
        r"okhttp"
    ]

    for i, pattern in enumerate(ssrf_patterns):

        features[f"ssrf_pattern_{i}"] = len(
            re.findall(
                pattern,
                text,
                re.IGNORECASE
            )
        )

    # --------------------------------------------------------
    # Authentication and authorization
    # --------------------------------------------------------

    auth_patterns = [
        r"password",
        r"authenticate",
        r"authorization",
        r"access_token",
        r"jwt",
        r"session",
        r"login",
        r"admin",
        r"role",
        r"permission",
        r"privilege",
        r"identity",
        r"authentication",
        r"authorize",
        r"access_control"
    ]

    for i, pattern in enumerate(auth_patterns):

        features[f"auth_pattern_{i}"] = len(
            re.findall(
                pattern,
                text,
                re.IGNORECASE
            )
        )

    # --------------------------------------------------------
    # Cryptography
    # --------------------------------------------------------

    crypto_patterns = [
        r"md5",
        r"sha1",
        r"md5\s*\(",
        r"sha1\s*\(",
        r"random\.random",
        r"crypto",
        r"password_hash",
        r"bcrypt",
        r"pbkdf2",
        r"argon2",
        r"scrypt",
        r"sha256",
        r"sha512",
        r"encrypt",
        r"decrypt"
    ]

    for i, pattern in enumerate(crypto_patterns):

        features[f"crypto_pattern_{i}"] = len(
            re.findall(
                pattern,
                text,
                re.IGNORECASE
            )
        )

    # --------------------------------------------------------
    # Secrets and credentials
    # --------------------------------------------------------

    secret_patterns = [
        r"api_key",
        r"apikey",
        r"secret_key",
        r"""password\s*=\s*["']""",
        r"""token\s*=\s*["']""",
        r"private_key",
        r"aws_access_key",
        r"aws_secret",
        r"client_secret",
        r"access_key",
        r"credentials",
        r"authorization\s*:",
        r"bearer\s+"
    ]

    for i, pattern in enumerate(secret_patterns):

        features[f"secret_pattern_{i}"] = len(
            re.findall(
                pattern,
                text,
                re.IGNORECASE
            )
        )

    # --------------------------------------------------------
    # Unsafe deserialization
    # --------------------------------------------------------

    deserialize_patterns = [
        r"pickle\.loads",
        r"pickle\.load",
        r"yaml\.load",
        r"marshal\.loads",
        r"objectinputstream",
        r"readobject",
        r"unserialize",
        r"deserialize",
        r"fromjson",
        r"loadobject",
        r"binaryformatter",
        r"unsafe"
    ]

    for i, pattern in enumerate(deserialize_patterns):

        features[f"deserialize_pattern_{i}"] = len(
            re.findall(
                pattern,
                text,
                re.IGNORECASE
            )
        )

    # --------------------------------------------------------
    # Security configuration
    # --------------------------------------------------------

    config_patterns = [
        r"debug\s*=\s*true",
        r"verify\s*=\s*false",
        r"ssl_verify\s*=\s*false",
        r"http://",
        r"""allow_origin\s*=\s*["']\*""",
        r"cors.*\*",
        r"csrf",
        r"httponly",
        r"secure\s*=\s*true",
        r"allow_all",
        r"disable_security",
        r"debug_mode",
        r"traceback",
        r"stacktrace",
        r"server_version"
    ]

    for i, pattern in enumerate(config_patterns):

        features[f"config_pattern_{i}"] = len(
            re.findall(
                pattern,
                text,
                re.IGNORECASE
            )
        )

    # --------------------------------------------------------
    # Dynamic execution
    # --------------------------------------------------------

    dynamic_patterns = [
        r"eval\s*\(",
        r"exec\s*\(",
        r"compile\s*\(",
        r"new\s+function",
        r"function\s*\(",
        r"expression\s*\(",
        r"scriptengine",
        r"javascriptengine"
    ]

    for i, pattern in enumerate(dynamic_patterns):

        features[f"dynamic_pattern_{i}"] = len(
            re.findall(
                pattern,
                text,
                re.IGNORECASE
            )
        )

    # --------------------------------------------------------
    # Code entropy
    # --------------------------------------------------------

    if code:

        counts = Counter(code)
        entropy = 0

        for count in counts.values():

            probability = count / len(code)

            entropy -= probability * math.log2(probability)

        features["character_entropy"] = entropy

    else:

        features["character_entropy"] = 0

    return features


# ============================================================
# TRANSFORMER FOR ENGINEERED SECURITY FEATURES
# ============================================================

class SecurityFeatureTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        return [
            extract_security_features(code)
            for code in X
        ]


# ============================================================
# V5 MACHINE LEARNING MODEL
# ============================================================

def build_model():

    feature_union = FeatureUnion([

        # ----------------------------------------------------
        # Character TF-IDF
        # ----------------------------------------------------

        (
            "char_tfidf",

            TfidfVectorizer(
                analyzer="char",
                ngram_range=(2, 5),
                max_features=30000,
                sublinear_tf=True
            )
        ),

        # ----------------------------------------------------
        # Word TF-IDF
        # ----------------------------------------------------

        (
            "word_tfidf",

            TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                max_features=20000,
                sublinear_tf=True
            )
        ),

        # ----------------------------------------------------
        # Engineered security features
        # ----------------------------------------------------

        (
            "security_features",

            Pipeline([

                (
                    "feature_extractor",

                    SecurityFeatureTransformer()
                ),

                (
                    "dict_vectorizer",

                    DictVectorizer()
                ),

                (
                    "scaler",

                    StandardScaler(
                        with_mean=False
                    )
                )

            ])
        )

    ])

    # --------------------------------------------------------
    # Complete ML pipeline
    # --------------------------------------------------------

    model = Pipeline([

        (
            "features",

            feature_union
        ),

        (
            "classifier",

            LogisticRegression(
                class_weight="balanced",
                C=0.5,
                max_iter=5000,
                solver="liblinear",
                random_state=42
            )
        )

    ])

    return model


# ============================================================
# V5 VERSION INFORMATION
# ============================================================

MODEL_VERSION = "V5"

MODEL_DESCRIPTION = (
    "Security vulnerability detection model using "
    "character TF-IDF, word TF-IDF, and engineered "
    "security features."
)