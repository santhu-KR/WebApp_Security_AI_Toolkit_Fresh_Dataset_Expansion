# ============================================================
# V8 FALSE NEGATIVE ANALYSIS
# Used to identify patterns missed by the V8 security ML model
# ============================================================

import os
import re
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ERROR_FILE = os.path.join(
    BASE_DIR,
    "results",
    "v8",
    "error_analysis",
    "v8_best_model_errors.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "v9",
    "error_analysis"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# PATTERN CATEGORIES
# ============================================================

PATTERNS = {

    "SQL Injection": [
        r"execute\s*\(",
        r"executeQuery",
        r"executeUpdate",
        r"SELECT",
        r"INSERT",
        r"UPDATE",
        r"DELETE",
        r"Statement",
        r"PreparedStatement",
        r"query\s*\("
    ],

    "XSS": [
        r"innerHTML",
        r"outerHTML",
        r"document\.write",
        r"response\.write",
        r"getWriter",
        r"printf",
        r"print\s*\(",
        r"render_template_string",
        r"Markup"
    ],

    "Command Injection": [
        r"os\.system",
        r"os\.popen",
        r"subprocess",
        r"Runtime\.getRuntime",
        r"ProcessBuilder",
        r"shell\s*=\s*True"
    ],

    "SSRF / Network": [
        r"requests\.get",
        r"requests\.post",
        r"urlopen",
        r"HttpURLConnection",
        r"socket",
        r"URL\s*\(",
        r"urllib",
        r"fetch\s*\(",
        r"axios"
    ],

    "Cryptography": [
        r"\bDES\b",
        r"\bMD5\b",
        r"\bSHA1\b",
        r"SHA-1",
        r"RC4",
        r"ECB",
        r"Cipher",
        r"MessageDigest",
        r"Math\.random",
        r"Random\s*\("
    ],

    "Authentication": [
        r"login",
        r"password",
        r"passwd",
        r"credential",
        r"authenticate",
        r"verifyPassword",
        r"checkPassword"
    ],

    "Session Management": [
        r"getSession",
        r"session",
        r"sessionId",
        r"session_id",
        r"putValue",
        r"setAttribute"
    ],

    "Access Control": [
        r"admin",
        r"role",
        r"permission",
        r"authorize",
        r"authorization",
        r"userId",
        r"ownerId"
    ],

    "File Operations": [
        r"open\s*\(",
        r"File\s*\(",
        r"FileInputStream",
        r"FileOutputStream",
        r"readFile",
        r"writeFile",
        r"path"
    ],

    "Deserialization": [
        r"ObjectInputStream",
        r"readObject",
        r"pickle",
        r"yaml\.load",
        r"deserialize"
    ],

    "TLS / SSL": [
        r"verify\s*=\s*False",
        r"CERT_NONE",
        r"check_hostname",
        r"TrustManager",
        r"HostnameVerifier",
        r"SSLContext"
    ],

    "Input Handling": [
        r"getParameter",
        r"request\.args",
        r"request\.form",
        r"request\.json",
        r"req\.body",
        r"req\.query",
        r"req\.params",
        r"getHeader",
        r"input\s*\("
    ],

    "Validation / Sanitization": [
        r"validate",
        r"sanitize",
        r"escape",
        r"encode",
        r"decode",
        r"strip",
        r"replace",
        r"regex",
        r"allowlist",
        r"whitelist"
    ],

    "Configuration": [
        r"DEBUG\s*=\s*True",
        r"debug\s*=\s*True",
        r"CORS",
        r"Access-Control-Allow-Origin",
        r"permitAll",
        r"allowAll"
    ],

    "Error Handling": [
        r"try\s*:",
        r"except",
        r"catch",
        r"printStackTrace",
        r"traceback",
        r"debug"
    ]
}


# ============================================================
# DETECT PATTERNS
# ============================================================

def detect_patterns(code):

    if pd.isna(code):
        code = ""

    code = str(code)

    detected = []

    for category, patterns in PATTERNS.items():

        for pattern in patterns:

            if re.search(
                pattern,
                code,
                flags=re.IGNORECASE
            ):

                detected.append(category)

                break

    if not detected:

        detected.append("No recognized pattern")

    return detected


# ============================================================
# LOAD ERROR DATA
# ============================================================

print("=" * 70)
print("V8 FALSE NEGATIVE ANALYSIS")
print("=" * 70)

if not os.path.exists(ERROR_FILE):

    print("\nERROR FILE NOT FOUND:")
    print(ERROR_FILE)

    print("\nCheck whether V8 training completed successfully.")

    raise SystemExit


df = pd.read_csv(ERROR_FILE)

print("\nLoaded error records:", len(df))

print("\nAvailable columns:")

print(df.columns.tolist())


# ============================================================
# FILTER FALSE NEGATIVES
# ============================================================

if "error_type" in df.columns:

    false_negatives = df[
        df["error_type"] == "False Negative"
    ].copy()

elif "actual" in df.columns and "predicted" in df.columns:

    false_negatives = df[
        (df["actual"] == 1) &
        (df["predicted"] == 0)
    ].copy()

else:

    print("Cannot identify False Negative rows.")

    raise SystemExit


print("\nFalse Negatives:", len(false_negatives))


# ============================================================
# IDENTIFY CODE COLUMN
# ============================================================

possible_code_columns = [

    "code_normalized",

    "code",

    "source_code",

    "snippet",

    "content",

    "text"
]

code_column = None

for column in possible_code_columns:

    if column in false_negatives.columns:

        code_column = column

        break


if code_column is None:

    print("\nCode column not found.")

    print("Available columns:", false_negatives.columns.tolist())

    raise SystemExit


print("\nCode column:", code_column)


# ============================================================
# IDENTIFY SOURCE COLUMN
# ============================================================

possible_source_columns = [

    "source",

    "dataset",

    "source_name",

    "origin"
]

source_column = None

for column in possible_source_columns:

    if column in false_negatives.columns:

        source_column = column

        break


# ============================================================
# PATTERN ANALYSIS
# ============================================================

false_negatives["detected_patterns"] = (
    false_negatives[code_column]
    .apply(detect_patterns)
)


false_negatives["pattern_string"] = (
    false_negatives["detected_patterns"]
    .apply(lambda x: ", ".join(x))
)


# ============================================================
# PATTERN FREQUENCY
# ============================================================

pattern_counts = {}

for patterns in false_negatives["detected_patterns"]:

    for pattern in patterns:

        pattern_counts[pattern] = (
            pattern_counts.get(pattern, 0) + 1
        )


pattern_summary = pd.DataFrame({

    "pattern": list(pattern_counts.keys()),

    "count": list(pattern_counts.values())

})


pattern_summary = pattern_summary.sort_values(
    "count",
    ascending=False
)


# ============================================================
# SOURCE ANALYSIS
# ============================================================

if source_column is not None:

    source_summary = (
        false_negatives[source_column]
        .value_counts()
        .reset_index()
    )

    source_summary.columns = [
        "source",
        "false_negative_count"
    ]

else:

    source_summary = pd.DataFrame()


# ============================================================
# SCORE ANALYSIS
# ============================================================

if "score_vulnerable" in false_negatives.columns:

    score_summary = (
        false_negatives["score_vulnerable"]
        .describe()
    )

else:

    score_summary = None


# ============================================================
# SAVE RESULTS
# ============================================================

fn_path = os.path.join(
    OUTPUT_DIR,
    "v8_false_negatives_detailed.csv"
)

false_negatives.to_csv(
    fn_path,
    index=False
)


pattern_path = os.path.join(
    OUTPUT_DIR,
    "v8_false_negative_pattern_summary.csv"
)

pattern_summary.to_csv(
    pattern_path,
    index=False
)


if not source_summary.empty:

    source_path = os.path.join(
        OUTPUT_DIR,
        "v8_false_negative_source_summary.csv"
    )

    source_summary.to_csv(
        source_path,
        index=False
    )


# ============================================================
# GENERATE REPORT
# ============================================================

report_path = os.path.join(
    OUTPUT_DIR,
    "v8_false_negative_analysis_report.txt"
)


with open(
    report_path,
    "w",
    encoding="utf-8"
) as report:

    report.write("=" * 70 + "\n")

    report.write("V8 FALSE NEGATIVE ANALYSIS REPORT\n")

    report.write("=" * 70 + "\n\n")

    report.write(
        f"Total False Negatives: "
        f"{len(false_negatives)}\n\n"
    )

    report.write("PATTERN SUMMARY\n")

    report.write("-" * 50 + "\n")

    report.write(
        pattern_summary.to_string(index=False)
    )

    report.write("\n\n")

    if not source_summary.empty:

        report.write("SOURCE SUMMARY\n")

        report.write("-" * 50 + "\n")

        report.write(
            source_summary.to_string(index=False)
        )

        report.write("\n\n")

    if score_summary is not None:

        report.write("VULNERABILITY SCORE SUMMARY\n")

        report.write("-" * 50 + "\n")

        report.write(
            score_summary.to_string()
        )

        report.write("\n\n")

    report.write("FALSE NEGATIVE CODE SAMPLES\n")

    report.write("-" * 50 + "\n\n")

    for i, row in false_negatives.head(20).iterrows():

        report.write(
            f"\nSample {i}\n"
        )

        report.write(
            f"Detected Patterns: "
            f"{row['pattern_string']}\n"
        )

        if "source" in row:

            report.write(
                f"Source: {row['source']}\n"
            )

        if "score_vulnerable" in row:

            report.write(
                f"Score: "
                f"{row['score_vulnerable']}\n"
            )

        report.write("\nCODE:\n")

        report.write(
            str(row[code_column])[:3000]
        )

        report.write("\n")

        report.write("-" * 70)

        report.write("\n")


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)

print("FALSE NEGATIVE PATTERN SUMMARY")

print("=" * 70)

print(
    pattern_summary.to_string(
        index=False
    )
)


if not source_summary.empty:

    print("\n" + "=" * 70)

    print("SOURCE SUMMARY")

    print("=" * 70)

    print(
        source_summary.to_string(
            index=False
        )
    )


if score_summary is not None:

    print("\n" + "=" * 70)

    print("VULNERABILITY SCORE SUMMARY")

    print("=" * 70)

    print(score_summary)


print("\n" + "=" * 70)

print("ANALYSIS COMPLETED")

print("=" * 70)

print("\nOutput folder:")

print(OUTPUT_DIR)

print("\nGenerated files:")

print("1. v8_false_negatives_detailed.csv")

print("2. v8_false_negative_pattern_summary.csv")

print("3. v8_false_negative_source_summary.csv")

print("4. v8_false_negative_analysis_report.txt")

print("\n")