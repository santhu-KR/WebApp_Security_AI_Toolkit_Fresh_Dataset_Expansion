import json
import csv
from pathlib import Path
from collections import Counter, defaultdict

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "osv_records.json"

OUTPUT_DIR = BASE_DIR / "analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# CWE CLASSIFICATION MAP
# ============================================================

CWE_CATEGORY_MAP = {

    # SQL Injection
    "CWE-89": "SQL Injection",
    "CWE-564": "SQL Injection",
    "CWE-943": "SQL Injection",

    # Cross-Site Scripting
    "CWE-79": "Cross-Site Scripting",
    "CWE-80": "Cross-Site Scripting",
    "CWE-81": "Cross-Site Scripting",
    "CWE-82": "Cross-Site Scripting",
    "CWE-83": "Cross-Site Scripting",
    "CWE-87": "Cross-Site Scripting",

    # Command Injection
    "CWE-77": "Command Injection",
    "CWE-78": "Command Injection",
    "CWE-88": "Command Injection",

    # Code Injection
    "CWE-94": "Code Injection",
    "CWE-95": "Code Injection",
    "CWE-96": "Code Injection",
    "CWE-97": "Code Injection",
    "CWE-99": "Code Injection",

    # Expression / Template Injection
    "CWE-917": "Expression Language Injection",
    "CWE-1336": "Template Injection",

    # XML / XPath / LDAP
    "CWE-611": "XXE",
    "CWE-643": "XPath Injection",
    "CWE-90": "LDAP Injection",
    "CWE-776": "XML Entity Expansion",
    "CWE-91": "XML Injection",

    # Path Traversal
    "CWE-22": "Path Traversal",
    "CWE-23": "Path Traversal",
    "CWE-36": "Path Traversal",
    "CWE-59": "Path Traversal",
    "CWE-219": "Path Traversal",
    "CWE-64": "Path Traversal",

    # File Handling
    "CWE-73": "File Path Manipulation",
    "CWE-434": "Unrestricted File Upload",
    "CWE-552": "External File Exposure",
    "CWE-377": "Insecure Temporary File",
    "CWE-378": "Insecure Temporary File",
    "CWE-379": "Insecure Temporary File",
    "CWE-668": "Resource Exposure",
    "CWE-732": "Incorrect Permissions",
    "CWE-548": "Directory Listing Exposure",

    # SSRF
    "CWE-918": "SSRF",
    "CWE-441": "Confused Deputy",
    "CWE-610": "External Resource Control",

    # Broken Access Control
    "CWE-862": "Broken Access Control",
    "CWE-863": "Broken Access Control",
    "CWE-639": "Broken Access Control",
    "CWE-284": "Broken Access Control",
    "CWE-285": "Broken Access Control",
    "CWE-425": "Authorization Bypass",
    "CWE-266": "Incorrect Privilege Assignment",
    "CWE-281": "Improper Permissions",

    # Privilege Escalation
    "CWE-269": "Privilege Escalation",
    "CWE-276": "Incorrect Permissions",
    "CWE-250": "Privilege Escalation",

    # Authentication
    "CWE-287": "Authentication",
    "CWE-306": "Authentication",
    "CWE-290": "Authentication Spoofing",
    "CWE-294": "Authentication Bypass",
    "CWE-289": "Authentication Bypass",
    "CWE-307": "Brute Force Protection Failure",
    "CWE-308": "Authentication Bypass",
    "CWE-640": "Password Recovery Weakness",
    "CWE-1390": "Weak Authentication",
    "CWE-1391": "Weak Authentication",

    # Password
    "CWE-521": "Weak Password",
    "CWE-522": "Weak Password",
    "CWE-258": "Empty Password",
    "CWE-259": "Hardcoded Password",
    "CWE-256": "Password Storage",
    "CWE-257": "Password Storage",
    "CWE-916": "Weak Password Hashing",
    "CWE-261": "Weak Password Storage",

    # Credentials
    "CWE-798": "Hardcoded Credentials",
    "CWE-321": "Hardcoded Cryptographic Key",

    # Session
    "CWE-384": "Session Fixation",
    "CWE-613": "Insufficient Session Expiration",
    "CWE-614": "Sensitive Cookie Without Secure Attribute",
    "CWE-315": "Sensitive Cookie Exposure",
    "CWE-539": "Persistent Cookie Information Exposure",
    "CWE-565": "Cookie Manipulation",
    "CWE-1275": "Cookie Security Weakness",

    # CSRF / Data Authenticity
    "CWE-352": "CSRF",
    "CWE-345": "Data Authenticity Failure",

    # Deserialization / Prototype Pollution
    "CWE-502": "Insecure Deserialization",
    "CWE-915": "Prototype Pollution",
    "CWE-1321": "Prototype Pollution",

    # Input Validation
    "CWE-20": "Improper Input Validation",
    "CWE-74": "Injection",
    "CWE-116": "Improper Encoding",
    "CWE-117": "Log Injection",
    "CWE-180": "Improper Input Validation",
    "CWE-181": "Improper Input Validation",
    "CWE-182": "Improper Input Validation",
    "CWE-183": "Improper Input Validation",
    "CWE-184": "Improper Input Validation",
    "CWE-185": "Improper Input Validation",
    "CWE-1284": "Improper Input Validation",
    "CWE-129": "Improper Input Validation",
    "CWE-130": "Improper Input Validation",
    "CWE-138": "Improper Input Validation",
    "CWE-141": "Improper Input Validation",
    "CWE-144": "Improper Input Validation",
    "CWE-233": "Improper Parameter Handling",
    "CWE-1286": "Improper Input Validation",

    # HTTP / Protocol
    "CWE-113": "HTTP Response Splitting",
    "CWE-444": "HTTP Protocol Interpretation Discrepancy",
    "CWE-436": "Protocol Confusion",

    # Redirect
    "CWE-601": "Open Redirect",

    # Cryptography
    "CWE-327": "Weak Cryptography",
    "CWE-328": "Weak Cryptography",
    "CWE-326": "Weak Cryptography",
    "CWE-330": "Weak Randomness",
    "CWE-338": "Weak Randomness",
    "CWE-759": "Weak Hashing",
    "CWE-760": "Weak Hashing",
    "CWE-347": "Cryptographic Verification Failure",
    "CWE-494": "Code Integrity Failure",
    "CWE-295": "Improper Certificate Validation",
    "CWE-297": "Improper Certificate Validation",

    # Information Disclosure
    "CWE-200": "Information Disclosure",
    "CWE-201": "Information Disclosure",
    "CWE-209": "Information Disclosure",
    "CWE-215": "Information Disclosure",
    "CWE-497": "Information Disclosure",
    "CWE-538": "Information Disclosure",
    "CWE-203": "Information Disclosure",
    "CWE-204": "Information Disclosure",
    "CWE-208": "Information Disclosure Through Timing",
    "CWE-524": "Sensitive Information in Cache",
    "CWE-525": "Sensitive Information in Cache",

    # Misconfiguration
    "CWE-16": "Security Misconfiguration",
    "CWE-2": "Security Misconfiguration",
    "CWE-489": "Debug Mode Enabled",
    "CWE-942": "CORS Misconfiguration",
    "CWE-693": "Protection Mechanism Failure",
    "CWE-670": "Security Configuration Weakness",
    "CWE-1023": "Security Configuration Weakness",

    # TLS
    "CWE-319": "Cleartext Transmission",

    # Denial of Service
    "CWE-400": "Denial of Service",
    "CWE-770": "Denial of Service",
    "CWE-834": "Denial of Service",
    "CWE-835": "Denial of Service",
    "CWE-1333": "ReDoS",
    "CWE-407": "Algorithmic Complexity",
    "CWE-674": "Uncontrolled Recursion",
    "CWE-789": "Excessive Memory Allocation",
    "CWE-409": "Compression Bomb",
    "CWE-1088": "Denial of Service",
    "CWE-755": "Improper Exception Handling",
    "CWE-683": "Denial of Service",

    # Cache
    "CWE-349": "Cache Poisoning",

    # Memory Safety
    "CWE-119": "Memory Corruption",
    "CWE-120": "Buffer Overflow",
    "CWE-121": "Stack Buffer Overflow",
    "CWE-122": "Heap Buffer Overflow",
    "CWE-125": "Out-of-Bounds Read",
    "CWE-126": "Out-of-Bounds Read",
    "CWE-127": "Out-of-Bounds Read",
    "CWE-787": "Out-of-Bounds Write",
    "CWE-416": "Use After Free",
    "CWE-476": "Null Pointer Dereference",
    "CWE-457": "Uninitialized Variable",
    "CWE-908": "Uninitialized Resource",
    "CWE-805": "Buffer Access Error",

    # Race Conditions
    "CWE-362": "Race Condition",
    "CWE-367": "Time-of-Check Time-of-Use",
    "CWE-364": "Race Condition",
    "CWE-366": "Race Condition",

    # Integer Errors
    "CWE-190": "Integer Overflow",
    "CWE-191": "Integer Underflow",
    "CWE-192": "Integer Coercion Error",
    "CWE-193": "Off-by-One Error",
    "CWE-194": "Integer Conversion Error",
    "CWE-195": "Signedness Error",
    "CWE-196": "Unsigned Conversion Error",
    "CWE-197": "Numeric Truncation",
    "CWE-198": "Numeric Conversion Error",

    # Resource Management
    "CWE-459": "Incomplete Cleanup",
    "CWE-664": "Resource Management Error",
    "CWE-772": "Missing Resource Release",
    "CWE-773": "Missing Resource Release",

    # Additional CWE mappings
    "CWE-150": "Improper Escaping",
    "CWE-134": "Format String Vulnerability",
    "CWE-676": "Unsafe Function Usage",
    "CWE-680": "Integer Overflow to Buffer Overflow",
    "CWE-226": "Sensitive Information Removal Failure",
    "CWE-1395": "Dependency Vulnerability",
    "CWE-252": "Unchecked Return Value",
    "CWE-763": "Invalid Pointer Release",
    "CWE-665": "Improper Initialization",
    "CWE-686": "Incorrect Argument Type",
    "CWE-471": "Modification of Assumed Immutable Data",
    "CWE-754": "Improper Handling of Exceptional Conditions",
    "CWE-93": "CRLF Injection",
    "CWE-626": "Null Byte Injection",
    "CWE-112": "Improper Output Encoding",
    "CWE-178": "Improper Character Encoding",
    "CWE-829": "Untrusted Functionality Inclusion",
    "CWE-913": "Improper Dynamic Resource Management",
    "CWE-470": "Unsafe Reflection",
    "CWE-346": "Origin Validation Error",
    "CWE-115": "Input Interpretation Error",

    # Other
    "CWE-102": "Security Weakness",
    "CWE-441": "Confused Deputy",
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_cwe(cwe):
    """
    Normalize CWE identifiers to a consistent format.
    """
    if not cwe:
        return ""

    cwe = str(cwe).strip().upper()

    if not cwe.startswith("CWE-"):
        cwe = "CWE-" + cwe

    return cwe


def get_category(cwes):
    """
    Determine the category from CWE identifiers.

    Returns the first mapped CWE category.
    """
    if not cwes:
        return "Uncategorized"

    for cwe in cwes:
        normalized = normalize_cwe(cwe)

        if normalized in CWE_CATEGORY_MAP:
            return CWE_CATEGORY_MAP[normalized]

    return "Uncategorized"


# ============================================================
# LOAD DATA
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}"
    )

with INPUT_FILE.open("r", encoding="utf-8") as file:
    records = json.load(file)

if not isinstance(records, list):
    raise ValueError("osv_records.json must contain a JSON list.")

print("=" * 70)
print("OSV DATASET ANALYSIS - CWE CLASSIFICATION")
print("=" * 70)

print("Total records:", len(records))

# ============================================================
# COUNTERS
# ============================================================

cve_count = 0
cwe_count = 0
github_commit_count = 0
repository_count = 0

ecosystem_counter = Counter()
package_counter = Counter()
cwe_counter = Counter()
severity_counter = Counter()
category_counter = Counter()
unmapped_cwe_counter = Counter()

# Store example records for unmapped CWEs
unmapped_cwe_examples = defaultdict(list)

candidate_records = []

# ============================================================
# PROCESS RECORDS
# ============================================================

for record in records:

    aliases = record.get("aliases", [])

    cves = [
        alias for alias in aliases
        if isinstance(alias, str) and alias.startswith("CVE-")
    ]

    if cves:
        cve_count += 1

    database_specific = record.get("database_specific", {})

    cwes = database_specific.get("cwe_ids", [])

    normalized_cwes = [
        normalize_cwe(cwe)
        for cwe in cwes
        if cwe
    ]

    if normalized_cwes:
        cwe_count += 1

    for cwe in normalized_cwes:
        cwe_counter[cwe] += 1

    references = record.get("references", [])

    has_github_commit = any(
        isinstance(ref, dict)
        and "github.com" in ref.get("url", "")
        and "/commit/" in ref.get("url", "")
        for ref in references
    )

    has_repository = any(
        isinstance(ref, dict)
        and ref.get("type") == "PACKAGE"
        for ref in references
    )

    if has_github_commit:
        github_commit_count += 1

    if has_repository:
        repository_count += 1

    severity = database_specific.get("severity", "UNKNOWN")

    severity_counter[severity] += 1

    ecosystem = record.get("ecosystem", "UNKNOWN")
    package = record.get("package_name", "UNKNOWN")

    ecosystem_counter[ecosystem] += 1
    package_counter[package] += 1

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    category = get_category(normalized_cwes)

    category_counter[category] += 1

    # --------------------------------------------------------
    # Unmapped CWE tracking
    # --------------------------------------------------------

    for cwe in normalized_cwes:

        if cwe not in CWE_CATEGORY_MAP:

            unmapped_cwe_counter[cwe] += 1

            if len(unmapped_cwe_examples[cwe]) < 5:

                unmapped_cwe_examples[cwe].append({
                    "osv_id": record.get("osv_id", ""),
                    "summary": record.get("summary", ""),
                    "package_name": package,
                    "ecosystem": ecosystem
                })

    # --------------------------------------------------------
    # Export candidate record
    # --------------------------------------------------------

    candidate_records.append({

        "osv_id": record.get("osv_id", ""),

        "cve_ids": ";".join(cves),

        "summary": record.get("summary", ""),

        "package_name": package,

        "ecosystem": ecosystem,

        "category_candidate": category,

        "cwe_ids": ";".join(normalized_cwes),

        "severity": severity,

        "github_commit_reference": has_github_commit,

        "repository_reference": has_repository,

        "verification_status": "UNVERIFIED_METADATA"
    })

# ============================================================
# COVERAGE
# ============================================================

total_records = len(records)

categorized_records = (
    total_records - category_counter.get("Uncategorized", 0)
)

uncategorized_records = category_counter.get("Uncategorized", 0)

classification_coverage = (
    round((categorized_records / total_records) * 100, 2)
    if total_records > 0
    else 0
)

# ============================================================
# CSV EXPORT HELPER
# ============================================================

def export_csv(filepath, headers, rows):

    with filepath.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.writer(file)

        writer.writerow(headers)

        writer.writerows(rows)


# ============================================================
# EXPORT CANDIDATE RECORDS
# ============================================================

csv_file = OUTPUT_DIR / "osv_cve_candidates.csv"

fieldnames = [
    "osv_id",
    "cve_ids",
    "summary",
    "package_name",
    "ecosystem",
    "category_candidate",
    "cwe_ids",
    "severity",
    "github_commit_reference",
    "repository_reference",
    "verification_status"
]

with csv_file.open(
    "w",
    newline="",
    encoding="utf-8-sig"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(candidate_records)

# ============================================================
# EXPORT CATEGORY SUMMARY
# ============================================================

category_file = OUTPUT_DIR / "osv_category_summary.csv"

export_csv(
    category_file,
    ["Category Candidate", "Record Count"],
    category_counter.most_common()
)

# ============================================================
# EXPORT CWE SUMMARY
# ============================================================

cwe_file = OUTPUT_DIR / "osv_cwe_summary.csv"

export_csv(
    cwe_file,
    ["CWE", "Record Count"],
    cwe_counter.most_common()
)

# ============================================================
# EXPORT UNMAPPED CWE SUMMARY
# ============================================================

unmapped_cwe_file = OUTPUT_DIR / "osv_unmapped_cwe_summary.csv"

export_csv(
    unmapped_cwe_file,
    ["Unmapped CWE", "Record Count"],
    unmapped_cwe_counter.most_common()
)

# ============================================================
# EXPORT UNMAPPED CWE EXAMPLES
# ============================================================

unmapped_examples_file = (
    OUTPUT_DIR / "osv_unmapped_cwe_examples.csv"
)

example_rows = []

for cwe, examples in unmapped_cwe_examples.items():

    for example in examples:

        example_rows.append([
            cwe,
            example["osv_id"],
            example["summary"],
            example["package_name"],
            example["ecosystem"]
        ])

export_csv(
    unmapped_examples_file,
    [
        "Unmapped CWE",
        "OSV ID",
        "Summary",
        "Package",
        "Ecosystem"
    ],
    example_rows
)

# ============================================================
# QUALITY REPORT
# ============================================================

quality_report = {

    "total_records": total_records,

    "records_with_cve": cve_count,

    "records_with_cwe": cwe_count,

    "records_with_github_commit_reference": github_commit_count,

    "records_with_repository_reference": repository_count,

    "records_with_severity": sum(
        count
        for severity, count in severity_counter.items()
        if severity != "UNKNOWN"
    ),

    "categorized_records": categorized_records,

    "uncategorized_records": uncategorized_records,

    "classification_coverage_percentage": classification_coverage,

    "ecosystem_distribution": dict(ecosystem_counter),

    "package_distribution": dict(package_counter),

    "cwe_distribution": dict(cwe_counter),

    "severity_distribution": dict(severity_counter),

    "category_candidates": dict(category_counter),

    "unmapped_cwe_distribution": dict(unmapped_cwe_counter),

    "total_unmapped_cwe_identifiers": len(unmapped_cwe_counter),

    "classification_method": "CWE-based candidate classification",

    "verification_status": "UNVERIFIED_METADATA",

    "warning": (
        "Categories are candidate classifications inferred from CWE "
        "metadata. OSV records are not verified training samples. "
        "Source code, affected versions, vulnerability evidence, "
        "and patches must be independently verified before ML training."
    )
}

quality_file = OUTPUT_DIR / "osv_quality_report.json"

with quality_file.open(
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        quality_report,
        file,
        indent=2,
        ensure_ascii=False
    )

# ============================================================
# DISPLAY SUMMARY
# ============================================================

print("\nDATA QUALITY SUMMARY")
print("-" * 70)

print("Total records:", total_records)
print("Records with CVE:", cve_count)
print("Records with CWE:", cwe_count)
print("GitHub commit references:", github_commit_count)
print("Repository references:", repository_count)

print("\nCLASSIFICATION COVERAGE")
print("-" * 70)

print("Categorized records:", categorized_records)
print("Uncategorized records:", uncategorized_records)
print("Coverage:", classification_coverage, "%")

print("\nSEVERITY DISTRIBUTION")
print("-" * 70)

for severity, count in severity_counter.most_common():
    print(f"{severity}: {count}")

print("\nVULNERABILITY CATEGORIES")
print("-" * 70)

for category, count in category_counter.most_common():
    print(f"{category}: {count}")

print("\nECOSYSTEM DISTRIBUTION")
print("-" * 70)

for ecosystem, count in ecosystem_counter.most_common():
    print(f"{ecosystem}: {count}")

print("\nTOP 20 CWE IDENTIFIERS")
print("-" * 70)

for cwe, count in cwe_counter.most_common(20):
    print(f"{cwe}: {count}")

print("\nUNMAPPED CWE IDENTIFIERS")
print("-" * 70)

for cwe, count in unmapped_cwe_counter.most_common(30):
    print(f"{cwe}: {count}")

print("\nFILES CREATED")
print("-" * 70)

print(csv_file)
print(category_file)
print(cwe_file)
print(unmapped_cwe_file)
print(unmapped_examples_file)
print(quality_file)

print("\nAnalysis completed successfully.")