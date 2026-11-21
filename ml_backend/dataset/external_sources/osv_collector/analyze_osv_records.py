import json, csv, re
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "osv_records.json"
RECOVERED_FILE = BASE_DIR / "analysis" / "osv_records_recovered.csv"
OUT = BASE_DIR / "analysis"
OUT.mkdir(exist_ok=True)

CWE_CATEGORY_MAP = {
  "CWE-89": "SQL Injection",
  "CWE-564": "SQL Injection",
  "CWE-943": "SQL Injection",
  "CWE-79": "Cross-Site Scripting",
  "CWE-80": "Cross-Site Scripting",
  "CWE-81": "Cross-Site Scripting",
  "CWE-82": "Cross-Site Scripting",
  "CWE-83": "Cross-Site Scripting",
  "CWE-87": "Cross-Site Scripting",
  "CWE-77": "Command Injection",
  "CWE-78": "Command Injection",
  "CWE-88": "Command Injection",
  "CWE-94": "Code Injection",
  "CWE-95": "Code Injection",
  "CWE-96": "Code Injection",
  "CWE-97": "Code Injection",
  "CWE-99": "Code Injection",
  "CWE-917": "Expression Language Injection",
  "CWE-1336": "Template Injection",
  "CWE-611": "XXE",
  "CWE-643": "XPath Injection",
  "CWE-90": "LDAP Injection",
  "CWE-776": "XML Entity Expansion",
  "CWE-91": "XML Injection",
  "CWE-22": "Path Traversal",
  "CWE-23": "Path Traversal",
  "CWE-36": "Path Traversal",
  "CWE-59": "Path Traversal",
  "CWE-219": "Path Traversal",
  "CWE-64": "Path Traversal",
  "CWE-73": "File Path Manipulation",
  "CWE-434": "Unrestricted File Upload",
  "CWE-552": "External File Exposure",
  "CWE-377": "Insecure Temporary File",
  "CWE-378": "Insecure Temporary File",
  "CWE-379": "Insecure Temporary File",
  "CWE-668": "Resource Exposure",
  "CWE-732": "Incorrect Permissions",
  "CWE-548": "Directory Listing Exposure",
  "CWE-918": "SSRF",
  "CWE-441": "Confused Deputy",
  "CWE-610": "External Resource Control",
  "CWE-862": "Broken Access Control",
  "CWE-863": "Broken Access Control",
  "CWE-639": "Broken Access Control",
  "CWE-284": "Broken Access Control",
  "CWE-285": "Broken Access Control",
  "CWE-425": "Authorization Bypass",
  "CWE-266": "Incorrect Privilege Assignment",
  "CWE-281": "Improper Permissions",
  "CWE-269": "Privilege Escalation",
  "CWE-276": "Incorrect Permissions",
  "CWE-250": "Privilege Escalation",
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
  "CWE-521": "Weak Password",
  "CWE-522": "Weak Password",
  "CWE-258": "Empty Password",
  "CWE-259": "Hardcoded Password",
  "CWE-256": "Password Storage",
  "CWE-257": "Password Storage",
  "CWE-916": "Weak Password Hashing",
  "CWE-261": "Weak Password Storage",
  "CWE-798": "Hardcoded Credentials",
  "CWE-321": "Hardcoded Cryptographic Key",
  "CWE-384": "Session Fixation",
  "CWE-613": "Insufficient Session Expiration",
  "CWE-614": "Sensitive Cookie Without Secure Attribute",
  "CWE-315": "Sensitive Cookie Exposure",
  "CWE-539": "Persistent Cookie Information Exposure",
  "CWE-565": "Cookie Manipulation",
  "CWE-1275": "Cookie Security Weakness",
  "CWE-352": "CSRF",
  "CWE-345": "Data Authenticity Failure",
  "CWE-502": "Insecure Deserialization",
  "CWE-915": "Prototype Pollution",
  "CWE-1321": "Prototype Pollution",
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
  "CWE-113": "HTTP Response Splitting",
  "CWE-444": "HTTP Protocol Interpretation Discrepancy",
  "CWE-436": "Protocol Confusion",
  "CWE-601": "Open Redirect",
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
  "CWE-16": "Security Misconfiguration",
  "CWE-2": "Security Misconfiguration",
  "CWE-489": "Debug Mode Enabled",
  "CWE-942": "CORS Misconfiguration",
  "CWE-693": "Protection Mechanism Failure",
  "CWE-670": "Security Configuration Weakness",
  "CWE-1023": "Security Configuration Weakness",
  "CWE-319": "Cleartext Transmission",
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
  "CWE-349": "Cache Poisoning",
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
  "CWE-362": "Race Condition",
  "CWE-367": "Time-of-Check Time-of-Use",
  "CWE-364": "Race Condition",
  "CWE-366": "Race Condition",
  "CWE-190": "Integer Overflow",
  "CWE-191": "Integer Underflow",
  "CWE-192": "Integer Coercion Error",
  "CWE-193": "Off-by-One Error",
  "CWE-194": "Integer Conversion Error",
  "CWE-195": "Signedness Error",
  "CWE-196": "Unsigned Conversion Error",
  "CWE-197": "Numeric Truncation",
  "CWE-198": "Numeric Conversion Error",
  "CWE-459": "Incomplete Cleanup",
  "CWE-664": "Resource Management Error",
  "CWE-772": "Missing Resource Release",
  "CWE-773": "Missing Resource Release",
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
  "CWE-102": "Security Weakness"
}

NORMALIZED = {
 "SQL Injection": {"SQL Injection"},
 "XSS": {"Cross-Site Scripting", "Improper Output Encoding", "Improper Escaping"},
 "Command Injection": {"Command Injection"},
 "Code / Template Injection": {"Code Injection", "Expression Language Injection", "Template Injection", "Injection", "Unsafe Reflection"},
 "XML / XPath / LDAP Injection": {"XXE", "XPath Injection", "LDAP Injection", "XML Entity Expansion", "XML Injection"},
 "Path Traversal / File Handling": {"Path Traversal", "File Path Manipulation", "Unrestricted File Upload", "External File Exposure", "Insecure Temporary File", "Resource Exposure", "Directory Listing Exposure"},
 "SSRF / External Resource Access": {"SSRF", "Confused Deputy", "External Resource Control"},
 "Broken Access Control": {"Broken Access Control", "Authorization Bypass", "Incorrect Privilege Assignment", "Improper Permissions", "Incorrect Permissions", "Privilege Escalation"},
 "Authentication / Session": {"Authentication", "Authentication Spoofing", "Authentication Bypass", "Brute Force Protection Failure", "Password Recovery Weakness", "Weak Authentication", "Weak Password", "Empty Password", "Hardcoded Password", "Password Storage", "Weak Password Hashing", "Weak Password Storage", "Hardcoded Credentials", "Session Fixation", "Insufficient Session Expiration", "Sensitive Cookie Without Secure Attribute", "Sensitive Cookie Exposure", "Persistent Cookie Information Exposure", "Cookie Manipulation", "Cookie Security Weakness"},
 "CSRF / Request Forgery": {"CSRF", "Data Authenticity Failure"},
 "Insecure Deserialization": {"Insecure Deserialization"},
 "Prototype Pollution": {"Prototype Pollution"},
 "Security Misconfiguration": {"Security Misconfiguration", "Debug Mode Enabled", "CORS Misconfiguration", "Protection Mechanism Failure", "Security Configuration Weakness"},
 "Information Disclosure": {"Information Disclosure", "Information Disclosure Through Timing", "Sensitive Information in Cache", "Cleartext Transmission", "Sensitive Information Removal Failure", "Log Injection"},
 "Denial of Service (DoS)": {"Denial of Service", "ReDoS", "Algorithmic Complexity", "Uncontrolled Recursion", "Excessive Memory Allocation", "Compression Bomb", "Improper Exception Handling"},
 "Memory Safety": {"Memory Corruption", "Buffer Overflow", "Stack Buffer Overflow", "Heap Buffer Overflow", "Out-of-Bounds Read", "Out-of-Bounds Write", "Use After Free", "Null Pointer Dereference", "Uninitialized Variable", "Uninitialized Resource", "Buffer Access Error", "Integer Overflow to Buffer Overflow", "Invalid Pointer Release"},
 "Cryptographic Issues": {"Weak Cryptography", "Weak Randomness", "Weak Hashing", "Cryptographic Verification Failure", "Code Integrity Failure", "Improper Certificate Validation", "Hardcoded Cryptographic Key"},
 "Open Redirect": {"Open Redirect"},
 "Race Condition": {"Race Condition", "Time-of-Check Time-of-Use"},
 "HTTP / Protocol Handling": {"HTTP Response Splitting", "HTTP Protocol Interpretation Discrepancy", "Protocol Confusion", "CRLF Injection", "Origin Validation Error", "Cache Poisoning"},
 "Input Validation / Data Handling": {"Improper Input Validation", "Improper Parameter Handling", "Improper Encoding", "Improper Character Encoding", "Input Interpretation Error", "Null Byte Injection", "Incorrect Argument Type", "Off-by-One Error", "Integer Underflow", "Integer Coercion Error", "Integer Conversion Error", "Signedness Error", "Unsigned Conversion Error", "Numeric Truncation", "Numeric Conversion Error"},
 "Resource / Logic Errors": {"Incomplete Cleanup", "Resource Management Error", "Missing Resource Release", "Unchecked Return Value", "Improper Initialization", "Modification of Assumed Immutable Data", "Improper Handling of Exceptional Conditions", "Unsafe Function Usage", "Security Weakness", "Dependency Vulnerability"},
}
DETAIL_TO_MAIN = {detail: main for main, details in NORMALIZED.items() for detail in details}

DESC_RULES = [
 ("SQL Injection", r"\bsql injection\b|\bsqli\b|sql.{0,40}(inject|injection)"),
 ("XSS", r"cross.site scripting|\bxss\b"),
 ("Command Injection", r"command injection|shell injection|os command injection"),
 ("Code / Template Injection", r"code injection|remote code execution|\brce\b|template injection|expression language injection"),
 ("XML / XPath / LDAP Injection", r"\bxxe\b|xml external entit|xpath injection|ldap injection"),
 ("Path Traversal / File Handling", r"path traversal|directory traversal|arbitrary file (read|write)|unrestricted file upload"),
 ("SSRF / External Resource Access", r"server.side request forgery|\bssrf\b"),
 ("Broken Access Control", r"authorization bypass|broken access control|unauthorized access|privilege escalation"),
 ("Authentication / Session", r"authentication bypass|session fixation|session hijack|weak password|password reset|credential leak|cookie security"),
 ("CSRF / Request Forgery", r"cross.site request forgery|\bcsrf\b"),
 ("Insecure Deserialization", r"insecure deserialization|unsafe deserialization"),
 ("Prototype Pollution", r"prototype pollution"),
 ("Security Misconfiguration", r"security misconfiguration|misconfigured|debug mode|cors misconfiguration"),
 ("Information Disclosure", r"information disclosure|sensitive information|information leak|data leak|stack trace disclosure"),
 ("Denial of Service (DoS)", r"denial.of.service|\bdos\b|\bredos\b|resource exhaustion|compression bomb"),
 ("Memory Safety", r"buffer overflow|out.of.bounds|memory corruption|use.after.free|null pointer dereference"),
 ("Cryptographic Issues", r"weak crypt|cryptographic|certificate validation|insecure random"),
 ("Open Redirect", r"open redirect|unvalidated redirect"),
 ("Race Condition", r"race condition|time.of.check.time.of.use|\btoctou\b"),
 ("HTTP / Protocol Handling", r"request smuggling|http response splitting|protocol confusion|crlf injection"),
]

def norm_cwe(x):
    x = str(x).strip().upper()
    return x if x.startswith("CWE-") else "CWE-" + x

def main_category(detail):
    if detail in DETAIL_TO_MAIN: return DETAIL_TO_MAIN[detail]
    if detail == "Uncategorized": return "Uncategorized"
    return "Other"

def classify_description(text):
    text = (text or "").lower()
    for label, pattern in DESC_RULES:
        if re.search(pattern, text, re.I): return label
    return None

def read_recovered():
    data = {}
    if RECOVERED_FILE.exists():
        with RECOVERED_FILE.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                data[row.get("osv_id", "").strip()] = row
    else:
        print("WARNING: recovered descriptions CSV not found.")
    return data

def export_csv(path, rows):
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def run():
    if not INPUT_FILE.exists(): raise FileNotFoundError(INPUT_FILE)
    records = json.loads(INPUT_FILE.read_text(encoding="utf-8"))
    recovered = read_recovered()
    rows, uncategorized = [], []
    cats, methods, severities, cwes, original = Counter(), Counter(), Counter(), Counter(), Counter()
    cve_total = cwe_total = 0
    unmapped = Counter()

    for r in records:
        osv = str(r.get("osv_id", "")).strip()
        rr = recovered.get(osv, {})
        aliases = r.get("aliases") or []
        cves = [a for a in aliases if isinstance(a, str) and a.startswith("CVE-")]
        db = r.get("database_specific") or {}
        raw_cwes = db.get("cwe_ids") or []
        cwe_ids = sorted(set(norm_cwe(c) for c in raw_cwes if c))
        cve_total += bool(cves); cwe_total += bool(cwe_ids)
        cwes.update(cwe_ids)

        detailed = sorted(set(CWE_CATEGORY_MAP[c] for c in cwe_ids if c in CWE_CATEGORY_MAP))
        for c in cwe_ids:
            if c not in CWE_CATEGORY_MAP: unmapped[c] += 1
        original_cat = rr.get("category_candidate") or (detailed[0] if detailed else "Uncategorized")
        summary = rr.get("summary") or r.get("summary", "")
        description = rr.get("recovered_description") or r.get("details", "")
        text = (summary + " " + description).strip()

        if detailed:
            norm = main_category(detailed[0])
            method = "CWE metadata"
            confidence = "High" if len(detailed) == 1 else "Medium"
        elif original_cat != "Uncategorized":
            norm = main_category(original_cat)
            method = "Original category candidate"
            confidence = "Medium"
        else:
            fallback = classify_description(text)
            if fallback:
                norm, method, confidence = fallback, "Description keyword rules", "Low"
            else:
                norm, method, confidence = "Uncategorized", "No classification evidence", "Low"

        severity = str(rr.get("severity") or db.get("severity") or "UNKNOWN").upper()
        if severity in ("", "NAN", "NONE"): severity = "UNKNOWN"
        refs = r.get("references") or []
        commit_ref = any(isinstance(x, dict) and "github.com" in str(x.get("url","")) and "/commit/" in str(x.get("url","")) for x in refs)
        package_ref = any(isinstance(x, dict) and x.get("type") == "PACKAGE" for x in refs)
        row = {
            "osv_id": osv, "cve_ids": ";".join(cves) or rr.get("cve_ids", ""),
            "summary": summary, "recovered_description": description,
            "description_source": rr.get("description_source", ""),
            "package_name": rr.get("package_name") or r.get("package_name", "UNKNOWN"),
            "ecosystem": rr.get("ecosystem") or r.get("ecosystem", "UNKNOWN"),
            "category_candidate": original_cat, "cwe_categories": ";".join(detailed),
            "normalized_category": norm, "classification_method": method,
            "classification_confidence": confidence, "cwe_ids": ";".join(cwe_ids),
            "severity": severity, "github_commit_reference": commit_ref,
            "repository_reference": package_ref,
            "verification_status": rr.get("verification_status", "UNVERIFIED_METADATA")
        }
        rows.append(row); cats[norm] += 1; methods[method] += 1
        severities[severity] += 1; original[original_cat] += 1
        if norm == "Uncategorized": uncategorized.append(row)

    total = len(rows); categorized = total - cats.get("Uncategorized", 0)
    coverage = round(categorized * 100 / total, 2) if total else 0
    export_csv(OUT / "osv_normalized_dataset.csv", rows)
    export_csv(OUT / "osv_cve_candidates.csv", rows)
    export_csv(OUT / "osv_normalized_category_summary.csv", [{"normalized_category":k,"record_count":v} for k,v in cats.most_common()])
    export_csv(OUT / "osv_original_category_summary.csv", [{"category_candidate":k,"record_count":v} for k,v in original.most_common()])
    export_csv(OUT / "osv_classification_method_summary.csv", [{"classification_method":k,"record_count":v} for k,v in methods.most_common()])
    export_csv(OUT / "osv_uncategorized_records.csv", uncategorized)
    export_csv(OUT / "osv_cwe_summary.csv", [{"cwe":k,"record_count":v} for k,v in cwes.most_common()])
    export_csv(OUT / "osv_unmapped_cwe_summary.csv", [{"unmapped_cwe":k,"record_count":v} for k,v in unmapped.most_common()])

    report = {
        "total_records": total, "records_with_cve": int(cve_total), "records_with_cwe": int(cwe_total),
        "categorized_records": categorized, "uncategorized_records": cats.get("Uncategorized",0),
        "classification_coverage_percentage": coverage,
        "normalized_category_distribution": dict(cats),
        "original_category_distribution": dict(original),
        "classification_method_distribution": dict(methods),
        "severity_distribution": dict(severities), "cwe_distribution": dict(cwes),
        "unmapped_cwe_distribution": dict(unmapped),
        "verification_status": "UNVERIFIED_METADATA",
        "notes": ["Candidate categories, not independently verified ground truth.",
                  "CWE classification is preferred; description fallback is heuristic and Low confidence.",
                  "UNKNOWN severity is never converted to LOW.",
                  "All available CWE category labels are retained in cwe_categories."]
    }
    (OUT / "osv_normalized_quality_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("="*65)
    print("OSV NORMALIZED DATASET ANALYSIS")
    print("="*65)
    print("Total records:", total, "| CVEs:", cve_total, "| records with CWE:", cwe_total)
    print("Categorized:", categorized, "| Uncategorized:", cats.get("Uncategorized",0), "| Coverage:", coverage, "%")
    print("\nCATEGORY DISTRIBUTION")
    for k,v in cats.most_common(): print(f"{k}: {v}")
    print("\nCLASSIFICATION METHODS")
    for k,v in methods.most_common(): print(f"{k}: {v}")
    print("\nSEVERITY")
    for k,v in severities.most_common(): print(f"{k}: {v}")
    print("\nReports saved in:", OUT)

if __name__ == "__main__":
    run()
