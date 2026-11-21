"""Audit and normalize vulnerability category metadata without changing source datasets.

Run:
    python audit_dataset.py

Outputs:
    dataset/processed/security_snippets_normalized.csv
    dataset/processed/category_audit.csv
    dataset/processed/duplicate_conflicts.csv
    dataset/processed/dataset_audit.json

This script does not train or overwrite the existing model.
"""
from pathlib import Path
import csv
import hashlib
import json
import re
from collections import defaultdict, Counter

HERE = Path(__file__).resolve().parent
FILES = [
    ("original_curated", HERE / "dataset" / "security_snippets.csv"),
    ("OWASP-Benchmark-Java", HERE / "dataset" / "external" / "owasp_benchmark_java.csv"),
    ("NIST-SARD-Juliet-Java-1.3", HERE / "dataset" / "external" / "nist_juliet_java.csv"),
]
OUT = HERE / "dataset" / "processed"

ALIASES = {
    "csrf": "Cross-Site Request Forgery (CSRF)",
    "cross-site request forgery (csrf)": "Cross-Site Request Forgery (CSRF)",
    "ssrf": "Server-Side Request Forgery (SSRF)",
    "server-side request forgery (ssrf)": "Server-Side Request Forgery (SSRF)",
    "broken access control": "Broken Access Control / IDOR",
    "broken access control / idor": "Broken Access Control / IDOR",
    "xxe": "XML External Entity (XXE)",
    "xml external entity (xxe)": "XML External Entity (XXE)",
    "password storage": "Weak Password Storage",
    "weak password storage": "Weak Password Storage",
    "security misconfiguration": "Security Misconfiguration / Debug",
    "security misconfiguration / debug": "Security Misconfiguration / Debug",
    "sql injection": "SQL Injection",
    "cross-site scripting (xss)": "Cross-Site Scripting (XSS)",
    "command injection": "Command Injection",
    "path traversal": "Path Traversal",
    "cors misconfiguration": "CORS Misconfiguration",
    "jwt misconfiguration": "JWT Misconfiguration",
    "insecure cookie": "Insecure Cookie",
    "open redirect": "Open Redirect",
    "server-side template injection": "Server-Side Template Injection",
    "hardcoded secrets": "Hardcoded Secrets",
    "insecure deserialization": "Insecure Deserialization",
    "insecure tls configuration": "Insecure TLS Configuration",
    "weak randomness": "Weak Randomness",
    "weak cryptography": "Weak Cryptography",
    "weak hashing": "Weak Hashing",
    "trust boundary violation": "Trust Boundary Violation",
    "ldap injection": "LDAP Injection",
    "xpath injection": "XPath Injection",
    "sql_injection": "SQL Injection",
    "xss": "Cross-Site Scripting (XSS)",
    "command_injection": "Command Injection",
    "path_traversal": "Path Traversal",
    "weak_randomness": "Weak Randomness",
    "weak_cryptography": "Weak Cryptography",
    "weak_hashing": "Weak Hashing",
    "trust_boundary_violation": "Trust Boundary Violation",
    "insecure_cookies": "Insecure Cookie",
    "ldap_injection": "LDAP Injection",
    "xpath_injection": "XPath Injection",
}

def normalize_category(value):
    raw = str(value or "").strip()
    raw = re.sub(r"\s*-\s*safe\s*$", "", raw, flags=re.IGNORECASE).strip()
    key = raw.lower()
    if key in {"", "safe", "vulnerable", "unspecified"}:
        return "Uncategorized"
    return ALIASES.get(key, raw)

def normalize_label(value):
    value = str(value or "").strip().lower()
    if value in {"1", "vulnerable", "true", "unsafe", "positive"}:
        return "vulnerable"
    if value in {"0", "safe", "false", "not_vulnerable", "negative"}:
        return "safe"
    return "unknown"

def fingerprint(code):
    normalized = " ".join(str(code or "").split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

def read_rows():
    rows = []
    for source, path in FILES:
        if not path.exists():
            print(f"WARNING: missing {path}")
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for index, row in enumerate(csv.DictReader(handle), start=2):
                code = (row.get("code") or row.get("source_code") or "").strip()
                if not code:
                    continue
                rows.append({
                    "source": source,
                    "source_row": index,
                    "code": code,
                    "label": normalize_label(row.get("label")),
                    "category_original": (row.get("category") or "").strip(),
                    "category_normalized": normalize_category(row.get("category")),
                    "source_id": (row.get("source_id") or "").strip(),
                    "cwe": (row.get("cwe") or "").strip(),
                    "fingerprint": fingerprint(code),
                })
    return rows

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = read_rows()

    # Identify identical normalized code with contradictory labels.
    by_hash = defaultdict(list)
    for row in rows:
        by_hash[row["fingerprint"]].append(row)
    conflict_hashes = {
        key for key, group in by_hash.items()
        if len({r["label"] for r in group if r["label"] != "unknown"}) > 1
    }
    conflicts = []
    for key in sorted(conflict_hashes):
        for row in by_hash[key]:
            conflicts.append({
                "fingerprint": key,
                "source": row["source"],
                "source_row": row["source_row"],
                "label": row["label"],
                "category_original": row["category_original"],
                "category_normalized": row["category_normalized"],
                "source_id": row["source_id"],
            })

    # Category/label distribution includes all rows so the audit is transparent.
    counts = Counter((r["category_normalized"], r["label"]) for r in rows)
    categories = sorted({r["category_normalized"] for r in rows})
    audit_rows = []
    for category in categories:
        safe = counts[(category, "safe")]
        vulnerable = counts[(category, "vulnerable")]
        unknown = counts[(category, "unknown")]
        total = safe + vulnerable + unknown
        audit_rows.append({
            "category": category,
            "safe": safe,
            "vulnerable": vulnerable,
            "unknown_label": unknown,
            "total": total,
            "status": (
                "uncategorized" if category == "Uncategorized"
                else "low sample count (<20)" if total < 20
                else "review label conflicts" if any(
                    r["category_normalized"] == category for r in conflicts
                )
                else "usable for further validation; not proof of generalization"
            ),
        })

    normalized_path = OUT / "security_snippets_normalized.csv"
    with normalized_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["source", "source_row", "label", "category_original",
                  "category_normalized", "source_id", "cwe", "code", "fingerprint"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fields})

    def write_csv(path, data, fields):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(data)

    write_csv(OUT / "category_audit.csv", audit_rows,
              ["category", "safe", "vulnerable", "unknown_label", "total", "status"])
    write_csv(OUT / "duplicate_conflicts.csv", conflicts,
              ["fingerprint", "source", "source_row", "label",
               "category_original", "category_normalized", "source_id"])

    summary = {
        "total_rows": len(rows),
        "source_rows": dict(Counter(r["source"] for r in rows)),
        "label_counts": dict(Counter(r["label"] for r in rows)),
        "normalized_category_count_including_uncategorized": len(categories),
        "conflicting_code_groups": len(conflict_hashes),
        "conflicting_rows": len(conflicts),
        "warning": (
            "Audit only: this does not change training data or model. "
            "Resolve contradictory duplicate labels before retraining. "
            "Category counts measure available labelled snippets, not model recall."
        ),
        "outputs": [
            "dataset/processed/security_snippets_normalized.csv",
            "dataset/processed/category_audit.csv",
            "dataset/processed/duplicate_conflicts.csv",
        ],
    }
    (OUT / "dataset_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("DATASET AUDIT COMPLETE")
    print(f"Rows inspected: {len(rows)}")
    print("Sources:", dict(Counter(r["source"] for r in rows)))
    print("Labels:", dict(Counter(r["label"] for r in rows)))
    print(f"Normalized categories: {len(categories)} (includes Uncategorized)")
    print(f"Conflicting duplicate code groups: {len(conflict_hashes)}")
    print("\nCATEGORY DISTRIBUTION")
    for item in audit_rows:
        print(f"{item['category']}: safe={item['safe']}, "
              f"vulnerable={item['vulnerable']}, total={item['total']} [{item['status']}]")
    print("\nCreated:", OUT)
    print("Existing model and training files were not changed.")

if __name__ == "__main__":
    main()
