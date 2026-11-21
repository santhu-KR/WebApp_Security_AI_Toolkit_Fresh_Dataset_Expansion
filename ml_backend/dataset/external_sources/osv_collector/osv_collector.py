import requests
import json
import time
from pathlib import Path
from datetime import datetime, timezone

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OSV_API = "https://api.osv.dev/v1/query"

HEADERS = {
    "User-Agent": "Academic-Security-Dataset-Collector/1.0"
}

# Ecosystems to collect vulnerability metadata from
ECOSYSTEMS = [
    "PyPI",
    "npm",
    "Maven",
    "Go",
]

# ============================================================
# COLLECTION
# ============================================================

session = requests.Session()
session.headers.update(HEADERS)

records = []
errors = []

print("=" * 70)
print("OSV VULNERABILITY METADATA COLLECTOR")
print("=" * 70)

print("\nThis collector retrieves metadata for known vulnerable packages.")
print("Records are candidates and require source-code verification.\n")

# Package targets for initial research
PACKAGES = {
    "PyPI": [
        "django",
        "flask",
        "requests",
        "jinja2",
        "pillow",
    ],
    "npm": [
        "express",
        "lodash",
        "axios",
        "jquery",
    ],
    "Maven": [
        "org.apache.struts:struts2-core",
        "org.springframework:spring-core",
        "com.fasterxml.jackson.core:jackson-databind",
    ],
    "Go": [
        "github.com/gin-gonic/gin",
        "golang.org/x/net",
    ],
}

# ============================================================
# QUERY OSV
# ============================================================

for ecosystem, package_list in PACKAGES.items():

    print(f"\n{'=' * 50}")
    print(f"ECOSYSTEM: {ecosystem}")
    print(f"{'=' * 50}")

    for package_name in package_list:

        print(f"\nQuerying package: {package_name}")

        payload = {
            "package": {
                "name": package_name,
                "ecosystem": ecosystem
            }
        }

        try:

            response = session.post(
                OSV_API,
                json=payload,
                timeout=30
            )

            response.raise_for_status()

            data = response.json()

            vulnerabilities = data.get("vulns", [])

            print(f"Found: {len(vulnerabilities)} vulnerability records")

            for vulnerability in vulnerabilities:

                record = {
                    "osv_id": vulnerability.get("id"),
                    "aliases": vulnerability.get("aliases", []),
                    "summary": vulnerability.get("summary", ""),
                    "details": vulnerability.get("details", ""),
                    "package_name": package_name,
                    "ecosystem": ecosystem,
                    "affected": vulnerability.get("affected", []),
                    "references": vulnerability.get("references", []),
                    "severity": vulnerability.get("severity", []),
                    "database_specific": vulnerability.get(
                        "database_specific", {}
                    ),
                    "source": "OSV",
                    "collected_at": datetime.now(
                        timezone.utc
                    ).isoformat(),
                    "verification_status": "UNVERIFIED_METADATA"
                }

                records.append(record)

            time.sleep(0.3)

        except Exception as exc:

            error_record = {
                "ecosystem": ecosystem,
                "package": package_name,
                "error": str(exc)
            }

            errors.append(error_record)

            print(f"ERROR: {exc}")

# ============================================================
# REMOVE DUPLICATE RECORDS
# ============================================================

unique_records = {}

for record in records:

    osv_id = record.get("osv_id")

    if osv_id:
        unique_records[osv_id] = record

records = list(unique_records.values())

# ============================================================
# SAVE DATA
# ============================================================

records_path = OUTPUT_DIR / "osv_records.json"

with records_path.open("w", encoding="utf-8") as file:

    json.dump(
        records,
        file,
        indent=2,
        ensure_ascii=False
    )

# ============================================================
# GENERATE REPORT
# ============================================================

report = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "source": "OSV",
    "api": OSV_API,
    "status": "completed",
    "ecosystems": ECOSYSTEMS,
    "packages_queried": sum(
        len(package_list)
        for package_list in PACKAGES.values()
    ),
    "records_collected": len(records),
    "unique_records": len(records),
    "errors_count": len(errors),
    "errors": errors,
    "verification_status": "UNVERIFIED_METADATA",
    "limitations": [
        "Records contain vulnerability metadata only.",
        "Source code has not been downloaded or verified.",
        "Vulnerable and patched code pairs are not included automatically.",
        "OSV vulnerability metadata must be independently checked.",
        "Records must not be directly added to the training dataset without validation."
    ]
}

report_path = OUTPUT_DIR / "collection_report.json"

with report_path.open("w", encoding="utf-8") as file:

    json.dump(
        report,
        file,
        indent=2,
        ensure_ascii=False
    )

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("COLLECTION COMPLETED")
print("=" * 70)

print(f"Packages queried : {report['packages_queried']}")
print(f"Records collected: {report['records_collected']}")
print(f"Errors           : {report['errors_count']}")

print("\nFiles saved:")

print(records_path)
print(report_path)

print("\nIMPORTANT:")
print("These are vulnerability metadata candidates.")
print("Verify CVE, vulnerable code, patches, CWE, and labels")
print("before using them for machine-learning training.")