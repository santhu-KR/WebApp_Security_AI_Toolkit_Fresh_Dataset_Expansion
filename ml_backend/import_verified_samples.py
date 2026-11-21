import json
import csv
from pathlib import Path
from datetime import datetime, timezone

BASE_DIR = Path(__file__).resolve().parent
VERIFIED_DIR = BASE_DIR / "dataset" / "verified_samples"
OUTPUT_DIR = BASE_DIR / "dataset" / "processed" / "verified"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_OUTPUT = OUTPUT_DIR / "verified_source_pairs.csv"
REPORT_OUTPUT = OUTPUT_DIR / "verified_import_report.json"

REQUIRED_FIELDS = [
    "sample_id",
    "project",
    "repository",
    "cve",
    "vulnerability_category",
    "cwe",
    "label",
    "language",
    "vulnerable_file",
    "vulnerable_commit",
    "fix_commit",
    "fix_version",
    "evidence_files",
    "verification_status",
]

records = []
errors = []
processed = 0

for sample_dir in VERIFIED_DIR.iterdir():
    if not sample_dir.is_dir():
        continue

    metadata_path = sample_dir / "metadata.json"

    if not metadata_path.exists():
        errors.append({
            "directory": sample_dir.name,
            "error": "metadata.json missing"
        })
        continue

    try:
        with metadata_path.open("r", encoding="utf-8-sig") as f:
            metadata = json.load(f)

        missing_fields = [
            field for field in REQUIRED_FIELDS
            if field not in metadata
        ]

        if missing_fields:
            errors.append({
                "sample": sample_dir.name,
                "error": "Missing metadata fields",
                "fields": missing_fields
            })
            continue

        vulnerable_path = sample_dir / "vulnerable.java"
        patched_path = sample_dir / "patched.java"

        if not vulnerable_path.exists():
            errors.append({
                "sample": sample_dir.name,
                "error": "vulnerable.java missing"
            })
            continue

        if not patched_path.exists():
            errors.append({
                "sample": sample_dir.name,
                "error": "patched.java missing"
            })
            continue

        vulnerable_code = vulnerable_path.read_text(
            encoding="utf-8-sig",
            errors="replace"
        )

        patched_code = patched_path.read_text(
            encoding="utf-8-sig",
            errors="replace"
        )

        if not vulnerable_code.strip():
            errors.append({
                "sample": sample_dir.name,
                "error": "Vulnerable source is empty"
            })
            continue

        if not patched_code.strip():
            errors.append({
                "sample": sample_dir.name,
                "error": "Patched source is empty"
            })
            continue

        record = {
            "sample_id": metadata["sample_id"],
            "project": metadata["project"],
            "repository": metadata["repository"],
            "cve": metadata["cve"],
            "category": metadata["vulnerability_category"],
            "cwe": metadata["cwe"],
            "language": metadata["language"],
            "vulnerable_file": metadata["vulnerable_file"],
            "vulnerable_commit": metadata["vulnerable_commit"],
            "fix_commit": metadata["fix_commit"],
            "fix_version": metadata["fix_version"],
            "vulnerable_label": "Vulnerable",
            "patched_label": "Patched",
            "verification_status": metadata["verification_status"],
            "label_confidence": metadata.get(
                "label_confidence", "Unspecified"
            ),
            "vulnerable_code": vulnerable_code,
            "patched_code": patched_code,
            "notes": metadata.get("notes", ""),
            "source_directory": str(sample_dir.relative_to(BASE_DIR)),
        }

        records.append(record)
        processed += 1

    except Exception as exc:
        errors.append({
            "sample": sample_dir.name,
            "error": str(exc)
        })

fieldnames = list(records[0].keys()) if records else []

if records:
    with CSV_OUTPUT.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

report = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "dataset_type": "Verified Vulnerability Source Pairs",
    "samples_processed": processed,
    "samples_exported": len(records),
    "errors": errors,
    "output_csv": str(CSV_OUTPUT),
    "methodology": (
        "Each record contains vulnerable and patched source code "
        "paired with provenance metadata. Patched code is labeled "
        "Patched, not universally Safe. These records require further "
        "review before conversion into ML training samples."
    )
}

with REPORT_OUTPUT.open("w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print("=" * 60)
print("VERIFIED DATASET IMPORT")
print("=" * 60)
print(f"Samples processed : {processed}")
print(f"Samples exported : {len(records)}")
print(f"Errors           : {len(errors)}")
print()

if records:
    print("CSV:")
    print(CSV_OUTPUT)

print("Report:")
print(REPORT_OUTPUT)

if errors:
    print("\nERRORS:")
    for error in errors:
        print(error)

print("\nImport completed.")
