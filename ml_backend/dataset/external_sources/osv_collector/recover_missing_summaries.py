import json
import pandas as pd
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).parent

JSON_FILE = BASE_DIR / "osv_records.json"

ANALYSIS_DIR = BASE_DIR / "analysis"

INPUT_CSV = ANALYSIS_DIR / "osv_cve_candidates.csv"

OUTPUT_CSV = ANALYSIS_DIR / "osv_records_recovered.csv"

REPORT_JSON = ANALYSIS_DIR / "summary_recovery_report.json"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_identifier(value):
    if value is None:
        return ""

    if pd.isna(value):
        return ""

    return str(value).strip().upper()


def extract_identifiers(record):

    identifiers = set()

    osv_id = record.get("id")

    if osv_id:
        identifiers.add(normalize_identifier(osv_id))

    aliases = record.get("aliases", [])

    if isinstance(aliases, list):

        for alias in aliases:

            if alias:
                identifiers.add(normalize_identifier(alias))

    return identifiers


def extract_csv_identifiers(row):

    identifiers = set()

    osv_id = row.get("osv_id")

    if osv_id:
        identifiers.add(normalize_identifier(osv_id))

    cve_ids = row.get("cve_ids")

    if cve_ids and not pd.isna(cve_ids):

        for identifier in str(cve_ids).split(","):

            identifier = normalize_identifier(identifier)

            if identifier:
                identifiers.add(identifier)

    return identifiers


# ============================================================
# LOAD ORIGINAL OSV JSON
# ============================================================

print("=" * 70)
print("OSV SUMMARY RECOVERY - MULTI IDENTIFIER MATCHING")
print("=" * 70)

with open(JSON_FILE, "r", encoding="utf-8") as file:

    osv_records = json.load(file)

print(f"Original OSV records: {len(osv_records)}")


# ============================================================
# LOAD CSV
# ============================================================

df = pd.read_csv(INPUT_CSV)

print(f"CSV records: {len(df)}")


# ============================================================
# BUILD IDENTIFIER LOOKUP
# ============================================================

identifier_lookup = {}

for record in osv_records:

    identifiers = extract_identifiers(record)

    for identifier in identifiers:

        if identifier:

            identifier_lookup[identifier] = record


print(f"Identifiers indexed: {len(identifier_lookup)}")


# ============================================================
# RECOVERY
# ============================================================

recovered_descriptions = []

description_sources = []

matched_records = 0

recovered_count = 0

still_missing_count = 0

unmatched_records = 0

for _, row in df.iterrows():

    original_summary = row.get("summary")

    # --------------------------------------------------------
    # Existing summary
    # --------------------------------------------------------

    if (
        not pd.isna(original_summary)
        and str(original_summary).strip()
    ):

        recovered_descriptions.append(
            str(original_summary).strip()
        )

        description_sources.append("OSV Summary")

        continue

    # --------------------------------------------------------
    # Missing summary: find original JSON record
    # --------------------------------------------------------

    csv_identifiers = extract_csv_identifiers(row)

    matched_record = None

    for identifier in csv_identifiers:

        if identifier in identifier_lookup:

            matched_record = identifier_lookup[identifier]

            break

    # --------------------------------------------------------
    # Record matched
    # --------------------------------------------------------

    if matched_record:

        matched_records += 1

        details = matched_record.get("details")

        summary = matched_record.get("summary")

        # Try details first
        if details and str(details).strip():

            recovered_descriptions.append(
                str(details).strip()
            )

            description_sources.append("OSV Details")

            recovered_count += 1

        # Try original JSON summary
        elif summary and str(summary).strip():

            recovered_descriptions.append(
                str(summary).strip()
            )

            description_sources.append("Recovered OSV Summary")

            recovered_count += 1

        # No description available
        else:

            recovered_descriptions.append("")

            description_sources.append("Missing")

            still_missing_count += 1

    # --------------------------------------------------------
    # No matching original record
    # --------------------------------------------------------

    else:

        recovered_descriptions.append("")

        description_sources.append("Unmatched")

        unmatched_records += 1

        still_missing_count += 1


# ============================================================
# ADD COLUMNS
# ============================================================

df["recovered_description"] = recovered_descriptions

df["description_source"] = description_sources


# ============================================================
# SAVE RECOVERED CSV
# ============================================================

df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8"
)


# ============================================================
# STATISTICS
# ============================================================

original_missing = df["summary"].isna().sum()

recovery_percentage = (
    recovered_count / original_missing * 100
    if original_missing > 0
    else 0
)

total_records = len(df)

final_missing = (
    df["recovered_description"]
    .fillna("")
    .str.strip()
    .eq("")
    .sum()
)


# ============================================================
# REPORT
# ============================================================

report = {

    "total_records": int(total_records),

    "original_missing_summaries": int(original_missing),

    "records_matched_using_identifiers": int(
        matched_records
    ),

    "descriptions_recovered": int(
        recovered_count
    ),

    "recovery_percentage": round(
        recovery_percentage,
        2
    ),

    "records_still_missing_descriptions": int(
        still_missing_count
    ),

    "unmatched_records": int(
        unmatched_records
    ),

    "final_missing_descriptions": int(
        final_missing
    ),

    "description_sources": {
        str(key): int(value)
        for key, value in
        df["description_source"].value_counts().items()
    },

    "output_file": str(OUTPUT_CSV)
}


# ============================================================
# SAVE JSON REPORT
# ============================================================

with open(
    REPORT_JSON,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        report,
        file,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)

print("RECOVERY RESULTS")

print("=" * 70)

print(f"Total records: {total_records}")

print(f"Original missing summaries: {original_missing}")

print(f"Records matched: {matched_records}")

print(f"Descriptions recovered: {recovered_count}")

print(f"Recovery percentage: {recovery_percentage:.2f}%")

print(f"Still missing: {still_missing_count}")

print(f"Unmatched records: {unmatched_records}")

print(f"Final missing descriptions: {final_missing}")


print("\nDescription sources:")

print(
    df["description_source"]
    .value_counts()
    .to_string()
)


print("\n" + "=" * 70)

print("FILES CREATED")

print("=" * 70)

print(OUTPUT_CSV)

print(REPORT_JSON)

print("\nRecovery completed successfully!")