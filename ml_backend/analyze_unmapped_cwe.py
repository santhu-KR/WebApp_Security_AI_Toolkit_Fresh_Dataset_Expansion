import json
from collections import Counter

file_path = r"dataset\external_sources\osv_collector\osv_records.json"

with open(file_path, "r", encoding="utf-8") as f:
    records = json.load(f)

# CWE identifiers already mapped in our analyzer
mapped_cwe = {
    "CWE-89", "CWE-79", "CWE-78", "CWE-94", "CWE-917",
    "CWE-611", "CWE-643", "CWE-90", "CWE-22", "CWE-23",
    "CWE-36", "CWE-59", "CWE-73", "CWE-434", "CWE-918",
    "CWE-862", "CWE-863", "CWE-639", "CWE-284", "CWE-285",
    "CWE-732", "CWE-425", "CWE-269", "CWE-276", "CWE-287",
    "CWE-306", "CWE-521", "CWE-522", "CWE-256", "CWE-257",
    "CWE-916", "CWE-798", "CWE-259", "CWE-384", "CWE-613",
    "CWE-352", "CWE-502", "CWE-20", "CWE-74", "CWE-116",
    "CWE-117", "CWE-180", "CWE-181", "CWE-182", "CWE-183",
    "CWE-184", "CWE-185", "CWE-601", "CWE-327", "CWE-328",
    "CWE-326", "CWE-330", "CWE-338", "CWE-759", "CWE-760",
    "CWE-200", "CWE-201", "CWE-209", "CWE-215", "CWE-497",
    "CWE-538", "CWE-548", "CWE-16", "CWE-2", "CWE-489",
    "CWE-942", "CWE-295", "CWE-297", "CWE-319", "CWE-400",
    "CWE-770", "CWE-834", "CWE-835", "CWE-1333", "CWE-119",
    "CWE-120", "CWE-121", "CWE-122", "CWE-125", "CWE-787",
    "CWE-416", "CWE-476", "CWE-362", "CWE-367", "CWE-190",
    "CWE-191"
}

unmapped = Counter()
examples = {}

for record in records:
    cwe_ids = record.get("database_specific", {}).get("cwe_ids", [])

    if not cwe_ids:
        unmapped["NO_CWE"] += 1
        continue

    for cwe in cwe_ids:
        if cwe not in mapped_cwe:
            unmapped[cwe] += 1

            if cwe not in examples:
                examples[cwe] = {
                    "OSV_ID": record.get("osv_id"),
                    "Summary": record.get("summary", "")
                }

print("=" * 70)
print("UNMAPPED CWE ANALYSIS")
print("=" * 70)

for cwe, count in unmapped.most_common(40):
    print(f"\n{cwe}: {count}")
    print("Example:", examples.get(cwe))

print("\nTotal unmapped CWE identifiers:", len(unmapped))
print("Total records containing unmapped CWEs:", sum(unmapped.values()))