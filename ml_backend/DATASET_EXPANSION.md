# Expanding the training corpus with real benchmark cases

The bundled teaching corpus remains **836 examples**. This upgrade does not inflate it with copied or cosmetic variants. Instead, it adds an importer for labelled test cases from the official OWASP Benchmark Java suite.

## Why this is the safer approach

- OWASP Benchmark supplies runnable Java test cases and an expected-results file mapping cases to true/false results and CWEs.
- The current project dataset covers 18 broad categories; OWASP Benchmark Java covers a smaller set of categories. Imported records are therefore supplemental, not a complete replacement.
- The imported Java code is language-specific. A model trained on it may learn Java patterns and should not be assumed to generalize to Python, JavaScript, PHP, or other languages.
- Keep test cases from the same source project together when creating evaluation splits. Do not randomly split near-identical files across train and test.

Official sources:
- OWASP Benchmark project: https://owasp.org/www-project-benchmark/
- BenchmarkJava repository: https://github.com/OWASP-Benchmark/BenchmarkJava
- The Java v1.2 suite documents 2,740 cases across 11 listed areas; the project page describes the expected-results CSV and CWE mapping.

## Import steps (Windows PowerShell)

From the directory containing this project:

```powershell
git clone https://github.com/OWASP-Benchmark/BenchmarkJava.git
cd webapp-security-ai-toolkit-trained-ml\ml_backend
python import_owasp_benchmark.py --benchmark-dir ..\..\BenchmarkJava
```

The importer creates `dataset/external/owasp_benchmark_java.csv`. It reads labels from `expectedresults-1.2.csv` and source code from the matching `BenchmarkTest*.java` files. It does not download or redistribute OWASP source code automatically; you clone the repository yourself and should review its GPL-2.0 license before redistribution.

## Merge only after inspection

The imported file has extra provenance columns (`source`, `source_id`, `cwe`, `group_id`). Preserve these fields in a master dataset. Do not blindly append it to `security_snippets.csv` and run a random row split. For a defensible experiment, use a group-based split keyed by `group_id` or project/repository, and report metrics separately for Java benchmark data and the original teaching corpus.

## Important model limitation

The current ML classifier is binary (vulnerable vs safe). It does **not** perform reliable 18-class vulnerability identification. Category names in the CSV are metadata, while the scanner's rules and Gemini review provide the category-level findings. A future multi-class model requires enough independently labelled examples per category and per-category evaluation.


## Additional dataset: NIST SARD Juliet Java 1.3

Recommended next expansion source: **NIST SARD Juliet Java 1.3 (28,881 test cases)**. NIST describes it as Java test cases organized under 112 CWEs. The suite is 73.2 MB and its page states it is public domain / CC0 1.0. It is a curated static-analysis benchmark, not production-project data.

Official suite page: https://samate.nist.gov/SARD/test-suites/111

1. Download the Juliet Java 1.3 ZIP from the official NIST page.
2. Extract it to a folder, for example `D:\Datasets\Juliet_Java_1.3`.
3. From this `ml_backend` folder run:

```powershell
python import_nist_juliet.py --juliet-dir "D:\Datasets\Juliet_Java_1.3"
```

4. Review `dataset/external/nist_juliet_import_summary.json` and the resulting `nist_juliet_java.csv`.
5. Run `python audit_dataset.py` to regenerate category and duplicate audits.
6. Only after reviewing the import and conflicts, run `python train_model.py`. Juliet examples are added to training; the held-out OWASP test-case evaluation remains the same evaluation source so the result can be compared to the previous baseline. This is still not project-separated or production evaluation.

The importer extracts Juliet `bad()` methods as vulnerable and `good*()` methods as safe. For class-based test cases it uses the documented `_bad` and `_goodN` file suffixes. It imports only CWEs mapped to the toolkit's vulnerability categories by default; unmapped CWEs are excluded unless `--include-other-cwe` is explicitly supplied. Always inspect the generated summary before training.
