"""Import labelled OWASP Benchmark Java test cases into this project's CSV format.

Usage (after cloning OWASP BenchmarkJava beside this project):
  python import_owasp_benchmark.py --benchmark-dir ../../BenchmarkJava

This imports only the Java CWE categories mapped below. It does not invent labels.
"""
from pathlib import Path
import argparse, csv, re, hashlib

CWE_CATEGORY = {
    '78': 'command_injection', '22': 'path_traversal', '89': 'sql_injection',
    '79': 'xss', '614': 'insecure_cookies', '90': 'ldap_injection',
    '327': 'weak_cryptography', '328': 'weak_hashing', '501': 'trust_boundary_violation',
    '330': 'weak_randomness', '643': 'xpath_injection'
}

def parse_expected(path):
    rows=[]
    # OWASP expected-results files have a comment/header row and tabular records;
    # normalize by locating the test name, TRUE/FALSE, and final CWE integer.
    with path.open(encoding='utf-8-sig', errors='replace', newline='') as f:
        for line in f:
            if 'BenchmarkTest' not in line:
                continue
            parts=[p.strip().strip('"') for p in re.split(r'[,\t]', line.strip())]
            name=next((p for p in parts if re.fullmatch(r'BenchmarkTest\d+', p)), None)
            if not name: continue
            bools=[p.upper() for p in parts if p.upper() in {'TRUE','FALSE'}]
            nums=[p for p in parts if p.isdigit()]
            if not bools or not nums: continue
            cwe=nums[-1]
            if cwe not in CWE_CATEGORY: continue
            rows.append((name, int(bools[0]=='TRUE'), CWE_CATEGORY[cwe], cwe))
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--benchmark-dir', required=True, type=Path)
    ap.add_argument('--output', type=Path, default=Path(__file__).parent/'dataset'/'external'/'owasp_benchmark_java.csv')
    args=ap.parse_args()
    expected=args.benchmark_dir/'expectedresults-1.2.csv'
    if not expected.exists(): raise SystemExit(f'Missing {expected}')
    labels=parse_expected(expected)
    if not labels: raise SystemExit('No supported labelled rows parsed; inspect the expected-results file format.')
    # Index Java test files by their class name.
    files={p.stem:p for p in args.benchmark_dir.rglob('BenchmarkTest*.java')}
    out=[]; missing=[]
    for name,label,category,cwe in labels:
        p=files.get(name)
        if not p: missing.append(name); continue
        code=p.read_text(encoding='utf-8', errors='replace')
        if not code.strip(): continue
        out.append({'code':code, 'label':label, 'category':category,
                    'source':'OWASP-Benchmark-Java-1.2', 'source_id':name,
                    'cwe':f'CWE-{cwe}', 'group_id':name})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8', newline='') as f:
        w=csv.DictWriter(f, fieldnames=['code','label','category','source','source_id','cwe','group_id'])
        w.writeheader(); w.writerows(out)
    print(f'Imported {len(out)} labelled Java test cases; skipped missing files: {len(missing)}')
    print(f'Output: {args.output}')
    print('These Java examples supplement the existing dataset; they do not cover all 18 project categories.')
if __name__=='__main__': main()
