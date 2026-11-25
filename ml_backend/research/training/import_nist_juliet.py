"""Import the official NIST SARD Juliet Java 1.3 test suite into a CSV.

Download/extract the suite yourself from the official NIST SARD page, then run:
  python import_nist_juliet.py --juliet-dir "C:\\path\\to\\Juliet_Java_1.3"

The importer extracts explicitly named bad()/good*() method bodies from regular
Juliet files. For class-based test cases, it uses the documented _bad/_goodN
file suffixes. It does not execute Java code. Unmapped CWEs are skipped unless
--include-other-cwe is supplied. Review the output and category audit before
training; Juliet is primarily a static-analysis test suite, not production code.
"""
from pathlib import Path
import argparse, csv, re, hashlib, json

OUT = Path(__file__).resolve().parent / "dataset" / "external" / "nist_juliet_java.csv"

# CWE -> toolkit category mapping. Only mappings directly relevant to the toolkit
# are included; unmapped CWEs are excluded by default to reduce unrelated noise.
CWE_CATEGORY = {
    22: "Path Traversal", 23: "Path Traversal", 36: "Path Traversal", 73: "Path Traversal",
    77: "Command Injection", 78: "Command Injection", 88: "Command Injection",
    79: "Cross-Site Scripting (XSS)", 80: "Cross-Site Scripting (XSS)",
    81: "Cross-Site Scripting (XSS)", 83: "Cross-Site Scripting (XSS)",
    84: "Cross-Site Scripting (XSS)", 85: "Cross-Site Scripting (XSS)",
    86: "Cross-Site Scripting (XSS)", 87: "Cross-Site Scripting (XSS)",
    89: "SQL Injection", 352: "Cross-Site Request Forgery (CSRF)",
    601: "Open Redirect", 611: "XML External Entity (XXE)",
    614: "Insecure Cookie", 639: "Broken Access Control / IDOR",
    862: "Broken Access Control / IDOR", 863: "Broken Access Control / IDOR",
    918: "Server-Side Request Forgery (SSRF)", 942: "CORS Misconfiguration",
    1004: "Insecure Cookie", 1336: "Server-Side Template Injection",
    259: "Hardcoded Secrets", 321: "Hardcoded Secrets", 798: "Hardcoded Secrets",
    295: "Insecure TLS Configuration", 326: "Insecure TLS Configuration",
    327: "Insecure TLS Configuration", 347: "JWT Misconfiguration",
    345: "JWT Misconfiguration", 502: "Insecure Deserialization",
    328: "Weak Password Storage", 759: "Weak Password Storage",
    760: "Weak Password Storage", 916: "Weak Password Storage",
    16: "Security Misconfiguration / Debug", 489: "Security Misconfiguration / Debug",
    693: "Security Misconfiguration / Debug",
}

METHOD_RE = re.compile(
    r"\b(?:public|protected|private)\s+(?:static\s+)?(?:final\s+)?"
    r"[\w.$<>?,\[\] ]+?\s+"
    r"(?P<name>bad|good(?:\d+|G2B\d*|B2G\d*)?)\s*\([^)]*\)"
    r"[^;{}]*\{", re.MULTILINE
)

def extract_block(text, open_brace):
    depth=0; in_string=None; escaped=False; line_comment=False; block_comment=False
    i=open_brace
    while i < len(text):
        ch=text[i]; nxt=text[i+1] if i+1<len(text) else ''
        if line_comment:
            if ch=='\n': line_comment=False
        elif block_comment:
            if ch=='*' and nxt=='/': block_comment=False; i+=1
        elif in_string:
            if escaped: escaped=False
            elif ch=='\\': escaped=True
            elif ch==in_string: in_string=None
        elif ch=='/' and nxt=='/': line_comment=True; i+=1
        elif ch=='/' and nxt=='*': block_comment=True; i+=1
        elif ch in ('"', "'"): in_string=ch
        elif ch=='{': depth+=1
        elif ch=='}':
            depth-=1
            if depth==0: return text[open_brace:i+1]
        i+=1
    return None

def method_snippets(text):
    found=[]
    for match in METHOD_RE.finditer(text):
        brace=text.find('{', match.start(), match.end())
        body=extract_block(text, brace)
        if body and len(body.strip()) >= 35:
            found.append((match.group('name'), text[match.start():brace]+body))
    return found

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--juliet-dir', required=True, help='Extracted root directory of Juliet Java 1.3')
    ap.add_argument('--include-other-cwe', action='store_true', help='Also import unmapped CWEs as category Other CWE')
    args=ap.parse_args()
    base=Path(args.juliet_dir).expanduser().resolve()
    if not base.exists(): raise SystemExit(f'Directory not found: {base}')
    files=list(base.rglob('*.java'))
    if not files: raise SystemExit('No .java files found. Select the extracted Juliet Java suite root.')
    rows=[]; skipped_cwe=0; skipped_unlabelled=0; skipped_empty=0
    seen=set()
    for path in files:
        m=re.search(r'CWE(\d+)_', path.name, re.I)
        if not m: continue
        cwe=int(m.group(1)); category=CWE_CATEGORY.get(cwe)
        if category is None and not args.include_other_cwe:
            skipped_cwe+=1; continue
        if category is None: category='Other CWE'
        try: text=path.read_text(encoding='utf-8', errors='replace')
        except OSError: continue
        snippets=[]
        # Juliet class-based flaws store the bad and good class in separate files.
        lower=path.stem.lower()
        if re.search(r'_bad$', lower):
            snippets=[('bad', text)]
        elif re.search(r'_good\d*$', lower):
            snippets=[('good', text)]
        else:
            snippets=method_snippets(text)
        if not snippets:
            skipped_unlabelled+=1; continue
        for method, code in snippets:
            label=1 if method=='bad' else 0
            normalized=' '.join(code.split())
            fp=hashlib.sha256(normalized.encode()).hexdigest()
            if fp in seen: continue
            seen.add(fp)
            rel=str(path.relative_to(base))
            rows.append({'code':code,'label':label,'category':category,
                         'source':'NIST-SARD-Juliet-Java-1.3',
                         'source_id':rel+'::'+method,'cwe':f'CWE-{cwe}'})
    if not rows: raise SystemExit('No labelled snippets extracted. Check the suite folder and Juliet file naming.')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['code','label','category','source','source_id','cwe'])
        w.writeheader(); w.writerows(rows)
    from collections import Counter
    dist=Counter((r['category'], 'vulnerable' if r['label']==1 else 'safe') for r in rows)
    summary={'source':'NIST SARD Juliet Java 1.3','source_url':'https://samate.nist.gov/SARD/test-suites/111',
             'input_java_files':len(files),'imported_method_snippets':len(rows),
             'safe':sum(r['label']==0 for r in rows),'vulnerable':sum(r['label']==1 for r in rows),
             'skipped_unmapped_cwe_files':skipped_cwe,'skipped_files_without_extractable_label':skipped_unlabelled,
             'category_label_counts':{f'{k[0]} | {k[1]}':v for k,v in sorted(dist.items())},
             'note':'Juliet is a curated static-analysis test suite. It contains synthetic test cases and is not equivalent to independent production repositories.'}
    (OUT.parent/'nist_juliet_import_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('NIST JULIET IMPORT COMPLETE')
    print('Java files found:',len(files)); print('Rows imported:',len(rows))
    print('Safe:',summary['safe'],'Vulnerable:',summary['vulnerable'])
    print('Skipped unmapped CWE files:',skipped_cwe)
    print('Skipped files without labelled methods:',skipped_unlabelled)
    print('Output:',OUT)
    print('Summary:',OUT.parent/'nist_juliet_import_summary.json')

if __name__=='__main__': main()
