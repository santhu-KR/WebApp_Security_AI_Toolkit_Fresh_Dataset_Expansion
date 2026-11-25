import os, re, joblib, numpy as np, pandas as pd
from scipy.sparse import hstack, csr_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_PATH = os.path.join(BASE_DIR, "dataset", "v7", "balanced", "test_v7.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "v7", "best_model_v7_balanced.pkl")
OUT_DIR = os.path.join(BASE_DIR, "results", "v8", "error_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

SECURITY_PATTERNS = {
    "sql_keywords":["select","insert","update","delete","drop","union","where"],
    "sqli_patterns":[" or 1=1","' or '1'='1","union select","drop table","sql injection"],
    "command_exec":["runtime.getruntime","processbuilder","exec(","system("],
    "xss_patterns":["<script","javascript:","innerhtml","document.write","eval("],
    "file_operations":["fileinputstream","fileoutputstream","filewriter","filereader","filepath"],
    "path_traversal":["../","..\\","/etc/passwd","path traversal"],
    "authentication":["password","login","authenticate","authentication","session"],
    "crypto":["md5","sha1","sha256","cipher","encrypt","decrypt"],
    "network":["http","https","socket","urlconnection","request"],
    "input_handling":["request.getparameter","getparameter","input","userinput","parameter"]
}
INDIVIDUAL_PATTERNS = [
"preparedstatement","statement.execute","executequery","executeupdate","parameterized",
"sanitize","escape","validation","allowlist","whitelist","blacklist","try","catch",
"finally","throw","throws","return","public","private","protected","static","new",
"null","true","false","getparameter","setparameter","request","response","cookie",
"session","authorization","csrf","cors","ssl","tls","certificate","deserialize",
"serialization","reflection","ldap","xpath","xml","json"
]
DANGEROUS_APIS = [
"runtime","processbuilder","exec","eval","system","scriptengine","class.forname",
"reflect","deserialize","objectinputstream","readobject","writeobject","random",
"math.random","secure random","md5","sha1","hardcoded","secret","token","api_key",
"password","passwd"
]

def extract_features(code):
    code=str(code); low=code.lower(); features=[]
    features += [len(code),len(code.split()),len(code.splitlines())]
    features += [code.count(x) for x in ["{","}","(",")",";","=",".",",",":"]]
    features += [code.count("//"),code.count("/*"),code.count("*/")]
    features += [code.count('"'),code.count("'")]
    tokens=re.findall(r"\b\w+\b",low); features.append(len(set(tokens)))
    for _, patterns in SECURITY_PATTERNS.items():
        features.append(sum(low.count(p.lower()) for p in patterns))
    features += [low.count(p) for p in INDIVIDUAL_PATTERNS]
    features += [low.count(p) for p in DANGEROUS_APIS]
    features += [low.count("http://"),low.count("https://"),low.count("<"),low.count(">"),low.count("&")]
    features += [code.count(" "),code.count("\t"),code.count("\n")]
    features.append(sum(c.isdigit() for c in code))
    features += [code.count(c) for c in ["$","%","&","*","!","?","#","@","/","\\","<",">","|"]]
    return features

def generate_features(codes, package):
    char_v=package["char_vectorizer"]; word_v=package["word_vectorizer"]; scaler=package["scaler"]
    char=char_v.transform(codes); word=word_v.transform(codes)
    eng=np.asarray([extract_features(c) for c in codes])
    eng=csr_matrix(scaler.transform(eng))
    return hstack([char,word,eng],format="csr")

def get_score(model, X):
    try:
        probs=model.predict_proba(X)
        classes=list(model.classes_)
        return probs[:,classes.index(1)]
    except Exception:
        try:
            scores=model.decision_function(X)
            return scores
        except Exception:
            return np.full(X.shape[0], np.nan)

def category_tags(code):
    s=str(code).lower()
    tags=[]
    checks={
      "SQL Injection":["select ","insert ","update ","delete ","executequery","statement.execute","sql"],
      "XSS":["<script","innerhtml","document.write","javascript:"],
      "Command Injection":["processbuilder","runtime.getruntime","exec(","system("],
      "Path Traversal":["../","..\\\\","/etc/passwd","path traversal"],
      "Insecure Deserialization":["objectinputstream","readobject","deserialize","pickle"],
      "Authentication/Session":["password","login","session","authenticate"],
      "Cryptography":["md5","sha1","random","cipher"],
      "SSRF/Network":["urlconnection","http://","https://","socket"],
      "Security Configuration":["debug=true","verify=false","cors","ssl","tls"],
      "Input Handling":["getparameter","request","userinput","parameter"]
    }
    for name, needles in checks.items():
        if any(n in s for n in needles): tags.append(name)
    return "; ".join(tags) if tags else "No simple pattern detected"

def main():
    print("="*68); print("V8 ERROR ANALYSIS"); print("="*68)
    df=pd.read_csv(TEST_PATH)
    package=joblib.load(MODEL_PATH)
    model=package["model"]
    code_col="code_normalized"
    df[code_col]=df[code_col].fillna("").astype(str)
    label_map=package.get("label_mapping",{"safe":0,"vulnerable":1})
    df["label"]=df["label"].astype(str).str.lower().str.strip()
    df["actual"]=df["label"].map(label_map).astype(int)
    X=generate_features(df[code_col].tolist(),package)
    df["predicted"]=model.predict(X).astype(int)
    df["score_vulnerable"]=get_score(model,X)
    df["error_type"]=np.where((df.actual==1)&(df.predicted==0),"FALSE_NEGATIVE",
                     np.where((df.actual==0)&(df.predicted==1),"FALSE_POSITIVE","CORRECT"))
    errors=df[df.error_type!="CORRECT"].copy()
    errors["pattern_tags"]=errors[code_col].apply(category_tags)
    fn=errors[errors.error_type=="FALSE_NEGATIVE"].copy()
    fp=errors[errors.error_type=="FALSE_POSITIVE"].copy()
    # Keep code samples for inspection, but CSV may contain source snippets.
    for subset,name in [(fn,"false_negatives.csv"),(fp,"false_positives.csv")]:
        keep=[c for c in ["source","source_id","label","actual","predicted","score_vulnerable","pattern_tags",code_col] if c in subset.columns]
        subset[keep].to_csv(os.path.join(OUT_DIR,name),index=False)
    summary=(errors.groupby(["error_type","source"],dropna=False).size().reset_index(name="count"))
    summary.to_csv(os.path.join(OUT_DIR,"error_summary.csv"),index=False)
    tags=(errors.assign(tag=errors.pattern_tags.str.split("; ")).explode("tag")
          .groupby(["error_type","tag"]).size().reset_index(name="count"))
    tags.to_csv(os.path.join(OUT_DIR,"error_pattern_summary.csv"),index=False)
    report=[
      "V8 ERROR ANALYSIS REPORT","="*65,
      f"Test rows: {len(df)}",
      f"False negatives: {len(fn)} (vulnerable predicted safe)",
      f"False positives: {len(fp)} (safe predicted vulnerable)",
      f"Correct: {(df.error_type=='CORRECT').sum()}","",
      "ERRORS BY SOURCE:",""
    ]
    report += summary.to_string(index=False).splitlines()
    report += ["","ERROR PATTERN TAGS:",""]
    report += tags.to_string(index=False).splitlines()
    report += ["","FALSE NEGATIVE SCORE DISTRIBUTION:",
       fn["score_vulnerable"].describe().to_string() if len(fn) else "No false negatives",
       "","Interpretation: tags are heuristic indicators only, not confirmed vulnerability categories."]
    with open(os.path.join(OUT_DIR,"error_analysis_report.txt"),"w",encoding="utf-8") as f:
        f.write("\n".join(report))
    print("\n".join(report))
    print("\nSaved results in:",OUT_DIR)
    print("Files: false_negatives.csv, false_positives.csv, error_summary.csv, error_pattern_summary.csv, error_analysis_report.txt")

if __name__=="__main__":
    main()
