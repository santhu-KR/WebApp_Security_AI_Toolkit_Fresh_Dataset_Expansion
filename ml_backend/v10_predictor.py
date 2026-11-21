from pathlib import Path
import re, json, joblib
import pandas as pd
from scipy.sparse import hstack, csr_matrix
HERE=Path(__file__).resolve().parent
MODEL_DIR=HERE/'models'/'v10'
FEATURE_PATTERNS={
'sql_select':r'\bselect\b','sql_insert':r'\binsert\b','sql_update':r'\bupdate\b','sql_delete':r'\bdelete\b','sql_union':r'\bunion\b','sql_where':r'\bwhere\b','sql_concat':r'(?:\+\s*\w+|%\s*\w+|f["\'].*\{)','sql_execute':r'\b(?:execute|exec|cursor\.execute|query)\b','sql_parameterized':r'\b(?:preparedstatement|parameterized|execute\s*\([^)]*,|%\s*s|:\w+|\?\s*\))',
'xss_innerhtml':r'\.innerhtml\s*=','xss_outerhtml':r'\.outerhtml\s*=','xss_document_write':r'\bdocument\.write\s*\(','xss_dangerously':r'\bdangerouslysetinnerhtml\b','xss_eval':r'\beval\s*\(','xss_location':r'\b(?:location\.hash|location\.search|location\.href)\b','xss_html_escape':r'\b(?:escape|htmlspecialchars|htmlencode|sanitizehtml)\b',
'command_os_system':r'\bos\.system\s*\(','command_shell_true':r'\bshell\s*=\s*true','command_subprocess':r'\bsubprocess\.(?:run|popen|call|check_output)\s*\(','command_exec':r'\b(?:exec|system|popen)\s*\(','code_eval':r'\b(?:eval|exec)\s*\(','path_traversal':r'\.\./|\.\\.\\','file_open':r'\b(?:open|file|readfile|writefile)\s*\(','file_upload':r'\b(?:upload|multipart|filename|file_name)\b','deserialization':r'\b(?:pickle\.loads|yaml\.load|unserialize|deserialize)\s*\(','weak_crypto_md5':r'\bmd5\s*\(','weak_crypto_sha1':r'\bsha1\s*\(','weak_crypto_des':r'\bdes\b','crypto_random':r'\b(?:random\.random|math\.random)\s*\(','strong_crypto':r'\b(?:bcrypt|scrypt|argon2|pbkdf2|sha256|sha512)\b',
'auth_password':r'\b(?:password|passwd|pwd)\b','auth_login':r'\b(?:login|signin|authenticate)\b','auth_session':r'\b(?:session|jwt|token)\b','auth_cookie':r'\b(?:cookie|setcookie)\b','cookie_secure':r'\bsecure\s*=\s*true|secure:\s*true','cookie_httponly':r'\bhttponly\s*=\s*true|httponly:\s*true','cookie_samesite':r'\bsamesite\b','csrf':r'\b(?:csrf|xsrf|csrf_token|xsrf_token)\b','validation':r'\b(?:validate|validation|validator|checkinput)\b','sanitization':r'\b(?:sanitize|sanitise|strip_tags|clean_input)\b','input_request':r'\b(?:request\.(?:get|post|args|data|form)|req\.(?:body|query|params))\b','network_url':r'\b(?:https?://|urlopen|requests\.get|fetch\s*\()\b','debug':r'\b(?:debug\s*=\s*true|debugger|traceback|stacktrace)\b','error_handling':r'\b(?:try\s*:|catch\s*\(|except\s*:|finally\s*:)\b','hardcoded_secret':r'\b(?:api[_-]?key|secret|password)\s*[:=]\s*["\'][^"\']+["\']','permissions':r'\b(?:chmod|permission|authorize|authorization|access_control)\b','redirect':r'\b(?:redirect|location\s*=|window\.location)\b','ldap':r'\b(?:ldap|directorysearch|searchfilter)\b','xml':r'\b(?:xml|etree|expat|documentbuilder)\b','reflection':r'\b(?:reflect|reflection|getattr|setattr)\s*\('
}
def extract_features(code):
 s=str(code); low=s.lower(); f={}
 for n,p in FEATURE_PATTERNS.items():
  try:f[n]=int(bool(re.search(p,low,re.I)))
  except re.error:f[n]=0
 f.update(code_length=min(len(s),100000),line_count=min(s.count('\n')+1,5000),word_count=min(len(re.findall(r'\b\w+\b',s)),20000),quote_count=s.count("'")+s.count('"'),semicolon_count=s.count(';'),bracket_count=sum(s.count(x) for x in '()[]{}'),comment_count=len(re.findall(r'(?m)^\s*(#|//|/\*|\*)',s)),uppercase_ratio=sum(c.isupper() for c in s)/max(len(s),1),digit_ratio=sum(c.isdigit() for c in s)/max(len(s),1),special_ratio=sum(not c.isalnum() and not c.isspace() for c in s)/max(len(s),1))
 f['has_user_input']=int(bool(re.search(r'user_input|userinput|request|req\.|params|query|form|input|argv',low)))
 f['has_sink']=int(bool(re.search(r'execute|eval|exec|innerhtml|system|popen|write|open|query',low)))
 f['input_sink_pair']=int(f['has_user_input'] and f['has_sink'])
 f['has_concat']=int(bool(re.search(r'\+\s*\w+|\w+\s*\+',s))); f['has_interpolation']=int(bool(re.search(r'\{[^}]+\}|%\s*[sdif]',s)))
 f['has_exception']=int(bool(re.search(r'\btry\b|\bcatch\b|\bexcept\b',low))); f['has_function']=int(bool(re.search(r'\bdef\b|\bfunction\b|=>',low)))
 f['has_variable']=int(bool(re.search(r'\b(?:var|let|const)\s+\w+|\b\w+\s*=',s))); f['has_comment']=int(bool(re.search(r'#|//|/\*|\*/',s)))
 f['has_http']=int(bool(re.search(r'http://|https://',low))); f['has_database']=int(bool(re.search(r'database|db\.|cursor|sql|query',low)))
 f['has_security_keyword']=int(bool(re.search(r'security|secure|sanitize|escape|validate|encrypt|hash|auth',low)))
 return f
class V10Predictor:
 def __init__(self):
  names=['v10_best_model.pkl','v10_char_vectorizer.pkl','v10_word_vectorizer.pkl','v10_feature_columns.pkl','v10_threshold.pkl']
  missing=[n for n in names if not (MODEL_DIR/n).exists()]
  if missing:raise FileNotFoundError(f'Missing V10 artifacts at {MODEL_DIR}: '+', '.join(missing))
  self.model=joblib.load(MODEL_DIR/names[0]);self.char=joblib.load(MODEL_DIR/names[1]);self.word=joblib.load(MODEL_DIR/names[2]);self.columns=joblib.load(MODEL_DIR/names[3]);self.threshold=float(joblib.load(MODEL_DIR/names[4]))
  meta=MODEL_DIR/'v10_metadata.json';self.metadata=json.loads(meta.read_text()) if meta.exists() else {}
 def predict(self,texts):
  texts=[str(t)[:12000] for t in texts]
  if not texts:return []
  eng=pd.DataFrame([extract_features(t) for t in texts]).reindex(columns=self.columns,fill_value=0)
  X=hstack([self.char.transform(texts),self.word.transform(texts),csr_matrix(eng.values.astype(float))],format='csr')
  probs=self.model.predict_proba(X)[:,1];out=[]
  for p in probs:
   p=float(p);v=p>=self.threshold
   out.append({'probability':p,'isVulnerable':v,'prediction':'Vulnerable' if v else 'Likely Safe','confidencePercent':round(p*100,1),'threshold':self.threshold,'confidenceBand':'high' if p>=.75 else 'moderate' if p>=.5 else 'low'})
  return out
