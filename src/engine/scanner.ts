import type { Finding, ProjectFile, Severity } from "../types";
import { scoreLine } from "./mlScorer";

const ALLOWED = /\.(py|js|ts|tsx|jsx|php|java|html|env(?:\.example)?|json|yml|yaml)$/i;
const IGNORED = /(^|\/)(node_modules|\.git|dist|build)(\/|$)/i;
const BINARY = /\0/;
const RULES: Array<{ id: string; title: string; category: string; severity: Severity; re: RegExp }> = [
 {id:"RULE-SQL-001",title:"Potential SQL injection via dynamic query",category:"SQL Injection",severity:"Critical",re:/(SELECT|INSERT|UPDATE|DELETE)[^\n]*(\+\s*|%s|\$\{|\.format\s*\()/i},
 {id:"RULE-XSS-001",title:"Unsafe HTML injection sink",category:"Cross-Site Scripting",severity:"High",re:/dangerouslySetInnerHTML|\.innerHTML\s*=/i},
 {id:"RULE-CMD-001",title:"Command execution with shell enabled",category:"Command Injection",severity:"Critical",re:/subprocess\.[\w]+\s*\([^\n]*shell\s*=\s*True/i},
 {id:"RULE-EVAL-001",title:"Dynamic code evaluation",category:"Code Injection",severity:"High",re:/\beval\s*\(/i},
 {id:"RULE-DESER-001",title:"Unsafe pickle deserialization",category:"Insecure Deserialization",severity:"High",re:/pickle\.loads\s*\(/i},
 {id:"RULE-CRYPTO-001",title:"Weak cryptographic hash",category:"Weak Cryptography",severity:"Normal",re:/\b(md5|sha1)\s*\(/i},
 {id:"RULE-PATH-001",title:"Potential path traversal from untrusted input",category:"Path Traversal",severity:"High",re:/(\.\.\/|\.\.\\)/},
 {id:"RULE-SECRET-001",title:"Possible hardcoded credential or API secret",category:"Hardcoded Secret",severity:"Critical",re:/(api[_-]?key|secret|password|token)\s*[:=]\s*["'][A-Za-z0-9_\-./+=]{12,}["']/i},
 {id:"RULE-DEBUG-001",title:"Debug mode enabled",category:"Security Misconfiguration",severity:"Normal",re:/(DEBUG\s*=\s*True|debug\s*:\s*true)/i},
 {id:"RULE-TLS-001",title:"TLS certificate verification disabled",category:"Insecure Transport",severity:"High",re:/verify\s*=\s*False/i},
 {id:"RULE-CSRF-001",title:"CSRF protection appears disabled",category:"Cross-Site Request Forgery",severity:"High",re:/csrf_exempt|WTF_CSRF_ENABLED\s*=\s*False|csrf_enabled\s*=\s*False/i},
 {id:"RULE-SSRF-001",title:"Potential server-side request using user-controlled URL",category:"Server-Side Request Forgery",severity:"High",re:/(requests\.(get|post)|fetch|axios\.(get|post))\([^\n]*(request\.|req\.|query|params|callbackUrl)/i},
 {id:"RULE-ACCESS-001",title:"Potential direct object access without ownership check",category:"Broken Access Control",severity:"High",re:/(filter_by|findOne|query)\([^\n]*(request\.(args|view_args)|req\.(params|query))/i},
 {id:"RULE-CORS-001",title:"Wildcard CORS origin configured",category:"Security Misconfiguration",severity:"Normal",re:/Access-Control-Allow-Origin['"]?\s*[,=:]\s*['"]?\*|origins\s*=\s*['"]\*['"]/i},
 {id:"RULE-JWT-001",title:"JWT signature verification disabled or weak algorithm",category:"Broken Authentication",severity:"Critical",re:/verify_signature\s*[:=]\s*False|algorithm\s*=\s*['"]none['"]/i},
 {id:"RULE-COOKIE-001",title:"Cookie security flag disabled",category:"Security Misconfiguration",severity:"Normal",re:/(httponly|secure|httpOnly)\s*[:=]\s*False/i},
 {id:"RULE-XXE-001",title:"XML external entity processing enabled",category:"XML External Entity",severity:"High",re:/resolve_entities\s*=\s*True|load_dtd\s*=\s*True|external-general-entities.*true/i},
 {id:"RULE-REDIRECT-001",title:"Potential unvalidated redirect",category:"Open Redirect",severity:"Normal",re:/(redirect|\.redirect)\([^\n]*(request\.|req\.|query|params)/i},
 {id:"RULE-SSTI-001",title:"Potential server-side template injection",category:"Server-Side Template Injection",severity:"High",re:/render_template_string\([^\n]*(request\.|req\.|\+)/i},
 {id:"RULE-PASSWORD-001",title:"Fast general-purpose hash used for password handling",category:"Cryptographic Failures",severity:"High",re:/hashlib\.(sha256|sha1|md5)\([^\n]*password/i}
];
const severityFor = (s: string): Severity => s === "Critical" || s === "High" || s === "Normal" ? s : "Low";
const idFor = (p:string,l:number,r:string) => `${p}:${l}:${r}`.replace(/[^a-zA-Z0-9:_-]/g,"_");

export function scanProject(files: ProjectFile[]): Finding[] {
 const findings: Finding[] = [];
 for (const file of files) {
  if (IGNORED.test(file.path) || !ALLOWED.test(file.path) || BINARY.test(file.content)) continue;
  const lines = file.content.split(/\r?\n/);
  lines.forEach((line, index) => {
   const matches = RULES.filter(rule => rule.re.test(line));
   if (!matches.length) return;
   const context = lines.slice(Math.max(0,index-2), Math.min(lines.length,index+3)).join("\n");
   const score = scoreLine(line, context);
   for (const rule of matches) {
    findings.push({
     id:idFor(file.path,index+1,rule.id),fileName:file.path.split("/").pop() || file.path,filePath:file.path,
     lineNumber:index+1,codeSnippet:lines.slice(Math.max(0,index-2),Math.min(lines.length,index+3)).map((v,i)=>`${Math.max(0,index-2)+i+1}: ${v}`).join("\n"),
     ruleId:rule.id,title:rule.title,predictedCategory:rule.category,mlConfidence:score.mlConfidence,
     detectionSource:score.detectionSource,severity:severityFor(rule.severity)
    });
   }
  });
 }
 return findings;
}

export function normalizePath(path: string): string {
 return path.replace(/\\/g,"/").replace(/^\/+/, "").replace(/^\.\//,"");
}