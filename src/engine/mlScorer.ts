import type { DetectionSource } from "../types";

export interface ScoreResult {
  mlConfidence: number; predictedCategory: string; detectionSource: DetectionSource;
  entropy: number; featureTokens: string[];
}

export function shannonEntropy(value: string): number {
  if (!value) return 0;
  const counts = new Map<string, number>();
  for (const char of value) counts.set(char, (counts.get(char) || 0) + 1);
  let entropy = 0;
  for (const count of counts.values()) {
    const p = count / value.length;
    entropy -= p * Math.log2(p);
  }
  return entropy;
}

const FEATURES: Array<{ token: RegExp; category: string; weight: number }> = [
  { token: /\beval\s*\(/i, category: "Code Injection", weight: 0.92 },
  { token: /\bexec\s*\(/i, category: "Command Injection", weight: 0.86 },
  { token: /subprocess\.[\w]+\s*\([^\n]*shell\s*=\s*True/i, category: "Command Injection", weight: 0.96 },
  { token: /dangerouslySetInnerHTML|\.innerHTML\s*=/i, category: "Cross-Site Scripting", weight: 0.86 },
  { token: /pickle\.loads\s*\(/i, category: "Insecure Deserialization", weight: 0.95 },
  { token: /\b(md5|sha1)\s*\(/i, category: "Weak Cryptography", weight: 0.66 },
  { token: /csrf_exempt|WTF_CSRF_ENABLED\s*=\s*False/i, category: "Cross-Site Request Forgery", weight: 0.9 },
  { token: /requests\.(get|post)\([^\n]*(request\.|req\.|query|params)/i, category: "Server-Side Request Forgery", weight: 0.88 },
  { token: /verify_signature\s*[:=]\s*False|algorithm\s*=\s*["']none["']/i, category: "Broken Authentication", weight: 0.94 },
  { token: /Access-Control-Allow-Origin.*\*|origins\s*=\s*["']\*["']/i, category: "Security Misconfiguration", weight: 0.7 },
  { token: /resolve_entities\s*=\s*True|load_dtd\s*=\s*True/i, category: "XML External Entity", weight: 0.9 },
  { token: /render_template_string\([^\n]*(request\.|\+)/i, category: "Server-Side Template Injection", weight: 0.9 },
  { token: /\.\.\/|\.\.\\|join\s*\([^\n]*(?:request|params|input)/i, category: "Path Traversal", weight: 0.62 },
  { token: /(SELECT|INSERT|UPDATE|DELETE)[^\n]*(\+\s*|%s|\$\{|\.format\s*\()/i, category: "SQL Injection", weight: 0.9 },
  { token: /(?:api[_-]?key|secret|password|token)\s*[:=]\s*["'][A-Za-z0-9_\-./+=]{12,}["']/i, category: "Hardcoded Secret", weight: 0.82 }
];

export function scoreLine(line: string, context = ""): ScoreResult {
  const windowText = `${context}\n${line}`;
  const matched = FEATURES.filter(f => f.token.test(windowText));
  const entropy = shannonEntropy(line.replace(/\s/g, ""));
  const secretLike = entropy >= 3.5 && line.length >= 20 && /(key|secret|token|password|api)/i.test(line);
  const tokens = matched.map(m => m.category);
  if (secretLike) tokens.push("High-Entropy Secret");
  const top = matched.sort((a, b) => b.weight - a.weight)[0];
  const confidence = Math.min(99, Math.round(Math.max(...matched.map(m => m.weight * 100), secretLike ? 76 : 0, 8)));
  const category = top?.category || (secretLike ? "Hardcoded Secret" : "Review Candidate");
  return {
    mlConfidence: confidence, predictedCategory: category,
    detectionSource: matched.length && (secretLike || matched.length > 1) ? "Hybrid (ML + Rule)" : matched.length ? "Static Rule" : "ML Feature Model",
    entropy: Number(entropy.toFixed(2)), featureTokens: tokens
  };
}