# Project brief — Web Application Security & Penetration Testing Toolkit with AI

## Purpose
A local, defensive workspace for reviewing source code and identifying potential web-application security weaknesses. It combines static source-code rules, a supervised ML triage signal, passive HTTP response-header inspection, and optional Gemini-assisted contextual review and remediation suggestions.

## Intended use and safety boundary
- Assess only source code, repositories, and endpoints that you own or are explicitly authorized to review.
- The scanner performs source-code pattern analysis and passive response-header inspection.
- It must not execute target code, exploit vulnerabilities, brute-force credentials, or send attack payloads.
- Findings are candidate indicators, not proof of exploitability. Require human review and supporting evidence.

## Vulnerability coverage
The project dataset and static-analysis rules aim to cover these 18 areas:
1. SQL Injection
2. Cross-Site Scripting (XSS)
3. Command Injection
4. Cross-Site Request Forgery (CSRF)
5. Server-Side Request Forgery (SSRF)
6. Broken Access Control / IDOR
7. CORS Misconfiguration
8. JWT Misconfiguration
9. Insecure Cookies
10. XML External Entity (XXE)
11. Open Redirect
12. Weak Password Storage
13. Path Traversal
14. Insecure Deserialization
15. TLS Misconfiguration
16. Server-Side Template Injection (SSTI)
17. Security Misconfiguration
18. Hardcoded Secrets

## Technology found in the uploaded project
- Frontend: React 18, TypeScript, Vite, Tailwind CSS, Recharts, Lucide React.
- ML/API backend: Python, Flask, scikit-learn, joblib.
- Model: character-level TF-IDF plus engineered security features and Logistic Regression.
- Dataset: curated security snippets plus imported OWASP Benchmark Java examples.
- Optional AI assistance: Google Gemini through `@google/genai`.

## Important ML limitation
The current classifier is binary (safe vs vulnerable). It does not independently provide reliable classification across all 18 vulnerability categories. Category names are dataset metadata and static rules provide named candidate findings. Some categories, especially CSRF and access-control flaws, require multi-file and framework-aware analysis.

## Research and reporting requirements
- Keep source provenance and labels intact.
- Do not invent or synthetically inflate training examples.
- Evaluate using project/source-group-aware splits where possible.
- Report per-category precision, recall, F1, and support alongside binary metrics.
- Do not describe benchmark holdout metrics as production accuracy.
- Preserve the current trained model and baseline metrics before any retraining.
