# Sentinel AI — Web Application Security & Penetration Testing Toolkit

A defensive, browser-based security review workspace featuring local static rules, an explainable heuristic feature scorer, passive HTTP-header inspection, optional Gemini verification, full-file remediation assistance, patch snapshots, and ZIP/Markdown exports.

## Run locally
1. Install Node.js 20+.
2. `npm install`
3. Copy `.env.example` to `.env` and set `GEMINI_API_KEY`.
4. `npm run dev`

Build: `npm run build`

## Important security notes
- This is a local assessment aid, not a certified scanner or a substitute for expert review.
- Only inspect code and endpoints you own or are explicitly authorized to assess.
- Endpoint inspection uses browser `fetch` and therefore requires CORS permission from the target.
- Static rules and ML-style feature scores are heuristic signals, not a trained ML model. Validate all findings.
- The requested browser-side `process.env.GEMINI_API_KEY` setup is provided for local demonstration. Vite substitutes the value into client assets, so **the key is visible to every user of a deployed site**. Do not deploy this configuration publicly. For production, put Gemini calls behind an authenticated server-side proxy, apply quotas/rate limits, and store the key only in server environment variables.
- Source files are sent to Google Gemini only when the user explicitly runs verification or sends a chat prompt. Avoid uploading secrets, credentials, customer data, or regulated information.
- Remediation suggestions are untrusted generated output. Review, test, and approve changes before applying them.

## Features
- Folder upload, ZIP upload, and four-file vulnerable sample.
- Exact 1-based line locations with five-line code context.
- Static rule engine + Shannon entropy and sink-weighted feature scorer.
- Passive security header and cookie-flag inspection.
- Structured Gemini finding verification (`gemini-3.8-flash`).
- Gemini full-file remediation chat (`gemini-2.5-pro`) with code extraction and snapshots.
- Search/filter findings, dashboard charts, Markdown report, patched ZIP.

## Scope
The scanner performs source-code pattern analysis and passive response-header inspection. It does not execute target code, exploit vulnerabilities, brute-force credentials, or send attack payloads.


## Real trained ML model (Python service)
This version adds a supervised scikit-learn classifier in `ml_backend/`. It combines character-level TF-IDF n-grams with engineered code features (including Shannon entropy and security-sink indicators), then trains Logistic Regression on labeled safe/vulnerable examples.

### Start the ML service (Terminal 1)
```powershell
cd ml_backend
python -m venv .venv
.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
python train_model.py
python app.py
```
The included `security_model.joblib` is already trained on the included 836-row curated teaching dataset. Run `python train_model.py` to retrain it and regenerate evaluation metrics.

### Start React (Terminal 2)
```powershell
npm install
npm run dev
```
The React scanner sends static-rule candidate snippets to `http://127.0.0.1:5001/api/ml/predict`. The dashboard displays whether the trained ML service is online. If it is offline, the original heuristic score is shown as a fallback.

**Research limitation:** the seed dataset is small and curated for demonstration. Its holdout score is not a reliable estimate of real-world performance. Expand the dataset with independently reviewed, legally usable samples and validate against a separate project-level test set before claiming model accuracy.


## Expanded ML and static-analysis coverage
The labelled training set now has 836 curated snippets, covering SQL injection, XSS, command injection, CSRF, SSRF, Broken Access Control / IDOR, CORS misconfiguration, JWT misconfiguration, insecure cookies, XXE, open redirects, weak password storage, path traversal, insecure deserialization, TLS misconfiguration, server-side template injection, security misconfiguration, and hardcoded secrets. The expanded set includes both vulnerable and safer code patterns. The examples are educational and mostly curated/synthetic; they are not a substitute for a professionally labelled benchmark.

The static rule engine includes explicit indicators for these additional patterns. Important limitation: some vulnerabilities (especially CSRF and access-control flaws) require multi-file/framework-aware reasoning; a single-line pattern is only a warning signal, not proof that the full application is vulnerable.


## Risk prioritization

Findings are prioritized in this order: **Critical → High → Normal → Low**.
- **Critical:** likely severe impact, such as exploitable injection or exposed credentials; investigate immediately.
- **High:** significant security weakness that may expose sensitive data or privileged actions; remediate promptly.
- **Normal:** moderate or context-dependent weakness / security hardening issue; schedule remediation.
- **Low:** lower-impact observation or defense-in-depth recommendation; address during routine maintenance.

Severity is assigned by static rule metadata and may be adjusted by human review. It is not a proof of exploitability. Validate findings in the application's full context before treating them as confirmed vulnerabilities.


## Module 7 — Gemini full-file secure code suggestions

1. Copy `.env.example` to `.env` in the project root.
2. Add your Google AI Studio key: `GEMINI_API_KEY=your_actual_key`.
3. Restart Vite after changing `.env`.
4. Load a project, open **Gemini AI Security Chat**, and choose the target file.
5. Ask Gemini to fix the findings or make a requested secure-code improvement.
6. Gemini receives the selected file's complete source and relevant scanner findings, then is instructed to return one complete replacement file in a single code block.
7. Review the generated file, copy it, or apply it to the in-memory project workspace. Applying creates a snapshot first; use **Rollback** to restore the prior file.
8. Export the patched project ZIP when you have reviewed the changes.

Example prompt: `Fix all security findings in this file. Preserve all existing features and return the complete corrected file.`

The Gemini key is currently embedded into browser assets by the local Vite configuration. This is suitable only for a trusted local demonstration. Do not publish or deploy this frontend with a real key; use a server-side Gemini proxy with authentication, rate limits, and server-only environment variables for deployment. Review and test generated code before applying it.


## Added engineering modules

### Context-aware vulnerability verification
- Static rules generate candidate findings; they do not claim that a pattern match proves exploitability.
- The ML classifier adds a second signal and returns `Vulnerable`, `Review Required`, or `Likely Safe` based on experimental probability thresholds.
- The optional Gemini verification action now receives the complete selected source file (up to 30,000 characters per file) alongside each candidate, so it can review nearby validation, authorization, and framework protections.
- Findings with insufficient evidence should remain unconfirmed. Always manually review results.

### GitHub repository import
- Open **Project Folder & URL Scanner** and paste a public GitHub repository URL.
- The local Flask service fetches the repository's default branch and imports supported source files.
- Import limits: 20 MB compressed archive, 1,500 supported files, and 500 KB per file.
- Private repositories and authenticated GitHub access are not included in this local demo. Scan only repositories you own or are authorized to assess. GitHub API rate limits apply.

### Local user authentication
- The app starts with a sign-in / create-account screen.
- Accounts are stored in `ml_backend/users.sqlite3` using PBKDF2-HMAC-SHA256 password hashes with per-user random salts.
- The backend issues signed, expiring bearer tokens. ML prediction and GitHub import endpoints require a valid token.
- The local SQLite account database and browser session token are for a single-machine academic demonstration, not a production identity system.
- Set a strong random `APP_TOKEN_SECRET` environment variable before starting the backend. Do not use the built-in demo fallback secret for deployment.

### Start the backend
PowerShell:
```powershell
cd ml_backend
.venv\Scripts\Activate.ps1
$env:APP_TOKEN_SECRET = "replace-with-a-long-random-secret"
python app.py
```

Start the React app in a second terminal from the project root:
```powershell
npm install
npm run dev
```

### Important accuracy and security notes
- The ML score is a triage signal, not a calibrated probability or proof of vulnerability.
- Gemini verification is contextual assistance, not a formal proof. It can still make mistakes and has file-context size limits.
- Source code sent for Gemini verification or chat is processed by Google's Gemini service. Do not submit secrets, credentials, customer data, or code you are not permitted to share.
- The browser-side Gemini API key configuration remains local-demo-only. For public deployment, move all Gemini calls behind an authenticated backend proxy and store the key only on the server.
- Before production use, add HTTPS, CSRF protections for cookie-based sessions if introduced, rate limiting, account recovery, audit logging, secret rotation, GitHub OAuth for private repos, and a proper deployment-grade identity provider.


## Training dataset expansion

See `ml_backend/DATASET_EXPANSION.md` for the official OWASP Benchmark import workflow. The bundled 836-example corpus is retained as-is; external benchmark examples are imported separately with provenance rather than being synthetically duplicated.


## Expand the ML dataset (optional)
The ML backend now includes an importer for the official NIST SARD Juliet Java 1.3 suite. Read `ml_backend/DATASET_EXPANSION.md` before importing. Do not train immediately after downloading: inspect the generated import summary and dataset audit first.


## Configure Gemini API (local development)

1. Create an API key in Google AI Studio: https://aistudio.google.com/app/apikey
2. From the repository root in PowerShell, run:
   ```powershell
   .\\scripts\\configure-gemini.ps1
   ```
3. Paste the key when prompted. The script writes it to the root `.env` file. That file is ignored by Git and must never be committed.
4. Restart Vite:
   ```powershell
   Ctrl+C
   npm run dev
   ```

**Security warning:** This project currently uses a local-demo Vite configuration that embeds the Gemini key in frontend assets. Use only locally with a key you can revoke. Do not deploy or share a build containing the key. For public deployment, Gemini requests must be moved behind a server-side authenticated proxy.
