# Trained ML classifier

This is a real supervised-learning model, trained by `train_model.py` using scikit-learn.

**Model:** character-level TF-IDF n-grams + engineered security features (including Shannon entropy and security sink signals) + Logistic Regression.

**Labels:** `0 = safe`, `1 = vulnerable`.

## Train the model (first time and whenever the dataset changes)

```powershell
cd ml_backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train_model.py
```

This creates `security_model.joblib` and `training_metrics.json`. The script first evaluates a held-out test split, then refits the model on the full curated dataset for local inference.

## Start the inference API

```powershell
python app.py
```

API: `http://127.0.0.1:5001/api/ml/status`

The React app calls `/api/ml/predict` after its static rules identify candidate code lines. The model provides an independently trained vulnerable-vs-safe probability; static rules continue to supply named vulnerability categories.

## Dataset limitation
The included 836-row dataset is a curated educational dataset with vulnerable and safer examples across 18 security categories, not a large independently validated benchmark. Its holdout metrics can be optimistic because related patterns are represented in both train and test sets. Do not treat its confidence as a real-world probability or claim production-grade accuracy. The examples are mostly curated/synthetic and may share coding patterns across the train/test split. Treat the reported metrics as a development check only. Validate on independently collected, legally usable projects and use a project-level holdout before making accuracy claims.


## Expanded vulnerability coverage
The labelled examples now also include:
- Cross-Site Request Forgery (CSRF) protection disabled
- Server-Side Request Forgery (SSRF)
- Broken Access Control / insecure direct object references
- CORS wildcard with credentials
- Weak JWT verification and `alg=none`
- Insecure cookie flags
- XML External Entity (XXE) parser configuration
- Open redirects
- Server-Side Template Injection (SSTI)
- Weak password hashing

The static scanner also has corresponding rules for explicit code patterns. These are candidate indicators only: for example, lack of a CSRF token cannot reliably be proven from a single source line, and access-control issues usually require tracing route, identity, and data ownership across multiple files.


## Additional local API endpoints

- `POST /api/auth/register` — create a local account (email + password of at least 10 characters).
- `POST /api/auth/login` — authenticate and receive a signed, expiring bearer token.
- `GET /api/auth/me` — validate the current bearer token.
- `POST /api/github/import` — authenticated import of a public GitHub repository's supported source files.
- `POST /api/ml/predict` — authenticated ML classification.

Set `APP_TOKEN_SECRET` to a long random value in the environment before starting the backend. The default secret is for local demonstration only. User records are stored in `users.sqlite3` beside this application. Do not expose this local service directly to the public internet.


### Add NIST Juliet Java 1.3
See `DATASET_EXPANSION.md`. Download and extract the official NIST suite, then run `python import_nist_juliet.py --juliet-dir "PATH_TO_EXTRACTED_JULIET"`. This writes `dataset/external/nist_juliet_java.csv`. Run `python audit_dataset.py` before `python train_model.py`. The model remains a binary safe/vulnerable classifier; `category` is metadata, not an 18-class prediction output.


## Gemini AI Explanation and Ask AI

AI requests now go through the Flask backend. The Gemini API key is **not placed in React frontend code**.

1. Copy `.env.example` to `.env` inside `ml_backend`.
2. Edit `ml_backend/.env` and set `GEMINI_API_KEY` to your own Gemini API key. Keep this file private; it is ignored by Git.
3. Install/update backend dependencies: `pip install -r requirements.txt`.
4. Start the backend with `python app.py`.
5. In the frontend, sign in, scan/load a project, select a finding, and use **Explain with AI**. For Ask AI, select a source file and use the Chat tab.

AI endpoints (authenticated):
- `GET /api/ai/status`
- `POST /api/ai/explain`
- `POST /api/ai/chat`

The explanation is contextual and is intended as assistance—not a guarantee that a finding is confirmed. Validate recommendations before applying generated changes.
