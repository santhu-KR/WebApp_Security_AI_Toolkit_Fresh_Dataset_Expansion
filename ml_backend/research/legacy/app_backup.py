from pathlib import Path
import os, json, re, hmac, hashlib, base64, sqlite3, secrets, time, io, zipfile, urllib.request, urllib.error
from functools import wraps
import joblib
from flask import Flask, jsonify, request, g
from flask_cors import CORS

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "security_model.joblib"
METRICS_PATH = HERE / "training_metrics.json"
AUTH_DB = HERE / "users.sqlite3"
TOKEN_SECRET = os.environ.get("APP_TOKEN_SECRET", "CHANGE_THIS_LOCAL_DEMO_SECRET_BEFORE_DEPLOYMENT").encode()
TOKEN_TTL = 12 * 60 * 60
MAX_REPO_BYTES = 20 * 1024 * 1024
MAX_REPO_FILES = 1500
ALLOWED_EXT = re.compile(r"\.(py|js|ts|tsx|jsx|php|java|html|env|json|ya?ml)$", re.I)
IGNORED_PARTS = {"node_modules", ".git", "dist", "build", "__pycache__", ".venv", "venv"}

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": ["http://localhost:5173", "http://127.0.0.1:5173"]}},
     supports_credentials=False)
model = None

def db():
    conn = sqlite3.connect(AUTH_DB)
    conn.row_factory = sqlite3.Row
    return conn

def init_auth_db():
    with db() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at INTEGER NOT NULL
        )""")

def load_model():
    global model
    try:
        model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
    except Exception as exc:
        print(f"Could not load trained model: {exc}. Re-run python train_model.py.")
        model = None

def make_password_hash(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 310_000).hex()

def issue_token(user_id, email):
    payload = json.dumps({"uid": user_id, "email": email, "exp": int(time.time()) + TOKEN_TTL},
                         separators=(",", ":")).encode()
    body = base64.urlsafe_b64encode(payload).rstrip(b"=")
    signature = hmac.new(TOKEN_SECRET, body, hashlib.sha256).digest()
    return (body + b"." + base64.urlsafe_b64encode(signature).rstrip(b"=")).decode()

def decode_token(token):
    try:
        body, sig = token.encode().split(b".", 1)
        expected = hmac.new(TOKEN_SECRET, body, hashlib.sha256).digest()
        supplied = base64.urlsafe_b64decode(sig + b"=" * (-len(sig) % 4))
        if not hmac.compare_digest(expected, supplied):
            return None
        payload = base64.urlsafe_b64decode(body + b"=" * (-len(body) % 4))
        data = json.loads(payload)
        if int(data.get("exp", 0)) < int(time.time()):
            return None
        return data
    except Exception:
        return None

def require_auth(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        header = request.headers.get("Authorization", "")
        token = header[7:].strip() if header.lower().startswith("bearer ") else ""
        data = decode_token(token)
        if not data:
            return jsonify({"error": "Authentication required. Please sign in again."}), 401
        g.user = data
        return fn(*args, **kwargs)
    return wrapped

init_auth_db()
load_model()

@app.get("/api/health")
def health():
    return jsonify({"ok": True, "service": "local-security-api"})

@app.post("/api/auth/register")
def register():
    payload = request.get_json(silent=True) or {}
    email = str(payload.get("email", "")).strip().lower()
    password = str(payload.get("password", ""))
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email) or len(email) > 254:
        return jsonify({"error": "Enter a valid email address."}), 400
    if len(password) < 10:
        return jsonify({"error": "Password must contain at least 10 characters."}), 400
    salt = secrets.token_bytes(16)
    try:
        with db() as conn:
            cur = conn.execute("INSERT INTO users(email,password_hash,salt,created_at) VALUES(?,?,?,?)",
                               (email, make_password_hash(password, salt), salt.hex(), int(time.time())))
            user_id = cur.lastrowid
    except sqlite3.IntegrityError:
        return jsonify({"error": "An account with this email already exists."}), 409
    return jsonify({"token": issue_token(user_id, email), "user": {"id": user_id, "email": email}}), 201

@app.post("/api/auth/login")
def login():
    payload = request.get_json(silent=True) or {}
    email = str(payload.get("email", "")).strip().lower()
    password = str(payload.get("password", ""))
    with db() as conn:
        user = conn.execute("SELECT id,email,password_hash,salt FROM users WHERE email=?", (email,)).fetchone()
    if not user:
        return jsonify({"error": "Invalid email or password."}), 401
    salt = bytes.fromhex(user["salt"])
    candidate = make_password_hash(password, salt)
    if not hmac.compare_digest(candidate, user["password_hash"]):
        return jsonify({"error": "Invalid email or password."}), 401
    return jsonify({"token": issue_token(user["id"], user["email"]), "user": {"id": user["id"], "email": user["email"]}})

@app.get("/api/auth/me")
@require_auth
def me():
    return jsonify({"user": {"id": g.user["uid"], "email": g.user["email"]}})

@app.get("/api/ml/status")
def status():
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else None
    return jsonify({
        "available": model is not None,
        "modelName": "SecuritySnippetClassifier-v2",
        "algorithm": "TF-IDF character n-grams + engineered security features + Logistic Regression",
        "datasetRows": metrics.get("dataset_rows_after_deduplication") if metrics else None,
        "metrics": metrics,
        "message": "Trained model ready." if model is not None else "Model not trained. Run python train_model.py."
    })

@app.post("/api/ml/predict")
@require_auth
def predict():
    if model is None:
        return jsonify({"error": "Trained model is not available. Run python train_model.py in ml_backend first."}), 503
    payload = request.get_json(silent=True) or {}
    items = payload.get("items", [])
    if not isinstance(items, list) or len(items) > 1000:
        return jsonify({"error": "Send an items array containing at most 1000 snippets."}), 400
    texts = [str(item.get("code", ""))[:12000] for item in items]
    if not texts:
        return jsonify({"predictions": []})
    probs = model.predict_proba(texts)
    classes = list(model.named_steps["classifier"].classes_)
    vulnerable_index = classes.index(1)
    predictions = []
    for item, row in zip(items, probs):
        probability = float(row[vulnerable_index])
        if probability >= 0.75:
            prediction, band = "Vulnerable", "high"
        elif probability >= 0.50:
            prediction, band = "Review Required", "moderate"
        else:
            prediction, band = "Likely Safe", "low"
        predictions.append({
            "id": str(item.get("id", "")),
            "prediction": prediction,
            "vulnerabilityProbability": round(probability, 4),
            "confidencePercent": round(probability * 100),
            "confidenceBand": band
        })
    return jsonify({"modelName": "SecuritySnippetClassifier-v2", "predictions": predictions})

@app.post("/api/github/import")
@require_auth
def github_import():
    payload = request.get_json(silent=True) or {}
    repo_url = str(payload.get("url", "")).strip()
    match = re.fullmatch(r"https?://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?/?", repo_url)
    if not match:
        return jsonify({"error": "Enter a public GitHub repository URL such as https://github.com/owner/repository"}), 400
    owner, repo = match.group(1), match.group(2)
    if owner in {".", ".."} or repo in {".", ".."}:
        return jsonify({"error": "Invalid repository path."}), 400
    api_url = f"https://api.github.com/repos/{owner}/{repo}"
    headers = {"User-Agent": "Sentinel-AI-Local-Scanner", "Accept": "application/vnd.github+json"}
    try:
        req = urllib.request.Request(api_url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as response:
            metadata = json.loads(response.read(1_000_000).decode("utf-8"))
        if metadata.get("private"):
            return jsonify({"error": "Private repositories are not supported by this public-repository importer."}), 400
        branch = metadata.get("default_branch") or "main"
        archive_url = f"https://codeload.github.com/{owner}/{repo}/zip/refs/heads/{branch}"
        req = urllib.request.Request(archive_url, headers={"User-Agent": "Sentinel-AI-Local-Scanner"})
        with urllib.request.urlopen(req, timeout=30) as response:
            archive = response.read(MAX_REPO_BYTES + 1)
        if len(archive) > MAX_REPO_BYTES:
            return jsonify({"error": "Repository archive exceeds the 20 MB import limit."}), 413
        files = []
        with zipfile.ZipFile(io.BytesIO(archive)) as zf:
            for info in zf.infolist():
                if info.is_dir() or len(files) >= MAX_REPO_FILES:
                    continue
                name = info.filename.replace("\\", "/")
                parts = name.split("/")
                if any(part in IGNORED_PARTS for part in parts) or not ALLOWED_EXT.search(name):
                    continue
                if info.file_size > 500_000:
                    continue
                # Reject symlink entries and oversized compression ratios.
                mode = (info.external_attr >> 16) & 0o170000
                if mode == 0o120000:
                    continue
                if info.file_size > max(1, info.compress_size) * 200:
                    continue
                try:
                    content = zf.read(info).decode("utf-8")
                except (UnicodeDecodeError, RuntimeError, zipfile.BadZipFile):
                    continue
                if "\x00" in content:
                    continue
                relative = "/".join(parts[1:]) if len(parts) > 1 else name
                files.append({"path": relative, "content": content})
        if not files:
            return jsonify({"error": "No supported source files were found in this repository."}), 404
        return jsonify({"repository": f"{owner}/{repo}", "defaultBranch": branch, "files": files,
                        "fileCount": len(files), "note": "Public repository import only. Scan only repositories you own or are authorized to assess."})
    except urllib.error.HTTPError as exc:
        message = "GitHub repository not found or API rate limit reached." if exc.code in (403, 404) else f"GitHub returned HTTP {exc.code}."
        return jsonify({"error": message}), 502
    except (urllib.error.URLError, TimeoutError) as exc:
        return jsonify({"error": f"Could not reach GitHub: {exc}"}), 502
    except zipfile.BadZipFile:
        return jsonify({"error": "GitHub returned an invalid repository archive."}), 502
    except Exception as exc:
        return jsonify({"error": f"Repository import failed: {exc}"}), 500

@app.post("/api/ml/reload")
def reload_model():
    load_model()
    return jsonify({"available": model is not None})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)
