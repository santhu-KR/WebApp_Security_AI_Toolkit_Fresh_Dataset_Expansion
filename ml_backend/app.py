from pathlib import Path
import os
import json
import re
import hmac
import hashlib
import base64
import sqlite3
import secrets
import time
import io
import zipfile
import urllib.request
import urllib.error
from functools import wraps
from urllib.parse import urlparse

from flask import Flask, jsonify, request, g
from flask_cors import CORS

# Load ml_backend/.env explicitly and allow it to override stale shell variables.
HERE = Path(__file__).resolve().parent
try:
    from dotenv import load_dotenv
    load_dotenv(HERE / ".env", override=True)
except Exception:
    pass

try:
    from v10_predictor import V10Predictor
except Exception:
    V10Predictor = None

try:
    from scanner import scan_website
except Exception as exc:
    scan_website = None
    SCANNER_IMPORT_ERROR = str(exc)
else:
    SCANNER_IMPORT_ERROR = None


# =========================================================
# CONFIGURATION
# =========================================================
V10_MODEL_DIR = HERE / "models" / "v10"
AUTH_DB = HERE / "users.sqlite3"
SCAN_DB = HERE / "scan_history.db"

TOKEN_SECRET = os.environ.get(
    "APP_TOKEN_SECRET",
    "LOCAL_DEMO_ONLY_CHANGE_THIS_SECRET"
).encode("utf-8")
TOKEN_TTL = 12 * 60 * 60

MAX_REPO_BYTES = 20 * 1024 * 1024
MAX_REPO_FILES = 1500
MAX_SCAN_URL_LENGTH = 2048

ALLOWED_EXT = re.compile(
    r"\.(py|js|ts|tsx|jsx|php|java|html|env|json|ya?ml)$",
    re.I
)
IGNORED_PARTS = {
    "node_modules", ".git", "dist", "build",
    "__pycache__", ".venv", "venv"
}
ALLOWED_HOSTS = {"127.0.0.1", "localhost", "::1"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024

CORS(
    app,
    resources={
        r"/api/*": {
            "origins": [
                "http://localhost:5173",
                "http://127.0.0.1:5173",
                "http://localhost:3000",
                "http://127.0.0.1:3000",
            ]
        }
    },
    supports_credentials=False,
)

model = None
v10_predictor = None


# =========================================================
# DATABASE
# =========================================================
def auth_db():
    conn = sqlite3.connect(AUTH_DB)
    conn.row_factory = sqlite3.Row
    return conn


def scan_db():
    conn = sqlite3.connect(SCAN_DB)
    conn.row_factory = sqlite3.Row
    return conn


def init_auth_db():
    with auth_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at INTEGER NOT NULL
            )
        """)


def init_scan_db():
    with scan_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                target TEXT NOT NULL,
                status TEXT NOT NULL,
                total_findings INTEGER DEFAULT 0,
                severity_counts TEXT,
                findings TEXT,
                scan_started TEXT NOT NULL,
                scan_finished TEXT NOT NULL,
                duration_seconds REAL DEFAULT 0,
                error TEXT
            )
        """)


def save_scan_record(record):
    with scan_db() as conn:
        cursor = conn.execute("""
            INSERT INTO scans (
                target, status, total_findings, severity_counts,
                findings, scan_started, scan_finished,
                duration_seconds, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.get("target", ""),
            record.get("status", "completed"),
            int(record.get("total_findings", 0)),
            json.dumps(record.get("severity_counts", {})),
            json.dumps(record.get("findings", [])),
            record.get("scan_started", ""),
            record.get("scan_finished", ""),
            float(record.get("duration_seconds", 0)),
            record.get("error"),
        ))
        return cursor.lastrowid


# =========================================================
# ML MODEL
# =========================================================
def load_model():
    global model, v10_predictor

    if V10Predictor is None:
        v10_predictor = None
        model = None
        print("V10 predictor module unavailable.")
        return

    try:
        v10_predictor = V10Predictor()
        model = getattr(v10_predictor, "model", None)
        print("V10 ML predictor loaded successfully.")
    except Exception as exc:
        v10_predictor = None
        model = None
        print(f"V10 model unavailable: {exc}")


# =========================================================
# AUTHENTICATION
# =========================================================
def make_password_hash(password, salt):
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        310_000
    ).hex()


def issue_token(user_id, email):
    payload = json.dumps(
        {
            "uid": user_id,
            "email": email,
            "exp": int(time.time()) + TOKEN_TTL
        },
        separators=(",", ":")
    ).encode("utf-8")

    body = base64.urlsafe_b64encode(payload).rstrip(b"=")
    signature = hmac.new(TOKEN_SECRET, body, hashlib.sha256).digest()
    signed = base64.urlsafe_b64encode(signature).rstrip(b"=")
    return (body + b"." + signed).decode("utf-8")


def decode_token(token):
    try:
        body, signature_part = token.encode("utf-8").split(b".", 1)
        expected = hmac.new(TOKEN_SECRET, body, hashlib.sha256).digest()
        supplied = base64.urlsafe_b64decode(
            signature_part + b"=" * (-len(signature_part) % 4)
        )

        if not hmac.compare_digest(expected, supplied):
            return None

        payload = base64.urlsafe_b64decode(
            body + b"=" * (-len(body) % 4)
        )
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
        token = (
            header[7:].strip()
            if header.lower().startswith("bearer ")
            else ""
        )
        data = decode_token(token)

        if not data:
            return jsonify({
                "error": "Authentication required. Please sign in again."
            }), 401

        g.user = data
        return fn(*args, **kwargs)

    return wrapped


# Initialize local databases and model on startup.
init_auth_db()
init_scan_db()
load_model()


# =========================================================
# HEALTH
# =========================================================
@app.get("/api/health")
def health():
    return jsonify({
        "ok": True,
        "service": "local-security-api",
        "scanner_available": scan_website is not None,
        "ml_available": v10_predictor is not None
    })


# =========================================================
# AUTH ROUTES
# =========================================================
@app.post("/api/auth/register")
def register():
    payload = request.get_json(silent=True) or {}
    email = str(payload.get("email", "")).strip().lower()
    password = str(payload.get("password", ""))

    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email) or len(email) > 254:
        return jsonify({"error": "Enter a valid email address."}), 400

    if len(password) < 10:
        return jsonify({
            "error": "Password must contain at least 10 characters."
        }), 400

    salt = secrets.token_bytes(16)

    try:
        with auth_db() as conn:
            cursor = conn.execute(
                """INSERT INTO users(email,password_hash,salt,created_at)
                   VALUES(?,?,?,?)""",
                (
                    email,
                    make_password_hash(password, salt),
                    salt.hex(),
                    int(time.time())
                )
            )
            user_id = cursor.lastrowid
    except sqlite3.IntegrityError:
        return jsonify({
            "error": "An account with this email already exists."
        }), 409

    return jsonify({
        "token": issue_token(user_id, email),
        "user": {"id": user_id, "email": email}
    }), 201


@app.post("/api/auth/login")
def login():
    payload = request.get_json(silent=True) or {}
    email = str(payload.get("email", "")).strip().lower()
    password = str(payload.get("password", ""))

    with auth_db() as conn:
        user = conn.execute(
            "SELECT id,email,password_hash,salt FROM users WHERE email=?",
            (email,)
        ).fetchone()

    if not user:
        return jsonify({"error": "Invalid email or password."}), 401

    salt = bytes.fromhex(user["salt"])
    candidate = make_password_hash(password, salt)

    if not hmac.compare_digest(candidate, user["password_hash"]):
        return jsonify({"error": "Invalid email or password."}), 401

    return jsonify({
        "token": issue_token(user["id"], user["email"]),
        "user": {"id": user["id"], "email": user["email"]}
    })


@app.get("/api/auth/me")
@require_auth
def me():
    return jsonify({
        "user": {
            "id": g.user["uid"],
            "email": g.user["email"]
        }
    })


# =========================================================
# ML ROUTES
# =========================================================
@app.get("/api/ml/status")
def ml_status():
    metadata = getattr(v10_predictor, "metadata", {}) if v10_predictor else {}

    return jsonify({
        "available": v10_predictor is not None,
        "modelName": metadata.get("model_name", "V10 ExtraTrees"),
        "algorithm": (
            "ExtraTrees + character TF-IDF + word TF-IDF "
            "+ engineered features"
        ),
        "datasetRows": metadata.get("training_samples"),
        "metrics": metadata,
        "threshold": getattr(v10_predictor, "threshold", None),
        "message": (
            "V10 model ready."
            if v10_predictor
            else "V10 artifacts are not loaded. Check ml_backend/models/v10."
        )
    })


@app.post("/api/ml/predict")
@require_auth
def predict():
    if v10_predictor is None:
        return jsonify({
            "error": "V10 artifacts are not loaded. Check ml_backend/models/v10."
        }), 503

    payload = request.get_json(silent=True) or {}
    items = payload.get("items", [])

    if not isinstance(items, list) or len(items) > 1000:
        return jsonify({
            "error": "Send an items array containing at most 1000 snippets."
        }), 400

    texts = [
        str(item.get("code", ""))[:12000]
        if isinstance(item, dict)
        else str(item)[:12000]
        for item in items
    ]

    if not texts:
        return jsonify({"predictions": []})

    try:
        scores = v10_predictor.predict(texts)
    except Exception as exc:
        return jsonify({"error": f"V10 prediction failed: {exc}"}), 500

    predictions = []
    for item, score in zip(items, scores):
        item_id = item.get("id", "") if isinstance(item, dict) else ""

        predictions.append({
            "id": str(item_id),
            "prediction": score.get("prediction"),
            "vulnerabilityProbability": round(
                float(score.get("probability", 0)), 4
            ),
            "confidencePercent": score.get("confidencePercent"),
            "confidenceBand": score.get("confidenceBand"),
            "isVulnerable": score.get("isVulnerable"),
            "threshold": score.get("threshold"),
            "modelName": "V10 ExtraTrees"
        })

    return jsonify({
        "modelName": "V10 ExtraTrees",
        "threshold": getattr(v10_predictor, "threshold", None),
        "predictions": predictions
    })


@app.post("/api/ml/reload")
def reload_model():
    load_model()
    return jsonify({"available": v10_predictor is not None})


# =========================================================
# GITHUB REPOSITORY IMPORT
# =========================================================
@app.post("/api/github/import")
@require_auth
def github_import():
    payload = request.get_json(silent=True) or {}
    repo_url = str(payload.get("url", "")).strip()

    match = re.fullmatch(
        r"https?://github\.com/([A-Za-z0-9_.-]+)/"
        r"([A-Za-z0-9_.-]+?)(?:\.git)?/?",
        repo_url
    )

    if not match:
        return jsonify({
            "error": "Enter a public GitHub repository URL such as "
                     "https://github.com/owner/repository"
        }), 400

    owner, repo = match.groups()

    if owner in {".", ".."} or repo in {".", ".."}:
        return jsonify({"error": "Invalid repository path."}), 400

    api_url = f"https://api.github.com/repos/{owner}/{repo}"
    headers = {
        "User-Agent": "Local-Security-Scanner",
        "Accept": "application/vnd.github+json"
    }

    try:
        req = urllib.request.Request(api_url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as response:
            metadata = json.loads(response.read(1_000_000).decode("utf-8"))

        if metadata.get("private"):
            return jsonify({
                "error": "Private repositories are not supported."
            }), 400

        branch = metadata.get("default_branch") or "main"
        archive_url = (
            f"https://codeload.github.com/{owner}/{repo}"
            f"/zip/refs/heads/{branch}"
        )

        req = urllib.request.Request(
            archive_url,
            headers={"User-Agent": "Local-Security-Scanner"}
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            archive = response.read(MAX_REPO_BYTES + 1)

        if len(archive) > MAX_REPO_BYTES:
            return jsonify({
                "error": "Repository archive exceeds the 20 MB import limit."
            }), 413

        files = []

        with zipfile.ZipFile(io.BytesIO(archive)) as zf:
            for info in zf.infolist():
                if info.is_dir() or len(files) >= MAX_REPO_FILES:
                    continue

                name = info.filename.replace("\\", "/")
                parts = name.split("/")

                if any(part in IGNORED_PARTS for part in parts):
                    continue
                if not ALLOWED_EXT.search(name):
                    continue
                if info.file_size > 500_000:
                    continue

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
                files.append({
                    "path": relative,
                    "content": content
                })

        if not files:
            return jsonify({
                "error": "No supported source files were found."
            }), 404

        return jsonify({
            "repository": f"{owner}/{repo}",
            "defaultBranch": branch,
            "files": files,
            "fileCount": len(files),
            "note": "Import only repositories you own or are authorized to assess."
        })

    except urllib.error.HTTPError as exc:
        message = (
            "GitHub repository not found or API rate limit reached."
            if exc.code in (403, 404)
            else f"GitHub returned HTTP {exc.code}."
        )
        return jsonify({"error": message}), 502
    except (urllib.error.URLError, TimeoutError) as exc:
        return jsonify({"error": f"Could not reach GitHub: {exc}"}), 502
    except zipfile.BadZipFile:
        return jsonify({"error": "GitHub returned an invalid archive."}), 502
    except Exception as exc:
        return jsonify({"error": f"Repository import failed: {exc}"}), 500


# =========================================================
# WEBSITE SCAN API - LOCAL AUTHORIZED TARGETS ONLY
# =========================================================
def validate_local_target(target_url):
    if not isinstance(target_url, str):
        return False, "Target URL must be a string."

    target_url = target_url.strip()

    if not target_url or len(target_url) > MAX_SCAN_URL_LENGTH:
        return False, "Target URL is empty or too long."

    parsed = urlparse(target_url)

    if parsed.scheme not in {"http", "https"}:
        return False, "Only HTTP and HTTPS URLs are supported."

    if not parsed.hostname:
        return False, "Invalid target URL."

    hostname = parsed.hostname.lower()

    if hostname not in ALLOWED_HOSTS:
        return False, "For this local demo, scan only localhost targets."

    if parsed.username or parsed.password:
        return False, "URLs containing credentials are not allowed."

    return True, target_url


def get_findings_from_result(results):
    if isinstance(results, list):
        return results

    if isinstance(results, dict):
        for key in ("findings", "vulnerabilities", "issues", "results"):
            value = results.get(key)
            if isinstance(value, list):
                return value

    return []


def calculate_severity_counts(findings):
    counts = {
        "Critical": 0,
        "High": 0,
        "Medium": 0,
        "Low": 0,
        "Informational": 0
    }

    for finding in findings:
        if not isinstance(finding, dict):
            continue

        severity = str(
            finding.get("severity", finding.get("risk", "Informational"))
        ).strip().title()

        if severity in counts:
            counts[severity] += 1
        elif severity in {"Info", "Information"}:
            counts["Informational"] += 1
        else:
            counts["Informational"] += 1

    return counts


@app.post("/api/scan")
def run_website_scan():
    if scan_website is None:
        return jsonify({
            "success": False,
            "error": f"Scanner could not be imported: {SCANNER_IMPORT_ERROR}"
        }), 503

    payload = request.get_json(silent=True) or {}
    target_url = payload.get("target_url", payload.get("url", ""))

    valid, message = validate_local_target(target_url)
    if not valid:
        return jsonify({
            "success": False,
            "error": message
        }), 400 if "Invalid" in message or "must be" in message else 403

    target_url = message
    started_at = datetime_iso()
    start_time = time.perf_counter()

    try:
        results = scan_website(target_url)
        findings = get_findings_from_result(results)
        severity_counts = calculate_severity_counts(findings)
        finished_at = datetime_iso()
        duration = round(time.perf_counter() - start_time, 3)

        record = {
            "target": target_url,
            "status": "completed",
            "total_findings": len(findings),
            "severity_counts": severity_counts,
            "findings": findings,
            "scan_started": started_at,
            "scan_finished": finished_at,
            "duration_seconds": duration,
            "error": None
        }

        scan_id = save_scan_record(record)

        return jsonify({
            "success": True,
            "scan_id": scan_id,
            "target": target_url,
            "status": "completed",
            "total_findings": len(findings),
            "severity_counts": severity_counts,
            "duration_seconds": duration,
            "results": results,
            "findings": findings
        }), 200

    except Exception as exc:
        finished_at = datetime_iso()
        duration = round(time.perf_counter() - start_time, 3)

        record = {
            "target": target_url,
            "status": "failed",
            "total_findings": 0,
            "severity_counts": {},
            "findings": [],
            "scan_started": started_at,
            "scan_finished": finished_at,
            "duration_seconds": duration,
            "error": str(exc)
        }

        try:
            save_scan_record(record)
        except Exception:
            pass

        return jsonify({
            "success": False,
            "target": target_url,
            "error": str(exc)
        }), 500


def datetime_iso():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# =========================================================
# SCAN HISTORY
# =========================================================
@app.get("/api/scans")
def get_scan_history():
    limit = request.args.get("limit", "50")

    try:
        limit = max(1, min(int(limit), 200))
    except ValueError:
        limit = 50

    with scan_db() as conn:
        rows = conn.execute(
            "SELECT * FROM scans ORDER BY id DESC LIMIT ?",
            (limit,)
        ).fetchall()

    history = []
    for row in rows:
        item = dict(row)
        try:
            item["severity_counts"] = json.loads(item["severity_counts"] or "{}")
        except json.JSONDecodeError:
            item["severity_counts"] = {}
        try:
            item["findings"] = json.loads(item["findings"] or "[]")
        except json.JSONDecodeError:
            item["findings"] = []
        history.append(item)

    return jsonify({"scans": history, "count": len(history)})


@app.get("/api/scans/<int:scan_id>")
def get_scan_details(scan_id):
    with scan_db() as conn:
        row = conn.execute(
            "SELECT * FROM scans WHERE id=?",
            (scan_id,)
        ).fetchone()

    if not row:
        return jsonify({"error": "Scan not found."}), 404

    item = dict(row)
    try:
        item["severity_counts"] = json.loads(item["severity_counts"] or "{}")
    except json.JSONDecodeError:
        item["severity_counts"] = {}
    try:
        item["findings"] = json.loads(item["findings"] or "[]")
    except json.JSONDecodeError:
        item["findings"] = []

    return jsonify({"scan": item})


@app.delete("/api/scans")
def delete_scan_history():
    with scan_db() as conn:
        conn.execute("DELETE FROM scans")
    return jsonify({"success": True, "message": "Scan history cleared."})


# =========================================================
# AI SERVICE IMPORT
# =========================================================
try:
    from ai_service import (
        gemini_generate,
        gemini_chat,
        get_ai_status as ai_service_status,
        GEMINI_MODEL,
        GEMINI_CHAT_MODEL,
        GEMINI_API_KEY,
    )
    AI_SERVICE_ERROR = None
except Exception as exc:
    gemini_generate = None
    gemini_chat = None
    ai_service_status = None
    GEMINI_MODEL = None
    GEMINI_CHAT_MODEL = None
    GEMINI_API_KEY = ""
    AI_SERVICE_ERROR = str(exc)


# =========================================================
# AI STATUS
# =========================================================
@app.get("/api/ai/status")
def ai_status():
    if ai_service_status:
        status = ai_service_status()
        status["available"] = bool(gemini_generate and GEMINI_API_KEY)
        status["model"] = GEMINI_CHAT_MODEL or status.get("model")
        status["explanation_model"] = GEMINI_MODEL
        if not status["available"]:
            status["message"] = "Configure GEMINI_API_KEY in ml_backend/.env"
        return jsonify(status)

    return jsonify({
        "available": False,
        "message": "AI service module unavailable: " + str(AI_SERVICE_ERROR),
        "model": None
    }), 503


# =========================================================
# AI EXPLAIN
# =========================================================
@app.post("/api/ai/explain")
@require_auth
def ai_explain():
    if not gemini_generate:
        return jsonify({
            "error": "AI service module unavailable: " + str(AI_SERVICE_ERROR)
        }), 503

    payload = request.get_json(silent=True) or {}
    finding = payload.get("finding", {})
    code_context = str(payload.get("codeContext", ""))[:20000]

    if not isinstance(finding, dict) or not finding:
        return jsonify({"error": "Provide a finding object."}), 400

    prompt = (
        "Analyze this defensive code-security finding using the supplied code context. "
        "Do not assume it is confirmed based only on a rule match. Clearly state whether evidence is "
        "confirmed, likely, or insufficient. Explain in simple English, include cause, impact, "
        "evidence, safe remediation, and a small secure code example if relevant. Do not provide "
        "exploit payloads or attack instructions.\n\nFINDING:\n"
        + json.dumps(finding, ensure_ascii=False)
        + "\n\nSOURCE CONTEXT:\n" + code_context
    )

    system = (
        "You are a defensive application security analyst. Be accurate, contextual, "
        "and avoid overstating scanner evidence."
    )

    try:
        answer = gemini_generate(prompt, system)
        return jsonify({
            "answer": answer,
            "model": GEMINI_MODEL
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 502


# =========================================================
# AI CHAT
# =========================================================
@app.post("/api/ai/chat")
@require_auth
def ai_chat():
    if not gemini_chat:
        return jsonify({
            "error": "AI service module unavailable: " + str(AI_SERVICE_ERROR)
        }), 503

    payload = request.get_json(silent=True) or {}
    prompt = str(payload.get("prompt", "")).strip()[:6000]
    file_path = str(payload.get("filePath", ""))[:500]
    file_content = str(payload.get("fileContent", ""))[:30000]
    findings = payload.get("findings", [])
    history = payload.get("history", [])

    if not prompt:
        return jsonify({"error": "Prompt cannot be empty."}), 400
    if not isinstance(findings, list):
        findings = []
    if not isinstance(history, list):
        history = []

    safe_history = []
    for item in history[-8:]:
        if isinstance(item, dict) and item.get("role") in ("user", "model"):
            safe_history.append(
                f"{item.get('role')}: {str(item.get('text', ''))[:3000]}"
            )

    context = (
        "Selected file: " + file_path + "\n\nSOURCE FILE:\n" + file_content +
        "\n\nFINDINGS FOR THIS FILE:\n" + json.dumps(findings[:50], ensure_ascii=False) +
        "\n\nRECENT CHAT:\n" + "\n".join(safe_history)
    )

    system = (
        "You are a defensive application-security assistant helping review an authorized source-code project. "
        "Answer questions using the selected file and findings context. Be honest if evidence is insufficient. "
        "When the user asks to fix or rewrite code, return a complete replacement file in one fenced code block; "
        "never omit sections or use ellipses. Avoid exploit payloads, credential theft, destructive actions, "
        "or offensive instructions. For ordinary questions, do not force code output."
    )

    try:
        answer = gemini_chat(context + "\n\nUSER QUESTION:\n" + prompt, system)
        return jsonify({
            "answer": answer,
            "model": GEMINI_CHAT_MODEL
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 502


# =========================================================
# RUN SERVER
# =========================================================
if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5001,
        debug=False,
        threaded=True
    )
