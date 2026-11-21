"""Server-side Gemini integration for defensive source-code analysis."""
import os
import json
import urllib.request
import urllib.error

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.0-flash")


def gemini_generate(prompt, system_instruction=None, model=None):
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured in the backend environment.")
    model_name = model or GEMINI_MODEL
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        + model_name + ":generateContent?key=" + GEMINI_API_KEY
    )
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 8192}
    }
    if system_instruction:
        body["systemInstruction"] = {"parts": [{"text": system_instruction}]}
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as response:
            data = json.loads(response.read().decode("utf-8"))
        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError("Gemini returned no response. Check model access or quota.")
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(part.get("text", "") for part in parts).strip()
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8")[:1200]
        except Exception:
            detail = str(exc)
        if exc.code == 429:
            raise RuntimeError("Gemini quota/rate limit reached. Please wait and retry.")
        raise RuntimeError(f"Gemini API error ({exc.code}): {detail}")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Cannot connect to Gemini: {exc.reason}")
