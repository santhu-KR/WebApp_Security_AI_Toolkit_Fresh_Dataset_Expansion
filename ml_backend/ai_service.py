"""
AI Service
Web Application Security AI Toolkit

Server-side Gemini integration.
API keys must remain in the backend .env file.
"""

import os
import json
import urllib.request
import urllib.error
import time
from pathlib import Path


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"

try:
    from dotenv import load_dotenv

    # Load environment variables from ml_backend/.env
    load_dotenv(ENV_FILE, override=True)

except ImportError:
    print("[WARNING] python-dotenv is not installed.")


# =========================================================
# CONFIGURATION
# =========================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# Default model
DEFAULT_MODEL = "gemini-3.8-flash"

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    DEFAULT_MODEL
).strip()

GEMINI_CHAT_MODEL = os.getenv(
    "GEMINI_CHAT_MODEL",
    GEMINI_MODEL
).strip()


# =========================================================
# CONFIGURATION STATUS
# =========================================================

def get_ai_status():

    return {
        "available": bool(GEMINI_API_KEY),
        "message": (
            "AI service ready."
            if GEMINI_API_KEY
            else "GEMINI_API_KEY is missing."
        ),
        "model": GEMINI_CHAT_MODEL
    }


# =========================================================
# GEMINI API REQUEST
# =========================================================

def gemini_generate(
    prompt,
    system_instruction=None,
    model=None
):
    """Call Gemini with retries and model fallback for temporary/API model failures."""

    if not GEMINI_API_KEY or GEMINI_API_KEY.lower().startswith(("put_your", "replace_", "your_")):
        raise RuntimeError(
            "GEMINI_API_KEY is missing or is still a placeholder. "
            "Configure a real key in ml_backend/.env and restart the backend."
        )

    primary_model = (model or GEMINI_CHAT_MODEL or GEMINI_MODEL).strip()
    if primary_model.startswith("models/"):
        primary_model = primary_model.removeprefix("models/")

    # Fallback models can be customized in ml_backend/.env:
    # GEMINI_FALLBACK_MODELS=gemini-model-1,gemini-model-2
    fallback_setting = os.getenv(
        "GEMINI_FALLBACK_MODELS",
        DEFAULT_MODEL
    )
    fallback_models = [
        item.strip().removeprefix("models/")
        for item in fallback_setting.split(",")
        if item.strip()
    ]

    model_candidates = []
    for candidate in [primary_model, *fallback_models]:
        if candidate and candidate not in model_candidates:
            model_candidates.append(candidate)

    body = {
        "contents": [{"role": "user", "parts": [{"text": str(prompt)}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 8192}
    }
    if system_instruction:
        body["systemInstruction"] = {"parts": [{"text": str(system_instruction)}]}

    request_data = json.dumps(body).encode("utf-8")
    retryable_statuses = {429, 500, 502, 503, 504}
    last_error = None

    for model_name in model_candidates:
        url = (
            "https://generativelanguage.googleapis.com/"
            "v1beta/models/" + model_name + ":generateContent"
        )

        # Retry temporary errors with exponential backoff: 1s, 2s, 4s.
        for attempt in range(3):
            req = urllib.request.Request(
                url,
                data=request_data,
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": GEMINI_API_KEY,
                    "User-Agent": "WebApp-Security-AI-Toolkit"
                },
                method="POST"
            )

            try:
                with urllib.request.urlopen(req, timeout=90) as response:
                    data = json.loads(response.read().decode("utf-8"))

                candidates = data.get("candidates", [])
                if not candidates:
                    last_error = RuntimeError(
                        "Gemini returned an empty response from model "
                        + model_name
                    )
                    break

                parts = candidates[0].get("content", {}).get("parts", [])
                answer = "".join(part.get("text", "") for part in parts).strip()
                if not answer:
                    last_error = RuntimeError(
                        "Gemini returned no text response from model "
                        + model_name
                    )
                    break

                if model_name != primary_model:
                    print("[INFO] Gemini fallback model succeeded: " + model_name)
                return answer

            except urllib.error.HTTPError as exc:
                try:
                    error_data = json.loads(exc.read().decode("utf-8"))
                    error_message = error_data.get("error", {}).get("message", "")
                except Exception:
                    error_message = str(exc)

                if exc.code in (401, 403):
                    raise RuntimeError(
                        "Gemini API authentication failed. Check your API key and permissions."
                    ) from exc

                if exc.code == 400:
                    raise RuntimeError(
                        "Gemini rejected the request: " + str(error_message)
                    ) from exc

                if exc.code == 404:
                    last_error = RuntimeError(
                        "Gemini model unavailable: " + model_name + ". " + str(error_message)
                    )
                    print("[WARNING] Model unavailable; trying next fallback: " + model_name)
                    break

                if exc.code in retryable_statuses:
                    last_error = RuntimeError(
                        "Gemini API error " + str(exc.code) + ": " + str(error_message)
                    )
                    if attempt < 2:
                        wait_seconds = 2 ** attempt
                        print(
                            "[WARNING] Gemini " + str(exc.code)
                            + " on " + model_name
                            + "; retrying in " + str(wait_seconds) + " seconds."
                        )
                        time.sleep(wait_seconds)
                        continue
                    print("[WARNING] Retries exhausted for model: " + model_name)
                    break

                raise RuntimeError(
                    "Gemini API error " + str(exc.code) + ": " + str(error_message)
                ) from exc

            except urllib.error.URLError as exc:
                last_error = RuntimeError(
                    "Cannot connect to Gemini API. Check your internet connection. "
                    + str(exc.reason)
                )
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                break

            except TimeoutError:
                last_error = RuntimeError(
                    "Gemini request timed out for model " + model_name
                )
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                break

            except json.JSONDecodeError as exc:
                raise RuntimeError("Gemini returned an invalid response.") from exc

    raise RuntimeError(
        "Gemini AI is temporarily unavailable. The primary model and fallback "
        "models failed. Please try again in a minute. Details: "
        + str(last_error or "Unknown error")
    )


# =========================================================
# CHAT MODEL
# =========================================================

def gemini_chat(
    prompt,
    system_instruction=None
):

    return gemini_generate(

        prompt=prompt,

        system_instruction=system_instruction,

        model=GEMINI_CHAT_MODEL

    )


# =========================================================
# SECURITY EXPLANATION
# =========================================================

def explain_vulnerability(
    finding,
    code_context=""
):

    system_instruction = """

You are an expert defensive application security analyst.

Analyze security findings carefully.

Do not automatically assume a vulnerability is confirmed.

Clearly classify the evidence as:

1. Confirmed
2. Likely
3. Potential
4. Insufficient evidence

Explain:

- Vulnerability name
- Severity
- What the vulnerability means
- Why it occurs
- Evidence from the source code
- Potential impact
- Recommended remediation
- Secure coding example

Use simple English suitable for MCA students.

Do not provide malicious exploitation instructions.

"""


    prompt = f"""

Analyze the following security finding.

FINDING:

{json.dumps(finding, indent=2, ensure_ascii=False)}


SOURCE CODE CONTEXT:

{code_context}


Provide a detailed defensive security analysis.

"""


    return gemini_generate(

        prompt=prompt,

        system_instruction=system_instruction,

        model=GEMINI_MODEL

    )


# =========================================================
# END OF AI SERVICE
# =========================================================