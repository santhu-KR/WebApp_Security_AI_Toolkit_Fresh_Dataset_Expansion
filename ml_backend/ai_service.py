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

    # -----------------------------------------------------
    # API KEY VALIDATION
    # -----------------------------------------------------

    if not GEMINI_API_KEY:

        raise RuntimeError(
            "GEMINI_API_KEY is missing or is still a placeholder. "
            "Configure a real key in ml_backend/.env and restart the backend."
        )


    # -----------------------------------------------------
    # SELECT MODEL
    # -----------------------------------------------------

    model_name = model or GEMINI_MODEL

    model_name = model_name.strip()

    if model_name.startswith("models/"):

        model_name = model_name.replace(
            "models/",
            "",
            1
        )


    # -----------------------------------------------------
    # API URL
    # -----------------------------------------------------

    url = (
        "https://generativelanguage.googleapis.com/"
        "v1beta/models/"
        + model_name
        + ":generateContent"
    )


    # -----------------------------------------------------
    # REQUEST BODY
    # -----------------------------------------------------

    body = {

        "contents": [

            {
                "role": "user",

                "parts": [

                    {
                        "text": str(prompt)
                    }

                ]
            }

        ],

        "generationConfig": {

            "temperature": 0.2,

            "maxOutputTokens": 8192

        }

    }


    # -----------------------------------------------------
    # SYSTEM INSTRUCTION
    # -----------------------------------------------------

    if system_instruction:

        body["systemInstruction"] = {

            "parts": [

                {
                    "text": str(system_instruction)
                }

            ]

        }


    # -----------------------------------------------------
    # BUILD REQUEST
    # -----------------------------------------------------

    request_data = json.dumps(
        body
    ).encode("utf-8")


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


    # -----------------------------------------------------
    # SEND REQUEST
    # -----------------------------------------------------

    try:

        with urllib.request.urlopen(
            req,
            timeout=90
        ) as response:

            response_data = response.read().decode(
                "utf-8"
            )

            data = json.loads(
                response_data
            )


        # -------------------------------------------------
        # EXTRACT RESPONSE
        # -------------------------------------------------

        candidates = data.get(
            "candidates",
            []
        )


        if not candidates:

            raise RuntimeError(
                "Gemini returned an empty response. "
                "Check model availability and API quota."
            )


        candidate = candidates[0]


        content = candidate.get(
            "content",
            {}
        )


        parts = content.get(
            "parts",
            []
        )


        answer = "".join(

            part.get(
                "text",
                ""
            )

            for part in parts

        ).strip()


        if not answer:

            raise RuntimeError(
                "Gemini returned no text response."
            )


        return answer


    # =====================================================
    # HTTP ERRORS
    # =====================================================

    except urllib.error.HTTPError as exc:

        try:

            error_body = exc.read().decode(
                "utf-8"
            )

            error_data = json.loads(
                error_body
            )

            error_message = (
                error_data
                .get("error", {})
                .get("message", error_body)
            )

        except Exception:

            error_message = str(exc)


        # -------------------------------------------------
        # 400
        # -------------------------------------------------

        if exc.code == 400:

            raise RuntimeError(
                "Gemini rejected the request: "
                + str(error_message)
            )


        # -------------------------------------------------
        # 401 / 403
        # -------------------------------------------------

        elif exc.code in (401, 403):

            raise RuntimeError(
                "Gemini API authentication failed. "
                "Check your API key and permissions."
            )


        # -------------------------------------------------
        # 404 MODEL NOT FOUND
        # -------------------------------------------------

        elif exc.code == 404:

            raise RuntimeError(

                "Gemini model not found: "
                + model_name
                + ". Check GEMINI_MODEL in ml_backend/.env "
                "and verify that your API key supports this model. "
                "Original error: "
                + str(error_message)

            )


        # -------------------------------------------------
        # 429 QUOTA
        # -------------------------------------------------

        elif exc.code == 429:

            raise RuntimeError(

                "Gemini API quota or rate limit exceeded. "
                "Wait and try again. "
                + str(error_message)

            )


        # -------------------------------------------------
        # OTHER HTTP ERRORS
        # -------------------------------------------------

        else:

            raise RuntimeError(

                "Gemini API error "
                + str(exc.code)
                + ": "
                + str(error_message)

            )


    # =====================================================
    # NETWORK ERRORS
    # =====================================================

    except urllib.error.URLError as exc:

        raise RuntimeError(

            "Cannot connect to Gemini API. "
            "Check your internet connection. "
            + str(exc.reason)

        )


    # =====================================================
    # TIMEOUT
    # =====================================================

    except TimeoutError:

        raise RuntimeError(
            "Gemini request timed out. Please try again."
        )


    # =====================================================
    # JSON ERRORS
    # =====================================================

    except json.JSONDecodeError:

        raise RuntimeError(
            "Gemini returned an invalid response."
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