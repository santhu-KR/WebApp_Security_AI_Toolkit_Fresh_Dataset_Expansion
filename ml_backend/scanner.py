"""
Web Application Security AI Toolkit
Scanner Module

Authorized testing only.
"""

import requests
import ssl
import socket
import re

from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from datetime import datetime


# =========================================================
# CONFIGURATION
# =========================================================

REQUEST_TIMEOUT = 10

HEADERS = {
    "User-Agent": "WebApplicationSecurityAIToolkit/1.0"
}


# =========================================================
# ML MODEL INTEGRATION
# =========================================================

# The website scanner is rule-based. The active V10 model is
# loaded and used by app.py for source-code prediction.
# The old V2 predictor is intentionally disconnected here.
ML_AVAILABLE = False

print("[INFO] Scanner uses rule-based checks; V10 ML is managed by app.py.")


# =========================================================
# URL VALIDATION
# =========================================================

def validate_url(target_url):

    try:

        parsed = urlparse(target_url)

        if parsed.scheme not in ("http", "https"):
            return False

        if not parsed.hostname:
            return False

        return True

    except Exception:

        return False


# =========================================================
# WEBSITE REQUEST
# =========================================================

def get_website(target_url):

    try:

        response = requests.get(
            target_url,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True
        )

        return response

    except requests.exceptions.RequestException as error:

        return {
            "error": str(error)
        }


# =========================================================
# FINDING CREATION
# =========================================================

def create_finding(
    vulnerability,
    severity,
    description,
    url,
    recommendation,
    category=None
):

    return {

        "vulnerability": vulnerability,

        "severity": severity,

        "description": description,

        "url": url,

        "recommendation": recommendation,

        "category": category,

        "detected_at": datetime.now().isoformat(),

        "source": "Scanner",

        "ml_prediction": None,

        "ml_model_prediction": None,

        "ml_enabled": False,

        "ml_agrees": None

    }


# =========================================================
# ML CLASSIFICATION
# =========================================================

def classify_finding(finding):
    """Return a rule-based finding without invoking the legacy V2 model."""
    finding["ml_prediction"] = None
    finding["ml_model_prediction"] = None
    finding["ml_enabled"] = False
    finding["ml_agrees"] = None
    return finding


# =========================================================
# SECURITY HEADER CHECK
# =========================================================

def check_security_headers(response, target_url):

    findings = []

    security_headers = {

        "Strict-Transport-Security": {

            "name": "Missing HSTS",

            "severity": "Medium",

            "category": "Security Misconfiguration",

            "recommendation":
                "Configure HTTP Strict Transport Security."
        },

        "Content-Security-Policy": {

            "name": "Missing Content Security Policy",

            "severity": "Medium",

            "category": "Security Misconfiguration",

            "recommendation":
                "Configure an appropriate Content Security Policy."
        },

        "X-Content-Type-Options": {

            "name": "Missing X-Content-Type-Options",

            "severity": "Low",

            "category": "Security Misconfiguration",

            "recommendation":
                "Set X-Content-Type-Options to nosniff."
        },

        "X-Frame-Options": {

            "name": "Missing X-Frame-Options",

            "severity": "Low",

            "category": "Security Misconfiguration",

            "recommendation":
                "Configure frame protection against clickjacking."
        },

        "Referrer-Policy": {

            "name": "Missing Referrer Policy",

            "severity": "Low",

            "category": "Security Misconfiguration",

            "recommendation":
                "Configure an appropriate Referrer-Policy."
        }

    }

    for header, details in security_headers.items():

        if header not in response.headers:

            finding = create_finding(

                vulnerability=details["name"],

                severity=details["severity"],

                description=(
                    f"Missing HTTP security header: {header}"
                ),

                url=target_url,

                recommendation=details["recommendation"],

                category=details["category"]

            )

            findings.append(
                classify_finding(finding)
            )

    return findings


# =========================================================
# COOKIE SECURITY CHECK
# =========================================================

def check_cookies(response, target_url):

    findings = []

    set_cookie_headers = response.headers.getlist(
        "Set-Cookie"
    ) if hasattr(response.headers, "getlist") else []

    # Requests may combine cookie information differently.
    # Inspect the response cookie jar as a fallback.

    for cookie in response.cookies:

        cookie_name = cookie.name

        if not cookie.secure:

            finding = create_finding(

                vulnerability="Cookie Missing Secure Flag",

                severity="Medium",

                description=(
                    f"Cookie '{cookie_name}' is missing "
                    "the Secure flag."
                ),

                url=target_url,

                recommendation=(
                    "Set the Secure attribute for cookies "
                    "transmitted over HTTPS."
                ),

                category="Authentication / Session"

            )

            findings.append(
                classify_finding(finding)
            )

        # More reliable HttpOnly detection using
        # the raw Set-Cookie headers.

        cookie_header = ""

        for header_value in set_cookie_headers:

            if cookie_name in header_value:

                cookie_header = header_value

                break

        httponly_present = (
            "httponly" in cookie_header.lower()
        )

        if not httponly_present:

            finding = create_finding(

                vulnerability="Cookie Missing HttpOnly Flag",

                severity="Medium",

                description=(
                    f"Cookie '{cookie_name}' may be "
                    "accessible to JavaScript."
                ),

                url=target_url,

                recommendation=(
                    "Set the HttpOnly attribute for "
                    "sensitive session cookies."
                ),

                category="Authentication / Session"

            )

            findings.append(
                classify_finding(finding)
            )

    return findings


# =========================================================
# INFORMATION DISCLOSURE CHECK
# =========================================================

def check_information_disclosure(response, target_url):

    findings = []

    disclosure_headers = [

        "Server",

        "X-Powered-By",

        "X-AspNet-Version",

        "X-AspNetMvc-Version"

    ]

    for header in disclosure_headers:

        value = response.headers.get(header)

        if value:

            finding = create_finding(

                vulnerability="Technology Information Disclosure",

                severity="Low",

                description=(
                    f"The server discloses information "
                    f"through {header}: {value}"
                ),

                url=target_url,

                recommendation=(
                    "Consider removing or minimizing "
                    "technology version information."
                ),

                category="Information Disclosure"

            )

            findings.append(
                classify_finding(finding)
            )

    return findings


# =========================================================
# HTML FORM ANALYSIS
# =========================================================

def analyze_forms(response, target_url):

    findings = []

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    forms = soup.find_all("form")

    for form in forms:

        method = form.get(
            "method",
            "GET"
        ).upper()

        action = form.get("action", "")

        form_url = urljoin(
            target_url,
            action
        )

        inputs = form.find_all("input")

        password_fields = [

            item for item in inputs

            if item.get(
                "type",
                ""
            ).lower() == "password"

        ]

        # Password transmitted using GET

        if password_fields and method == "GET":

            finding = create_finding(

                vulnerability="Password Submission Through GET",

                severity="High",

                description=(
                    "A form containing a password field "
                    "uses the GET method."
                ),

                url=form_url,

                recommendation=(
                    "Use POST for authentication forms "
                    "and ensure HTTPS is enabled."
                ),

                category="Authentication / Session"

            )

            findings.append(
                classify_finding(finding)
            )

        # Potential CSRF issue

        if method == "POST":

            csrf_fields = [

                item for item in inputs

                if "csrf" in item.get(
                    "name",
                    ""
                ).lower()

                or "token" in item.get(
                    "name",
                    ""
                ).lower()

            ]

            if not csrf_fields:

                finding = create_finding(

                    vulnerability="Potential Missing CSRF Token",

                    severity="Medium",

                    description=(
                        "A POST form has no obvious "
                        "CSRF token field."
                    ),

                    url=form_url,

                    recommendation=(
                        "Verify that server-side CSRF "
                        "protection is implemented."
                    ),

                    category="CSRF / Request Forgery"

                )

                findings.append(
                    classify_finding(finding)
                )

    return findings


# =========================================================
# HTML CONTENT ANALYSIS
# =========================================================

def check_html_content(response, target_url):

    findings = []

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    page_text = response.text.lower()

    # Error information disclosure indicators

    debug_indicators = [

        "traceback",

        "stack trace",

        "debug mode",

        "exception occurred",

        "sql syntax",

        "database error"

    ]

    for indicator in debug_indicators:

        if indicator in page_text:

            finding = create_finding(

                vulnerability="Potential Error Information Disclosure",

                severity="Medium",

                description=(
                    "The webpage contains a potential "
                    f"error disclosure indicator: {indicator}"
                ),

                url=target_url,

                recommendation=(
                    "Ensure production applications do not "
                    "expose detailed internal errors."
                ),

                category="Information Disclosure"

            )

            findings.append(
                classify_finding(finding)
            )

            break

    # External JavaScript resources

    scripts = soup.find_all(
        "script",
        src=True
    )

    for script in scripts:

        script_url = urljoin(
            target_url,
            script.get("src")
        )

        parsed_script = urlparse(
            script_url
        )

        parsed_target = urlparse(
            target_url
        )

        if parsed_script.netloc != parsed_target.netloc:

            finding = create_finding(

                vulnerability="External JavaScript Resource",

                severity="Informational",

                description=(
                    "The page loads JavaScript from "
                    f"an external domain: {script_url}"
                ),

                url=target_url,

                recommendation=(
                    "Review external JavaScript dependencies "
                    "and ensure their integrity."
                ),

                category="Security Misconfiguration"

            )

            findings.append(
                classify_finding(finding)
            )

    return findings


# =========================================================
# DIRECTORY LISTING INDICATOR
# =========================================================

def check_directory_listing(response, target_url):

    findings = []

    content = response.text.lower()

    indicators = [

        "index of /",

        "directory listing",

        "parent directory"

    ]

    for indicator in indicators:

        if indicator in content:

            finding = create_finding(

                vulnerability="Potential Directory Listing",

                severity="Medium",

                description=(
                    "The webpage contains an indicator "
                    "commonly associated with directory listing."
                ),

                url=target_url,

                recommendation=(
                    "Disable directory listing on the "
                    "web server where it is not required."
                ),

                category="Security Misconfiguration"

            )

            findings.append(
                classify_finding(finding)
            )

            break

    return findings


# =========================================================
# HTTPS CHECK
# =========================================================

def check_https(target_url):

    findings = []

    parsed = urlparse(target_url)

    if parsed.scheme.lower() != "https":

        finding = create_finding(

            vulnerability="Website Uses HTTP",

            severity="High",

            description=(
                "The target website is accessed "
                "using HTTP instead of HTTPS."
            ),

            url=target_url,

            recommendation=(
                "Enable HTTPS with a valid TLS certificate "
                "and redirect HTTP to HTTPS."
            ),

            category="Cryptographic Issues"

        )

        findings.append(
            classify_finding(finding)
        )

    return findings


# =========================================================
# SSL CERTIFICATE CHECK
# =========================================================

def check_ssl_certificate(target_url):

    findings = []

    parsed = urlparse(target_url)

    if parsed.scheme != "https":

        return findings

    hostname = parsed.hostname

    port = parsed.port or 443

    if not hostname:

        return findings

    try:

        context = ssl.create_default_context()

        with socket.create_connection(

            (hostname, port),

            timeout=REQUEST_TIMEOUT

        ) as sock:

            with context.wrap_socket(

                sock,

                server_hostname=hostname

            ) as ssl_socket:

                certificate = ssl_socket.getpeercert()

                if not certificate:

                    finding = create_finding(

                        vulnerability="SSL Certificate Information Missing",

                        severity="Medium",

                        description=(
                            "The TLS connection did not "
                            "return certificate information."
                        ),

                        url=target_url,

                        recommendation=(
                            "Verify that the website has "
                            "a valid TLS certificate."
                        ),

                        category="Cryptographic Issues"

                    )

                    findings.append(
                        classify_finding(finding)
                    )

    except ssl.SSLError as error:

        finding = create_finding(

            vulnerability="SSL/TLS Certificate Problem",

            severity="High",

            description=str(error),

            url=target_url,

            recommendation=(
                "Configure a valid TLS certificate "
                "and secure TLS settings."
            ),

            category="Cryptographic Issues"

        )

        findings.append(
            classify_finding(finding)
        )

    except Exception:

        pass

    return findings


# =========================================================
# MAIN SCANNER
# =========================================================

def scan_website(target_url):

    findings = []

    scan_started = datetime.now()

    # Validate URL

    if not validate_url(target_url):

        return {

            "success": False,

            "error": (
                "Invalid URL. Enter a valid "
                "HTTP or HTTPS URL."
            ),

            "target": target_url

        }

    # HTTPS check

    findings.extend(
        check_https(target_url)
    )

    # SSL check

    findings.extend(
        check_ssl_certificate(target_url)
    )

    # Fetch website

    response = get_website(target_url)

    if isinstance(response, dict) and "error" in response:

        return {

            "success": False,

            "error": response["error"],

            "target": target_url,

            "findings": findings

        }

    # Security headers

    findings.extend(
        check_security_headers(
            response,
            target_url
        )
    )

    # Cookie analysis

    findings.extend(
        check_cookies(
            response,
            target_url
        )
    )

    # Information disclosure

    findings.extend(
        check_information_disclosure(
            response,
            target_url
        )
    )

    # HTML forms

    findings.extend(
        analyze_forms(
            response,
            target_url
        )
    )

    # HTML content

    findings.extend(
        check_html_content(
            response,
            target_url
        )
    )

    # Directory listing

    findings.extend(
        check_directory_listing(
            response,
            target_url
        )
    )

    scan_finished = datetime.now()

    # Severity counts

    severity_counts = {

        "Critical": 0,

        "High": 0,

        "Medium": 0,

        "Low": 0,

        "Informational": 0

    }

    for finding in findings:

        severity = finding.get(
            "severity"
        )

        if severity in severity_counts:

            severity_counts[severity] += 1

    # ML statistics

    ml_predictions = [

        finding.get("ml_model_prediction")

        for finding in findings

        if finding.get("ml_model_prediction")

    ]

    ml_agreements = [

        finding.get("ml_agrees")

        for finding in findings

        if finding.get("ml_agrees") is not None

    ]

    ml_agreement_count = sum(
        ml_agreements
    )

    ml_mismatch_count = (
        len(ml_agreements)
        - ml_agreement_count
    )

    return {

        "success": True,

        "target": target_url,

        "status_code": response.status_code,

        "scan_started": scan_started.isoformat(),

        "scan_finished": scan_finished.isoformat(),

        "duration_seconds": round(
            (
                scan_finished
                - scan_started
            ).total_seconds(),
            2
        ),

        "total_findings": len(findings),

        "severity_counts": severity_counts,

        "ml_enabled": ML_AVAILABLE,

        "ml_classifications": len(ml_predictions),

        "ml_agreements": ml_agreement_count,

        "ml_mismatches": ml_mismatch_count,

        "findings": findings

    }


# =========================================================
# TEST EXECUTION
# =========================================================

if __name__ == "__main__":

    print("=" * 60)

    print(
        "WEB APPLICATION SECURITY AI TOOLKIT"
    )

    print("=" * 60)

    target = input(
        "Enter authorized target URL: "
    ).strip()

    result = scan_website(target)

    print("\nSCAN RESULTS")

    print("=" * 60)

    if result.get("success"):

        print(
            "Target:",
            result["target"]
        )

        print(
            "Status Code:",
            result["status_code"]
        )

        print(
            "Total Findings:",
            result["total_findings"]
        )

        print("\nSeverity Summary:")

        for severity, count in result[
            "severity_counts"
        ].items():

            print(
                f"{severity}: {count}"
            )

        print("\nML Summary:")

        print(
            "ML Classifications:",
            result["ml_classifications"]
        )

        print(
            "ML Agreements:",
            result["ml_agreements"]
        )

        print(
            "ML Mismatches:",
            result["ml_mismatches"]
        )

        print("\nFindings:")

        for index, finding in enumerate(

            result["findings"],

            start=1

        ):

            print(
                f"\n{index}. "
                f"{finding['vulnerability']}"
            )

            print(
                "Severity:",
                finding["severity"]
            )

            print(
                "Rule Category:",
                finding.get("category")
            )

            print(
                "ML Prediction:",
                finding.get("ml_prediction")
            )

            print(
                "ML Agreement:",
                finding.get("ml_agrees")
            )

            print(
                "Description:",
                finding["description"]
            )

            print(
                "Recommendation:",
                finding["recommendation"]
            )

            if finding.get("ml_model_prediction"):

                print(
                    "ML Algorithm:",
                    finding.get("ml_algorithm")
                )

                print(
                    "ML Version:",
                    finding.get("ml_model_version")
                )

    else:

        print("Scan failed:")

        print(
            result.get("error")
        )

    print("\nScan completed.")