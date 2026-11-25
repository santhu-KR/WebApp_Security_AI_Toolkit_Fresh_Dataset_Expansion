# Configure Gemini for the Flask backend.
# This script stores the key only in ml_backend/.env (ignored by Git).
$ErrorActionPreference = "Stop"
$backendDir = Join-Path (Get-Location) "ml_backend"
if (!(Test-Path $backendDir)) { throw "Run this script from the repository root; ml_backend folder not found." }
$envPath = Join-Path $backendDir ".env"

$keySecure = Read-Host "Paste your Gemini API key" -AsSecureString
$ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($keySecure)
try { $key = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr) }
finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr) }

if ([string]::IsNullOrWhiteSpace($key)) { throw "No API key entered." }
$key = $key.Trim()
if ($key -match "[\r\n]") { throw "API key must be a single line." }

$content = @"
GEMINI_API_KEY=$key
GEMINI_MODEL=gemini-2.5-pro
GEMINI_CHAT_MODEL=gemini-2.5-pro
APP_TOKEN_SECRET=replace_with_a_long_random_secret
"@
Set-Content -Path $envPath -Value $content -Encoding ascii
Write-Host "Created ml_backend/.env successfully. API key is not printed." -ForegroundColor Green
Write-Host "Now restart Flask from ml_backend: python app.py" -ForegroundColor Yellow
