# Configure Gemini for the Flask backend.
# Run this script from the repository root.
# The API key is saved only to ml_backend/.env (ignored by Git).
$ErrorActionPreference = "Stop"

$backendDir = Join-Path (Get-Location) "ml_backend"
if (!(Test-Path $backendDir)) {
    throw "Run this script from the repository root; ml_backend folder not found."
}

$envPath = Join-Path $backendDir ".env"
$keySecure = Read-Host "Paste your Gemini API key" -AsSecureString
$ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($keySecure)
try {
    $key = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr)
}
finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr)
}

if ([string]::IsNullOrWhiteSpace($key)) { throw "No API key entered." }
$key = $key.Trim()
if ($key -match "[\r\n]") { throw "API key must be a single line." }

# Create a random local app secret.
$bytes = New-Object byte[] 32
$rng = [Security.Cryptography.RandomNumberGenerator]::Create()
$rng.GetBytes($bytes)
$rng.Dispose()
$appSecret = [Convert]::ToBase64String($bytes)

$content = @"
GEMINI_API_KEY=$key
GEMINI_MODEL=gemini-3.8-flash
GEMINI_CHAT_MODEL=gemini-3.8-flash
APP_TOKEN_SECRET=$appSecret
"@

Set-Content -Path $envPath -Value $content -Encoding ascii
Write-Host "Created ml_backend/.env successfully. API key is not printed." -ForegroundColor Green
Write-Host "Model configured: gemini-3.8-flash" -ForegroundColor Cyan
Write-Host "Restart Flask from ml_backend: python app.py" -ForegroundColor Yellow
