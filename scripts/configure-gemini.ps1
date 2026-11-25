# Configure Gemini API key for local Vite development.
# This script writes the key only to the ignored root .env file.
$ErrorActionPreference = "Stop"

$keySecure = Read-Host "Paste your Gemini API key" -AsSecureString
$ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($keySecure)
try {
    $key = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr)
} finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr)
}

if ([string]::IsNullOrWhiteSpace($key)) {
    Write-Error "No API key entered. Nothing was changed."
    exit 1
}

$envContent = @"
# Local demo only. Vite embeds this key into browser assets.
# Do not deploy this configuration publicly.
GEMINI_API_KEY=$key
"@

Set-Content -Path ".env" -Value $envContent -Encoding UTF8
Write-Host "Created .env in project root. It is ignored by Git." -ForegroundColor Green
Write-Host "Restart Vite: stop it with Ctrl+C, then run npm run dev." -ForegroundColor Yellow
