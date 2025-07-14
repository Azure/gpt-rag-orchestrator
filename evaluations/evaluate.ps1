param(
    [switch]$SkipEval
)

# 1) Validate
if (-not $Env:APP_CONFIG_ENDPOINT) {
    Write-Error "❌ APP_CONFIG_ENDPOINT environment variable is required"
    exit 1
}

# 2) Create & activate venv
python -m venv evaluations\.venv
# Activate the venv
& "evaluations\.venv\Scripts\Activate.ps1"

# 3) Install dependencies
pip install --upgrade pip
pip install -r evaluations/requirements.txt

# 4) Ensure Python can see your src/ package
# In PowerShell, PYTHONPATH entries are separated by semicolons
$pwdPath = (Get-Location).Path
$Env:PYTHONPATH = "$pwdPath;$pwdPath\src"

# 5) Generate eval-input
Write-Host "▶ Generating eval input…"
python evaluations/generate_eval_input.py

# 6) Conditionally run evaluation
if (-not $SkipEval) {
    Write-Host "▶ Running evaluation…"
    python evaluations/evaluate.py
} else {
    Write-Host "▶ Skipping evaluation as requested (-SkipEval)."
}

# 7) Teardown
# Deactivate the venv
# The Activate.ps1 script defines a function 'Deactivate' or 'deactivate'
# In many venv setups, 'deactivate' is available
if (Get-Command deactivate -ErrorAction SilentlyContinue) {
    deactivate
} elseif (Get-Command Deactivate -ErrorAction SilentlyContinue) {
    Deactivate
}

# Remove the virtual environment folder
Remove-Item -Recurse -Force evaluations\.venv

Write-Host "✅ All done."
