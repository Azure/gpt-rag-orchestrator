<#
.SYNOPSIS
    deploy.ps1 — validate APP_CONFIG_ENDPOINT, load App Config (label=gpt-rag), then build & push

.DESCRIPTION
    - Checks for APP_CONFIG_ENDPOINT in environment; if missing, tries to fetch from `azd env get-values`.
    - Parses App Configuration name from endpoint.
    - Checks Azure CLI login.
    - Fetches required keys (containerRegistryName, containerRegistryLoginServer, resourceGroupName, orchestratorApp) from Azure App Configuration with label "gpt-rag".
    - Logs into ACR, builds Docker image (tag from git short HEAD unless $env:tag is set), pushes image, and updates the Container App.

.NOTES
    - Requires Azure CLI installed and logged in.
    - Running in PowerShell 5.1+ or PowerShell Core.
#>

#region Helper: color output functions
function Write-Green($msg) {
    Write-Host $msg -ForegroundColor Green
}
function Write-Blue($msg) {
    Write-Host $msg -ForegroundColor Cyan
}
function Write-Yellow($msg) {
    Write-Host $msg -ForegroundColor Yellow
}
function Write-ErrorColored($msg) {
    Write-Host $msg -ForegroundColor Red
}
#endregion

#region Debug toggle
# If environment variable DEBUG is "true", enable verbose output
if ($env:DEBUG -eq 'true') {
    $VerbosePreference = 'Continue'
    Write-Verbose "DEBUG mode is ON"
} else {
    $VerbosePreference = 'SilentlyContinue'
}
#endregion

Write-Host ""  # blank line

#region APP_CONFIG_ENDPOINT check
if ($null -ne $env:APP_CONFIG_ENDPOINT -and $env:APP_CONFIG_ENDPOINT.Trim() -ne '') {
    Write-Green "✅ Using APP_CONFIG_ENDPOINT from environment: $($env:APP_CONFIG_ENDPOINT)"
    $APP_CONFIG_ENDPOINT = $env:APP_CONFIG_ENDPOINT.Trim()
} else {
    Write-Blue "🔍 Fetching APP_CONFIG_ENDPOINT from azd env…"
    # Try to get azd env values; ignore errors
    try {
        $envValues = azd env get-values 2>$null
    } catch {
        $envValues = $null
    }
    if ($envValues) {
        # Parse lines like KEY="value" or KEY=value
        foreach ($line in $envValues -split "`n") {
            if ($line -match '^\s*APP_CONFIG_ENDPOINT\s*=\s*"?([^"]+)"?\s*$') {
                $APP_CONFIG_ENDPOINT = $Matches[1].Trim()
                break
            }
        }
    }
}
if (-not $APP_CONFIG_ENDPOINT) {
    Write-Yellow "⚠️  Missing APP_CONFIG_ENDPOINT."
    Write-Host "  • `Set it with: azd env set APP_CONFIG_ENDPOINT <your-endpoint>`"
    Write-Host "  • Or export in shell: `$env:APP_CONFIG_ENDPOINT = '<your-endpoint>'` before running this script."
    exit 1
}
Write-Green "✅ APP_CONFIG_ENDPOINT: $APP_CONFIG_ENDPOINT"
Write-Host ""
#endregion

#region Parse configName from endpoint
# Remove leading https:// and trailing .azconfig.io
# Use regex replace
$configName = $APP_CONFIG_ENDPOINT -replace '^https?://', ''
$configName = $configName -replace '\.azconfig\.io/?$', ''
if (-not $configName) {
    Write-Yellow "⚠️  Could not parse config name from endpoint '$APP_CONFIG_ENDPOINT'."
    exit 1
}
Write-Green "✅ App Configuration name: $configName"
Write-Host ""
#endregion

#region Azure CLI login check
Write-Blue "🔐 Checking Azure CLI login and subscription…"
try {
    az account show > $null 2>&1
} catch {
    Write-Yellow "⚠️  Not logged in. Please run 'az login'."
    exit 1
}
Write-Green "✅ Azure CLI is logged in."
Write-Host ""
#endregion

#region Fetch App Configuration values
$label = "gpt-rag"
Write-Green "⚙️ Loading App Configuration settings (label=$label)…"
Write-Host ""

function Get-ConfigValue {
    param(
        [Parameter(Mandatory=$true)][string]$Key
    )
    Write-Blue "🛠️  Retrieving '$Key' (label=$label) from App Configuration…"
    try {
        # az appconfig kv show returns JSON; with -o tsv --query value, it outputs plain value or error.
        $val = az appconfig kv show `
            --name $configName `
            --key $Key `
            --label $label `
            --auth-mode login `
            --query value -o tsv 2>&1
        $exitCode = $LASTEXITCODE
    } catch {
        $val = $_.Exception.Message
        $exitCode = 1
    }
    if ($exitCode -ne 0 -or [string]::IsNullOrWhiteSpace($val)) {
        Write-Yellow "⚠️  Failed to retrieve key '$Key'. CLI output: $val"
        return $null
    }
    return $val.Trim()
}

# keys to fetch
$keys = @{
    containerRegistryName = $null
    containerRegistryLoginServer = $null
    resourceGroupName = $null
    orchestratorApp = $null
}
$missing = @()
foreach ($k in $keys.Keys) {
    $v = Get-ConfigValue -Key $k
    if ($null -eq $v) {
        $missing += $k
    } else {
        $keys[$k] = $v
    }
}
if ($missing.Count -gt 0) {
    Write-Yellow "⚠️  Missing or invalid App Config keys: $($missing -join ', ')"
    exit 1
}

Write-Green "✅ All App Configuration values retrieved:"
Write-Host "   containerRegistryName = $($keys.containerRegistryName)"
Write-Host "   containerRegistryLoginServer = $($keys.containerRegistryLoginServer)"
Write-Host "   resourceGroupName = $($keys.resourceGroupName)"
Write-Host "   orchestratorApp = $($keys.orchestratorApp)"
Write-Host ""
#endregion

#region Login to ACR
Write-Green "🔐 Logging into ACR ($($keys.containerRegistryName))…"
try {
    az acr login --name $keys.containerRegistryName
    Write-Green "✅ Logged into ACR."
} catch {
    Write-Yellow "⚠️  Failed to login to ACR: $_"
    exit 1
}
Write-Host ""
#endregion

#region Determine tag
Write-Blue "🛢️ Defining tag…"
if ($env:tag) {
    $tag = $env:tag.Trim()
    Write-Verbose "Using tag from environment: $tag"
} else {
    try {
        $gitTag = & git rev-parse --short HEAD 2>$null
        if ($LASTEXITCODE -eq 0 -and $gitTag) {
            $tag = $gitTag.Trim()
        } else {
            Write-Yellow "⚠️  Could not get git short HEAD. Please set environment variable `tag`."
            exit 1
        }
    } catch {
        Write-Yellow "⚠️  Error running git: $_"
        exit 1
    }
}
Write-Green "✅ tag set to: $tag"
Write-Host ""
#endregion

#region Build Docker image
Write-Green "🛠️  Building Docker image…"
$fullImageName = "$($keys.containerRegistryLoginServer)/azure-gpt-rag/orchestrator-build:$tag"
try {
    docker build -t $fullImageName .
    Write-Green "✅ Docker build succeeded."
} catch {
    Write-Yellow "⚠️  Docker build failed: $_"
    exit 1
}
Write-Host ""
#endregion

#region Push Docker image
Write-Host ""
Write-Green "📤 Pushing image…"
try {
    docker push $fullImageName
    Write-Green "✅ Image pushed."
} catch {
    Write-Yellow "⚠️  Docker push failed: $_"
    exit 1
}
Write-Host ""
#endregion

#region Update Container App
Write-Green "🔄 Updating container app…"
try {
    az containerapp update `
        --name $keys.orchestratorApp `
        --resource-group $keys.resourceGroupName `
        --image $fullImageName
    Write-Green "✅ Container app updated."
} catch {
    Write-Yellow "⚠️  Failed to update container app: $_"
    exit 1
}
#endregion
