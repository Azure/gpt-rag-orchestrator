<#
.SYNOPSIS
    deploy.ps1 — validate APP_CONFIG_ENDPOINT, load App Config (label=gpt-rag), then build & push

.DESCRIPTION
    - Checks for APP_CONFIG_ENDPOINT in environment; if missing, tries to fetch from `azd env get-values`.
    - Parses App Configuration name from endpoint.
    - Checks Azure CLI login.
    - Fetches required keys (CONTAINER_REGISTRY_NAME, CONTAINER_REGISTRY_LOGIN_SERVER, AZURE_RESOURCE_GROUP, ORCHESTRATOR_APP_NAME) from Azure App Configuration with label "gpt-rag".
      If a key is not found with original casing, tries uppercase.
    - Logs into ACR, builds Docker image (tag from git short HEAD unless $env:tag is set). If local Docker is unavailable, uses `az acr build`.
    - Pushes image and updates the Container App.
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

function Invoke-ExternalCommand {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter()][string[]]$Arguments = @(),
        [Parameter(Mandatory = $true)][string]$What
    )

    Write-Verbose ("Running: {0} {1}" -f $FilePath, ($Arguments -join ' '))
    $output = & $FilePath @Arguments 2>&1
    $exitCode = $LASTEXITCODE

    if ($exitCode -ne 0) {
        Write-ErrorColored ("❌ Failed: {0} (exit {1})" -f $What, $exitCode)
        if ($output) {
            Write-Host ($output | Out-String)
        }
        exit 1
    }

    return $output
}
#endregion

#region Debug toggle
if ($env:DEBUG -eq 'true') {
    $VerbosePreference = 'Continue'
    Write-Verbose "DEBUG mode is ON"
} else {
    $VerbosePreference = 'SilentlyContinue'
}
#endregion

Write-Host ""  # blank line

#region Early Docker validation
$pausedPattern   = 'Docker Desktop is manually paused'
$daemonDownRegex = '((?i)error during connect|Cannot connect to the Docker daemon|Is the docker daemon running|The Docker daemon is not running|dockerDesktopLinuxEngine|dockerDesktopWindowsEngine|The system cannot find the file specified|open \\./pipe/|context deadline exceeded)'

# Optional: try service check, but do NOT fail based on it
try {
    $dockerSvc = Get-Service -Name 'com.docker.service' -ErrorAction SilentlyContinue
    if ($dockerSvc) {
        Write-Blue "🔍 Docker Desktop service status: $($dockerSvc.Status)"
    }
} catch { }

if (Get-Command docker -ErrorAction SilentlyContinue) {
    Write-Blue "🔍 Checking Docker availability…"
    $probeOutput = & docker info 2>&1
    $probeExit   = $LASTEXITCODE
    $probeText   = ($probeOutput | Out-String)

    if ($probeText -match $pausedPattern -or $probeText -match $daemonDownRegex -or $probeExit -ne 0) {
        if ($probeText -match $pausedPattern) {
            Write-ErrorColored '❌ Docker Desktop is manually paused. Unpause it via the Whale menu or Dashboard.'
        } else {
            Write-ErrorColored '❌ Docker Desktop is not running.'
        }
        Write-Yellow '⚠️  Please start/unpause Docker Desktop and re-run this script.'
        exit 1
    }
} else {
    Write-ErrorColored '❌ Docker CLI not found on this system.'
    Write-Yellow '⚠️  Please install Docker Desktop and re-run this script.'
    exit 1
}
Write-Green "✅ Docker is available."
Write-Host ""
#endregion

#region APP_CONFIG_ENDPOINT check
if ($null -ne $env:APP_CONFIG_ENDPOINT -and $env:APP_CONFIG_ENDPOINT.Trim() -ne '') {
    Write-Green "✅ Using APP_CONFIG_ENDPOINT from environment: $($env:APP_CONFIG_ENDPOINT)"
    $APP_CONFIG_ENDPOINT = $env:APP_CONFIG_ENDPOINT.Trim()
} else {
    Write-Blue "🔍 Fetching APP_CONFIG_ENDPOINT from azd env…"
    try {
        $envValues = azd env get-values 2>$null
    } catch {
        $envValues = $null
    }
    if ($envValues) {
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
    Write-Host "  • Set it with: azd env set APP_CONFIG_ENDPOINT <your-endpoint>"
    Write-Host "  • Or in PowerShell: `$env:APP_CONFIG_ENDPOINT = '<your-endpoint>'` before running."
    exit 1
}
Write-Green "✅ APP_CONFIG_ENDPOINT: $APP_CONFIG_ENDPOINT"
Write-Host ""
#endregion

#region Parse configName from endpoint
$configName = $APP_CONFIG_ENDPOINT -replace '^https?://', ''
$configName = $configName -replace '\.azconfig\.io/?$', ''
if (-not $configName) {
    Write-Yellow ("⚠️ Could not parse config name from endpoint '{0}'." -f $APP_CONFIG_ENDPOINT)
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
    Write-Blue ("🛠️  Retrieving '{0}' (label={1}) from App Configuration…" -f $Key, $label)
    try {
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
        Write-Yellow ("⚠️  Key '{0}' not found or empty. CLI output: {1}" -f $Key, $val)
        return $null
    }
    return $val.Trim()
}

# Define required keys
$keyNames = @('CONTAINER_REGISTRY_NAME', 'CONTAINER_REGISTRY_LOGIN_SERVER', 'SUBSCRIPTION_ID', 'AZURE_RESOURCE_GROUP', 'RESOURCE_TOKEN', 'ORCHESTRATOR_APP_NAME')
$values = @{}
$missing = @()

foreach ($k in $keyNames) {
    $v = Get-ConfigValue -Key $k
    if ($null -eq $v) {
        # try uppercase fallback
        $upperKey = $k.ToUpper()
        if ($upperKey -ne $k) {
            Write-Blue ("🔍 Trying uppercase key '{0}'…" -f $upperKey)
            $v = Get-ConfigValue -Key $upperKey
        }
    }
    if ($null -eq $v) {
        $missing += $k
    } else {
        $values[$k] = $v
    }
}
if ($missing.Count -gt 0) {
    Write-Yellow ("⚠️  Missing or invalid App Config keys: {0}" -f ($missing -join ', '))
    exit 1
}

Write-Green "✅ All App Configuration values retrieved:"
Write-Host ("   CONTAINER_REGISTRY_NAME = {0}" -f $values.CONTAINER_REGISTRY_NAME)
Write-Host ("   CONTAINER_REGISTRY_LOGIN_SERVER = {0}" -f $values.CONTAINER_REGISTRY_LOGIN_SERVER)
Write-Host ("   AZURE_RESOURCE_GROUP = {0}" -f $values.AZURE_RESOURCE_GROUP)
Write-Host ("   ORCHESTRATOR_APP_NAME = {0}" -f $values.ORCHESTRATOR_APP_NAME)
Write-Host ""
#endregion

#region Login to ACR
Write-Green ("🔐 Logging into ACR ({0} in {1})…" -f $values.CONTAINER_REGISTRY_NAME, $values.AZURE_RESOURCE_GROUP)
Invoke-ExternalCommand -FilePath 'az' -Arguments @('acr','login','--name',$values.CONTAINER_REGISTRY_NAME,'--resource-group',$values.AZURE_RESOURCE_GROUP) -What 'ACR login'
Write-Green "✅ Logged into ACR."
Write-Host ""
#endregion

#region Determine tag
Write-Blue "Defining tag..."
if ($env:tag) {
    $tag = $env:tag.Trim()
    Write-Verbose ("Using tag from environment: {0}" -f $tag)
} else {
    try {
        $gitTag = & git rev-parse --short HEAD 2>$null
        if ($LASTEXITCODE -eq 0 -and $gitTag) {
            $tag = $gitTag.Trim()
            Write-Verbose ("Using Git short HEAD as tag: {0}" -f $tag)

            # If working tree is dirty, append a timestamp so local changes actually deploy.
            try {
                $dirty = & git status --porcelain 2>$null
                if ($LASTEXITCODE -eq 0 -and $dirty) {
                    $suffix = Get-Date -Format 'yyyyMMddHHmmss'
                    $tag = "$tag-dirty-$suffix"
                    Write-Verbose ("Git working tree is dirty; using tag: {0}" -f $tag)
                }
            } catch { }
        } else {
            Write-Yellow "Could not get Git short HEAD. Generating random tag."
            $randomNumber = Get-Random -Minimum 100000 -Maximum 999999
            $tag = "GPT$randomNumber"
            Write-Verbose ("Generated random tag: {0}" -f $tag)
        }
    } catch {
        $errMsg = $_.Exception.Message
        Write-Yellow ("Error running Git: {0}. Generating random tag." -f $errMsg)
        $randomNumber = Get-Random -Minimum 100000 -Maximum 999999
        $tag = "GPT$randomNumber"
        Write-Verbose ("Generated random tag: {0}" -f $tag)
    }
}
#endregion

#region Build or ACR build image
$fullImageName = "$($values.CONTAINER_REGISTRY_LOGIN_SERVER)/azure-gpt-rag/orchestrator:$tag"
Write-Green "🛠️  Building Docker image…"
if (Get-Command docker -ErrorAction SilentlyContinue) {
    $buildOut = & docker build -t $fullImageName . 2>&1
    $buildExit = $LASTEXITCODE
    if ($buildExit -ne 0) {
        Write-ErrorColored ("❌ Docker build failed (exit {0})." -f $buildExit)
        if ($buildOut) { Write-Host ($buildOut | Out-String) }
        exit 1
    }
    Write-Green "✅ Docker build succeeded."
} else {
    Write-Blue "⚠️  Docker CLI not found locally. Falling back to 'az acr build'."
    Invoke-ExternalCommand -FilePath 'az' -Arguments @('acr','build','--registry',$values.CONTAINER_REGISTRY_NAME,'--image',"azure-gpt-rag/orchestrator:$tag",'--file','Dockerfile','.') -What 'ACR cloud build'
    Write-Green "✅ ACR cloud build succeeded."
}
Write-Host ""
#endregion

#Make sure container registry is registered
Write-Green "🔄 Updating container app registry…"
$idsJson = Invoke-ExternalCommand -FilePath 'az' -Arguments @('containerapp','identity','show','--name',$values.ORCHESTRATOR_APP_NAME,'--resource-group',$values.AZURE_RESOURCE_GROUP,'--output','json') -What 'fetch container app identity'
$ids = ($idsJson | Out-String) | ConvertFrom-Json

if ($ids.type.tostring().contains("UserAssigned")) {
    Invoke-ExternalCommand -FilePath 'az' -Arguments @(
        'containerapp','registry','set',
        '--name',$values.ORCHESTRATOR_APP_NAME,
        '--resource-group',$values.AZURE_RESOURCE_GROUP,
        '--server',"$($values.CONTAINER_REGISTRY_NAME).azurecr.io",
        '--identity',"/subscriptions/$($values.SUBSCRIPTION_ID)/resourceGroups/$($values.AZURE_RESOURCE_GROUP)/providers/Microsoft.ManagedIdentity/userAssignedIdentities/uai-ca-$($values.RESOURCE_TOKEN)-orchestrator"
    ) -What 'set container app registry (user-assigned identity)'
} else {
    Invoke-ExternalCommand -FilePath 'az' -Arguments @(
        'containerapp','registry','set',
        '--name',$values.ORCHESTRATOR_APP_NAME,
        '--resource-group',$values.AZURE_RESOURCE_GROUP,
        '--server',"$($values.CONTAINER_REGISTRY_NAME).azurecr.io",
        '--identity','system'
    ) -What 'set container app registry (system identity)'
}

Write-Green "✅ Container app registry updated."

#region Push Docker image (if local build used)
if (Get-Command docker -ErrorAction SilentlyContinue) {
    Write-Green "📤 Pushing image…"
    $pushOut = & docker push $fullImageName 2>&1
    $pushExit = $LASTEXITCODE
    if ($pushExit -ne 0) {
        Write-ErrorColored ("❌ Docker push failed (exit {0})." -f $pushExit)
        if ($pushOut) { Write-Host ($pushOut | Out-String) }
        exit 1
    }
    Write-Green "✅ Image pushed."
    Write-Host ""
} else {
    # If using az acr build, image is already in ACR
    Write-Green "ℹ️  Image built in ACR; no local push needed."
    Write-Host ""
}
#endregion

#region Update Container App
Write-Green "🔄 Updating container app…"
Invoke-ExternalCommand -FilePath 'az' -Arguments @('containerapp','update','--name',$values.ORCHESTRATOR_APP_NAME,'--resource-group',$values.AZURE_RESOURCE_GROUP,'--image',$fullImageName) -What 'update container app image'
Write-Green "✅ Container app updated."

#get the current revision
Write-Blue "🔍 Fetching current revision…"
$currentRevisionOut = Invoke-ExternalCommand -FilePath 'az' -Arguments @('containerapp','revision','list','--name',$values.ORCHESTRATOR_APP_NAME,'--resource-group',$values.AZURE_RESOURCE_GROUP,'--query','[0].name','-o','tsv') -What 'fetch current revision'
$currentRevision = ($currentRevisionOut | Out-String).Trim()
if ([string]::IsNullOrWhiteSpace($currentRevision)) {
    Write-ErrorColored '❌ Could not determine current revision; stopping.'
    exit 1
}


#region Restart Container App
Write-Green "🔄 Restarting container app…"
Invoke-ExternalCommand -FilePath 'az' -Arguments @('containerapp','revision','restart','--name',$values.ORCHESTRATOR_APP_NAME,'--resource-group',$values.AZURE_RESOURCE_GROUP,'--revision',$currentRevision) -What 'restart container app revision'
Write-Green "✅ Container app restarted."
#endregion
