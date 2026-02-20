# PowerShell menu script to create ONE test document at a time (userIds, groupIds, rbacScope, or public)
# It also reports the current permissionFilterOption status (no index mutation).

param(
  [ValidateSet('1','2','3','4','5','6','7','q','Q')]
  [string]$Selection,

  [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# This script intentionally avoids hardcoded tenant-specific values.
# It prompts for context and caches the last used selections in a temp state file.

$StateFile = Join-Path $env:TEMP 'gpt-rag-rbac-last.ps1'

$script:SUB = $null
$script:RG = $null
$script:Search = $null
$script:Index = $null
$script:Tenant = $null
$script:UserOid = $null
$script:GroupOid = $null
$script:RbacScope = $null

# Preview API used by this test script (permissionFilterOption).
$Api = "2025-11-01-preview"

# Vector fields (contentVector/captionVector) in this index are 3072 dimensions.
$VectorDimensions = 3072

# Query behavior:
# - When permissionFilterOption=enabled, queries can return 0 results unless you provide
#   an Entra ID token (Authorization header) OR enable elevated read for debugging.
$UseElevatedRead = $false
$IncludeQueryAuthToken = $true

function _Read-YesNo {
  param([string]$Prompt, [bool]$DefaultYes = $true)
  $suffix = if ($DefaultYes) { "[Y/n]" } else { "[y/N]" }
  $ans = Read-Host "${Prompt} ${suffix}"
  if ([string]::IsNullOrWhiteSpace($ans)) { return $DefaultYes }
  return ($ans -match '^(y|yes)$')
}

function _Try-LoadState {
  if (Test-Path -Path $StateFile) {
    try {
      . $StateFile
      if ($SUB) { $script:SUB = $SUB }
      if ($RG) { $script:RG = $RG }
      if ($SEARCH) { $script:Search = $SEARCH }
      if ($INDEX) { $script:Index = $INDEX }
      if ($TENANT) { $script:Tenant = $TENANT }
      if ($USER_OID) { $script:UserOid = $USER_OID }
      if ($GROUP_OID) { $script:GroupOid = $GROUP_OID }
      if ($RBAC_SCOPE) { $script:RbacScope = $RBAC_SCOPE }
    } catch {
      # Ignore state load failures.
    }
  }
}

function _Save-State {
  $safeSub = $(if ($null -ne $script:SUB) { $script:SUB } else { '' }) -replace "'","''"
  $safeRg = $(if ($null -ne $script:RG) { $script:RG } else { '' }) -replace "'","''"
  $safeSearch = $(if ($null -ne $script:Search) { $script:Search } else { '' }) -replace "'","''"
  $safeIndex = $(if ($null -ne $script:Index) { $script:Index } else { '' }) -replace "'","''"
  $safeTenant = $(if ($null -ne $script:Tenant) { $script:Tenant } else { '' }) -replace "'","''"
  $safeUserOid = $(if ($null -ne $script:UserOid) { $script:UserOid } else { '' }) -replace "'","''"
  $safeGroupOid = $(if ($null -ne $script:GroupOid) { $script:GroupOid } else { '' }) -replace "'","''"
  $safeRbacScope = $(if ($null -ne $script:RbacScope) { $script:RbacScope } else { '' }) -replace "'","''"

  Set-Content -Path $StateFile -Encoding UTF8 -Value @(
    "`$SUB = '$safeSub'",
    "`$RG = '$safeRg'",
    "`$SEARCH = '$safeSearch'",
    "`$INDEX = '$safeIndex'",
    "`$TENANT = '$safeTenant'",
    "`$USER_OID = '$safeUserOid'",
    "`$GROUP_OID = '$safeGroupOid'",
    "`$RBAC_SCOPE = '$safeRbacScope'"
  )
}

function _Select-FromList {
  param(
    [Parameter(Mandatory=$true)][string]$Title,
    [Parameter(Mandatory=$true)][object[]]$Items,
    [Parameter(Mandatory=$true)][scriptblock]$Render
  )

  if (-not $Items -or $Items.Count -eq 0) {
    throw "${Title}: no items available for selection."
  }

  Write-Host "" 
  Write-Host $Title -ForegroundColor Cyan
  for ($i = 0; $i -lt $Items.Count; $i++) {
    $line = & $Render $Items[$i]
    Write-Host " [$i] $line"
  }

  while ($true) {
    $raw = Read-Host "Select index"
    try {
      $idx = [int]$raw
      if ($idx -ge 0 -and $idx -lt $Items.Count) { return $Items[$idx] }
    } catch { }
    Write-Host "Invalid selection. Enter a number between 0 and $($Items.Count - 1)." -ForegroundColor Red
  }
}

function _Decode-JwtPayload {
  param([Parameter(Mandatory=$true)][string]$Jwt)
  $parts = $Jwt -split '\.'
  if ($parts.Count -lt 2) { return $null }
  $payload = $parts[1]
  $payload = $payload.Replace('-', '+').Replace('_', '/')
  switch ($payload.Length % 4) {
    2 { $payload += '==' }
    3 { $payload += '=' }
    0 { }
    default { }
  }
  try {
    $bytes = [Convert]::FromBase64String($payload)
    $json = [System.Text.Encoding]::UTF8.GetString($bytes)
    return ($json | ConvertFrom-Json)
  } catch {
    return $null
  }
}

function Resolve-CurrentUserOid {
  # Prefer Azure CLI AD helper when available.
  try {
    $oid = (az ad signed-in-user show --query id -o tsv 2>$null).Trim()
    if ($oid) { return $oid }
  } catch { }

  # Fallback: decode an ARM access token and read the oid claim.
  try {
    $jwt = (az account get-access-token --resource https://management.azure.com --query accessToken -o tsv 2>$null).Trim()
    if (-not $jwt) { return $null }
    $payload = _Decode-JwtPayload -Jwt $jwt
    return ($payload.oid)
  } catch {
    return $null
  }
}

function Prompt-Context {
  _Try-LoadState

  if ($script:SUB -or $script:RG -or $script:Search -or $script:Index) {
    Write-Host "" 
    Write-Host "Last used context (from cache):" -ForegroundColor DarkCyan
    if ($script:SUB) { Write-Host "  Subscription: $script:SUB" }
    if ($script:RG) { Write-Host "  Resource group: $script:RG" }
    if ($script:Search) { Write-Host "  Search service: $script:Search" }
    if ($script:Index) { Write-Host "  Index: $script:Index" }
  }

  # Subscription
  $reuseSub = $false
  if ($script:SUB) { $reuseSub = _Read-YesNo -Prompt "Reuse last subscription '$script:SUB'?" -DefaultYes $true }
  if (-not $reuseSub -or -not $script:SUB) {
    $raw = az account list --query "[].{id:id,name:name}" -o tsv 2>$null
    if (-not $raw) { throw "No subscriptions found. Run 'az login' and try again." }
    $subs = @()
    foreach ($line in ($raw -split "`n")) {
      if ([string]::IsNullOrWhiteSpace($line)) { continue }
      $p = $line -split "`t"
      if ($p.Count -ge 2) { $subs += [pscustomobject]@{ Id=$p[0]; Name=$p[1] } }
    }
    $chosen = _Select-FromList -Title "Available subscriptions" -Items $subs -Render { param($s) "$($s.Name) ($($s.Id))" }
    $script:SUB = $chosen.Id
  }
  az account set --subscription $script:SUB | Out-Null

  # Tenant
  try { $script:Tenant = (az account show --query tenantId -o tsv 2>$null).Trim() } catch { $script:Tenant = $null }

  # Resource group
  $reuseRg = $false
  if ($script:RG) { $reuseRg = _Read-YesNo -Prompt "Reuse last resource group '$script:RG'?" -DefaultYes $true }
  if (-not $reuseRg -or -not $script:RG) {
    $raw = az group list --query "[].{name:name,location:location}" -o tsv 2>$null
    if (-not $raw) { throw "No resource groups found in subscription. Check your permissions." }
    $rgs = @()
    foreach ($line in ($raw -split "`n")) {
      if ([string]::IsNullOrWhiteSpace($line)) { continue }
      $p = $line -split "`t"
      if ($p.Count -ge 2) { $rgs += [pscustomobject]@{ Name=$p[0]; Location=$p[1] } }
    }
    $chosen = _Select-FromList -Title "Available resource groups" -Items $rgs -Render { param($g) "$($g.Name) (location=$($g.Location))" }
    $script:RG = $chosen.Name
  }

  # Search service
  $reuseSearch = $false
  if ($script:Search) { $reuseSearch = _Read-YesNo -Prompt "Reuse last Search service '$script:Search'?" -DefaultYes $true }
  if (-not $reuseSearch -or -not $script:Search) {
    $raw = az search service list -g $script:RG --query "[].{name:name}" -o tsv 2>$null
    if (-not $raw) { throw "No Azure AI Search services found in resource group '$script:RG'." }
    $svcs = @()
    foreach ($line in ($raw -split "`n")) {
      if ([string]::IsNullOrWhiteSpace($line)) { continue }
      $name = ($line -split "`t")[0]
      if ($name) { $svcs += [pscustomobject]@{ Name=$name } }
    }
    $chosen = _Select-FromList -Title "Available Azure AI Search services" -Items $svcs -Render { param($s) $s.Name }
    $script:Search = $chosen.Name
  }

  # Resolve current user oid
  $script:UserOid = Resolve-CurrentUserOid
  if (-not $script:UserOid) {
    Write-Host "Could not resolve your user object id from 'az login'." -ForegroundColor Yellow
    Write-Host "The user-based test document requires your Entra object id (oid)." -ForegroundColor Yellow
    $script:UserOid = (Read-Host "Enter your user object id (oid)").Trim()
  }

  _Save-State
}

function Get-SearchContext {
  $endpoint = az search service show -g $script:RG -n $script:Search --query "endpoint" -o tsv
  if (-not $endpoint) { throw "Failed to resolve Search endpoint. Check RG/service name and az login context." }

  $adminKey = az search admin-key show -g $script:RG --service-name $script:Search --query "primaryKey" -o tsv
  if (-not $adminKey) { throw "Failed to resolve Search admin key." }

  $base = if ($endpoint -match '^https?://') { $endpoint } else { "https://$endpoint" }
  $adminHeaders = @{
    "api-key"      = $adminKey
    "Content-Type" = "application/json"
    "Accept"       = "application/json;odata.metadata=none"
  }

  return @{ Base = $base; AdminHeaders = $adminHeaders }
}

function Show-PermissionFilterOptionStatus($Base, $Headers) {
  # NOTE: For this preview feature, the index definition endpoint uses indexes('{name}')
  $indexUri = "$Base/indexes('$($script:Index)')?api-version=$Api"
  $indexDef = Invoke-RestMethod -Method GET -Uri $indexUri -Headers $Headers

  if (-not ($indexDef.PSObject.Properties.Name -contains 'permissionFilterOption')) {
    throw "Index definition returned, but missing permissionFilterOption. Check api-version=$Api."
  }

  if ($indexDef.permissionFilterOption -eq "enabled") {
    Write-Host "Index permissionFilterOption=enabled" -ForegroundColor DarkGreen
  } else {
    Write-Host "Index permissionFilterOption=$($indexDef.permissionFilterOption)" -ForegroundColor Yellow
  }

  return $indexDef.permissionFilterOption
}

function Ensure-PermissionFiltersEnabled($Base, $Headers) {
  # Back-compat wrapper: this script no longer mutates index settings.
  Show-PermissionFilterOptionStatus -Base $Base -Headers $Headers
}

function New-ZeroVector([int]$Dimensions) {
  # Creates a float32 array filled with zeros.
  return [single[]]::new($Dimensions)
}

function New-TestDocPayload {
  param(
    [Parameter(Mandatory=$true)][ValidateSet("user","group","rbac","public","publicAll")]$Mode
  )

  # Synthetic, concise English sentence about how a "Contoso computer" works (same content across scenarios)
  $contosoSentence = "A Contoso computer boots a signed firmware, loads the Contoso OS kernel, and uses the Security Hub to enforce device health and app access policies."

  $common = @{
    "@search.action" = "upload"
    chunk_id = 0
    title = "ACL Test - $Mode"
    category = "acl-test"
    source = "manual"
    content = $contosoSentence
    contentVector = (New-ZeroVector -Dimensions $VectorDimensions)
    captionVector = (New-ZeroVector -Dimensions $VectorDimensions)
  }

  switch ($Mode) {
    "user" {
      $doc = $common.Clone()
      $doc.id = "acl-test-user-001"
      $doc.parent_id = "acl-test-user"
      $doc.metadata_storage_path = "manual://acl-test/user"
      $doc.metadata_storage_name = "contoso-acl-user.txt"
      $doc.metadata_security_user_ids  = @($script:UserOid)
      $doc.metadata_security_group_ids = @()
      $doc.metadata_security_rbac_scope = $null
      return @{ value = @($doc) }
    }
    "group" {
      $doc = $common.Clone()
      $doc.id = "acl-test-group-001"
      $doc.parent_id = "acl-test-group"
      $doc.metadata_storage_path = "manual://acl-test/group"
      $doc.metadata_storage_name = "contoso-acl-group.txt"
      $doc.metadata_security_user_ids  = @()
      $doc.metadata_security_group_ids = @($script:GroupOid)
      $doc.metadata_security_rbac_scope = $null
      return @{ value = @($doc) }
    }
    "rbac" {
      $doc = $common.Clone()
      $doc.id = "acl-test-rbac-001"
      $doc.parent_id = "acl-test-rbac"
      $doc.metadata_storage_path = "manual://acl-test/rbac"
      $doc.metadata_storage_name = "contoso-acl-rbac.txt"
      $doc.metadata_security_user_ids  = @()
      $doc.metadata_security_group_ids = @()
      $doc.metadata_security_rbac_scope = $script:RbacScope
      return @{ value = @($doc) }
    }
    "public" {
      $doc = $common.Clone()
      $doc.id = "acl-test-public-naive-001"
      $doc.parent_id = "acl-test-public-naive"
      $doc.metadata_storage_path = "manual://acl-test/public"
      $doc.metadata_storage_name = "contoso-acl-public-naive.txt"
      # Naive "public" document: do not set any ACL/RBAC fields.
      # Note: if permissionFilterOption=enabled, a document without ACL/RBAC fields may not be visible
      # to non-elevated reads depending on service behavior.
      return @{ value = @($doc) }
    }
    "publicAll" {
      $doc = $common.Clone()
      $doc.id = "acl-test-public-all-001"
      $doc.parent_id = "acl-test-public-all"
      $doc.metadata_storage_path = "manual://acl-test/public-all"
      $doc.metadata_storage_name = "contoso-acl-public-all.txt"
      # Special ACL value: ["all"] => any user can access the document (when permission filtering is enabled).
      $doc.metadata_security_user_ids  = @("all")
      $doc.metadata_security_group_ids = @()
      $doc.metadata_security_rbac_scope = $null
      return @{ value = @($doc) }
    }
  }
}

function Upload-Doc($Base, $Headers, $Payload) {
  $body = ($Payload | ConvertTo-Json -Depth 50)
  $result = Invoke-RestMethod -Method POST -Uri "$Base/indexes('$($script:Index)')/docs/search.index?api-version=$Api" -Headers $Headers -Body $body
  if (-not $result -or -not $result.value) {
    throw "Indexing call returned no result."
  }

  $failures = @($result.value | Where-Object { $_.status -ne $true })
  if ($failures.Count -gt 0) {
    Write-Host "Indexing failures:" -ForegroundColor Red
    $failures | Format-List | Out-String | Write-Host
    throw "One or more documents failed indexing."
  }

  return $result
}

function Search-DocById($Base, $Headers, $DocId) {
  $searchBody = @{
    search = "*"
    filter = "id eq '$DocId'"
    top = 1
  } | ConvertTo-Json -Depth 10

  # 2025-11-01-preview uses the search POST endpoint named search.post.search
  $uri = "$Base/indexes('$($script:Index)')/docs/search.post.search?api-version=$Api"
  try {
    return Invoke-RestMethod -Method POST -Uri $uri -Headers $Headers -Body $searchBody
  }
  catch {
    $status = $null
    $requestId = $null
    if ($_.Exception -and $_.Exception.Response) {
      $status = $_.Exception.Response.StatusCode
      if ($_.Exception.Response.Headers) {
        $vals = $null
        if ($_.Exception.Response.Headers.TryGetValues('request-id', [ref]$vals)) {
          $requestId = ($vals | Select-Object -First 1)
        }
      }
    }

    $details = $null
    if ($_.ErrorDetails -and $_.ErrorDetails.Message) { $details = $_.ErrorDetails.Message }

    throw "Search query failed (status=$status, request-id=$requestId). $details"
  }
}

function Search-DocsByQuery($Base, $Headers, $Query, $Top = 10) {
  $searchBody = @{
    search = $Query
    top = $Top
  } | ConvertTo-Json -Depth 10

  # 2025-11-01-preview uses the search POST endpoint named search.post.search
  $uri = "$Base/indexes('$($script:Index)')/docs/search.post.search?api-version=$Api"
  try {
    return Invoke-RestMethod -Method POST -Uri $uri -Headers $Headers -Body $searchBody
  }
  catch {
    $status = $null
    $requestId = $null
    if ($_.Exception -and $_.Exception.Response) {
      $status = $_.Exception.Response.StatusCode
      if ($_.Exception.Response.Headers) {
        $vals = $null
        if ($_.Exception.Response.Headers.TryGetValues('request-id', [ref]$vals)) {
          $requestId = ($vals | Select-Object -First 1)
        }
      }
    }

    $details = $null
    if ($_.ErrorDetails -and $_.ErrorDetails.Message) { $details = $_.ErrorDetails.Message }

    throw "Search query failed (status=$status, request-id=$requestId). $details"
  }
}

function Format-DocForDisplay {
  param([Parameter(Mandatory=$true)][object]$Doc)

  $vectorFields = @('contentVector', 'captionVector')
  $out = [ordered]@{}

  foreach ($p in $Doc.PSObject.Properties) {
    $name = $p.Name
    $value = $p.Value

    if ($vectorFields -contains $name) {
      # Vectors can be huge; show only the first 50 characters of their JSON representation.
      try {
        if ($null -eq $value) {
          $out[$name] = $null
        } else {
          $json = ($value | ConvertTo-Json -Compress -Depth 5)
          if ($json.Length -gt 50) {
            $out[$name] = $json.Substring(0, 50)
          } else {
            $out[$name] = $json
          }
        }
      } catch {
        $out[$name] = '<vector:unavailable>'
      }

      continue
    }

    # Make common arrays more readable.
    if ($value -is [System.Array] -and -not ($value -is [byte[]])) {
      $items = @($value)
      if ($items.Count -eq 0) {
        $out[$name] = ""
      } elseif ($items[0] -is [string] -or $items[0] -is [guid]) {
        $out[$name] = ($items | ForEach-Object { "$_" }) -join ','
      } else {
        $out[$name] = $value
      }
    } else {
      $out[$name] = $value
    }
  }

  return [pscustomobject]$out
}

function Get-DocIdsBatch($Base, $Headers, $Top = 500) {
  $searchBody = @{
    search = "*"
    top = $Top
    select = "id"
  } | ConvertTo-Json -Depth 10

  $uri = "$Base/indexes('$($script:Index)')/docs/search.post.search?api-version=$Api"
  try {
    $result = Invoke-RestMethod -Method POST -Uri $uri -Headers $Headers -Body $searchBody
    $items = @($result.value)
    return @($items | ForEach-Object { $_.id } | Where-Object { $_ })
  }
  catch {
    $status = $null
    $requestId = $null
    if ($_.Exception -and $_.Exception.Response) {
      $status = $_.Exception.Response.StatusCode
      if ($_.Exception.Response.Headers) {
        $vals = $null
        if ($_.Exception.Response.Headers.TryGetValues('request-id', [ref]$vals)) {
          $requestId = ($vals | Select-Object -First 1)
        }
      }
    }

    $details = $null
    if ($_.ErrorDetails -and $_.ErrorDetails.Message) { $details = $_.ErrorDetails.Message }

    throw "Search query failed while enumerating ids (status=$status, request-id=$requestId). $details"
  }
}

function Remove-DocsByIds($Base, $Headers, $Ids) {
  if (-not $Ids -or $Ids.Count -eq 0) { return $null }

  $value = @()
  foreach ($id in $Ids) {
    $value += @{
      "@search.action" = "delete"
      id = $id
    }
  }

  $payload = @{ value = $value }
  $body = ($payload | ConvertTo-Json -Depth 20)
  $result = Invoke-RestMethod -Method POST -Uri "$Base/indexes('$($script:Index)')/docs/search.index?api-version=$Api" -Headers $Headers -Body $body

  if (-not $result -or -not $result.value) {
    throw "Delete call returned no result."
  }

  $failures = @($result.value | Where-Object { $_.status -ne $true })
  if ($failures.Count -gt 0) {
    Write-Host "Delete failures:" -ForegroundColor Red
    $failures | Format-List | Out-String | Write-Host
    throw "One or more documents failed deletion."
  }

  return $result
}

function Remove-AllDocumentsInIndex($Base, $AdminHeaders, $QueryHeaders, $PermissionFilteringEnabled) {
  $batchSize = 500
  $totalDeleted = 0
  $round = 0

  while ($true) {
    $round++
    $ids = @()

    # Prefer admin headers to avoid per-user trimming; fall back to query headers if needed.
    try {
      $ids = @(Get-DocIdsBatch -Base $Base -Headers $AdminHeaders -Top $batchSize)
    }
    catch {
      Write-Host "Failed to enumerate ids with admin headers. Error:" -ForegroundColor Yellow
      Write-Host $_
    }

    if ($ids.Count -eq 0 -and $PermissionFilteringEnabled) {
      # With permission filtering enabled, admin queries might not surface documents without elevated read.
      try {
        $ids = @(Get-DocIdsBatch -Base $Base -Headers $QueryHeaders -Top $batchSize)
        if ($ids.Count -gt 0) {
          Write-Host "Warning: deleting only documents visible to your query token (permission filtering enabled)." -ForegroundColor Yellow
        }
      }
      catch {
        # Ignore; we'll treat as empty.
      }
    }

    if ($ids.Count -eq 0) { break }

    [void](Remove-DocsByIds -Base $Base -Headers $AdminHeaders -Ids $ids)
    $totalDeleted += $ids.Count
    Write-Host "Deleted batch ${round}: $($ids.Count) docs (total=$totalDeleted)" -ForegroundColor DarkYellow

    Start-Sleep -Seconds 1
  }

  return $totalDeleted
}

function Read-Doc($Base, $Headers, $DocId) {
  $attempts = 6
  for ($i = 1; $i -le $attempts; $i++) {
    $searchResult = Search-DocById -Base $Base -Headers $Headers -DocId $DocId
    if ($searchResult.value -and $searchResult.value.Count -gt 0) {
      return $searchResult.value[0]
    }

    if ($i -lt $attempts) {
      Start-Sleep -Seconds 1
    }
  }

  if (-not $UseElevatedRead -and -not $IncludeQueryAuthToken) {
    throw "No results for '$DocId'. When permissionFilterOption=enabled you must query as a user (set IncludeQueryAuthToken=true) or enable elevated read."
  }

  if ($DocId -like 'acl-test-rbac-*') {
    throw "No results for '$DocId' (search returned 0 results). For rbacScope enforcement, ensure the querying user has a storage data-plane role (for example, Storage Blob Data Reader) on: $($script:RbacScope)"
  }

  throw "No results for '$DocId' (search returned 0 results)."
}

# --- Main ---
Prompt-Context

$ctx = Get-SearchContext
$Base = $ctx.Base
$AdminHeaders = $ctx.AdminHeaders

Write-Host "Search endpoint: $Base" -ForegroundColor Cyan

function Prompt-Index {
  $reuseIndex = $false
  if ($script:Index) { $reuseIndex = _Read-YesNo -Prompt "Reuse last index '$script:Index'?" -DefaultYes $true }
  if ($reuseIndex -and $script:Index) { return }

  $apisToTry = @($Api, '2024-07-01', '2023-11-01') | Select-Object -Unique
  $names = @()
  foreach ($v in $apisToTry) {
    try {
      $resp = Invoke-RestMethod -Method GET -Uri "$Base/indexes?api-version=$v" -Headers $AdminHeaders
      $names = @($resp.value | ForEach-Object { $_.name } | Where-Object { $_ })
      if ($names.Count -gt 0) { break }
    } catch {
      # Try next version
    }
  }
  if (-not $names -or $names.Count -eq 0) {
    throw "Failed to enumerate indexes. Verify you can access the Search data plane and that the admin key is valid."
  }

  $items = @($names | Sort-Object | ForEach-Object { [pscustomobject]@{ Name = $_ } })
  $chosen = _Select-FromList -Title "Available Search indexes" -Items $items -Render { param($i) $i.Name }
  $script:Index = $chosen.Name
  _Save-State
}

Prompt-Index

# Show native ACL enforcement status at index level (read-only)
$permissionFilterOption = Show-PermissionFilterOptionStatus -Base $Base -Headers $AdminHeaders
$permissionFilteringEnabled = ($permissionFilterOption -eq "enabled")

if (-not $permissionFilteringEnabled) {
  Write-Host "Permission filtering is not enabled; queries will run without query auth token headers (untrimmed results)." -ForegroundColor Yellow
}

if ($IncludeQueryAuthToken -and $permissionFilteringEnabled) {
  $token = az account get-access-token --resource https://search.azure.com --query accessToken -o tsv
  if (-not $token) { throw "Failed to obtain access token for https://search.azure.com" }
  $QueryHeaders = @{
    "Content-Type" = "application/json"
    "Accept"       = "application/json;odata.metadata=none"
    "Authorization" = "Bearer $token"
    # Required for query-time ACL/RBAC enforcement in the 2025-11-01-preview API.
    "x-ms-query-source-authorization" = $token
  }
} else {
  if ($IncludeQueryAuthToken -and -not $permissionFilteringEnabled) {
    Write-Host "Skipping query auth token headers because permissionFilterOption is not enabled." -ForegroundColor Yellow
  }
  $QueryHeaders = $AdminHeaders.Clone()
}

if ($UseElevatedRead -and $permissionFilteringEnabled) {
  # Note: requires RBAC action Microsoft.Search/searchServices/indexes/contentSecurity/elevatedOperations/read
  $QueryHeaders["x-ms-enable-elevated-read"] = "true"
}


$selectionParam = $Selection
$usedSelectionParam = $false

while ($true) {
  $payload = $null
  $docId = $null

  if (-not $usedSelectionParam -and $selectionParam) {
    $choice = $selectionParam
    $usedSelectionParam = $true
  } else {
    Write-Host ""
    Write-Host "Choose which test document to create:" -ForegroundColor Yellow
    Write-Host "  1) userIds only (your OID)"
    Write-Host "  2) groupIds only (group you belong to)"
    Write-Host "  3) rbacScope only (storage container scope)"
    Write-Host "  4) create PUBLIC doc (naive: no ACL/RBAC fields)"
    Write-Host "  5) create PUBLIC doc (userIds=['all'])" -ForegroundColor Cyan
    Write-Host "  6) list accessible docs (search 'contoso')"
    Write-Host "  7) DELETE ALL documents in index" -ForegroundColor Red
    Write-Host "  Q) Quit"
    $choice = Read-Host "Selection"
  }

  if ($choice -match '^(q|Q)$') { break }

  # Option 6: list accessible documents for the current user.
  # Important: handle this outside the switch so we can reliably continue the *while* loop
  # (a `continue` inside a switch can behave differently than expected).
  if ($choice -eq "6") {
    try {
      $result = Search-DocsByQuery -Base $Base -Headers $QueryHeaders -Query "contoso" -Top 10
      $items = @($result.value)
      if ($items.Count -eq 0) {
        Write-Host "No accessible documents matched query 'contoso'." -ForegroundColor Yellow
      } else {
        Write-Host "Top $($items.Count) accessible documents for query 'contoso':" -ForegroundColor Green
        foreach ($d in $items) {
          Write-Host "" 
          (Format-DocForDisplay -Doc $d) | Format-List
        }
      }
    }
    catch {
      Write-Host "Failed to list documents. Error:" -ForegroundColor Red
      Write-Host $_
    }

    if ($selectionParam) { return }
    continue
  }

  # Option 7: delete all documents in the index.
  if ($choice -eq "7") {
    if ($selectionParam -and -not $Force) {
      Write-Host "Refusing to delete without -Force in non-interactive mode." -ForegroundColor Red
      return
    }

    if (-not $Force) {
      Write-Host "DANGER: This will attempt to delete ALL documents in index '$Index'." -ForegroundColor Red
      $confirm = Read-Host "Type DELETE ALL to confirm"
      if ($confirm -ne "DELETE ALL") {
        Write-Host "Cancelled." -ForegroundColor Yellow
        if ($selectionParam) { return }
        continue
      }
    }

    try {
      $deleted = Remove-AllDocumentsInIndex -Base $Base -AdminHeaders $AdminHeaders -QueryHeaders $QueryHeaders -PermissionFilteringEnabled $permissionFilteringEnabled
      Write-Host "Delete complete. Deleted $deleted documents." -ForegroundColor Green
    }
    catch {
      Write-Host "Failed to delete all documents. Error:" -ForegroundColor Red
      Write-Host $_
    }

    if ($selectionParam) { return }
    continue
  }

  switch ($choice) {
    "1" {
      $payload = New-TestDocPayload -Mode "user"
      $docId = "acl-test-user-001"
    }
    "2" {
      if (-not $script:GroupOid) {
        $example = "d34c4ebe-4984-4903-a64d-8c20283d516b"
        Write-Host "" 
        Write-Host "Group-based test document requires a group object id." -ForegroundColor Yellow
        Write-Host "Example format: $example" -ForegroundColor DarkYellow
        $script:GroupOid = (Read-Host "Enter group object id (group oid)").Trim()
        _Save-State
      }
      $payload = New-TestDocPayload -Mode "group"
      $docId = "acl-test-group-001"
    }
    "3" {
      if (-not $script:RbacScope) {
        Write-Host "" 
        Write-Host "RBAC-scope test document requires a resource scope string." -ForegroundColor Yellow
        Write-Host "Example:" -ForegroundColor DarkYellow
        Write-Host "/subscriptions/$($script:SUB)/resourceGroups/$($script:RG)/providers/Microsoft.Storage/storageAccounts/<storageAccount>/blobServices/default/containers/<container>" -ForegroundColor DarkYellow
        $script:RbacScope = (Read-Host "Enter RBAC scope").Trim()
        _Save-State
      }
      $payload = New-TestDocPayload -Mode "rbac"
      $docId = "acl-test-rbac-001"
    }
    "4" {
      $payload = New-TestDocPayload -Mode "public"
      $docId = "acl-test-public-naive-001"
    }
    "5" {
      $payload = New-TestDocPayload -Mode "publicAll"
      $docId = "acl-test-public-all-001"
    }
    default {
      Write-Host "Invalid selection. Try again." -ForegroundColor Red
      continue
    }
  }

  # Options 6/7 don't set a payload/docId. If we got here without a payload, skip upload/read.
  if (-not $payload -or -not $docId) {
    if ($selectionParam) { break }
    continue
  }

  try {
    $indexResult = Upload-Doc -Base $Base -Headers $AdminHeaders -Payload $payload
    $item = $indexResult.value | Where-Object { $_.key -eq $docId } | Select-Object -First 1
    if ($item) {
      Write-Host "Uploaded: $docId (statusCode=$($item.statusCode))" -ForegroundColor Green
    } else {
      Write-Host "Uploaded: $docId" -ForegroundColor Green
    }

    $doc = Read-Doc -Base $Base -Headers $QueryHeaders -DocId $docId
    Write-Host "Stored ACL fields for ${docId}:" -ForegroundColor DarkCyan
    [pscustomobject]@{
      id = $doc.id
      metadata_security_user_ids  = ($doc.metadata_security_user_ids -join ",")
      metadata_security_group_ids = ($doc.metadata_security_group_ids -join ",")
      metadata_security_rbac_scope = $doc.metadata_security_rbac_scope
      content = $doc.content
    } | Format-List

    if ($docId -eq "acl-test-public-naive-001" -and $permissionFilteringEnabled) {
      Write-Host "Note: permissionFilterOption=enabled and this doc has no ACL/RBAC fields; it may not be visible to normal reads." -ForegroundColor Yellow
    }
  }
  catch {
    Write-Host "Failed to upload/read document. Error:" -ForegroundColor Red
    Write-Host $_
  }

  # Non-interactive mode: run a single selection then exit.
  if ($selectionParam) { break }
}

Write-Host "Done." -ForegroundColor Cyan
