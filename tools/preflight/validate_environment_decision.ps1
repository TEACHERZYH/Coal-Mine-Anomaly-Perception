param(
    [string]$DecisionPath = "configs/environment_decision.lock.json",
    [string]$OutputPath = "evidence/preimplementation/environment_decision_validation.json"
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path -LiteralPath $DecisionPath -PathType Leaf)) {
    throw "Environment decision not found: $DecisionPath"
}

$decision = Get-Content -LiteralPath $DecisionPath -Raw | ConvertFrom-Json
$required = @(
    "schema_version",
    "status",
    "local_snapshot",
    "local_snapshot_sha256",
    "remote_snapshot",
    "remote_snapshot_sha256",
    "local_strategy",
    "remote_strategy",
    "selected_local_python",
    "selected_remote_python",
    "local_activation_command",
    "remote_activation_command",
    "selected_local_framework",
    "selected_remote_framework",
    "selected_remote_cuda_runtime",
    "selected_remote_compiler",
    "framework_cuda_compatibility",
    "missing_local_packages",
    "missing_remote_packages",
    "package_source_policy",
    "local_environment_specification",
    "remote_environment_specification",
    "local_setup_command",
    "remote_setup_command",
    "local_environment_verification_command",
    "remote_environment_verification_command",
    "decision_rationale",
    "compatibility_risks",
    "reviewed_at",
    "reviewed_by"
)

foreach ($field in $required) {
    if (-not $decision.PSObject.Properties.Name.Contains($field)) {
        throw "Missing required environment decision field: $field"
    }
}

if ([int]$decision.schema_version -ne 2) {
    throw "Environment decision schema_version must be 2"
}
if ($decision.status -ne "locked") {
    throw "Environment decision status must be locked"
}
foreach ($scope in @("local", "remote")) {
    $strategyField = "${scope}_strategy"
    if ($decision.$strategyField -notin @("reuse_existing", "install_compatible_missing", "new_isolated")) {
        throw "Unsupported $scope environment strategy: $($decision.$strategyField)"
    }
}
foreach ($snapshot in @($decision.local_snapshot, $decision.remote_snapshot)) {
    if (-not (Test-Path -LiteralPath $snapshot -PathType Leaf)) {
        throw "Referenced snapshot not found: $snapshot"
    }
}

$localHash = (Get-FileHash -LiteralPath $decision.local_snapshot -Algorithm SHA256).Hash.ToLowerInvariant()
$remoteHash = (Get-FileHash -LiteralPath $decision.remote_snapshot -Algorithm SHA256).Hash.ToLowerInvariant()
if ($decision.local_snapshot_sha256.ToLowerInvariant() -ne $localHash) {
    throw "Local snapshot hash mismatch"
}
if ($decision.remote_snapshot_sha256.ToLowerInvariant() -ne $remoteHash) {
    throw "Remote snapshot hash mismatch"
}

$remoteSnapshot = Get-Content -LiteralPath $decision.remote_snapshot -Raw | ConvertFrom-Json
if ($remoteSnapshot.exit_code -ne 0 -or $remoteSnapshot.remote_host -ne "xinxi-zhyh@211.87.115.228") {
    throw "Remote snapshot is not a successful snapshot of the approved host"
}

$requiredResolvedStrings = @(
    "selected_local_python",
    "selected_remote_python",
    "local_activation_command",
    "remote_activation_command",
    "selected_local_framework",
    "selected_remote_framework",
    "selected_remote_cuda_runtime",
    "selected_remote_compiler",
    "framework_cuda_compatibility",
    "package_source_policy",
    "local_environment_specification",
    "remote_environment_specification",
    "local_setup_command",
    "remote_setup_command",
    "local_environment_verification_command",
    "remote_environment_verification_command",
    "decision_rationale",
    "reviewed_by"
)
foreach ($field in $requiredResolvedStrings) {
    $value = [string]$decision.$field
    if ([string]::IsNullOrWhiteSpace($value)) {
        throw "Environment decision field is unresolved: $field"
    }
}

$serializedDecision = $decision | ConvertTo-Json -Depth 12
if ($serializedDecision -match '(?i)TBD|template_not_locked|review_required') {
    throw "Environment decision contains an unresolved placeholder token"
}

$localProbe = "deferred_to_P025_for_new_isolated_strategy"
if ($decision.local_strategy -in @("reuse_existing", "install_compatible_missing")) {
    if (-not (Test-Path -LiteralPath $decision.selected_local_python -PathType Leaf)) {
        throw "Selected local Python does not exist: $($decision.selected_local_python)"
    }
    $localProbe = & $decision.selected_local_python -c "import json,sys; print(json.dumps({'executable':sys.executable,'version':sys.version.split()[0]}))" 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Selected local Python failed the execution probe: $localProbe"
    }
}

if ($decision.remote_strategy -in @("reuse_existing", "install_compatible_missing")) {
    $remoteText = @($remoteSnapshot.raw_lines) -join "`n"
    if ($remoteText -notmatch [regex]::Escape([string]$decision.selected_remote_python)) {
        throw "Selected reusable remote Python was not observed in the bounded remote snapshot"
    }
}

foreach ($scope in @("local", "remote")) {
    $strategyField = "${scope}_strategy"
    $missingField = "missing_${scope}_packages"
    if ($decision.$strategyField -eq "reuse_existing" -and @($decision.$missingField).Count -ne 0) {
        throw "reuse_existing requires an empty $missingField list"
    }
    if ($decision.$strategyField -eq "install_compatible_missing" -and @($decision.$missingField).Count -eq 0) {
        throw "install_compatible_missing requires at least one named package in $missingField"
    }
}

$reviewedAt = [datetimeoffset]::MinValue
if (-not [datetimeoffset]::TryParse([string]$decision.reviewed_at, [ref]$reviewedAt)) {
    throw "reviewed_at must be a valid timestamp"
}

$receipt = [ordered]@{
    schema_version = 2
    status = "pass"
    validated_at = (Get-Date).ToString("o")
    decision_path = $DecisionPath
    decision_sha256 = (Get-FileHash -LiteralPath $DecisionPath -Algorithm SHA256).Hash.ToLowerInvariant()
    local_snapshot_sha256 = $localHash
    remote_snapshot_sha256 = $remoteHash
    approved_remote_host = $remoteSnapshot.remote_host
    local_strategy = $decision.local_strategy
    remote_strategy = $decision.remote_strategy
    selected_local_python = $decision.selected_local_python
    selected_remote_python = $decision.selected_remote_python
    local_python_probe = ($localProbe | Out-String).Trim()
}
$parent = Split-Path -Parent $OutputPath
if ($parent) {
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
}
$receipt | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $OutputPath -Encoding UTF8
Write-Output $OutputPath
