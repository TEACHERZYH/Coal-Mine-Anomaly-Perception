param(
    [string]$TemplatePath = "configs/environment_decision.template.json",
    [string]$DecisionPath = "configs/environment_decision.lock.json"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $TemplatePath -PathType Leaf)) {
    throw "Environment decision template not found: $TemplatePath"
}

if (Test-Path -LiteralPath $DecisionPath -PathType Leaf) {
    $existing = Get-Content -LiteralPath $DecisionPath -Raw | ConvertFrom-Json
    if ($existing.status -eq "locked") {
        throw "Refusing to overwrite a locked environment decision: $DecisionPath"
    }
}

$decision = Get-Content -LiteralPath $TemplatePath -Raw | ConvertFrom-Json
foreach ($snapshotField in @("local_snapshot", "remote_snapshot")) {
    $snapshotPath = [string]$decision.$snapshotField
    if (-not (Test-Path -LiteralPath $snapshotPath -PathType Leaf)) {
        throw "Required environment snapshot not found: $snapshotPath"
    }
}

$decision.status = "review_required"
$decision.local_snapshot_sha256 = (Get-FileHash -LiteralPath $decision.local_snapshot -Algorithm SHA256).Hash.ToLowerInvariant()
$decision.remote_snapshot_sha256 = (Get-FileHash -LiteralPath $decision.remote_snapshot -Algorithm SHA256).Hash.ToLowerInvariant()

$parent = Split-Path -Parent $DecisionPath
if ($parent) {
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
}
$decision | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $DecisionPath -Encoding UTF8
Write-Output $DecisionPath
