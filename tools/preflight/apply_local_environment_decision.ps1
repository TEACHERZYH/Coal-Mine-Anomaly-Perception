param(
    [string]$DecisionPath = "configs/environment_decision.lock.json",
    [string]$ValidationPath = "evidence/preimplementation/environment_decision_validation.json",
    [string]$OutputPath = "evidence/preimplementation/local_environment_ready.json"
)

$ErrorActionPreference = "Stop"
function Get-Utf8StringSha256 {
    param([Parameter(Mandatory = $true)][string]$Value)
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($Value)
        return (($sha.ComputeHash($bytes) | ForEach-Object { $_.ToString("x2") }) -join "")
    }
    finally {
        $sha.Dispose()
    }
}

function Invoke-PowerShellCommandCaptured {
    param([Parameter(Mandatory = $true)][string]$CommandText)

    $previousPreference = $ErrorActionPreference
    try {
        # Native stderr can contain non-fatal warnings. Preserve it and use the
        # child process exit code as the only success criterion.
        $ErrorActionPreference = "Continue"
        $captured = @(& powershell -NoProfile -NonInteractive -Command $CommandText 2>&1)
        $exitCode = [int]$LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }

    return [pscustomobject]@{
        exit_code = $exitCode
        output = @($captured | ForEach-Object { $_.ToString() })
    }
}

foreach ($path in @($DecisionPath, $ValidationPath)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required environment artifact not found: $path"
    }
}

$decision = Get-Content -LiteralPath $DecisionPath -Raw | ConvertFrom-Json
$validation = Get-Content -LiteralPath $ValidationPath -Raw | ConvertFrom-Json
$decisionHash = (Get-FileHash -LiteralPath $DecisionPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($decision.status -ne "locked" -or $validation.status -ne "pass") {
    throw "Environment decision and validation must both be locked/pass"
}
if ([string]$validation.decision_sha256 -ne $decisionHash) {
    throw "Environment decision changed after P020 validation"
}

$setupOutput = @()
if ([string]$decision.local_setup_command -ne "none_required") {
    $setupResult = Invoke-PowerShellCommandCaptured -CommandText ([string]$decision.local_setup_command)
    $setupOutput = @($setupResult.output)
    if ($setupResult.exit_code -ne 0) {
        throw "Locked local setup command failed with exit code $($setupResult.exit_code)"
    }
}

if (-not (Test-Path -LiteralPath $decision.selected_local_python -PathType Leaf)) {
    throw "Selected local Python does not exist after setup: $($decision.selected_local_python)"
}
$verificationResult = Invoke-PowerShellCommandCaptured -CommandText ([string]$decision.local_environment_verification_command)
$verificationOutput = @($verificationResult.output)
if ($verificationResult.exit_code -ne 0) {
    throw "Locked local environment verification failed with exit code $($verificationResult.exit_code)"
}
$pythonProbe = @(& $decision.selected_local_python -c "import json,sys; print(json.dumps({'executable':sys.executable,'version':sys.version.split()[0]}))" 2>&1)
if ($LASTEXITCODE -ne 0) {
    throw "Selected local Python failed the final probe"
}
$packageInventory = @(& $decision.selected_local_python -c "import importlib.metadata as m,json; rows=sorted({'{}=={}'.format(d.metadata.get('Name'),d.version) for d in m.distributions() if d.metadata.get('Name')}); print(json.dumps(rows,separators=(',',':')))" 2>&1)
if ($LASTEXITCODE -ne 0) {
    throw "Selected local Python failed the package inventory probe"
}
$packageInventoryText = ($packageInventory | Out-String).Trim()
$environmentSpecification = [string]$decision.local_environment_specification
if (Test-Path -LiteralPath $environmentSpecification -PathType Leaf) {
    $specificationHashMode = "file_content"
    $environmentSpecificationHash = (Get-FileHash -LiteralPath $environmentSpecification -Algorithm SHA256).Hash.ToLowerInvariant()
}
else {
    $specificationHashMode = "canonical_text"
    $environmentSpecificationHash = Get-Utf8StringSha256 -Value $environmentSpecification
}

$receipt = [ordered]@{
    schema_version = 2
    scope = "local"
    status = "pass"
    ready_at = (Get-Date).ToString("o")
    host = $env:COMPUTERNAME
    decision_sha256 = $decisionHash
    strategy = $decision.local_strategy
    selected_python = $decision.selected_local_python
    environment_specification = $environmentSpecification
    specification_hash_mode = $specificationHashMode
    environment_specification_sha256 = $environmentSpecificationHash
    activation_command_sha256 = (Get-Utf8StringSha256 -Value ([string]$decision.local_activation_command))
    setup_command_sha256 = (Get-Utf8StringSha256 -Value ([string]$decision.local_setup_command))
    verification_command_sha256 = (Get-Utf8StringSha256 -Value ([string]$decision.local_environment_verification_command))
    package_inventory_sha256 = (Get-Utf8StringSha256 -Value $packageInventoryText)
    setup_command_executed = ([string]$decision.local_setup_command -ne "none_required")
    setup_output_excerpt = @($setupOutput | Select-Object -Last 80 | ForEach-Object { $_.ToString() })
    verification_output_excerpt = @($verificationOutput | Select-Object -Last 80 | ForEach-Object { $_.ToString() })
    python_probe = @($pythonProbe | ForEach-Object { $_.ToString() })
    package_inventory = @($packageInventory | ForEach-Object { $_.ToString() })
    slurm_job_id = $null
    compute_node = $null
    remote_closeout_reference = $null
}
$parent = Split-Path -Parent $OutputPath
if ($parent) {
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
}
$receipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $OutputPath -Encoding UTF8
Write-Output $OutputPath
