[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Mandatory = $true)]
    [string]$Module,
    [string]$EnvironmentDecisionPath = "configs/environment_decision.lock.json",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArguments
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $EnvironmentDecisionPath -PathType Leaf)) {
    throw "Environment decision not found: $EnvironmentDecisionPath"
}

$decision = Get-Content -LiteralPath $EnvironmentDecisionPath -Raw | ConvertFrom-Json
$python = [string]$decision.selected_local_python
if ([string]::IsNullOrWhiteSpace($python) -or -not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Selected local Python is missing or invalid: $python"
}

& $python -m $Module @RemainingArguments
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
