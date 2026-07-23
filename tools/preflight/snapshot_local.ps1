param(
    [string]$OutputPath = "evidence/preimplementation/local_environment.json",
    [string]$HashReceiptPath = "evidence/preimplementation/local_environment.sha256.json"
)

$ErrorActionPreference = "Stop"
$python = "C:\Users\zyh\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Bundled Python not found: $python"
}

$parent = Split-Path -Parent $OutputPath
if ($parent) {
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
}

$pythonVersion = (& $python --version 2>&1 | Out-String).Trim()
$pythonPrefix = (& $python -c "import json,sys; print(json.dumps({'executable':sys.executable,'prefix':sys.prefix,'version':sys.version}))" | ConvertFrom-Json)
$packagesJson = (& $python -m pip list --format=json 2>&1 | Out-String).Trim()
if ($LASTEXITCODE -ne 0) {
    throw "pip inventory failed with exit code ${LASTEXITCODE}: $packagesJson"
}
$packages = @(($packagesJson | ConvertFrom-Json) | ForEach-Object { $_ })
$commands = foreach ($name in @("git", "ssh", "scp", "rsync", "nvidia-smi", "nvcc", "gcc", "g++", "cl", "cmake")) {
    $cmd = Get-Command $name -ErrorAction SilentlyContinue
    $version = $null
    $versionProbeExitCode = $null
    $versionProbeError = $null
    if ($cmd) {
        $versionArgs = @(switch ($name) {
            "git" { @("--version") }
            "ssh" { @("-V") }
            "scp" { @("-V") }
            "rsync" { @("--version") }
            "nvidia-smi" { @("--query-gpu=driver_version,name", "--format=csv,noheader") }
            "nvcc" { @("--version") }
            "gcc" { @("--version") }
            "g++" { @("--version") }
            "cmake" { @("--version") }
            default { @() }
        })
        $previousErrorActionPreference = $ErrorActionPreference
        try {
            $ErrorActionPreference = "Continue"
            $versionOutput = & $cmd.Source @versionArgs 2>&1
            $versionProbeExitCode = $LASTEXITCODE
            $version = ($versionOutput | Select-Object -First 4 | Out-String).Trim()
            if ($versionProbeExitCode -ne 0) {
                $versionProbeError = "exit_code=$versionProbeExitCode"
            }
        }
        catch {
            $versionProbeError = $_.Exception.Message
        }
        finally {
            $ErrorActionPreference = $previousErrorActionPreference
        }
    }
    [ordered]@{
        name = $name
        available = $null -ne $cmd
        source = if ($cmd) { $cmd.Source } else { $null }
        version_excerpt = $version
        version_probe_exit_code = $versionProbeExitCode
        version_probe_error = $versionProbeError
    }
}

$payload = [ordered]@{
    schema_version = 1
    captured_at = (Get-Date).ToString("o")
    host = $env:COMPUTERNAME
    cwd = (Get-Location).Path
    powershell_version = $PSVersionTable.PSVersion.ToString()
    python_version = $pythonVersion
    python = $pythonPrefix
    package_inventory_shape = "flat_list"
    package_count = $packages.Count
    packages = $packages
    commands = $commands
    cuda_path = $env:CUDA_PATH
}

$payload | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $OutputPath -Encoding UTF8
$resolvedOutput = (Resolve-Path -LiteralPath $OutputPath).Path
$artifactHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $resolvedOutput).Hash.ToLowerInvariant()
$hashParent = Split-Path -Parent $HashReceiptPath
if ($hashParent) {
    New-Item -ItemType Directory -Force -Path $hashParent | Out-Null
}
$hashReceipt = [ordered]@{
    schema_version = 1
    artifact_path = $resolvedOutput
    artifact_sha256 = $artifactHash
    artifact_bytes = (Get-Item -LiteralPath $resolvedOutput).Length
    hash_algorithm = "sha256"
    created_at = (Get-Date).ToString("o")
}
$hashReceipt | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $HashReceiptPath -Encoding UTF8
Write-Output $OutputPath
Write-Output $HashReceiptPath
