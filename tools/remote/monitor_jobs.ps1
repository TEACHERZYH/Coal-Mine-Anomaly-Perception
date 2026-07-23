[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$StepId,
    [string[]]$JobIds = @(),
    [switch]$GpuExpected,
    [string]$Target = 'xinxi-zhyh@211.87.115.228',
    [string]$RemoteRunRoot = '',
    [Parameter(Mandatory = $true)]
    [string]$OutputPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if ($Target -ne 'xinxi-zhyh@211.87.115.228') {
    throw 'Only the current Slurm host is permitted.'
}
foreach ($jobId in $JobIds) {
    if ($jobId -notmatch '^\d+(?:_\d+)?$') {
        throw "Invalid Slurm job ID: $jobId"
    }
}
if ($RemoteRunRoot -and $RemoteRunRoot -notmatch '^/(?:[-A-Za-z0-9._@+=:]+/)*[-A-Za-z0-9._@+=:]+$') {
    throw 'RemoteRunRoot must be an absolute path containing only safe path characters.'
}

$jobFilter = if ($JobIds.Count -gt 0) { ($JobIds -join ',') } else { '' }
$remoteScript = @'
set -euo pipefail
printf '%s\n' '__SQUEUE__'
squeue -u "$USER" -h -o '%i|%P|%j|%T|%M|%D|%R'
printf '%s\n' '__SACCT__'
if test -n '__JOB_FILTER__'; then
  sacct -n -X -P -j '__JOB_FILTER__' -o JobIDRaw,Partition,State,Elapsed,ExitCode,NodeList
fi
printf '%s\n' '__PROCESSES__'
ps -u "$USER" -o pid=,ppid=,etimes=,cmd= | grep -E '[m]ining1|[t]orchrun|[p]ython.*[m]ining1' || true
printf '%s\n' '__SESSIONS__'
tmux list-sessions 2>/dev/null || true
screen -ls 2>/dev/null || true
printf '%s\n' '__GPU_TELEMETRY_EXCERPT__'
if test -n '__RUN_ROOT__' && test -d '__RUN_ROOT__'; then
  for artifact in '__RUN_ROOT__'/telemetry*.json; do
    test -f "$artifact" || continue
    printf 'FILE|%s\n' "$artifact"
    tail -n 1 "$artifact"
  done
fi
printf '%s\n' '__GPU_TELEMETRY_ARTIFACTS__'
if test -n '__RUN_ROOT__' && test -d '__RUN_ROOT__'; then
  for artifact in '__RUN_ROOT__'/telemetry*.json; do test -f "$artifact" && sha256sum -- "$artifact"; done
fi
printf '%s\n' '__LOG_ARTIFACTS__'
if test -n '__RUN_ROOT__' && test -d '__RUN_ROOT__'; then
  for artifact in '__RUN_ROOT__'/*.log; do test -f "$artifact" && sha256sum -- "$artifact"; done
fi
printf '%s\n' '__CHECKPOINT_ARTIFACTS__'
if test -n '__RUN_ROOT__' && test -d '__RUN_ROOT__'; then
  for artifact in '__RUN_ROOT__'/last_checkpoint; do test -f "$artifact" && sha256sum -- "$artifact"; done
fi
printf '%s\n' '__RESULT_ARTIFACTS__'
if test -n '__RUN_ROOT__' && test -d '__RUN_ROOT__'; then
  for artifact in '__RUN_ROOT__'/results*; do test -f "$artifact" && sha256sum -- "$artifact"; done
fi
printf '%s\n' '__END__'
'@
$remoteScript = $remoteScript.Replace('__JOB_FILTER__', $jobFilter).Replace('__RUN_ROOT__', $RemoteRunRoot)
$capturedAt = [DateTimeOffset]::UtcNow
$raw = & ssh -o BatchMode=yes -o ConnectTimeout=8 $Target $remoteScript 2>&1
$sshExitCode = $LASTEXITCODE

function Get-MarkedSection {
    param([object[]]$Lines, [string]$StartMarker, [string]$EndMarker)
    $capturing = $false
    $values = @()
    foreach ($line in $Lines) {
        $text = [string]$line
        if ($text -eq $StartMarker) { $capturing = $true; continue }
        if ($capturing -and $text -eq $EndMarker) { break }
        if ($capturing) { $values += $text }
    }
    return @($values)
}

function Convert-DigestLines {
    param([object[]]$Lines)
    $records = @()
    foreach ($line in $Lines) {
        if ([string]$line -match '^([0-9a-f]{64})\s+\*?(.*)$') {
            $records += [ordered]@{ sha256 = $Matches[1]; path = $Matches[2] }
        }
    }
    return @($records)
}

$workflowState = 'connection_failed'
$remoteError = $null
$nextCheck = $null
$squeueLines = @()
$sacctLines = @()
$requestedSqueue = @()
$requestedSacct = @()
if ($sshExitCode -eq 0) {
    $squeueLines = @(Get-MarkedSection $raw '__SQUEUE__' '__SACCT__')
    $sacctLines = @(Get-MarkedSection $raw '__SACCT__' '__PROCESSES__')
    $squeueRecords = @($squeueLines | Where-Object { $_ -match '^\d+(?:_\d+)?\|' } | ForEach-Object {
        $fields = $_ -split '\|', 7
        [ordered]@{ job_id=$fields[0]; partition=$fields[1]; name=$fields[2]; state=$fields[3]; elapsed=$fields[4]; nodes=$fields[5]; reason_or_nodelist=$fields[6] }
    })
    $sacctRecords = @($sacctLines | Where-Object { $_ -match '^\d+(?:_\d+)?\|' } | ForEach-Object {
        $fields = $_ -split '\|', 6
        [ordered]@{ job_id=$fields[0]; partition=$fields[1]; state=$fields[2].TrimEnd('+'); elapsed=$fields[3]; exit_code=$fields[4]; node_list=$fields[5] }
    })
    $requestedSqueue = @($squeueRecords | Where-Object { $JobIds -contains $_.job_id })
    $requestedSacct = @($sacctRecords | Where-Object { $JobIds -contains $_.job_id })
    $states = @($requestedSqueue | ForEach-Object { $_.state })
    if ($JobIds.Count -eq 0) {
        $workflowState = 'completed'
    }
    elseif ($states -contains 'RUNNING' -or $states -contains 'COMPLETING') {
        $workflowState = 'running'
        $nextCheck = $capturedAt.AddMinutes(30).ToString('o')
    }
    elseif ($states -contains 'PENDING' -or $states -contains 'CONFIGURING') {
        $workflowState = 'pending'
        $nextCheck = $capturedAt.AddMinutes(30).ToString('o')
    }
    else {
        $terminalStates = @('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE')
        $allAccounted = $requestedSacct.Count -eq $JobIds.Count
        $allTerminal = $allAccounted -and @($requestedSacct | Where-Object { $terminalStates -notcontains $_.state }).Count -eq 0
        $allSuccessful = $allAccounted -and @($requestedSacct | Where-Object { $_.state -ne 'COMPLETED' -or $_.exit_code -ne '0:0' }).Count -eq 0
        if ($allSuccessful) {
            $workflowState = 'completed'
        }
        elseif ($allTerminal) {
            $workflowState = 'failed'
            $remoteError = 'One or more Slurm jobs ended with a non-success terminal state or exit code.'
        }
        else {
            $workflowState = 'blocked'
            $remoteError = 'Requested jobs left squeue before complete terminal sacct evidence was available.'
        }
    }
}
else {
    $remoteError = (($raw | Select-Object -Last 8) -join "`n")
}
$telemetryExcerpt = @(Get-MarkedSection $raw '__GPU_TELEMETRY_EXCERPT__' '__GPU_TELEMETRY_ARTIFACTS__')
$telemetryArtifacts = @(Convert-DigestLines (Get-MarkedSection $raw '__GPU_TELEMETRY_ARTIFACTS__' '__LOG_ARTIFACTS__'))
$logArtifacts = @(Convert-DigestLines (Get-MarkedSection $raw '__LOG_ARTIFACTS__' '__CHECKPOINT_ARTIFACTS__'))
$checkpointArtifacts = @(Convert-DigestLines (Get-MarkedSection $raw '__CHECKPOINT_ARTIFACTS__' '__RESULT_ARTIFACTS__'))
$resultArtifacts = @(Convert-DigestLines (Get-MarkedSection $raw '__RESULT_ARTIFACTS__' '__END__'))
if ($workflowState -eq 'running' -and $GpuExpected.IsPresent -and ($telemetryArtifacts.Count -eq 0 -or $telemetryExcerpt.Count -eq 0)) {
    $workflowState = 'blocked'
    $remoteError = 'Running GPU job has no compute-node telemetry artifact or excerpt.'
    $nextCheck = $null
}

$receipt = [ordered]@{
    step_id = $StepId
    monitored_at = $capturedAt.ToString('o')
    host = $Target
    slurm_job_ids = @($JobIds)
    gpu_expected = $GpuExpected.IsPresent
    squeue_state = [ordered]@{ records = @($requestedSqueue); raw_excerpt = @($squeueLines | Select-Object -First 40) }
    sacct_state = [ordered]@{ records = @($requestedSacct); requested_job_filter = $jobFilter; raw_excerpt = @($sacctLines | Select-Object -First 40) }
    relevant_processes = @(Get-MarkedSection $raw '__PROCESSES__' '__SESSIONS__' | Select-Object -First 20)
    tmux_and_screen_sessions = @(Get-MarkedSection $raw '__SESSIONS__' '__GPU_TELEMETRY_EXCERPT__' | Where-Object { $_ -match ':\s+\d+\s+windows|^\s*\d+\.[^\s]+' })
    gpu_telemetry_artifacts = $telemetryArtifacts
    gpu_telemetry_excerpt = $telemetryExcerpt
    recent_log_artifacts = $logArtifacts
    checkpoint_artifacts = $checkpointArtifacts
    result_artifacts = $resultArtifacts
    workflow_state = $workflowState
    remote_error = $remoteError
    next_check_due_at = $nextCheck
}
$targetPath = [System.IO.Path]::GetFullPath($OutputPath)
if (Test-Path -LiteralPath $targetPath) {
    throw "Monitor receipt is immutable: $targetPath"
}
$parent = Split-Path -Parent $targetPath
New-Item -ItemType Directory -Force -Path $parent | Out-Null
$temporary = "$targetPath.tmp.$PID"
$receipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $temporary -Encoding utf8
Move-Item -LiteralPath $temporary -Destination $targetPath
if ($sshExitCode -ne 0) { exit 2 }
