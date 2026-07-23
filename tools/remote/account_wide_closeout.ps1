[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$TriggerStepId,
    [Parameter(Mandatory = $true)]
    [string]$RemoteActionId,
    [Parameter(Mandatory = $true)]
    [DateTimeOffset]$RemoteActionAt,
    [ValidateSet('remote_action_end', 'connection_check_end', 'final_reconcile')]
    [string]$TriggerKind = 'remote_action_end',
    [string]$Target = 'xinxi-zhyh@211.87.115.228',
    [string[]]$CancelJobIds = @(),
    [hashtable]$RetainedResourcesWithReason = @{},
    [string]$CurrentProjectJobPattern = '(?i)mining1',
    [string[]]$UnsynchronizedArtifacts = @(),
    [Parameter(Mandatory = $true)]
    [string]$OutputPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if ($Target -ne 'xinxi-zhyh@211.87.115.228') {
    throw 'Only the current Slurm host is permitted.'
}
if ($RemoteActionId -notmatch '^[-A-Za-z0-9._:]+$') {
    throw 'RemoteActionId contains unsafe characters.'
}
foreach ($jobId in $CancelJobIds) {
    if ($jobId -notmatch '^\d+(?:_\d+)?$') { throw "Invalid Slurm job ID: $jobId" }
}
if ([string]::IsNullOrWhiteSpace($CurrentProjectJobPattern)) {
    throw 'CurrentProjectJobPattern must be a non-empty regular expression.'
}
if ($CancelJobIds.Count -gt 0 -and $UnsynchronizedArtifacts.Count -gt 0) {
    throw 'Critical artifacts must be synchronized before resource cancellation.'
}

$auditScript = @'
set -euo pipefail
printf '%s\n' '__SQUEUE__'
squeue -u "$USER" -h -o '%i|%P|%j|%T|%M|%D|%R'
printf '%s\n' '__SACCT__'
sacct -n -X -u "$USER" -S now-2days -o JobIDRaw,Partition,State,Elapsed,ExitCode,NodeList | tail -n 100
printf '%s\n' '__SESSIONS__'
tmux list-sessions 2>/dev/null || true
screen -ls 2>/dev/null || true
printf '%s\n' '__PROCESSES__'
ps -u "$USER" -o pid=,ppid=,etimes=,cmd= | grep -E '[m]ining1|[t]orchrun|[r]sync|[s]cp|[r]clone|[m]onitor_jobs' || true
printf '%s\n' '__END__'
'@
$auditScript = $auditScript -replace "`r`n", "`n"
$initial = & ssh -o BatchMode=yes -o ConnectTimeout=8 $Target $auditScript 2>&1
if ($LASTEXITCODE -ne 0) {
    $initialError = (($initial | Select-Object -Last 8) -join "`n")
    throw "Account-wide closeout audit failed: $initialError"
}

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

$retainedReasons = @{}
foreach ($entry in $RetainedResourcesWithReason.GetEnumerator()) {
    $retainedReasons[$entry.Key] = $entry.Value
}
$initialSqueueLines = @(Get-MarkedSection $initial '__SQUEUE__' '__SACCT__')
foreach ($line in $initialSqueueLines) {
    if ($line -notmatch '^\d+(?:_\d+)?\|') { continue }
    $fields = $line -split '\|', 7
    $jobId = $fields[0]
    $jobName = $fields[2]
    if ($jobName -notmatch $CurrentProjectJobPattern -and -not $retainedReasons.ContainsKey($jobId)) {
        $retainedReasons[$jobId] = "active job for another named project ($jobName); retained by account-wide closeout"
    }
}

$actions = @()
if ($CancelJobIds.Count -gt 0) {
    $jobList = $CancelJobIds -join ' '
    $cancelScript = "set -euo pipefail; scancel --signal=TERM $jobList"
    $cancelOutput = & ssh -o BatchMode=yes -o ConnectTimeout=8 $Target $cancelScript 2>&1
    if ($LASTEXITCODE -ne 0) { throw "Slurm cancellation failed: $cancelOutput" }
    $actions += "scancel --signal=TERM $jobList"
}

$final = & ssh -o BatchMode=yes -o ConnectTimeout=8 $Target $auditScript 2>&1
if ($LASTEXITCODE -ne 0) { throw 'Final account-wide billing verification failed.' }
$squeueLines = @(Get-MarkedSection $final '__SQUEUE__' '__SACCT__')
$sessionLines = @(Get-MarkedSection $final '__SESSIONS__' '__PROCESSES__')
$processLines = @(Get-MarkedSection $final '__PROCESSES__' '__END__')
$activeJobs = @($squeueLines | Where-Object { $_ -match '^\d+(?:_\d+)?\|' })
$relevantProcesses = @($processLines | Where-Object { $_ -match 'mining1|torchrun' })
$transfersAndMonitors = @($processLines | Where-Object { $_ -match 'rsync|scp|rclone|monitor_jobs' })
$activeSessions = @($sessionLines | Where-Object {
    $_ -match ':\s+\d+\s+windows' -or $_ -match '^\s*\d+\.[^\s]+'
})
$retained = @($retainedReasons.GetEnumerator() | ForEach-Object {
    [ordered]@{ resource = $_.Key; reason = $_.Value }
})
$billingState = 'unverified'
if (
    $UnsynchronizedArtifacts.Count -eq 0 -and
    $activeJobs.Count -eq 0 -and
    $activeSessions.Count -eq 0 -and
    $relevantProcesses.Count -eq 0 -and
    $transfersAndMonitors.Count -eq 0
) {
    $billingState = 'verified_nonbilling'
}
elseif ($UnsynchronizedArtifacts.Count -eq 0 -and $retained.Count -gt 0) {
    $unexplained = @($activeJobs | Where-Object {
        $jobId = ($_ -split '\|')[0]
        -not $retainedReasons.ContainsKey($jobId)
    })
    $nonJobResourcesExplained = (
        ($activeSessions.Count -eq 0 -or $retainedReasons.ContainsKey('tmux_and_screen_sessions')) -and
        ($relevantProcesses.Count -eq 0 -or $retainedReasons.ContainsKey('relevant_processes')) -and
        ($transfersAndMonitors.Count -eq 0 -or $retainedReasons.ContainsKey('transfers_and_monitors'))
    )
    if ($unexplained.Count -eq 0 -and $nonJobResourcesExplained) {
        $billingState = 'verified_other_work_only'
    }
}

$receipt = [ordered]@{
    trigger_step_id = $TriggerStepId
    remote_action_id = $RemoteActionId
    remote_action_at = $RemoteActionAt.ToUniversalTime().ToString('o')
    trigger_kind = $TriggerKind
    closed_at = [DateTimeOffset]::UtcNow.ToString('o')
    host = $Target
    account_scope = 'all_user_owned_work'
    current_project_job_pattern = $CurrentProjectJobPattern
    slurm_jobs_and_allocations = $activeJobs
    interactive_sessions = @()
    tmux_and_screen_sessions = $activeSessions
    relevant_processes = $relevantProcesses
    transfers_and_monitors = $transfersAndMonitors
    unsynchronized_artifacts = @($UnsynchronizedArtifacts)
    retained_resources_with_reason = $retained
    cancellation_or_release_actions = $actions
    provider_stop_action = if ($billingState -eq 'verified_nonbilling') { 'not_applicable_shared_slurm_host_no_active_allocation' } elseif ($billingState -eq 'verified_other_work_only') { 'not_stopped_other_user_work_retained' } else { $null }
    final_billing_state = $billingState
    verification_note = 'Logging out or closing SSH is not billing-stop evidence; final state comes from the second account-wide Slurm audit.'
}
$targetPath = [System.IO.Path]::GetFullPath($OutputPath)
if (Test-Path -LiteralPath $targetPath) {
    throw "Closeout receipt is immutable: $targetPath"
}
$parent = Split-Path -Parent $targetPath
New-Item -ItemType Directory -Force -Path $parent | Out-Null
$temporary = "$targetPath.tmp.$PID"
$receipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $temporary -Encoding utf8
Move-Item -LiteralPath $temporary -Destination $targetPath
if ($billingState -eq 'unverified') { exit 3 }
