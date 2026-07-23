param(
    [string]$RemoteHost = "xinxi-zhyh@211.87.115.228",
    [string]$OutputPath = "evidence/preimplementation/remote_environment.json"
)

$ErrorActionPreference = "Stop"
$parent = Split-Path -Parent $OutputPath
if ($parent) {
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
}

$remoteCommand = @'
set -eu
printf 'hostname='; hostname
printf 'captured_at='; date -Iseconds
printf 'kernel='; uname -srmo
printf 'project_root_exists='; test -d /data/home/xinxi-zhyh/xinxi-zhyh && printf 'yes\n' || printf 'no\n'
printf 'python3_path='; command -v python3 || true
printf 'python3_version='; python3 --version 2>&1 || true
conda_path=$(command -v conda 2>/dev/null || true)
printf 'conda_path=%s\n' "$conda_path"
printf 'conda_envs_begin\n'; conda env list 2>&1 | sed -n '1,40p' || true; printf 'conda_envs_end\n'
printf 'candidate_env_probe_begin\n'
if command -v conda >/dev/null 2>&1; then
  conda env list 2>/dev/null | awk 'NF && $1 !~ /^#/ {print $NF}' | sed -n '1,8p' | while IFS= read -r prefix; do
    if [ -x "$prefix/bin/python" ]; then
      "$prefix/bin/python" -c "import importlib,importlib.util,json,sys; names=['torch','torchvision','ultralytics','numpy','pandas','scipy','sklearn','yaml']; out={'executable':sys.executable,'prefix':sys.prefix,'python':sys.version.split()[0]}; [(out.update({n:getattr(importlib.import_module(n),'__version__','installed_version_unknown')})) if importlib.util.find_spec(n) else out.update({n:'missing'}) for n in names]; out['torch_cuda_runtime']=getattr(importlib.import_module('torch').version,'cuda',None) if importlib.util.find_spec('torch') else None; print(json.dumps(out,sort_keys=True))" 2>&1 || true
    fi
  done
fi
printf 'candidate_env_probe_end\n'
printf 'tool_versions_begin\n'
for tool in gcc g++ nvcc cmake; do
  printf '%s_path=' "$tool"; command -v "$tool" || true
  case "$tool" in
    *) "$tool" --version 2>&1 | sed -n '1,4p' || true ;;
  esac
done
printf 'tool_versions_end\n'
printf 'python_core_probe_begin\n'
python3 - <<'PY' 2>&1 || true
import importlib
import json
import sys
names = ['torch', 'torchvision', 'ultralytics', 'numpy', 'pandas', 'scipy', 'sklearn', 'yaml']
result = {'executable': sys.executable, 'version': sys.version}
for name in names:
    try:
        module = importlib.import_module(name)
        result[name] = getattr(module, '__version__', 'installed_version_unknown')
    except Exception as exc:
        result[name] = 'missing:' + type(exc).__name__
try:
    import torch
    result['torch_cuda_runtime'] = torch.version.cuda
except Exception:
    pass
print(json.dumps(result, sort_keys=True))
PY
printf 'python_core_probe_end\n'
printf 'module_list_begin\n'; module -t list 2>&1 || true; printf 'module_list_end\n'
printf 'sinfo_begin\n'; sinfo -h -o '%P|%a|%l|%D|%G' 2>&1 || true; printf 'sinfo_end\n'
printf 'squeue_begin\n'; squeue -h -u "$USER" -o '%i|%P|%j|%T|%M|%D|%R' 2>&1 || true; printf 'squeue_end\n'
printf 'active_job_count='; squeue -h -u "$USER" 2>/dev/null | wc -l || true
printf 'tmux_sessions_begin\n'; tmux ls 2>&1 | sed -n '1,40p' || true; printf 'tmux_sessions_end\n'
printf 'screen_sessions_begin\n'; screen -ls 2>&1 | sed -n '1,40p' || true; printf 'screen_sessions_end\n'
printf 'user_processes_begin\n'; ps -u "$USER" -o pid=,ppid=,etime=,comm=,args= --sort=-etime 2>&1 | sed -n '1,80p' || true; printf 'user_processes_end\n'
printf 'transfer_processes_begin\n'; ps -u "$USER" -o pid=,etime=,comm=,args= 2>/dev/null | grep -E '(^|[ /])(rsync|scp|sftp|rclone|wget|curl)([ ]|$)' | grep -v grep | sed -n '1,40p' || true; printf 'transfer_processes_end\n'
printf 'project_df_begin\n'; df -h /data/home/xinxi-zhyh/xinxi-zhyh 2>&1 || true; printf 'project_df_end\n'
printf 'project_root='; pwd
'@
$remoteCommand = (($remoteCommand -replace "`r`n", "`n") -replace "`r", "`n").TrimEnd() + "`n"

$job = Start-Job -ScriptBlock {
    param($HostName, $CommandText)
    if ($HostName -notmatch '^[A-Za-z0-9._-]+@[A-Za-z0-9.:-]+$') {
        throw "Unsupported remote host syntax: $HostName"
    }

    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName = "ssh"
    $startInfo.Arguments = "-o BatchMode=yes -o ConnectionAttempts=1 -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=1 $HostName bash -s"
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardInput = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $startInfo
    if (-not $process.Start()) {
        throw "Failed to start ssh process"
    }

    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()
    $encoding = New-Object System.Text.UTF8Encoding($false)
    $commandBytes = $encoding.GetBytes($CommandText)
    $process.StandardInput.BaseStream.Write($commandBytes, 0, $commandBytes.Length)
    $process.StandardInput.BaseStream.Flush()
    $process.StandardInput.Close()
    $process.WaitForExit()

    $stdout = $stdoutTask.Result
    $stderr = $stderrTask.Result
    $lines = @()
    if (-not [string]::IsNullOrEmpty($stdout)) {
        $lines += @($stdout -split "`r?`n" | Where-Object { $_ -ne "" })
    }
    if (-not [string]::IsNullOrEmpty($stderr)) {
        $lines += @($stderr -split "`r?`n" | Where-Object { $_ -ne "" } | ForEach-Object { "ssh_stderr: $_" })
    }
    [ordered]@{
        exit_code = $process.ExitCode
        raw_lines = @($lines | ForEach-Object { $_.ToString() })
    }
} -ArgumentList $RemoteHost, $remoteCommand

$completed = Wait-Job -Job $job -Timeout 90
if ($null -eq $completed) {
    Stop-Job -Job $job
    $exitCode = 124
    $raw = @("remote probe exceeded 90 second hard timeout")
} else {
    $result = Receive-Job -Job $job
    $exitCode = [int]$result.exit_code
    $raw = @($result.raw_lines)
}
Remove-Job -Job $job -Force
$payload = [ordered]@{
    schema_version = 1
    captured_at = (Get-Date).ToString("o")
    remote_host = $RemoteHost
    command_scope = "bounded_login_environment_slurm_and_account_inventory"
    transport_line_endings = "lf"
    transport_mode = "dotnet_process_utf8_lf_stdin"
    exit_code = $exitCode
    raw_lines = @($raw | ForEach-Object { $_.ToString() })
    created_compute_allocation = $false
}
$payload | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $OutputPath -Encoding UTF8

if ($exitCode -ne 0) {
    throw "Remote preflight failed with exit code $exitCode; evidence was retained at $OutputPath"
}

Write-Output $OutputPath
