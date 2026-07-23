from __future__ import annotations

import base64
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
from typing import Any, Dict, Mapping

from .governance.immutable import write_once_bytes, write_once_json
from .provenance import sha256_file
from .workflow_common import WorkflowExecutionError


REMOTE_HOST = "xinxi-zhyh@211.87.115.228"
REMOTE_PROJECT_ROOT = "/data/home/xinxi-zhyh/xinxi-zhyh/projects/mining1"
STAGED_FILES = (
    "configs/environment_decision.lock.json",
    "env/remote-mining1-py39-cu121.lock.txt",
    "tools/preflight/verify_locked_environment.py",
    "tools/preflight/write_remote_environment_receipt.py",
    "slurm/setup_remote_environment.sbatch",
)


def _load_object(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise WorkflowExecutionError(f"Expected a JSON object: {path}")
    return payload


def _stage_archive(project_root: Path) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for relative in STAGED_FILES:
            source = project_root / relative
            if not source.is_file():
                raise WorkflowExecutionError(f"P030 staging input is missing: {relative}")
            archive.add(source, arcname=relative, recursive=False)
    return buffer.getvalue()


def _remote_submission_script(archive: bytes) -> str:
    encoded = base64.b64encode(archive).decode("ascii")
    return f"""set -euo pipefail
PROJECT_ROOT='{REMOTE_PROJECT_ROOT}'
mkdir -p "$PROJECT_ROOT/logs" "$PROJECT_ROOT/runs/slurm/P030"
ARCHIVE=$(mktemp)
cleanup() {{ rm -f "$ARCHIVE"; }}
trap cleanup EXIT TERM INT
base64 -d > "$ARCHIVE" <<'MINING1_P030_ARCHIVE'
{encoded}
MINING1_P030_ARCHIVE
if tar -tzf "$ARCHIVE" | grep -E '(^/|(^|/)\\.\\.(/|$))' >/dev/null; then
  printf '%s\n' 'unsafe archive member' >&2
  exit 2
fi
tar -xzf "$ARCHIVE" -C "$PROJECT_ROOT"
chmod 700 "$PROJECT_ROOT/slurm/setup_remote_environment.sbatch"
JOB_ID=$(cd "$PROJECT_ROOT" && sbatch --parsable slurm/setup_remote_environment.sbatch)
JOB_ID=${{JOB_ID%%;*}}
printf 'P030_JOB_ID=%s\n' "$JOB_ID"
while squeue -h -j "$JOB_ID" | grep -q .; do sleep 10; done
STATE=''
EXIT_CODE=''
for attempt in 1 2 3 4 5 6; do
  LINE=$(sacct -n -X -j "$JOB_ID" -o State,ExitCode | awk 'NF {{print $1"|"$2; exit}}')
  if test -n "$LINE"; then STATE=${{LINE%%|*}}; EXIT_CODE=${{LINE#*|}}; break; fi
  sleep 5
done
printf 'P030_STATE=%s\n' "$STATE"
printf 'P030_EXIT_CODE=%s\n' "$EXIT_CODE"
test "$STATE" = 'COMPLETED'
test "$EXIT_CODE" = '0:0'
RECEIPT="$PROJECT_ROOT/runs/slurm/P030/$JOB_ID/remote_environment_ready.compute.json"
test -f "$RECEIPT"
printf 'P030_RECEIPT_B64='
base64 -w0 "$RECEIPT"
printf '\n'
"""


def _run_remote_setup(script: str) -> subprocess.CompletedProcess[str]:
    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectionAttempts=1",
        "-o",
        "ConnectTimeout=8",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=4",
        REMOTE_HOST,
        "bash",
        "-s",
    ]
    completed = subprocess.run(
        command,
        input=script.replace("\r\n", "\n").encode("utf-8"),
        capture_output=True,
        check=False,
        timeout=7200,
    )
    return subprocess.CompletedProcess(
        completed.args,
        completed.returncode,
        stdout=completed.stdout.decode("utf-8", errors="replace"),
        stderr=completed.stderr.decode("utf-8", errors="replace"),
    )


def _parse_remote_result(stdout: str) -> tuple[str, Dict[str, Any], bytes]:
    fields = {}
    for line in stdout.splitlines():
        if line.startswith("P030_") and "=" in line:
            key, value = line.split("=", 1)
            fields[key] = value.strip()
    required = {"P030_JOB_ID", "P030_STATE", "P030_EXIT_CODE", "P030_RECEIPT_B64"}
    if required.difference(fields):
        raise WorkflowExecutionError("P030 remote result is incomplete")
    if (
        not fields["P030_JOB_ID"].isdigit()
        or fields["P030_STATE"] != "COMPLETED"
        or fields["P030_EXIT_CODE"] != "0:0"
    ):
        raise WorkflowExecutionError("P030 Slurm environment job did not complete cleanly")
    raw = base64.b64decode(fields["P030_RECEIPT_B64"], validate=True)
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise WorkflowExecutionError("P030 compute receipt is not a JSON object")
    return fields["P030_JOB_ID"], payload, raw


def _validate_compute_receipt(
    receipt: Mapping[str, Any], decision_path: Path, job_id: str
) -> None:
    required = {
        "schema_version",
        "scope",
        "status",
        "ready_at",
        "host",
        "decision_sha256",
        "strategy",
        "selected_python",
        "environment_specification_sha256",
        "package_inventory_sha256",
        "slurm_job_id",
        "compute_node",
    }
    if required.difference(receipt):
        raise WorkflowExecutionError("P030 compute receipt lacks required fields")
    if (
        receipt.get("schema_version") != 2
        or receipt.get("scope") != "remote"
        or receipt.get("status") != "pass"
        or receipt.get("host") != REMOTE_HOST
        or str(receipt.get("slurm_job_id")) != job_id
        or str(receipt.get("compute_node", "")).lower() == "mu01"
        or receipt.get("decision_sha256") != sha256_file(decision_path)
        or receipt.get("remote_closeout_reference") is not None
    ):
        raise WorkflowExecutionError("P030 compute receipt violates the remote-ready contract")


def _run_closeout(project_root: Path, job_id: str, action_at: str, output: Path) -> None:
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if powershell is None:
        raise WorkflowExecutionError("PowerShell is unavailable for account-wide closeout")
    script = project_root / "tools/remote/account_wide_closeout.ps1"
    result = subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            "-TriggerStepId",
            "P030",
            "-RemoteActionId",
            f"P030-{job_id}",
            "-RemoteActionAt",
            action_at,
            "-TriggerKind",
            "remote_action_end",
            "-OutputPath",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()[-4:]
        raise WorkflowExecutionError(f"P030 account-wide closeout failed: {' | '.join(detail)}")
    closeout = _load_object(output)
    if closeout.get("final_billing_state") not in {
        "verified_nonbilling",
        "verified_other_work_only",
    }:
        raise WorkflowExecutionError("P030 final billing state is not verified")


def apply_remote_environment(
    project_root: Path, row: Mapping[str, str], arguments: Mapping[str, Any]
) -> Dict[str, Any]:
    del row, arguments
    decision_path = project_root / "configs/environment_decision.lock.json"
    validation_path = (
        project_root / "evidence/preimplementation/environment_decision_validation.json"
    )
    decision = _load_object(decision_path)
    validation = _load_object(validation_path)
    if (
        decision.get("status") != "locked"
        or decision.get("remote_strategy") not in {
            "reuse_existing",
            "install_compatible_missing",
            "new_isolated",
        }
        or validation.get("status") != "pass"
        or validation.get("decision_sha256") != sha256_file(decision_path)
        or validation.get("approved_remote_host") != REMOTE_HOST
    ):
        raise WorkflowExecutionError("P030 environment decision validation drifted")
    archive = _stage_archive(project_root)
    action_at = datetime.now(timezone.utc).isoformat()
    remote = _run_remote_setup(_remote_submission_script(archive))
    if remote.returncode != 0:
        detail = (remote.stderr or remote.stdout).strip().splitlines()[-6:]
        raise WorkflowExecutionError(f"P030 remote setup failed: {' | '.join(detail)}")
    job_id, compute_receipt, raw_receipt = _parse_remote_result(remote.stdout)
    _validate_compute_receipt(compute_receipt, decision_path, job_id)
    compute_path = (
        project_root / f"evidence/preimplementation/P030_compute_{job_id}.json"
    )
    write_once_bytes(compute_path, raw_receipt)
    closeout_path = project_root / f"evidence/remote_closeout/P030-{job_id}.json"
    _run_closeout(project_root, job_id, action_at, closeout_path)
    final = dict(compute_receipt)
    final["remote_closeout_reference"] = closeout_path.relative_to(project_root).as_posix()
    output = project_root / "evidence/preimplementation/remote_environment_ready.json"
    write_once_json(output, final)
    return {
        "status": "pass",
        "output_paths": [
            output.relative_to(project_root).as_posix(),
            compute_path.relative_to(project_root).as_posix(),
            closeout_path.relative_to(project_root).as_posix(),
        ],
        "inputs": [
            {"path": decision_path.relative_to(project_root).as_posix(), "sha256": sha256_file(decision_path)},
            {"path": validation_path.relative_to(project_root).as_posix(), "sha256": sha256_file(validation_path)},
        ],
        "details": {
            "slurm_job_id": job_id,
            "compute_node": compute_receipt["compute_node"],
            "single_setup_ssh_session": True,
            "login_node_heavy_work": False,
        },
    }
