from __future__ import annotations

from pathlib import Path
import re
from typing import Mapping, Union

from .immutable import GovernanceContractError


PathLike = Union[str, Path]
ALLOWED_PARTITIONS = {"cu", "fat", "3090", "A100"}
DAMAGED_REMOTE_HOST = "180.209.128.66"
DAMAGED_REMOTE_USER_PREFIX = "zhangl@"
DAMAGED_REMOTE_LOGIN = DAMAGED_REMOTE_USER_PREFIX + DAMAGED_REMOTE_HOST
REQUIRED_DIRECTIVES = {
    "job-name",
    "partition",
    "nodes",
    "cpus-per-task",
    "mem",
    "time",
    "output",
    "error",
    "signal",
}
BARE_PYTHON = re.compile(
    r"(?m)(?:^\s*|[;&|]\s*|\bexec\s+|\bsrun\s+)"
    r"(?:python|python3|/usr/bin/env\s+python3?)(?:\s|$)"
)


def parse_sbatch_directives(path: PathLike) -> dict[str, str]:
    directives: dict[str, str] = {}
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped.startswith("#SBATCH "):
            continue
        token = stripped[len("#SBATCH ") :].strip()
        if not token.startswith("--") or "=" not in token:
            raise GovernanceContractError("Slurm directives must use --name=value")
        name, value = token[2:].split("=", 1)
        if name in directives or not name or not value:
            raise GovernanceContractError("Slurm directive is duplicate or empty")
        directives[name] = value
    return directives


def validate_sbatch_script(path: PathLike, *, resource_kind: str) -> Mapping[str, str]:
    script_path = Path(path)
    content = script_path.read_text(encoding="utf-8")
    directives = parse_sbatch_directives(script_path)
    missing = sorted(REQUIRED_DIRECTIVES.difference(directives))
    if missing:
        raise GovernanceContractError(f"Slurm directives are missing: {missing}")
    if directives["partition"] not in ALLOWED_PARTITIONS:
        raise GovernanceContractError("Slurm partition is not allowed")
    if not content.startswith("#!/bin/bash\n"):
        raise GovernanceContractError("Slurm script must use the Bash interpreter")
    if re.fullmatch(r"[-A-Za-z0-9_.]+", directives["job-name"]) is None:
        raise GovernanceContractError("Slurm job name is invalid")
    if directives["nodes"] != "1":
        raise GovernanceContractError("Slurm jobs must request exactly one node")
    if not directives["cpus-per-task"].isdigit() or int(
        directives["cpus-per-task"]
    ) <= 0:
        raise GovernanceContractError("Slurm CPU request must be positive")
    memory_match = re.fullmatch(r"(\d+)([KMGTP]?)", directives["mem"])
    if memory_match is None or int(memory_match.group(1)) <= 0:
        raise GovernanceContractError("Slurm memory request is invalid")
    time_match = re.fullmatch(r"(?:(\d+)-)?(\d+):(\d{2}):(\d{2})", directives["time"])
    if (
        time_match is None
        or int(time_match.group(3)) >= 60
        or int(time_match.group(4)) >= 60
        or all(int(value or 0) == 0 for value in time_match.groups())
    ):
        raise GovernanceContractError("Slurm time limit is invalid")
    signal_match = re.fullmatch(r"B:TERM@(\d+)", directives["signal"])
    if signal_match is None or int(signal_match.group(1)) < 60:
        raise GovernanceContractError("Slurm SIGTERM warning directive is invalid")
    for field in ("output", "error"):
        value = directives[field]
        if not value.startswith("/data/home/xinxi-zhyh/xinxi-zhyh/logs/mining1/"):
            raise GovernanceContractError("Slurm log path is outside the project account")
        if "%x" not in value or not any(token in value for token in ("%j", "%A")):
            raise GovernanceContractError("Slurm log path lacks stable job placeholders")
    if directives["output"] == directives["error"]:
        raise GovernanceContractError("Slurm stdout and stderr paths must differ")
    if resource_kind not in {"cpu", "gpu"}:
        raise GovernanceContractError("Slurm resource kind is invalid")
    if resource_kind == "gpu":
        if directives.get("gres") not in {"gpu:1", "gpu:2"} or directives["partition"] not in {"3090", "A100"}:
            raise GovernanceContractError("GPU script lacks a supported GPU request")
        if directives["partition"] not in {"3090", "A100"}:
            raise GovernanceContractError("GPU script uses a non-GPU partition")
    elif "gres" in directives:
        raise GovernanceContractError("CPU script must not request a GPU")
    elif directives["partition"] not in {"cu", "fat"}:
        raise GovernanceContractError("CPU script uses a GPU partition")
    if "array" in directives and re.fullmatch(
        r"\d+-\d+(?::\d+)?(?:%\d+)?", directives["array"]
    ) is None:
        raise GovernanceContractError("Slurm array directive is invalid")
    if "set -euo pipefail" not in content:
        raise GovernanceContractError("Slurm script lacks strict shell failure handling")
    if 'MINING1_RESOURCE_KIND="' + resource_kind + '"' not in content:
        raise GovernanceContractError("Slurm script resource declaration drifted")
    if "MINING1_STEP_ID=" not in content or "slurm/run_locked_step.sh" not in content:
        raise GovernanceContractError("Slurm script lacks the locked step runner")
    if 'exec bash "$PROJECT_ROOT/slurm/run_locked_step.sh"' not in content:
        raise GovernanceContractError("Slurm script does not propagate runner exit status")
    if BARE_PYTHON.search(content):
        raise GovernanceContractError("Slurm script invokes bare Python")
    if DAMAGED_REMOTE_HOST in content or DAMAGED_REMOTE_USER_PREFIX in content:
        raise GovernanceContractError("Slurm script references the damaged host")
    if re.search(r"(?m)^\s*ssh\s", content):
        raise GovernanceContractError("compute script must not open nested SSH")
    return directives


def validate_locked_step_runner(path: PathLike) -> None:
    content = Path(path).read_text(encoding="utf-8")
    required_tokens = {
        "environment_decision.lock.json",
        "selected_remote_python",
        "verify_locked_environment.py",
        "slurm_environment_probe.json",
        "framework_version",
        "cuda_runtime",
        "environment_fingerprint_sha256",
        'cd "$PROJECT_ROOT"',
        'export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"',
        "-m mining1_exp.cli execute-step",
        '"$REMOTE_PYTHON"',
        "trap handle_termination TERM INT",
        "wait \"$CHILD_PID\"",
        "exit \"$STATUS\"",
    }
    missing = sorted(token for token in required_tokens if token not in content)
    if missing:
        raise GovernanceContractError(f"locked Slurm runner tokens are missing: {missing}")
    if BARE_PYTHON.search(content):
        raise GovernanceContractError("locked Slurm runner invokes bare Python")
    if "hostname" not in content or "SLURM_JOB_ID" not in content:
        raise GovernanceContractError("locked Slurm runner lacks compute-job evidence")
