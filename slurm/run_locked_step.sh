#!/bin/bash
set -euo pipefail

: "${MINING1_STEP_ID:?MINING1_STEP_ID is required}"
: "${MINING1_RESOURCE_KIND:?MINING1_RESOURCE_KIND is required}"
: "${SLURM_JOB_ID:?This runner must execute inside a Slurm job}"

PROJECT_ROOT="${MINING1_PROJECT_ROOT:-/data/home/xinxi-zhyh/xinxi-zhyh/projects/mining1}"
DECISION_PATH="${MINING1_ENV_DECISION:-$PROJECT_ROOT/configs/environment_decision.lock.json}"
REMOTE_PYTHON="${MINING1_SELECTED_REMOTE_PYTHON:-/data/home/xinxi-zhyh/xinxi-zhyh/envs/mining1-py39-cu121/bin/python}"
RUN_ROOT="${MINING1_RUN_ROOT:-$PROJECT_ROOT/runs/slurm/$MINING1_STEP_ID/$SLURM_JOB_ID}"

test -f "$DECISION_PATH"
test -x "$REMOTE_PYTHON"
LOCKED_REMOTE_PYTHON="$("$REMOTE_PYTHON" - "$DECISION_PATH" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8-sig"))
decision = payload.get("decision", payload)
print(decision["selected_remote_python"])
PY
)"
test "$REMOTE_PYTHON" = "$LOCKED_REMOTE_PYTHON"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$RUN_ROOT"
test ! -e "$RUN_ROOT/environment_verification.log"
"$REMOTE_PYTHON" "$PROJECT_ROOT/tools/preflight/verify_locked_environment.py" \
  --scope remote --decision "$DECISION_PATH" | tee "$RUN_ROOT/environment_verification.log"
"$REMOTE_PYTHON" - "$RUN_ROOT/slurm_environment_probe.json" \
  "$RUN_ROOT/environment_verification.log" "$DECISION_PATH" <<'PY'
import hashlib
import json
import os
import pathlib
import platform
import sys

verification = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
if verification.get("status") != "pass":
    raise RuntimeError("locked environment verification did not pass")
packages = verification.get("packages", {})
canonical_verification = json.dumps(
    verification, sort_keys=True, separators=(",", ":")
).encode("utf-8")
canonical_packages = json.dumps(
    packages, sort_keys=True, separators=(",", ":")
).encode("utf-8")
payload = {
    "selected_remote_python": sys.executable,
    "python_version": platform.python_version(),
    "framework_version": packages.get("torch", "not_applicable"),
    "cuda_runtime": verification.get("torch_cuda_runtime") or "not_applicable",
    "environment_fingerprint_sha256": hashlib.sha256(canonical_verification).hexdigest(),
    "package_inventory_sha256": hashlib.sha256(canonical_packages).hexdigest(),
    "environment_decision_sha256": hashlib.sha256(
        pathlib.Path(sys.argv[3]).read_bytes()
    ).hexdigest(),
    "compute_node": platform.node(),
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "allocation_id": os.environ.get("SLURM_ARRAY_JOB_ID", os.environ["SLURM_JOB_ID"]),
    "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    "resource_kind": os.environ["MINING1_RESOURCE_KIND"],
}
target = pathlib.Path(sys.argv[1])
temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
temporary.write_text(
    json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
)
try:
    os.link(temporary, target)
except FileExistsError:
    raise FileExistsError(f"immutable environment receipt exists: {target}")
finally:
    temporary.unlink(missing_ok=True)
PY
test ! -e "$RUN_ROOT/compute_node.txt"
hostname > "$RUN_ROOT/compute_node.txt"

CHILD_PID=""
handle_termination() {
  printf '%s\n' "termination_requested" > "$RUN_ROOT/termination_requested.txt"
  if test -n "$CHILD_PID" && kill -0 "$CHILD_PID" 2>/dev/null; then
    kill -TERM "$CHILD_PID"
    wait "$CHILD_PID" || true
  fi
  exit 143
}
trap handle_termination TERM INT

ARGS=(-m mining1_exp.cli execute-step --step "$MINING1_STEP_ID" --run-root "$RUN_ROOT")
if test -n "${MINING1_STEP_CONFIG:-}"; then
  ARGS+=(--config "$MINING1_STEP_CONFIG")
fi
"$REMOTE_PYTHON" "${ARGS[@]}" &
CHILD_PID=$!
set +e
wait "$CHILD_PID"
STATUS=$?
set -e
printf '%s\n' "$STATUS" > "$RUN_ROOT/exit_code.txt"
exit "$STATUS"
