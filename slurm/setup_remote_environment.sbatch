#!/bin/bash
#SBATCH --job-name=mining1_P030_env
#SBATCH --partition=cu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

: "${SLURM_JOB_ID:?P030 environment setup must run through Slurm}"
PROJECT_ROOT="${MINING1_PROJECT_ROOT:-/data/home/xinxi-zhyh/xinxi-zhyh/projects/mining1}"
ACCOUNT_ROOT="/data/home/xinxi-zhyh/xinxi-zhyh"
DECISION="$PROJECT_ROOT/configs/environment_decision.lock.json"
LOCK_FILE="$PROJECT_ROOT/env/remote-mining1-py39-cu121.lock.txt"
WHEELHOUSE="$ACCOUNT_ROOT/cache/mining1/wheelhouse-cu121"
TARGET_ENV="$ACCOUNT_ROOT/envs/mining1-py39-cu121"
RUN_ROOT="$PROJECT_ROOT/runs/slurm/P030/$SLURM_JOB_ID"
COMPUTE_NODE="$(hostname)"

test "$COMPUTE_NODE" != "mu01"
test -f "$DECISION"
test -f "$LOCK_FILE"
test -f "$WHEELHOUSE/SHA256SUMS"
mkdir -p "$RUN_ROOT"
(cd "$WHEELHOUSE" && sha256sum -c SHA256SUMS)

SETUP_EXECUTED=false
STAGE_ENV="$ACCOUNT_ROOT/envs/.mining1-py39-cu121.$SLURM_JOB_ID.stage"
cleanup() {
  if test -d "$STAGE_ENV"; then rm -rf "$STAGE_ENV"; fi
}
trap cleanup EXIT TERM INT

if ! test -x "$TARGET_ENV/bin/python"; then
  test ! -e "$TARGET_ENV"
  test ! -e "$STAGE_ENV"
  python3 -m venv "$STAGE_ENV"
  "$STAGE_ENV/bin/python" -m pip install \
    --disable-pip-version-check --no-index --find-links "$WHEELHOUSE" \
    -r "$LOCK_FILE" > "$RUN_ROOT/setup.log" 2>&1
  "$STAGE_ENV/bin/python" -m pip check >> "$RUN_ROOT/setup.log" 2>&1
  mv "$STAGE_ENV" "$TARGET_ENV"
  SETUP_EXECUTED=true
else
  printf '%s\n' 'existing locked environment selected for verification' > "$RUN_ROOT/setup.log"
fi

"$TARGET_ENV/bin/python" "$PROJECT_ROOT/tools/preflight/verify_locked_environment.py" \
  --scope remote --decision "$DECISION" > "$RUN_ROOT/verification.json"
"$TARGET_ENV/bin/python" "$PROJECT_ROOT/tools/preflight/write_remote_environment_receipt.py" \
  --decision "$DECISION" \
  --verification "$RUN_ROOT/verification.json" \
  --setup-executed "$SETUP_EXECUTED" \
  --output "$RUN_ROOT/remote_environment_ready.compute.json"

printf '%s\n' '0' > "$RUN_ROOT/exit_code.txt"
