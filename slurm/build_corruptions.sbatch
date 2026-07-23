#!/bin/bash
#SBATCH --job-name=mining1_E220
#SBATCH --partition=cu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/data/home/xinxi-zhyh/xinxi-zhyh/logs/mining1/%x_%j.out
#SBATCH --error=/data/home/xinxi-zhyh/xinxi-zhyh/logs/mining1/%x_%j.err
#SBATCH --signal=B:TERM@120
set -euo pipefail
export MINING1_STEP_ID="E220"
export MINING1_RESOURCE_KIND="cpu"
PROJECT_ROOT="${MINING1_PROJECT_ROOT:-/data/home/xinxi-zhyh/xinxi-zhyh/projects/mining1}"
exec bash "$PROJECT_ROOT/slurm/run_locked_step.sh"
