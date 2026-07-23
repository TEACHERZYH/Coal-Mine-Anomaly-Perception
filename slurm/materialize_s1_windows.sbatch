#!/bin/bash
#SBATCH --job-name=mining1_E059
#SBATCH --partition=cu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/data/home/xinxi-zhyh/xinxi-zhyh/logs/mining1/%x_%j.out
#SBATCH --error=/data/home/xinxi-zhyh/xinxi-zhyh/logs/mining1/%x_%j.err
#SBATCH --signal=B:TERM@120
set -euo pipefail
export MINING1_STEP_ID="E059"
export MINING1_RESOURCE_KIND="cpu"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export ARROW_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
PROJECT_ROOT="${MINING1_PROJECT_ROOT:-/data/home/xinxi-zhyh/xinxi-zhyh/projects/mining1}"
exec bash "$PROJECT_ROOT/slurm/run_locked_step.sh"
