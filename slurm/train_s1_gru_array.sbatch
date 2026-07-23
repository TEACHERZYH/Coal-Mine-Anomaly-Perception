#!/bin/bash
#SBATCH --job-name=mining1_E122
#SBATCH --partition=3090
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-2%2
#SBATCH --output=/data/home/xinxi-zhyh/xinxi-zhyh/logs/mining1/%x_%A_%a.out
#SBATCH --error=/data/home/xinxi-zhyh/xinxi-zhyh/logs/mining1/%x_%A_%a.err
#SBATCH --signal=B:TERM@120
set -euo pipefail
export MINING1_STEP_ID="E122"
export MINING1_RESOURCE_KIND="gpu"
PROJECT_ROOT="${MINING1_PROJECT_ROOT:-/data/home/xinxi-zhyh/xinxi-zhyh/projects/mining1}"
exec bash "$PROJECT_ROOT/slurm/run_locked_step.sh"
