#!/usr/bin/env bash
#SBATCH --job-name=phasemoe-midpoint
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=phasemoe-%j.log

# Template only: select your actual partition/account/GPU constraint at submission.
# Activate the project's CUDA Python environment before sbatch, or adapt this file
# to the cluster's module/conda policy. No cluster configuration is assumed here.
set -euo pipefail
bash scripts/run_midpoint.sh "runs/slurm-${SLURM_JOB_ID}"
