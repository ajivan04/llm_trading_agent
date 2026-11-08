#!/usr/bin/env bash
#SBATCH --job-name=nominal-price-source
#SBATCH --partition=academic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=turing-%x-%j.out
#SBATCH --error=turing-%x-%j.err
set -Eeuo pipefail
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
SOURCE_ROOT="${SOURCE_ROOT:?Set SOURCE_ROOT to the existing archived source}"
NOMINAL_ROOT="${NOMINAL_ROOT:?Set NOMINAL_ROOT to a new nonnested directory}"
export PYTHONUNBUFFERED=1
"$PWD/.venv/bin/python" -m agent.normalize_prices --source "$SOURCE_ROOT" --output "$NOMINAL_ROOT"
