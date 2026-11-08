#!/usr/bin/env bash
#SBATCH --job-name=trading-agent
#SBATCH --partition=academic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:A30:1
#SBATCH --time=12:00:00
#SBATCH --output=turing-%x-%j.out
#SBATCH --error=turing-%x-%j.err
set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[[ -x .venv/bin/python ]] || { echo 'Missing project .venv/bin/python' >&2; exit 1; }
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TQDM_DISABLE=1
# All paths and budgets live in project.json; arguments are the same as local runs.
exec .venv/bin/python -m agent "$@" --cuda
