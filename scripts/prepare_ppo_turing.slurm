#!/usr/bin/env bash
#SBATCH --job-name=prepare-nlp-ppo
#SBATCH --partition=academic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=turing-%x-%j.out
#SBATCH --error=turing-%x-%j.err
set -Eeuo pipefail
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PREPARED_ROOT="${PREPARED_ROOT:?Set PREPARED_ROOT to hydrated train/validation source}"
TRAJECTORY_ROOT="${TRAJECTORY_ROOT:?Set TRAJECTORY_ROOT to new actual GOOD dataset}"
PPO_PREPARED_ROOT="${PPO_PREPARED_ROOT:?Set PPO_PREPARED_ROOT to a new output directory}"
export PYTHONUNBUFFERED=1
"$PWD/.venv/bin/python" -m agent.training.ppo_prepare \
  --prepared "$PREPARED_ROOT" --sft-dataset "$TRAJECTORY_ROOT/dataset" \
  --output "$PPO_PREPARED_ROOT" --sessions 60 --max-length "${MAX_LENGTH:-4096}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-512}"
