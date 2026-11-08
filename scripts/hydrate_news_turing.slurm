#!/usr/bin/env bash
#SBATCH --job-name=raw-news-refresh
#SBATCH --partition=academic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=turing-%x-%j.out
#SBATCH --error=turing-%x-%j.err
set -Eeuo pipefail
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
SOURCE_ROOT="${SOURCE_ROOT:?Set SOURCE_ROOT to the existing complete source artifact}"
HYDRATED_ROOT="${HYDRATED_ROOT:?Set HYDRATED_ROOT to a new output directory}"
ENV_FILE="${ENV_FILE:-$PWD/.env}"
PYTHON="$PWD/.venv/bin/python"
[[ -x "$PYTHON" ]] || { echo 'Missing project .venv/bin/python' >&2; exit 1; }
export PYTHONUNBUFFERED=1
args=(--source "$SOURCE_ROOT" --output "$HYDRATED_ROOT" --env-file "$ENV_FILE")
[[ "${HYDRATE_RESUME:-0}" != 1 ]] || args+=(--resume)
if [[ -n "${FROZEN_RELEASE:-}" ]]; then
  args+=(--release "$FROZEN_RELEASE")
else
  args+=(--historical-sessions "${HISTORICAL_SESSIONS:-20}")
fi
"$PYTHON" -m agent.hydrate_news "${args[@]}" --plan-only
"$PYTHON" -m agent.hydrate_news "${args[@]}"
