#!/usr/bin/env bash
# CPU-only, read-only target and saved-validation-prediction audit.
# sbatch --export=ALL,PREPARED_ROOT=/path/to/prepared scripts/audit_turing.slurm
# Optionally set PREDICTIONS=/path/to/eval-sft/predictions.jsonl.
# Exit 2 means the artifact loaded but contains unsupported supervised targets.
#SBATCH --job-name=stock-audit
#SBATCH --partition=academic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=turing-%x-%j.out
#SBATCH --error=turing-%x-%j.err

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PREPARED_ROOT="${PREPARED_ROOT:?Set PREPARED_ROOT to the artifact to audit}"
cd "$PROJECT_ROOT"
[[ -x "$PROJECT_ROOT/.venv/bin/python" ]] || {
  echo "Missing project virtual environment" >&2
  exit 1
}

prediction_args=()
if [[ -n "${PREDICTIONS:-}" ]]; then
  prediction_args=(--predictions "$PREDICTIONS")
fi
"$PROJECT_ROOT/.venv/bin/python" -m agent.audit_cli \
  --prepared "$PREPARED_ROOT" "${prediction_args[@]}"
