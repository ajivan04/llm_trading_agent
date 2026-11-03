#!/usr/bin/env bash
# Validation-only evaluation job for the frozen training artifacts.
# Submit from the repository root:
#   sbatch scripts/evaluate_turing.slurm

# Turing academic currently provides A30 GPUs; verify local availability with
# `scluster`/`idle_gpus` before changing these requests.
#SBATCH --job-name=stock-eval
#SBATCH --partition=academic
##SBATCH --account=REPLACE_WITH_YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:A30:1
#SBATCH --time=1-00:00:00
#SBATCH --output=turing-%x-%j.out
#SBATCH --error=turing-%x-%j.err

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PREPARED_ROOT="${PREPARED_ROOT:-$PROJECT_ROOT/artifacts/midterm-training}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"

cd "$PROJECT_ROOT"

if [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi
if ! command -v uv >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
  module load uv
fi
command -v uv >/dev/null 2>&1 || {
  echo "uv is unavailable; run 'module load uv' and resubmit." >&2
  exit 1
}

[[ -f "$PREPARED_ROOT/dataset/validation.jsonl" ]] || {
  echo "Missing validation dataset: $PREPARED_ROOT/dataset/validation.jsonl" >&2
  exit 1
}

echo "Host: $(hostname)"
echo "Job: ${SLURM_JOB_ID:-local}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

run_eval() {
  local name="$1"
  shift || true
  local output="$PREPARED_ROOT/eval-$name"
  if [[ -f "$output/evaluation.json" ]]; then
    echo "$name evaluation already exists; skipping."
    return 0
  fi
  echo "Evaluating $name on validation..."
  uv run stock-agent-evaluate \
    --prepared "$PREPARED_ROOT" \
    --split validation \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --output "$output" \
    "$@"
}

run_portfolio() {
  local name="$1"
  local predictions="$PREPARED_ROOT/eval-$name/predictions.jsonl"
  local output="$PREPARED_ROOT/portfolio-$name"
  [[ -f "$predictions" ]] || { echo "Missing predictions: $predictions" >&2; exit 1; }
  if [[ -f "$output/portfolio_metrics.json" ]]; then
    echo "$name portfolio replay already exists; skipping."
    return 0
  fi
  uv run stock-agent-portfolio-evaluate \
    --prepared "$PREPARED_ROOT" \
    --predictions "$predictions" \
    --split validation \
    --output "$output" \
    --initial-cash 100000 \
    --cost-bps 5 \
    --max-qty 50
}

run_eval base
run_portfolio base
run_eval sft --adapter "$PREPARED_ROOT/sft"
run_portfolio sft
run_eval dpo --adapter "$PREPARED_ROOT/dpo"
run_portfolio dpo
run_eval ppo --adapter "$PREPARED_ROOT/ppo/adapter"
run_portfolio ppo

echo "Validation evaluations completed under: $PREPARED_ROOT"
