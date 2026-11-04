#!/usr/bin/env bash
# CPU-only offline evidence compaction job.
# Submit with:
#   sbatch --export=ALL,INPUT_ROOT=$HOME/llm_trading_agent/artifacts/mixed-2026-test-v3,OUTPUT_ROOT=$HOME/llm_trading_agent/artifacts/mixed-2026-test-v3-compact scripts/compact_turing.slurm

#SBATCH --job-name=stock-compact
#SBATCH --partition=academic
##SBATCH --account=REPLACE_WITH_YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=turing-%x-%j.out
#SBATCH --error=turing-%x-%j.err

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
INPUT_ROOT="${INPUT_ROOT:?Set INPUT_ROOT to an existing prepared artifact}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT to a new compacted-artifact directory}"

cd "$PROJECT_ROOT"

[[ -d "$INPUT_ROOT" ]] || {
  echo "Input artifact does not exist: $INPUT_ROOT" >&2
  exit 1
}
[[ ! -e "$OUTPUT_ROOT" ]] || {
  echo "Refusing to overwrite existing output: $OUTPUT_ROOT" >&2
  exit 1
}
[[ -x "$PROJECT_ROOT/.venv/bin/python" ]] || {
  echo "Missing virtual environment: $PROJECT_ROOT/.venv/bin/python" >&2
  exit 1
}

echo "Host: $(hostname)"
echo "Job: ${SLURM_JOB_ID:-local}"
echo "Input: $INPUT_ROOT"
echo "Output: $OUTPUT_ROOT"

"$PROJECT_ROOT/.venv/bin/python" -m agent.compact_cli \
  --input "$INPUT_ROOT" \
  --output "$OUTPUT_ROOT" \
  --max-news "${MAX_NEWS:-8}" \
  --max-filings "${MAX_FILINGS:-3}" \
  --max-text-chars "${MAX_TEXT_CHARS:-1200}"

echo "Offline compaction completed."
