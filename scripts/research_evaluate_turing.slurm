#!/usr/bin/env bash
# Same 20-session validation plan, default two-session prefix; no held-out calls.
#SBATCH --job-name=research-eval
#SBATCH --partition=academic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:A30:1
#SBATCH --time=06:00:00
#SBATCH --output=turing-%x-%j.out
#SBATCH --error=turing-%x-%j.err

set -Eeuo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PREPARED_ROOT="${PREPARED_ROOT:?Set PREPARED_ROOT to complete NLP evidence artifact}"
EVALUATION_ROOT="${EVALUATION_ROOT:?Set EVALUATION_ROOT to a separate candidate evaluation directory}"
cd "$PROJECT_ROOT"
PYTHON="$PROJECT_ROOT/.venv/bin/python"
[[ -x "$PYTHON" ]] || { echo "Missing project .venv/bin/python" >&2; exit 1; }
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TQDM_DISABLE=1
args=(--prepared "$PREPARED_ROOT" --output "$EVALUATION_ROOT"
  --sessions "${EVAL_SESSIONS:-20}" --max-sessions "${EVAL_MAX_SESSIONS:-2}"
  --max-context-tokens "${MAX_LENGTH:-4096}"
  --max-new-tokens "${MAX_NEW_TOKENS:-512}" --max-turns "${MAX_TURNS:-6}")
[[ -z "${POLICY_ADAPTER:-}" ]] || args+=(--adapter "$POLICY_ADAPTER")
[[ "${EVAL_RESUME:-0}" != 1 ]] || args+=(--resume)
"$PYTHON" -m agent.research_eval_cli "${args[@]}" --plan-only
"$PYTHON" -m agent.research_eval_cli "${args[@]}" --require-cuda
