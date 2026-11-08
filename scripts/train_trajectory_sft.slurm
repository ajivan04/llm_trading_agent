#!/usr/bin/env bash
# One fixed SFT plan: smoke stops at2 steps, then resumes the SAME optimizer schedule.
#SBATCH --job-name=trajectory-sft
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
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
TRAJECTORY_ROOT="${TRAJECTORY_ROOT:?Set TRAJECTORY_ROOT to the exported GOOD trajectory artifact}"
RUN_MODE="${RUN_MODE:-preflight}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$TRAJECTORY_ROOT/sft-assignment}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
cd "$PROJECT_ROOT"
PYTHON="$PROJECT_ROOT/.venv/bin/python"
[[ -x "$PYTHON" ]] || { echo "Missing project .venv/bin/python" >&2; exit 1; }
[[ -f "$TRAJECTORY_ROOT/trajectory-manifest.json" ]] || {
  echo "Missing actual GOOD trajectory manifest" >&2; exit 1;
}
case "$RUN_MODE" in
  preflight) STEPS="${SFT_STEPS:-1}" ;;
  smoke|full)
    STEPS="${SFT_STEPS:?Choose the fixed total SFT_STEPS after reviewing corpus-size preflight}"
    ;;
  *) echo "RUN_MODE must be preflight, smoke, or full" >&2; exit 2 ;;
esac
[[ "$STEPS" =~ ^[1-9][0-9]*$ ]] || { echo "SFT_STEPS must be positive" >&2; exit 2; }
if [[ "$RUN_MODE" == full && "${REVIEWED_SMOKE:-0}" != 1 ]]; then
  echo "Review the same-plan smoke first; set REVIEWED_SMOKE=1 afterward." >&2; exit 2
fi
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TQDM_DISABLE=1
args=(sft --require-cuda --train "$TRAJECTORY_ROOT/dataset/train.jsonl"
  --validation "$TRAJECTORY_ROOT/dataset/validation.jsonl"
  --output "$OUTPUT_ROOT" --prompt-version bounded-tool-research-v1
  --supervision trajectory-final --max-length "$MAX_LENGTH" --max-steps "$STEPS"
  --train-batch-size 1 --eval-batch-size 1 --gradient-accumulation-steps 8
  --save-steps 2 --logging-steps 1)
if [[ -n "${SFT_INITIAL_ADAPTER:-}" ]]; then
  args+=(--adapter-checkpoint "$SFT_INITIAL_ADAPTER")
fi
"$PYTHON" -m agent.training.cli "${args[@]}" --preflight-only
[[ "$RUN_MODE" != preflight ]] || exit 0
if [[ -f "$OUTPUT_ROOT/training_manifest.json" ]]; then
  echo "This fixed SFT plan already completed; evaluate its adapter instead of retraining."
  exit 0
fi
if [[ -n "${RESUME_FROM:-}" ]]; then
  [[ -d "$RESUME_FROM" ]] || { echo "RESUME_FROM checkpoint missing" >&2; exit 1; }
  args+=(--resume-from-checkpoint "$RESUME_FROM")
elif [[ -d "$OUTPUT_ROOT" && -n "$(find "$OUTPUT_ROOT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Output is nonempty; set RESUME_FROM to a verified original checkpoint." >&2; exit 2
fi
if [[ "$RUN_MODE" == smoke ]]; then
  stop="${SFT_SMOKE_STEPS:-2}"
  [[ "$stop" =~ ^[1-9][0-9]*$ ]] || { echo "SFT_SMOKE_STEPS must be positive" >&2; exit 2; }
  (( stop <= STEPS )) || stop="$STEPS"
  args+=(--stop-after-steps "$stop")
fi
echo "SFT mode=$RUN_MODE fixed_total_steps=$STEPS output=$OUTPUT_ROOT"
"$PYTHON" -m agent.training.cli "${args[@]}"
