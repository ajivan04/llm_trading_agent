#!/usr/bin/env bash
# Existing project .venv only; no uv required. New GOOD-trajectory artifact only.
# sbatch --export=ALL,TRAJECTORY_ROOT=/path/to/trajectories,RUN_MODE=preflight scripts/train_trajectory_sft.slurm
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
OUTPUT_ROOT="${OUTPUT_ROOT:-$TRAJECTORY_ROOT/sft-$RUN_MODE}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
cd "$PROJECT_ROOT"
PYTHON="$PROJECT_ROOT/.venv/bin/python"
[[ -x "$PYTHON" ]] || { echo "Missing project .venv/bin/python" >&2; exit 1; }
[[ -f "$TRAJECTORY_ROOT/trajectory-manifest.json" ]] || {
  echo "Missing trajectory-manifest.json; counterfactual datasets are not GOOD trajectories." >&2
  exit 1
}
case "$RUN_MODE" in
  preflight) STEPS=10 ;;
  smoke) STEPS=10 ;;
  full)
    [[ "${REVIEWED_SMOKE:-0}" == 1 ]] || {
      echo "Review collection and ten-step SFT smoke results; set REVIEWED_SMOKE=1 afterward." >&2
      exit 2
    }
    STEPS="${SFT_STEPS:?Set SFT_STEPS explicitly after reviewing the smoke run}"
    ;;
  *) echo "RUN_MODE must be preflight, smoke, or full" >&2; exit 2 ;;
esac
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TQDM_DISABLE=1
args=(sft --train "$TRAJECTORY_ROOT/dataset/train.jsonl"
  --validation "$TRAJECTORY_ROOT/dataset/validation.jsonl"
  --output "$OUTPUT_ROOT" --prompt-version bounded-tool-research-v1
  --supervision trajectory --max-length "$MAX_LENGTH" --max-steps "$STEPS"
  --train-batch-size 1 --eval-batch-size 1 --gradient-accumulation-steps 8
  --save-steps 5 --logging-steps 1)
"$PYTHON" -m agent.training.cli "${args[@]}" --preflight-only
[[ "$RUN_MODE" != preflight ]] || exit 0
if [[ -f "$OUTPUT_ROOT/training_manifest.json" ]]; then
  echo "Output already has a completed training manifest; use a new output directory." >&2
  exit 2
fi
if [[ -n "${RESUME_FROM:-}" ]]; then
  [[ -d "$RESUME_FROM" ]] || { echo "RESUME_FROM checkpoint does not exist" >&2; exit 1; }
  args+=(--resume-from-checkpoint "$RESUME_FROM")
elif [[ -d "$OUTPUT_ROOT" && -n "$(find "$OUTPUT_ROOT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Output is nonempty; choose RESUME_FROM explicitly or use a new directory." >&2
  exit 2
fi
echo "Training trajectory SFT: mode=$RUN_MODE steps=$STEPS output=$OUTPUT_ROOT"
"$PYTHON" -m agent.training.cli "${args[@]}"
