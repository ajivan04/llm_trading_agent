#!/usr/bin/env bash
# Collect actual base-Qwen historical research trajectories; never submits orders.
# sbatch --export=ALL,PREPARED_ROOT=/path/to/prepared scripts/collect_turing.slurm
#SBATCH --job-name=trajectory-collect
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
PREPARED_ROOT="${PREPARED_ROOT:?Set PREPARED_ROOT to a complete prepared evidence artifact}"
RUN_MODE="${RUN_MODE:-smoke}"
ACTION="${ACTION:-collect}"
cd "$PROJECT_ROOT"
PYTHON="$PROJECT_ROOT/.venv/bin/python"
[[ -x "$PYTHON" ]] || { echo "Missing project .venv/bin/python" >&2; exit 1; }
case "$RUN_MODE" in
  smoke) TRAIN_SESSIONS=60; VALIDATION_SESSIONS=20 ;;
  full)
    [[ "${REVIEWED_SMOKE:-0}" == 1 ]] || {
      echo "Review smoke source coverage, decisions and labels; set REVIEWED_SMOKE=1 afterward." >&2
      exit 2
    }
    TRAIN_SESSIONS=60; VALIDATION_SESSIONS=20 ;;
  *) echo "RUN_MODE must be smoke or full" >&2; exit 2 ;;
esac
case "$ACTION" in collect|export|both) ;; *) echo "ACTION must be collect, export, or both" >&2; exit 2 ;; esac
COLLECTION_ROOT="${COLLECTION_ROOT:-$PREPARED_ROOT/assignment-final}"
TRAJECTORY_ROOT="${TRAJECTORY_ROOT:-$COLLECTION_ROOT/export}"
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TQDM_DISABLE=1
if [[ "$ACTION" != export ]]; then
  for cohort in recent midterm validation; do
    split=train
    cohort_args=()
    journal_name=train
    sessions="$TRAIN_SESSIONS"
    if [[ "$cohort" == midterm ]]; then
      sessions="${HISTORICAL_SESSIONS:-20}"
      [[ "$sessions" != 0 ]] || continue
      cohort_args=(--cohort midterm)
      journal_name=train-midterm
    elif [[ "$cohort" == validation ]]; then
      split=validation; journal_name=validation; sessions="$VALIDATION_SESSIONS"
    fi
    resume_args=()
    if [[ "${RESUME_COLLECTION:-0}" == 1 && -f "$COLLECTION_ROOT/$journal_name/collection-manifest.json" ]]; then
      resume_args=(--resume)
    fi
    if [[ "$RUN_MODE" == smoke ]]; then
      resume_args+=(--max-sessions "${COLLECTION_SMOKE_SESSIONS:-2}")
    fi
    echo "Collecting split=$split planned_sessions=$sessions mode=$RUN_MODE"
    "$PYTHON" -m agent.trajectory_cli collect \
      --prepared "$PREPARED_ROOT" --journal "$COLLECTION_ROOT/$journal_name" \
      --split "$split" --sessions "$sessions" --require-cuda \
      ${cohort_args[@]+"${cohort_args[@]}"} \
      --max-context-tokens "${MAX_LENGTH:-4096}" \
      --max-new-tokens "${MAX_NEW_TOKENS:-512}" --max-turns "${MAX_TURNS:-6}" \
      ${resume_args[@]+"${resume_args[@]}"}
  done
fi
if [[ "$ACTION" != collect ]]; then
  extra_args=()
  if [[ "${HISTORICAL_SESSIONS:-20}" != 0 ]]; then
    extra_args=(--extra-train-journal "$COLLECTION_ROOT/train-midterm")
  fi
  "$PYTHON" -m agent.trajectory_cli export \
    --train-journal "$COLLECTION_ROOT/train" \
    --validation-journal "$COLLECTION_ROOT/validation" --output "$TRAJECTORY_ROOT" \
    ${extra_args[@]+"${extra_args[@]}"}
fi
echo "Collection artifacts: $COLLECTION_ROOT"
