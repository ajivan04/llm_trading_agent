#!/usr/bin/env bash
# Turing Slurm job for the leakage-resistant Qwen training workflow.
# Submit from the repository root after editing the account directive below
# if your course/lab requires one: sbatch scripts/train_turing.slurm
#
# Override stages or paths at submission time, for example:
#   sbatch --export=ALL,STAGE=sft scripts/train_turing.slurm

# Turing's academic partition currently provides A30 GPUs. Adjust these
# resource requests only after checking `scluster`/`idle_gpus` on Turing.
#SBATCH --job-name=stock-agent
#SBATCH --partition=academic
##SBATCH --account=ahjivan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:A30:1
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:SIGUSR1@120
#SBATCH --open-mode=append
#SBATCH --output=turing-%x-%j.out
#SBATCH --error=turing-%x-%j.err

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/artifacts/midterm-training}"
STAGE="${STAGE:-all}" # prepare, eval, sft, dpo, ppo, or all
ENV_FILE="${ENV_FILE:-$PROJECT_ROOT/.env}"
PROMPT_VERSION="${PROMPT_VERSION:-v1-midterm-election}"

case "$STAGE" in
  prepare|eval|sft|dpo|ppo|all) ;;
  *) echo "Invalid STAGE=$STAGE (use prepare, eval, sft, dpo, ppo, or all)." >&2; exit 2 ;;
esac

cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_ROOT"

# Source credentials only when explicitly requested. Keep this file outside
# version control with mode 600; never print its contents or the environment.
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
elif [[ "$STAGE" == "prepare" || "$STAGE" == "all" ]]; then
  echo "Missing environment file for data preparation: $ENV_FILE" >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  # Turing provides uv as a module; batch shells may not have modules
  # initialized automatically.
  if [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
  fi
  if command -v module >/dev/null 2>&1; then
    module load uv
  fi
fi
command -v uv >/dev/null 2>&1 || {
  echo "uv is unavailable; run 'module load uv' on Turing and resubmit." >&2
  exit 1
}

echo "Host: $(hostname)"
echo "Job: ${SLURM_JOB_ID:-local}"
echo "Stage: $STAGE"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# If Turing sends SIGUSR1 near the time limit, stop the active child and
# requeue this job. Each stage resumes from its latest checkpoint on restart.
child_pid=""
handle_usr1() {
  echo "Received SIGUSR1; stopping active stage and requeueing." >&2
  if [[ -n "$child_pid" ]]; then
    kill -TERM "$child_pid" 2>/dev/null || true
    wait "$child_pid" 2>/dev/null || true
  fi
  scontrol requeue "${SLURM_JOB_ID:?}"
  exit 0
}
trap handle_usr1 SIGUSR1

run_child() {
  "$@" &
  child_pid=$!
  wait "$child_pid"
  child_pid=""
}

echo "Synchronizing the locked environment..."
run_child uv sync --all-extras
python3 -m unittest discover -s tests

if [[ "$STAGE" == "prepare" || "$STAGE" == "all" ]]; then
  if [[ -f "$OUTPUT_ROOT/preparation-manifest.json" ]]; then
    echo "Preparation already complete; skipping."
  else
    run_child uv run stock-agent-prepare \
      --start "${START_DATE:-2006-10-30}" \
      --end "${DATA_END:-2022-11-15}" \
      --train-end "${TRAIN_END:-2018-11-13T23:59:59Z}" \
      --validation-end "${VALIDATION_END:-2022-11-15T23:59:59Z}" \
      --test-end "${TEST_END:-2026-11-10T23:59:59Z}" \
      --output "$OUTPUT_ROOT" \
      --regime midterm-election
  fi
fi

if [[ "$STAGE" == "eval" || "$STAGE" == "all" ]]; then
  if [[ -f "$OUTPUT_ROOT/eval-base/evaluation.json" ]]; then
    echo "Base evaluation already complete; skipping."
  else
    run_child uv run stock-agent-evaluate \
      --prepared "$OUTPUT_ROOT" --split validation \
      --output "$OUTPUT_ROOT/eval-base"
  fi
fi

if [[ "$STAGE" == "sft" || "$STAGE" == "dpo" || "$STAGE" == "ppo" ]]; then
  if [[ ! -f "$OUTPUT_ROOT/dataset/train.jsonl" ]]; then
    echo "Missing prepared dataset: $OUTPUT_ROOT/dataset/train.jsonl" >&2
    echo "Run STAGE=prepare first or copy the prepared artifact to OUTPUT_ROOT." >&2
    exit 1
  fi
fi

latest_checkpoint() {
  find "$1" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1 || true
}

if [[ "$STAGE" == "sft" || "$STAGE" == "all" ]]; then
  SFT_DIR="$OUTPUT_ROOT/sft"
  mkdir -p "$SFT_DIR"
  if [[ -f "$SFT_DIR/training_manifest.json" ]]; then
    echo "SFT already complete; skipping."
  else
    resume="$(latest_checkpoint "$SFT_DIR")"
    resume_args=()
    [[ -z "$resume" ]] || resume_args=(--resume-from-checkpoint "$resume")
    run_child uv run stock-agent-train sft \
      --train "$OUTPUT_ROOT/dataset/train.jsonl" \
      --validation "$OUTPUT_ROOT/dataset/validation.jsonl" \
      --output "$SFT_DIR" --prompt-version "$PROMPT_VERSION" \
      --max-steps "${SFT_STEPS:-1000}" --save-steps 25 --logging-steps 5 \
      "${resume_args[@]}"
  fi
fi

if [[ "$STAGE" == "dpo" || "$STAGE" == "all" ]]; then
  DPO_DIR="$OUTPUT_ROOT/dpo"
  mkdir -p "$DPO_DIR"
  if [[ -f "$DPO_DIR/training_manifest.json" ]]; then
    echo "DPO already complete; skipping."
  else
    resume="$(latest_checkpoint "$DPO_DIR")"
    resume_args=()
    [[ -z "$resume" ]] || resume_args=(--resume-from-checkpoint "$resume")
    run_child uv run stock-agent-train dpo \
      --train "$OUTPUT_ROOT/dpo-train.jsonl" \
      --validation "$OUTPUT_ROOT/dpo-validation.jsonl" \
      --adapter-checkpoint "$OUTPUT_ROOT/sft" \
      --output "$DPO_DIR" --prompt-version "$PROMPT_VERSION" \
      --max-steps "${DPO_STEPS:-500}" --save-steps 25 --logging-steps 5 \
      "${resume_args[@]}"
  fi
fi

if [[ "$STAGE" == "ppo" || "$STAGE" == "all" ]]; then
  PPO_DIR="$OUTPUT_ROOT/ppo"
  mkdir -p "$PPO_DIR"
  if [[ -f "$PPO_DIR/training_manifest.json" ]]; then
    echo "PPO already complete; skipping."
  else
    resume="$(latest_checkpoint "$PPO_DIR")"
    ppo_args=(--adapter-checkpoint "$OUTPUT_ROOT/sft")
    if [[ -n "$resume" ]]; then
      ppo_args=(--resume-from-checkpoint "$resume")
    fi
    run_child uv run stock-agent-train ppo \
      --environment-factory agent.training.environment:create_training_environment \
      --environment-config "$OUTPUT_ROOT/ppo-environment.json" \
      --output "$PPO_DIR" --prompt-version "$PROMPT_VERSION" \
      --updates "${PPO_UPDATES:-100}" --save-steps 25 \
      "${ppo_args[@]}"
  fi
fi

echo "Training stage '$STAGE' completed. Artifacts: $OUTPUT_ROOT"
