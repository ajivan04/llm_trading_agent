#!/usr/bin/env bash
# Corrected PPO: default two-update smoke; extend to 10 then 100 only after review.
# Required: PREPARED_ROOT, SFT_ADAPTER, PPO_OUTPUT_ROOT. No uv or pip installation.
# Resume explicitly with PPO_RESUME_FROM; PPO_UPDATES is a TOTAL target.
#SBATCH --job-name=stock-ppo-v2
#SBATCH --partition=academic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:A30:1
#SBATCH --time=04:30:00
#SBATCH --output=turing-%x-%j.out
#SBATCH --error=turing-%x-%j.err

set -Eeuo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PREPARED_ROOT="${PREPARED_ROOT:?Set PREPARED_ROOT to the repaired prepared artifact}"
SFT_ADAPTER="${SFT_ADAPTER:?Set SFT_ADAPTER to the chosen complete SFT adapter}"
PPO_OUTPUT_ROOT="${PPO_OUTPUT_ROOT:?Set PPO_OUTPUT_ROOT to a new PPO-v2 experiment directory}"
cd "$PROJECT_ROOT"
PYTHON="$PROJECT_ROOT/.venv/bin/python"
[[ -x "$PYTHON" ]] || { echo "Missing project .venv/bin/python" >&2; exit 1; }
[[ -f "$PREPARED_ROOT/ppo-environment.json" ]] || {
  echo "Missing prepared PPO environment" >&2; exit 1;
}
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_PROGRESS_BARS=1

echo "Job: ${SLURM_JOB_ID:-local}; corrected PPO total target: ${PPO_UPDATES:-2}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"$PYTHON" -m unittest tests.test_qwen_ppo_numerics tests.test_ppo_runner

resume_args=(--adapter-checkpoint "$SFT_ADAPTER")
if [[ -n "${PPO_RESUME_FROM:-}" ]]; then
  resume_args=(--resume-from-checkpoint "$PPO_RESUME_FROM")
fi

"$PYTHON" -m agent.training.cli ppo \
  --require-cuda \
  --environment-factory agent.training.environment:create_training_environment \
  --environment-config "$PREPARED_ROOT/ppo-environment.json" \
  --output "$PPO_OUTPUT_ROOT" \
  --prompt-version "${PROMPT_VERSION:-v2-visible-evidence}" \
  --max-length "${MAX_LENGTH:-4096}" \
  --max-new-tokens "${PPO_MAX_NEW_TOKENS:-512}" \
  --updates "${PPO_UPDATES:-2}" \
  --learning-rate "${PPO_LEARNING_RATE:-0.00001}" \
  --value-learning-rate "${PPO_VALUE_LEARNING_RATE:-0.000001}" \
  --rollout-episodes "${PPO_ROLLOUT_EPISODES:-8}" \
  --ppo-epochs "${PPO_EPOCHS:-2}" \
  --stats-chunk-size "${PPO_STATS_CHUNK_SIZE:-32}" \
  --target-kl "${PPO_TARGET_KL:-0.1}" \
  --max-invalid-fraction "${PPO_MAX_INVALID_FRACTION:-0.5}" \
  --invalid-patience "${PPO_INVALID_PATIENCE:-2}" \
  --max-seconds "${PPO_MAX_SECONDS:-7200}" \
  --save-steps "${PPO_SAVE_STEPS:-10}" \
  "${resume_args[@]}"

echo "PPO target completed. Review training_status.json and ppo_metrics.jsonl before extending."
