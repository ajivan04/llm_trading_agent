#!/usr/bin/env bash
set -Eeuo pipefail

# Full pre-event training workflow for the declared midterm regime.
# This script intentionally stops at validation. The 2026 target week is a
# frozen, one-shot evaluation and must be prepared separately after outcomes
# mature.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

trap 'printf "Training workflow failed at line %s.\n" "$LINENO" >&2' ERR

: "${OUTPUT_ROOT:=artifacts/midterm-training}"
: "${START_DATE:=2006-10-30}"
: "${DATA_END:=2022-11-15}"
: "${TRAIN_END:=2018-11-13T23:59:59Z}"
: "${VALIDATION_END:=2022-11-15T23:59:59Z}"
: "${TEST_END:=2026-11-10T23:59:59Z}"
: "${PROMPT_VERSION:=v1-midterm-election}"
: "${SFT_STEPS:=1000}"
: "${DPO_STEPS:=500}"
: "${PPO_UPDATES:=100}"
: "${SYMBOLS:=}"
: "${SKIP_SMOKE:=0}"
: "${SKIP_DPO:=0}"
: "${SKIP_PPO:=0}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it, then rerun this script." >&2
  exit 1
fi

echo "Installing the locked project environment..."
uv sync --all-extras

echo "Running repository tests..."
python3 -m unittest discover -s tests

prepare_args=(
  --start "$START_DATE"
  --end "$DATA_END"
  --train-end "$TRAIN_END"
  --validation-end "$VALIDATION_END"
  --test-end "$TEST_END"
  --output "$OUTPUT_ROOT"
  --regime midterm-election
)
if [[ -n "$SYMBOLS" ]]; then
  read -r -a symbol_args <<< "$SYMBOLS"
  prepare_args+=(--symbols "${symbol_args[@]}")
fi

echo "Preparing the frozen historical training artifact..."
uv run stock-agent-prepare "${prepare_args[@]}"

DATASET_DIR="$OUTPUT_ROOT/dataset"
SFT_DIR="$OUTPUT_ROOT/sft"
DPO_DIR="$OUTPUT_ROOT/dpo"
PPO_DIR="$OUTPUT_ROOT/ppo"

echo "Evaluating the frozen base policy on validation..."
uv run stock-agent-evaluate \
  --prepared "$OUTPUT_ROOT" \
  --split validation \
  --output "$OUTPUT_ROOT/eval-base"

echo "Training SFT for $SFT_STEPS steps..."
uv run stock-agent-train sft \
  --train "$DATASET_DIR/train.jsonl" \
  --validation "$DATASET_DIR/validation.jsonl" \
  --output "$SFT_DIR" \
  --prompt-version "$PROMPT_VERSION" \
  --max-steps "$SFT_STEPS"

if [[ "$SKIP_SMOKE" != "1" ]]; then
  echo "Running the CUDA adapter save/reload smoke test..."
  uv run stock-agent-train smoke \
    --adapter-checkpoint "$SFT_DIR" \
    --output "$OUTPUT_ROOT/sft-smoke"
fi

echo "Evaluating SFT on validation..."
uv run stock-agent-evaluate \
  --prepared "$OUTPUT_ROOT" \
  --split validation \
  --adapter "$SFT_DIR" \
  --output "$OUTPUT_ROOT/eval-sft"

if [[ "$SKIP_DPO" != "1" ]]; then
  echo "Training audited DPO for $DPO_STEPS steps..."
  uv run stock-agent-train dpo \
    --train "$OUTPUT_ROOT/dpo-train.jsonl" \
    --validation "$OUTPUT_ROOT/dpo-validation.jsonl" \
    --adapter-checkpoint "$SFT_DIR" \
    --output "$DPO_DIR" \
    --prompt-version "$PROMPT_VERSION" \
    --max-steps "$DPO_STEPS"

  echo "Evaluating DPO on validation..."
  uv run stock-agent-evaluate \
    --prepared "$OUTPUT_ROOT" \
    --split validation \
    --adapter "$DPO_DIR" \
    --output "$OUTPUT_ROOT/eval-dpo"
fi

if [[ "$SKIP_PPO" != "1" ]]; then
  echo "Training native Qwen PPO for $PPO_UPDATES updates..."
  uv run stock-agent-train ppo \
    --environment-factory agent.training.environment:create_training_environment \
    --environment-config "$OUTPUT_ROOT/ppo-environment.json" \
    --adapter-checkpoint "$SFT_DIR" \
    --output "$PPO_DIR" \
    --prompt-version "$PROMPT_VERSION" \
    --updates "$PPO_UPDATES"

  echo "Evaluating PPO on validation..."
  uv run stock-agent-evaluate \
    --prepared "$OUTPUT_ROOT" \
    --split validation \
    --adapter "$PPO_DIR/adapter" \
    --output "$OUTPUT_ROOT/eval-ppo"
fi

echo
echo "Training workflow completed. Review metrics under: $OUTPUT_ROOT"
echo "Do not use this artifact for 2026 target-week test tuning."
