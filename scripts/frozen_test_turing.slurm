#!/usr/bin/env bash
#SBATCH --job-name=frozen-october-test
#SBATCH --partition=academic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:A30:1
#SBATCH --time=04:00:00
#SBATCH --output=turing-%x-%j.out
#SBATCH --error=turing-%x-%j.err
set -Eeuo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
FROZEN_RELEASE="${FROZEN_RELEASE:?Set FROZEN_RELEASE to immutable selected release}"
TEST_SOURCE="${TEST_SOURCE:?Set TEST_SOURCE to its separate hydrated held-out source}"
TEST_OUTPUT="${TEST_OUTPUT:?Set TEST_OUTPUT to one stable result directory}"
ASSESSMENT_REGISTRY="${ASSESSMENT_REGISTRY:?Set ASSESSMENT_REGISTRY to durable shared registry outside results}"
PYTHON="$PROJECT_ROOT/.venv/bin/python"
# Run the packaged implementation, so later working-tree changes cannot alter this release.
cd "$FROZEN_RELEASE/code"
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
args=(test --release "$FROZEN_RELEASE" --prepared "$TEST_SOURCE" --output "$TEST_OUTPUT"
  --registry "$ASSESSMENT_REGISTRY")
[[ "${TEST_RESUME:-0}" != 1 ]] || args+=(--resume)
"$PYTHON" -m agent.release_cli "${args[@]}" --plan-only
"$PYTHON" -m agent.release_cli "${args[@]}"
