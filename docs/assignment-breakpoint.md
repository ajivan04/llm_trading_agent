# Breakpoint 3: tools, language evidence and actual trajectories

This breakpoint implements the notebook's research-tool interfaces and a new
GOOD/BAD trajectory collection path. The policy remains the pinned
`Qwen/Qwen2.5-1.5B-Instruct`. Historical collection uses existing prepared prices,
news and SEC evidence. It does not call Alpaca's order API, fetch provider data,
or train a model. Alpaca paper execution integration and end-to-end reconciliation
remain breakpoint 4; these changes do not certify live execution readiness.

## What changed

Research receives the complete allowed universe and account state. A deterministic
candidate shortlist limits initial deep context; registered tools can research any
allowed symbol. Tool results have an immutable decision cutoff. Price history,
local MACD, RSI, quote, news, SEC and combined context interfaces expose their
available source data and explicit unavailable/quota results. Historical replay
does not treat a present-day quote or indicator response as past information.

New preparation includes actual eligible headlines/summaries beside aggregate
news features, retains all downloaded raw news in the evidence archive, and keeps
SEC text. Old aggregate-only archives cannot reconstruct missing article text.
Check source coverage before collecting: a missing-news marker does not mean news
was researched successfully. The Alpaca adapter dates a returned article version
at the later of creation and update, conservatively excluding later revisions.
Alpaca does not thereby provide an original historical version of each article.

Context uses the actual pinned tokenizer, bounded source text, explicit output
headroom, and exact short evidence aliases with an original-ID audit transcript.
Required records cannot be silently truncated. Tool calls and final responses are
subject to context and turn limits. A valid trade requires supporting evidence
for its symbol. These checks validate information and schema; they cannot ensure
that a model chooses a profitable trade.

The new training source is actual base-model tool research and decisions, followed
by private simulated portfolio outcomes and GOOD/BAD labels from net equity
change after modeled costs. Only eligible GOOD trajectories become SFT targets.
This differs from the existing oracle/counterfactual-best-action targets. Invalid
JSON, missing real tool use, failed execution/risk checks and unprofitable outcomes
remain diagnostic records rather than fabricated GOOD examples. SFT supervises
assistant tool requests and final output; tool/provider text is masked. All
native token spans and dataset identities are checked before allocating weights.

The prepared manifest controls chronological boundaries. Collection never invokes
the policy on test episodes or incorporates their outcomes into training. The
assignment's held-out week is **October 26–30, 2026**, the five XNYS sessions in
the last full October week. Preserve that static week and inspect the source
manifest; do not replace it with a rolling “last week” at execution time. If the
artifact contains other boundaries, correct the artifact plan before collection.

## Review gates and Turing commands

Run from the updated repository with the existing project `.venv`. These scripts
do not require `uv`, install packages or read `.env`. Collection uses archived
sources; the pinned tokenizer/model may need its existing Hugging Face cache or
download access. Keep original datasets, adapters and evaluation artifacts intact.

```bash
cd ~/llm_trading_agent
export PREPARED_ROOT="$HOME/llm_trading_agent/artifacts/mixed-2026-test-v5"
export COLLECTION_ROOT="$PREPARED_ROOT/assignment-smoke"
export TRAJECTORY_ROOT="$COLLECTION_ROOT/export"

# CPU-only archive/window checks before reserving a GPU.
.venv/bin/python -m agent.trajectory_cli plan \
  --prepared "$PREPARED_ROOT" --split train --sessions 2
.venv/bin/python -m agent.trajectory_cli plan \
  --prepared "$PREPARED_ROOT" --split validation --sessions 2

# Gate 1: two real sessions from each split, complete journals, no test calls.
sbatch --export=ALL,RUN_MODE=smoke,ACTION=collect scripts/collect_turing.slurm
```

Review the Slurm output and both journal directories after completion. Inspect
actual assistant requests, tool evidence and missing-source markers, final JSON,
citations, risk rejections, fills, labels, and episode/account dates. Confirm that
tool use comes from model requests rather than fabricated prefetch counts.
An all-invalid or zero-GOOD smoke run is a diagnosis to fix before larger jobs;
do not relax label rules or insert oracle targets merely to make export succeed.
Two sessions are a format/execution smoke test, not a meaningful performance
estimate. A schema-valid HOLD also does not demonstrate order execution.

For example, inspect saved episode envelopes without rerunning inference:

```bash
for SPLIT in train validation; do
  jq '.record | {session,status,label,portfolio_change_label,tool_call_count,exclusion_reason}' \
    "$COLLECTION_ROOT/$SPLIT"/episode-*.json
done
```

```bash
# Gate 2: export after review; requires nonempty GOOD train and validation.
.venv/bin/python -m agent.trajectory_cli export \
  --train-journal "$COLLECTION_ROOT/train" \
  --validation-journal "$COLLECTION_ROOT/validation" --output "$TRAJECTORY_ROOT"

# CPU-only preflight can run on the login node; no model weights are loaded.
.venv/bin/python -m agent.training.cli sft \
  --train "$TRAJECTORY_ROOT/dataset/train.jsonl" \
  --validation "$TRAJECTORY_ROOT/dataset/validation.jsonl" \
  --output "$TRAJECTORY_ROOT/sft-preflight" \
  --prompt-version bounded-tool-research-v1 --supervision trajectory \
  --max-length 4096 --preflight-only

# Gate 3: ten optimizer steps, separate output, checks every native mask first.
sbatch --export=ALL,RUN_MODE=smoke scripts/train_trajectory_sft.slurm
```

Inspect the ten-step run's loss, supervised spans, checkpoint files and validation
responses before approving longer collection/training. No production collection
or training was run as part of implementing this breakpoint.

```bash
# Gate 4: distinct full journals, 60 train / 20 validation sessions.
export COLLECTION_ROOT="$PREPARED_ROOT/assignment-full"
export TRAJECTORY_ROOT="$COLLECTION_ROOT/export"
sbatch --export=ALL,RUN_MODE=full,REVIEWED_SMOKE=1,ACTION=collect \
  scripts/collect_turing.slurm

# After reviewing completed journals, repeat the export command above.
# Choose SFT_STEPS deliberately; 100 is an initial bounded run, not an optimum.
sbatch --export=ALL,RUN_MODE=full,REVIEWED_SMOKE=1,SFT_STEPS=100 \
  scripts/train_trajectory_sft.slurm
```

Interrupted collection resumes the same journal contract with
`RESUME_COLLECTION=1`; keep its source, settings and planned session counts
unchanged. If one split is already complete, resume verifies/reuses its durable
prefix while continuing the other. Resume interrupted SFT using an explicit
`RESUME_FROM=/absolute/path/to/checkpoint-N` and the original output/settings.
Completed output directories are protected from accidental retraining. Full and
smoke directories intentionally differ.

DPO, PPO and final selection still need the subsequent integration gates. Existing
corrected PPO scripts remain available, but this breakpoint does not claim that
PPO already uses the complete new research loop. Evaluate and select on validation,
then freeze the chosen adapter, tools, context and risk loop together before the
single held-out-week comparison. Schema success, GOOD labels and a small training
loss do not establish out-of-sample profit.

GOOD-only validation loss measures imitation on a filtered subset. It is not an
estimate of trading profit or failure rate on all validation decisions. The next
breakpoint must run the complete account-aware loop over the full validation
schedule, including HOLDs, invalid outputs and rejected proposals, before model
selection. Labels based on total portfolio change can also reflect existing
holdings; the journal separately records incremental PnL against no action.

The default full plan selects the most recent 60 eligible training sessions and
20 validation sessions from the existing split boundaries. This does not collect
additional historical midterm folds. Dates before 2016 are excluded. Coverage
counts are per decision window; one article can appear on multiple dates.

Legacy compacted datasets can retain the original preparation manifest while
changing the target dataset hash. The collection gate checks the compacted target
integrity, preserved raw price hash and evidence count and binds the compaction
manifest into the collection protocol. Old v2 manifests cannot fully prove their
original source-to-target link; the plan explicitly reports
`legacy_compaction_source_link_unverified`. Collection does not reuse their
counterfactual targets as actual base-model decisions.
