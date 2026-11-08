# Run the assignment workflow

The old per-stage Slurm scripts and shell-export instructions are retired.
Run from the repository root. Edit **project.json once**: `source` points to the
already completed `artifacts/final-nlp-source`; `run_dir` defaults to
`artifacts/assignment`. Paths resolve relative to that config file, not your login
shell. No new download, normalization, hydration or offline compaction is needed.
Do not delete old data, adapters, failed smoke journals or evaluation results.

## Setup and source check

```bash
cd ~/llm_trading_agent
git pull origin main
.venv/bin/python -m agent check
```

If the environment is missing, create it once with Python and install the
project (`python -m venv .venv`, then `.venv/bin/python -m pip install -e '.[data,train]'`).
GPU jobs do not install packages or need `uv`. An uncached pinned tokenizer/model
still needs download access. Never run long model inference on the login node.

Credentials stay in ignored `.env`: `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`,
`SEC_USER_AGENT="Your Name email@example.com"`, optional `HF_TOKEN`, and
`ALPHA_VANTAGE_API_KEY` for the live RSI/quote tools. Empty optional values are
allowed. Never put credential values in notebook outputs or Slurm arguments.

`check` prints the three cohort summaries and verifies source/boundary/NLP guards.
It does not load policy weights or call providers. Existing old source bundles
load in place; their obsolete train/test targets are ignored.

## Collect actual base decisions

```bash
sbatch scripts/run_turing.slurm collect --smoke
```

Inspect `run_dir/collections/{train,train-midterm,validation}/episode-*.json`.
Each cohort uses a two-session prefix of its fixed full plan. Review raw model
output, tool calls, citations, risk rejections, fills and labels. The job stops
when a cohort has no researched, executable simulated trade. A successful Slurm
exit alone is not evidence of a profitable or usable model.

Then continue the same journals and export GOOD examples automatically:

```bash
sbatch scripts/run_turing.slurm collect --resume
```

The corpus appears at `run_dir/corpus`. Folds reset independently. No valid GOOD
examples means no SFT dataset; the code does not substitute oracle targets.
A changed prompt/code identity needs a new `run_dir`, not a forced resume.
This cleanup changes research tools/prompts, so old `final-collections*` journals
must stay separate from the new default run.

## SFT

```bash
sbatch scripts/run_turing.slurm sft --preflight
sbatch scripts/run_turing.slurm sft --smoke
# Review losses, measured parameter fraction and saved checkpoint; then:
sbatch scripts/run_turing.slurm sft --resume
```

Only the final assistant completion contributes loss. By default the total
optimizer target approximates three passes over the actual GOOD corpus at an
effective batch size of eight; it is not 1,000 updates by habit. The two-step
smoke uses that same scheduler target. `--resume` finds the newest complete
checkpoint. Actual data/model/settings and optimizer/scheduler/RNG checks remain.
If you use `--steps N`, use the same N for smoke and resume.

Token/target checks happen internally **before weights** on every real training
run; standalone preflight is for reviewing the budget, not a mandatory duplicate
stage. Checkpoint saves occur every two steps; loss validation has an independent
25-step cadence. Statefully evaluate the final saved adapter afterward.

## Optional PPO

```bash
sbatch scripts/run_turing.slurm ppo --smoke
# Inspect ppo/training_status.json and ppo/ppo_metrics.jsonl; then optionally:
sbatch scripts/run_turing.slurm ppo --resume --updates 10
sbatch scripts/run_turing.slurm ppo --resume
```

The command compiles train-only NLP episodes automatically and uses the completed
new SFT adapter. Updates are total targets: the default full target is the
`ppo_updates` value in project.json. Review invalid fraction, reward, reference KL
and update ratio before extension. A stopped/failed pilot is not a reason to
blindly increase its budget. This trains final decisions in a simpler one-stock
environment; the tool-using portfolio loop remains the comparison that matters.

## Compare base, SFT and optional PPO

```bash
sbatch scripts/run_turing.slurm evaluate --model all --smoke
# Review two-session prefixes, then complete their SAME 20-session plans:
sbatch scripts/run_turing.slurm evaluate --model all --resume
```

`all` includes base/SFT and PPO if its adapter exists. Use `--model base|sft|ppo`
for individual runs. Results are at `run_dir/eval-MODEL/research_metrics.json`.
Compare returns, actual trade activity, validity, drawdown and benchmark; include
failed/invalid decisions. Select on validation, not on the held-out October week.
Do not retrain/reselect on the November live days.

## Paper rehearsal and live week

Use an always-on machine with the repo environment, source archive and paper keys.
These commands default to dry order proposals, but still read live broker/data:

```bash
.venv/bin/python -m agent paper --model sft --cycle-id rehearsal-001 --iterations 10 --interval 60
```

After checking account, data freshness, research, risk and proposals, explicitly
submit paper orders with a **different** session prefix:

```bash
.venv/bin/python -m agent paper --model sft --cycle-id live-2026-11-09 --iterations 240 --interval 60 --submit-paper
```

Use `--model ppo` only if that is the selected validated model. Reuse the same
prefix on restart: completed cycle IDs are reused, not re-traded. Use a new prefix
for a new intended session. Closed/uncertain broker state pauses the loop;
recoverable model/data failures do not manufacture orders. The model is loaded
once for the session. Logs are saved by ET date under `run_dir/paper/MODEL/trading_logs`;
outbox and memory persist beside them. See [paper operations](paper-operations.md)
for low-level reconciliation/repair commands. Never change dry/submitting mode
for an already used cycle ID.

The notebook requires ten consecutive iterations with tool/memory/order evidence,
then November 9–13 daily logs, Alpaca order history and final balance. Offline
unit tests, synthetic fills and October replay do not satisfy those empirical gates.

## Final frozen assessment (separate from the baseline demo)

`agent.release_cli` still packages exact code, validated final adapter and risk
settings; its strict gate requires completed full NLP validation, schema validity
and executed trades. `agent.hydrate_news --release` creates the separate frozen
test source only after selection. `scripts/frozen_test_turing.slurm` reserves the
single October assessment in a durable registry and resumes only that attempt.
Use the module `--help` and release tests for these explicit archival operations.
They are not prerequisites for running the required base/SFT/replay/paper workflow.

Original data migrations remain available through `agent.normalize_prices` and
`agent.hydrate_news` (or their two CPU job scripts). They are one-time compatibility
operations, not repeated stages of each model experiment. New sources contain
only `source.json`, `price-bars.jsonl`, `evidence.jsonl` and optional provenance
sidecars; they do not carry obsolete training datasets.
