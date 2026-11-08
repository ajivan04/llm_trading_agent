# Final retraining and release workflow

This replaces earlier breakpoint commands. Run one stage at a time and inspect
its result before submitting the next. No jobs below depend on another job ID.
Nothing here requires `uv` on compute nodes: jobs use the repository `.venv`.
Do not overwrite or delete the old v5 archive, trained models or evaluation results.

## Choices now fixed

- Policy: pinned `Qwen/Qwen2.5-1.5B-Instruct`. Raw news and SEC filing text reach
  Qwen through bounded initial context and actual registered research tools.
- SFT labels: collect real **base-model** trajectories, replay account-aware
  trades with costs, label GOOD/BAD from net next-session portfolio equity change,
  and train only GOOD trajectories. Research/tool history stays in context;
  **only the final assistant decision is supervised**. No oracle target replacement.
- Training cohorts: latest 60 eligible 2026 trading sessions plus 20 complete
  October sessions from the latest earlier midterm year (2022 in v5). Each cohort
  starts an independent $100,000 account. Validation has its own 20-session account.
  All outcome horizons remain in their own split. No pre-2016 data is selected.
- PPO: corrected final-decision-only NLP episodes, initialized from the completed
  new SFT adapter; cash-only and held-share scenarios, actual risk and next-session
  rewards. This is **not multi-turn tool-use reinforcement learning**. The chosen
  adapter is subsequently evaluated through the same tool-using research loop.
- DPO remains an optional legacy experiment, not required for this recovery path.
  The previous counterfactual DPO labels do not implement GOOD-trajectory SFT.
- Historical held-out execution dates: **October 26–30, 2026**, decisions at the
  preceding session closes (October 23, 26–29). Historical memory is disabled.
  The notebook's **November 9–13 live paper week** is separate. Live confirmed
  trade reflections are enabled under frozen rules and must be reported as such.

## 1. Environment and paths on Turing

```bash
cd ~/llm_trading_agent
git pull origin main
export PROJECT_ROOT="$PWD"
export SOURCE_ROOT="$PWD/artifacts/mixed-2026-test-v5"
export NOMINAL_ROOT="$PWD/artifacts/final-nominal-source"
export HYDRATED_ROOT="$PWD/artifacts/final-nlp-source"
export ENV_FILE="$PWD/.env"
export HISTORICAL_SESSIONS=20
export PREPARED_ROOT="$HYDRATED_ROOT"
export COLLECTION_ROOT="$PWD/artifacts/final-collections"
export TRAJECTORY_ROOT="$COLLECTION_ROOT/export"
export OUTPUT_ROOT="$TRAJECTORY_ROOT/sft-assignment"
export PPO_PREPARED_ROOT="$PWD/artifacts/final-ppo-source"
export PPO_OUTPUT_ROOT="$PWD/artifacts/final-ppo"
export SFT_ADAPTER="$OUTPUT_ROOT"
test -x .venv/bin/python
.venv/bin/python -c 'import torch, transformers, peft, datasets, yfinance, exchange_calendars; print("Dependencies import successfully")'
```

These variables must be exported again in a new shell. If the existing `.venv`
is missing, install with Python (no `uv` required):

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[data,train]'
```

Credentials stay in ignored `$ENV_FILE`: `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`,
`SEC_USER_AGENT` (name and contact email); optional `HF_TOKEN` for Hub downloads
and `ALPHA_VANTAGE_API_KEY` for live RSI/quote tools. News uses Alpaca, MACD is
local. Never paste credentials into Slurm arguments or logs. Live paper endpoints
are enforced; do not substitute a real-money account. See `paper-breakpoint.md`.

## 2. Repair the source basis, then hydrate raw news (CPU jobs)

```bash
sbatch scripts/normalize_prices_turing.slurm
# After successful completion:
sbatch scripts/hydrate_news_turing.slurm
```

Yahoo OHLC with `auto_adjust=False` is still split-adjusted. Replay previously
also applied splits. Normalization restores nominal historical share units,
including dividends and volume, while leaving the original archive intact.
It requests only missing split-action tail coverage between the last archived
bar and original retrieval time; absent coverage fails rather than assuming no
split. This is a reconstruction from a later vendor capture, not proof of an
untouched historical vendor vintage. Adjustment factors stay private.
Provider references: [yfinance split repair](https://ranaroussi.github.io/yfinance/advanced/price_repair.html),
[Yahoo adjusted close convention](https://in.help.yahoo.com/kb/adjusted-close-sln28256.html).

Hydration preserves prices and SEC text and fetches complete weekly news windows
for the selected cohorts; it does not query the intervening years. Created and
updated times both constrain availability. Empty symbol/day news is valid.
Each completed chunk is checksummed and reusable after a disconnect:

```bash
sbatch --export=ALL,HYDRATE_RESUME=1 scripts/hydrate_news_turing.slurm
```

These source directories retain the old dataset only as split/calendar anchors.
Training directly from those old labels is rejected. Use the new GOOD export.
Inspect `price-basis-manifest.json`, `hydration-manifest.json` and job logs;
then check each plan without loading weights:

```bash
.venv/bin/python -m agent.trajectory_cli plan --prepared "$PREPARED_ROOT" --split train --sessions 60
.venv/bin/python -m agent.trajectory_cli plan --prepared "$PREPARED_ROOT" --split train --cohort midterm --sessions 20
.venv/bin/python -m agent.trajectory_cli plan --prepared "$PREPARED_ROOT" --split validation --sessions 20
```

## 3. Collect actual base trajectories, then export GOOD examples

```bash
sbatch --export=ALL,RUN_MODE=smoke scripts/collect_turing.slurm
```

Review the two-session prefix in each journal (`train`, `train-midterm`,
`validation`). Inspect `episode-*.json`, schema errors, actual tool calls,
citations, risk rejections, fills and GOOD/BAD counts. Do not continue if outputs
are malformed or news/SEC are missing. Then resume the **same** planned journals:

```bash
sbatch --export=ALL,RUN_MODE=full,REVIEWED_SMOKE=1,RESUME_COLLECTION=1,ACTION=both scripts/collect_turing.slurm
```

Collection prints progress after every session and persists each result. Export
fails if either split has zero GOOD examples; it never invents targets. It pools
recent and historical training cohorts without carrying holdings across years.

## 4. Choose the SFT budget before the two-step smoke

```bash
sbatch --export=ALL,RUN_MODE=preflight scripts/train_trajectory_sft.slurm
```

Read token lengths, GOOD train/validation counts, effective batch size, estimated
exposure and `three_pass_step_recommendation` in the log. With this small selective
corpus, blindly repeating 1,000 updates can massively overfit. Choose a fixed
`SFT_STEPS` using that report and available time; the recommendation is a budget
heuristic, not evidence of optimal performance. Save the choice in your shell:

```bash
export SFT_STEPS=12  # EXAMPLE ONLY: replace using your actual preflight report
sbatch --export=ALL,RUN_MODE=smoke scripts/train_trajectory_sft.slurm
```

Review `training_status.json`, `sft_run.json`, checkpoint state and losses.
The measured trainable fraction must be below 0.1%. Smoke stops after two steps
of the **full chosen scheduler**, preserving optimizer/scheduler/RNG state.
Then continue without restarting those steps:

```bash
export RESUME_FROM="$OUTPUT_ROOT/checkpoint-2"
sbatch --export=ALL,RUN_MODE=full,REVIEWED_SMOKE=1 scripts/train_trajectory_sft.slurm
unset RESUME_FROM
```

Keep `SFT_STEPS`, dataset and training settings unchanged when resuming. Use the
actual checkpoint name if a disconnect saved another step. Legacy checkpoints
cannot resume this new training contract. `training_manifest.json` appears only
when the fixed total target is completed; partial smokes are not release models.

## 5. Compile and train PPO in reviewed increments

```bash
sbatch scripts/prepare_ppo_turing.slurm
# After successful compilation and completed SFT:
sbatch --export=ALL,PREPARED_ROOT="$PPO_PREPARED_ROOT",PPO_UPDATES=2 scripts/ppo_turing.slurm
```

Review `training_status.json`, `ppo_metrics.jsonl`, invalid fraction, reward,
reference KL, update ratio and nonzero gradients. Correct numerical tests do not
prove useful trades. PPO may stop safely for invalid output, KL or wall time.
Only after the pilot is acceptable:

```bash
sbatch --export=ALL,PREPARED_ROOT="$PPO_PREPARED_ROOT",PPO_UPDATES=10,PPO_RESUME_FROM="$PPO_OUTPUT_ROOT" scripts/ppo_turing.slurm
# Review again, then optionally:
sbatch --export=ALL,PREPARED_ROOT="$PPO_PREPARED_ROOT",PPO_UPDATES=100,PPO_RESUME_FROM="$PPO_OUTPUT_ROOT" scripts/ppo_turing.slurm
```

Updates are total targets. They resume the same run; they are not extra 10/100
updates. Do not extend a failed pilot just to match SFT wall time. Keep the new
SFT adapter if PPO degrades validity or returns.

## 6. Compare candidates on the same validation plan

Run separately for base (unset `POLICY_ADAPTER`), SFT (`$SFT_ADAPTER`) and corrected
PPO (`$PPO_OUTPUT_ROOT/adapter`), using distinct evaluation directories:

```bash
export POLICY_ADAPTER="$SFT_ADAPTER"
export EVALUATION_ROOT="$PWD/artifacts/final-eval-sft"
sbatch scripts/research_evaluate_turing.slurm
# Review its two-session prefix; then resume to all 20:
sbatch --export=ALL,EVAL_MAX_SESSIONS=20,EVAL_RESUME=1 scripts/research_evaluate_turing.slurm
```

Repeat with a different adapter and `EVALUATION_ROOT` for each candidate. Keep
`PREPARED_ROOT="$HYDRATED_ROOT"`, not the PPO compiler output. Include invalid
outputs, rejected orders and holds in the comparison. Read `research_metrics.json`
for dollar P&L, ending equity, benchmark return, drawdown, schema rate, fills and
inference latency. Full validation may take hours; journal progress is resumable.
Select using validation only. The release gate requires completed 20-session NLP
validation, at least 95% valid schema, actual research calls and executed trades.
It does not assert profit or statistical significance.

## 7. Freeze one model and run one historical test

Create a paper config directly from the chosen validation protocol so replay and
paper risk/universe settings agree:

```bash
export SELECTED_ADAPTER="$SFT_ADAPTER"  # or "$PPO_OUTPUT_ROOT/adapter"
export SELECTED_VALIDATION="$PWD/artifacts/final-eval-sft"
export PAPER_CONFIG="$PWD/artifacts/final-paper-config.json"
.venv/bin/python - <<'PY'
import json, os
from dataclasses import asdict
from pathlib import Path
from agent.config import Config
p = json.loads((Path(os.environ['SELECTED_VALIDATION']) / 'research_manifest.json').read_text())['protocol']['settings']
c = Config(universe=tuple(p['universe']), sector_map=tuple(map(tuple, p['sector_map'])) or tuple((s, 'unknown') for s in p['universe']), initial_cash=p['initial_cash'], max_shares=p['max_qty'], slippage_bps=p['cost_bps'])
Path(os.environ['PAPER_CONFIG']).write_text(json.dumps(asdict(c), indent=2) + '\n')
PY
export FROZEN_RELEASE="$PWD/artifacts/final-release"
.venv/bin/python -m agent.release_cli freeze --adapter "$SELECTED_ADAPTER" --prepared "$HYDRATED_ROOT" --validation "$SELECTED_VALIDATION" --config "$PAPER_CONFIG" --output "$FROZEN_RELEASE"
.venv/bin/python -m agent.release_cli verify --release "$FROZEN_RELEASE"
```

The release copies inference weights, all Python implementation files, dependency
lock, risk/research settings and validation/training provenance. It excludes
credentials and optimizer files. It requires the pinned base-model download or
HF cache at inference time; it does not duplicate base weights. SFT release uses
the verified final saved output, not an unbound intermediate checkpoint.

Only **after freezing** hydrate the separate test source:

```bash
export TEST_SOURCE="$PWD/artifacts/final-heldout-source"
sbatch --export=ALL,SOURCE_ROOT="$HYDRATED_ROOT",HYDRATED_ROOT="$TEST_SOURCE" scripts/hydrate_news_turing.slurm
# After successful hydration:
export TEST_OUTPUT="$PWD/artifacts/final-heldout-result"
export ASSESSMENT_REGISTRY="$PWD/artifacts/october-assessment.sqlite"
sbatch scripts/frozen_test_turing.slurm
```

The durable registry reserves a single October assessment before any model call.
For an interrupted pending attempt, reuse all paths and run with `TEST_RESUME=1`.
Do not delete/change the registry or use new directories to rerun selection on
the test. The report is five execution sessions starting with $100,000; Friday's
November 2 outcome is excluded. A five-day Sharpe is descriptive only.

## 8. Paper rehearsal and submission evidence

Use the packaged code and an always-on runner with access to the Alpaca paper
account; a historical backtest does not establish broker execution readiness.
Keep model/settings frozen. First run status and a dry cycle:

```bash
cd "$FROZEN_RELEASE/code"
"$PROJECT_ROOT/.venv/bin/python" -m agent.paper_cli status --release "$FROZEN_RELEASE" --prepared "$HYDRATED_ROOT" --env-file "$ENV_FILE" --outbox "$PROJECT_ROOT/artifacts/paper-outbox.sqlite" --memory "$PROJECT_ROOT/artifacts/paper-memory.sqlite" --run-id final-paper
"$PROJECT_ROOT/.venv/bin/python" -m agent.paper_cli cycle --release "$FROZEN_RELEASE" --prepared "$HYDRATED_ROOT" --env-file "$ENV_FILE" --outbox "$PROJECT_ROOT/artifacts/paper-outbox.sqlite" --memory "$PROJECT_ROOT/artifacts/paper-memory.sqlite" --run-id final-paper --cycle-id rehearsal-001
```

Submission is explicitly enabled with `--submit-paper` on a new, stable cycle ID.
Closed markets skip orders. Reuse that cycle ID on restart; do not blindly replay
with a new ID. Confirm broker order status/fills and account reconciliation, then
save ten consecutive loop traces, tool use, risk outcomes, confirmed reflections,
restart behavior and daily ET equity for the actual live week. No real-money path
exists. No training, real-model pilot or broker rehearsal was run during this
implementation; those empirical release gates remain yours to execute.

`agent.notebook_api` supplies `NotebookSession`, tool registration/dispatch,
`llm_agent_decide`, `run_backtest`, `run_trading_loop` and reflection entrypoints.
Use an explicitly configured session, not a raw Torch model passed into a second
implementation. The facade preserves assignment concepts and delegates to tested
modules; see its docstrings and `tests/test_notebook_api.py` for minimal wiring.
