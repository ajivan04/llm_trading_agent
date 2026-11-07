# Account-aware validation research evaluation

This path calls the actual bounded research policy with the evolving replay
portfolio. It preserves every valid HOLD, malformed output and risk rejection in
the denominator and measures ending wealth from $100,000 after modeled costs.
It does not score the old static per-symbol prediction files or filter to GOOD
training trajectories. It is historical simulation, not Alpaca fills.

The schedule uses consecutive validation exchange sessions. The policy sees only
eligible completed bars, raw news/SEC text and current account state. Independent
risk runs at decision close and execution open; fills use the next exchange
session's open and positions are marked at that session's close. Output includes
profit, drawdown, turnover, schema/citation failure rates, execution rejections,
recorded decision-cycle latency and initial prompt token counts. No schema repair
converts invalid output to an intentional HOLD.

The default benchmark holds a fractional, equal-weight fixed universe from the
first execution open with zero benchmark transaction costs, reinvesting no
cash dividends. It adjusts held units for splits and credits subsequent ex-date
dividends. A requested single-symbol benchmark must have archived marks; the
cohort never changes using later coverage. Strategy costs and benchmark assumptions
are in the manifest.

Memory is **disabled** in this CLI comparison so base/SFT/PPO have an identical
empty memory seed. This is explicitly recorded; it does not evaluate reflection
enrichment. Paper runtime memory has a separate run/fold namespace. To compare
memory-enabled policies, first declare and freeze a common initialization and
matched automatic-update protocol; do not reuse another model's enriched file.

Only validation is exposed. This command does not claim a held-out-week score.
The October 26–30, 2026 test week still requires explicit schedule alignment and a
separately frozen release gate. No adapter selection may depend on test outcomes.

Run from the updated repository using the existing `.venv`; no `uv` is needed:

```bash
cd ~/llm_trading_agent
export PREPARED_ROOT="$HOME/llm_trading_agent/artifacts/complete-news-sec"
export EVALUATION_ROOT="$PREPARED_ROOT/research-eval-sft"
export POLICY_ADAPTER="$PREPARED_ROOT/sft"

# CPU archive/coverage/schedule checks; does not load tokenizer/model weights.
.venv/bin/python -m agent.research_eval_cli \
  --prepared "$PREPARED_ROOT" --output "$EVALUATION_ROOT" \
  --adapter "$POLICY_ADAPTER" --sessions 20 --plan-only

# A two-session prefix of the same fixed 20-session plan.
sbatch --export=ALL scripts/research_evaluate_turing.slurm
```

Review the per-decision flushed log events, journal prompts/tool results, strict
final outputs, source coverage and `research_metrics.json`. Zero valid outputs or
zero fills are results to investigate before extending; don't relax the schema or
invent trades. The downloaded v5 archive has news aggregates but no raw articles
and correctly fails the default NLP-source gate. Refresh raw articles first.

```bash
# After review, continue the same plan and candidate to all 20 sessions.
sbatch --export=ALL,EVAL_RESUME=1,EVAL_MAX_SESSIONS=20 \
  scripts/research_evaluate_turing.slurm
```

Use separate output directories for base, SFT and corrected PPO. Omit
`POLICY_ADAPTER` for base; set it to one immutable completed adapter for SFT/PPO.
Keep the same evidence archive, schedule, token budget, tools, independent risk
and benchmark. Compare all outcomes, including invalid outputs, before selecting
on validation. A partial prefix is marked `complete_plan=false` and is not a
full-run model comparison.

The journal binds source hashes, model/adapter identity and protocol. Changed
source files, adapter, code, schedule or settings require a new experiment. A
restart resumes saved account state and calls only the unfinished prefix. Summary
files are replaced atomically after a completed invocation; the verified journal
is the source of truth if a process stops before writing a summary. No provider
calls, model inference or training were run while implementing this path.
