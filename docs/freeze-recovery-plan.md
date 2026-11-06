# Two-day assignment and paper-trading release plan

Planning baseline: November 6, 2026; freeze within 48 hours, approximately November
8. Confirm the exact cutoff/time zone before scheduling final jobs. This document
plans implementation; it does not claim the gates below have passed.

## Release objective and scope

Deliver the notebook's required Qwen 2.5 1.5B research/tool workflow, GOOD/BAD
trajectory SFT, persistent BM25 and LLM reflections, plus a runnable Alpaca
paper-only loop. Preserve the design's SEC/news evidence, independent risk,
historical availability, durable intents and reconciliation. Freeze one chosen
adapter and its complete decision loop. Do not interpret a model's invalid output
as a trade or invent a trade merely to demonstrate activity.

Include one corrected, bounded PPO training experiment and compare it with SFT,
as explicitly requested. DPO retraining remains deferred; retain its existing
results as a negative experiment. Preserve the original failed PPO run separately.
PPO correctness work runs alongside tools/broker implementation, then GPU training
follows the selected SFT initialization. Reserve final integration/freeze time.

## Changes required

| Workstream | Existing modules / proposed additions | Required change and evidence |
| --- | --- | --- |
| Decision contract | policy/core.py, datasets.py, compaction.py, compact_cli.py | Explicit final schema, allowed symbols/actions, short exact evidence references, token-aware budgets, targets that cite only visible evidence; strict validation after alias resolution |
| Data lineage | preparation.py, compact_cli.py | Preserve archives; rewrite prompts, SFT/DPO citations and PPO allowlists consistently; recompute identities/audits; do not copy stale evaluations or claim old training used new data |
| News/SEC | prepare_cli.py, data/news.py, data/sec.py | Feed bounded eligible headlines/summaries plus relevant filing excerpts to Qwen; archive raw articles; recover existing cached text first; clearly represent missing coverage |
| Required tools | proposed agent/tools/ | Register exact notebook interfaces with typed validated inputs, cutoff-bound replay adapters, live adapters, caching, quota/backoff and redacted errors |
| Agent workflow | policy/core.py, policy/loader.py, proposed agent/runtime.py | Bounded multi-turn research loop; record full prompts, assistant completions, tool results/counts; final decisions validated separately from risk/order execution |
| Reflections/RAG | retrieval/bm25.py, proposed agent/reflection.py | Persistent situation/reflection records, Qwen-generated stock/action/gain-or-loss lessons, delayed broker-confirmed outcomes, eligible-only retrieval and split isolation |
| GOOD/BAD trajectories | proposed agent/trajectories.py, training/core.py, training/execution.py | Collect actual base decisions in historical replay, label matured outcomes, train on GOOD completions with prompt/tool-result masking; count and explain every filtered sample |
| Paper execution | proposed agent/execution/{alpaca_paper,outbox,reconcile}.py | Replace blocked broker with paper-only client, durable intents/stable client IDs, pending reservations, partial-fill handling, startup/presubmit reconciliation and broker-status logs |
| Stateful replay | evaluation/portfolio.py, proposed replay runner | Generate from current portfolio, same decision/tools/risk path, coherent fills, correct marks and benchmark keyed by symbol; avoid static predictions masquerading as dynamic decisions |
| Evaluation operations | evaluate_cli.py, evaluation/model.py, scripts/ | Categorize failures, token counts/finish reasons, flushed progress/ETA, identity-checked per-example resume, small smoke option, Slurm stage wrappers |
| Submission | design.md, docs/, release command | Correct stale dates/status, requirement evidence map, immutable release manifest and instructions; lightweight notebook entrypoints only after modules pass |

Keep existing pure modules and tests where possible. Do not broadly refactor the
single-file demo during this deadline. New runtime/execution modules can compose
its validated concepts without turning the demo into the production runner.

## Required tool interfaces and modes

Expose the notebook signatures:

1. get_news_for_symbols(symbols, limit, date): Alpaca headlines/summaries, HTML
   unescaping, empty results valid, cached calls visibly report cache hits.
2. get_price_history(symbol, period): yfinance OHLCV with correct exchange dates.
3. get_rsi(symbol, interval="daily", time_period=14): Alpha Vantage RSI and signal;
   quota-aware.
4. get_macd(symbol, interval="daily"): EMA12 minus EMA26, EMA9 signal and histogram from yfinance
   closes, with sufficient warm-up.
5. get_global_quote(symbol): Alpha Vantage quote/change/volume with observed date,
   freshness and entitlement stated. Never label a free daily quote real time.
6. SEC filing research: the design extension, using the same registry/evidence
   contract and historical availability checks.

Also preserve notebook-facing register_tool/dispatch_tool/get_extra_context,
llm_agent_decide (decisions, full prompt, completion, tool-call count),
TradingMemory.add_memory/get_memories, reflect_on_trade, run_backtest and
run_trading_loop signatures through thin wrappers. Required notebook cells can
then call the tested implementation without maintaining separate logic.

Production tools must never expose submit-order capability to Qwen. In historical
mode, tool closures bind the decision cutoff; do not call latest quotes or today's
RSI as if available historically. Unsupported historical data returns missing,
or an explicitly labeled local historical calculation, never silently substituted
live data. Live tools expose only the actual entitled feed and age.

Free Alpha Vantage capacity is a shared daily budget, not per-tool capacity.
Implement RSI and GLOBAL_QUOTE exactly, but allow cache/missing/quota outcomes.
Preserve the full assignment universe; a deterministic, recorded research budget
can restrict deep tools to candidate symbols without deleting universe members.
Choose and freeze the candidate rule/budget during validation. Entitled Alpaca
market data and broker state supply order checks; the model cannot rely on a
stale Alpha Vantage quote as an executable price.

## Decision and compaction contract

Use short local evidence handles (for example E1, E2) with a deterministic mapping
to full IDs in the archived prompt envelope. The resolver only accepts exact
handles supplied for that decision. Keep legacy full-ID output support only for
exact visible IDs. No fuzzy matching or guessed replacement of SEC hashes.

Token budget includes system prompt, tool definitions, account state, retrieved
memory, selected evidence, tool turns and reserved final output. Prefer useful
headline/summary and filing sections; do not truncate raw JSON. Budget from the
actual tokenizer and profile memory on A30. Test 512/768 output caps initially
for legacy outputs, then size the shorter schema from measured completions.

When input selection changes, rebuild SFT completion citations, DPO chosen/rejected
citations and PPO allowed IDs from that exact input. Preserve original provenance
in sidecars. Validate complete target semantics, not merely json.loads().

Version the new prompt/alias contract. Existing adapter recovery under a new
prompt is an experiment, not proof of compatibility. Archive the old scores.
As tools return eligible evidence during a turn, update that decision's exact
visible-evidence allowlist; the current tool policy only validates initial packed
IDs. Do not accept arbitrary IDs just to accommodate dynamically fetched results.

## Compute strategy and assignment-label compliance

Reserve one GPU slot for a sequential critical path; implement independent CPU
work while it runs. Bound new inference before launching long jobs:

- Time 10 representative calls, including long SEC prompts and tool turns.
- Screen the final SFT and a small number of existing checkpoints (e.g. best-loss
  step 550 and an early checkpoint) on the same 20–50 validation cases.
- Stop a candidate on repeated malformed/unknown-reference output. Report failures
  rather than force a format-compliant HOLD to inflate validity.
- If a checkpoint passes, advance to about 100 representative cases and a short
  stateful replay. Only finalists get broader evaluation.

In parallel, collect the notebook-compliant baseline: roughly 60 historical
training sessions with actual frozen base-model decisions, tool transcripts,
portfolio state and matured equity-change GOOD/BAD labels. Use one bounded
portfolio research cycle per session rather than 100 independent full research
cycles by default, while retaining the full universe. Freeze this scheduling rule
and use it identically in before/after comparisons. Benchmark runtime first.

Do not relabel invalid outputs GOOD or copy oracle targets into base trajectories.
Train only on supported, valid GOOD completions; report GOOD/BAD/invalid counts
and all filtering. Account changes/rewards used for labels stay outside prompts.
Train/validation/test and reflection memory remain isolated.

Budget at most one bounded assignment SFT run, preceded by a short smoke stage
and validity checks. Prefer continuation from the existing SFT checkpoint if
permitted, while clearly reporting its prior counterfactual training history.
If assignment requires base-only initialization, choose that before the run.
This is a modeling/protocol choice, not something to switch silently afterward.
Completion-only loss, masked user/tool tokens, exact model identity and measured
trainable-parameter fraction must be recorded. Start with a measured budget
(roughly 100–200 steps, subject to corpus size and observed speed), not 1,000
steps by habit. Fit update count to data exposure and the remaining evaluation
window; do not promise a quality threshold from a step count alone.

Resume the same run only with identical data/model/optimizer contract. A new
dataset or prompt version is a new experiment; initializing from its adapter is
different from resuming old optimizer state.

An existing SFT may be a useful recovery candidate, but it does not retroactively
meet the notebook's GOOD-trajectory training protocol. If compliant collection/
training fails, document that blocker rather than claiming the old model passed.
Do not spend the remaining deadline retraining DPO. Include one corrected PPO
experiment under the protocol below. Preserve old failed results and identify the
new PPO experiment separately in the report.

## Included PPO experiment

Correct the objective, sampling probabilities/dropout consistency, value-head
seeding and full-vocabulary memory allocations before launching training. Add
tests for KL gradients, unchanged-policy ratios, action masks, invalid-action
rewards and resume identity. Use train-only episodes with prompt-visible citations
and start from the chosen SFT adapter, with an immutable matching SFT reference.
Do not resume optimizer state from the previous defective PPO run.

First profile one complete collect/update/checkpoint/reload cycle on the A30.
Use actual bounded prompt lengths, useful news/SEC content and reserved output;
do not solve OOM by blindly cutting context below prompt length. Collect multiple
independent episodes per rollout so updates are not based on one random decision.
Record the sampled-decision count as well as optimizer-update count.

Allocate a 10-update pilot, then continue the same experiment to an initial
100-update target only if numerical and generation gates pass. This is a measured
starting budget, not a claim that 100 updates is sufficient. Set a separate,
conservative PPO learning rate and tested KL control. Use a target of two GPU
hours and a four-hour hard training budget, adjusted from the measured pilot.
Stop on nonfinite loss, sustained invalid outputs, KL breach, or memory failure.
Do not consume the final packaging window to reach a nominal update count.

Save checkpoints every 10 updates plus the final checkpoint, and persist rollout
summaries, actions, parse/risk failure categories, reward components, KL, entropy,
value metrics, elapsed time and peak VRAM. Keep sampled actions and evidence
available for audit without secrets. Evaluate only a few declared milestones
(e.g. pilot, update 50 and final), not all checkpoints over the whole split.

Use the same fixed validation cases and stateful replay settings as base/SFT.
Report first-pass schema/citation validity, invalid/risk rejection rates, profit,
drawdown, turnover and latency. Stop a PPO continuation on material regression
against its SFT initialization. Select on validation; the chosen model may be
SFT or PPO. A PPO experiment that fails its gate is retained as a measured failure,
not promoted to the paper runner. Freeze and package both adapters if both pass,
with one explicitly chosen for trading. Run one held-out test of the chosen
release; any preregistered comparison of both on test must not trigger reselection.

## Proposed 48-hour schedule

Time windows are work budgets, not guarantees. GPU allocation, free API access,
model throughput and market availability must be checked in hour 0.

| Window | Parallel CPU work | GPU work / decision gate |
| --- | --- | --- |
| H0–4 | Complete transfers/checksums; explicit schema, visible evidence checks, progress/resume; source cache inventory | Profile 10 calls; screen existing SFT; stop full 3,700-case reruns |
| H4–12 | Tools/news/SEC budget and trajectory recorder; independently implement broker/outbox/reconciliation and PPO math/memory fixes | Once recorder/tools pass: base historical trajectory collection; fix contract before collecting labels |
| H12–24 | Integrate account-aware runtime, BM25/delayed LLM reflections; test failure/restart semantics | At most one bounded GOOD-trajectory SFT run; smoke first, continuing the same run only if the gate passes |
| H24–30 | Correct portfolio benchmark/marks; finish command wrappers; verify PPO numerical tests | Corrected PPO smoke/pilot then bounded continuation from SFT; preserve checkpoints and enforce runtime/validity limits |
| H30–36 | Complete requirement evidence map and failure/restart tests | Same-sample base/SFT/PPO comparison and finalist stateful replay; choose on validation only |
| H36–42 | Ten-iteration integration/restart rehearsal; real paper service connectivity/order-status rehearsal when market permits | Finish selected-model validation and inference latency/memory profiling; no new training runs |
| H42–48 | Freeze manifest/release; package logs/report and clean-start rehearsal | One frozen held-out historical test after selection; no tuning against it |

Start broker/tools work independently rather than waiting for model training.
Reserve the final six hours for verification/packaging. If timing slips, cut
extra checkpoint sweeps, DPO retraining and extended ablations, not required broker
reconciliation or assignment evidence. PPO stays an included bounded experiment;
its correctness gates and stop conditions still apply. A 60-session trajectory run that profiles
too slowly must trigger an immediate batch/call-budget decision, not an overnight
unbounded job.

## Acceptance gates

### Before training

- Complete checksummed artifacts; unique IDs, nonempty splits, valid hashes.
- Every target passes strict parser and cites only prompt-visible evidence.
- Historical tool/evidence/memory availability is tested, including SEC publication
  bounds and absent news; current quote tools cannot leak into replay.
- All required tools are registered and called in tests; API failure/empty/quota
  cases work, repeat news calls demonstrate cache hit.
- GOOD/BAD trajectory counts and filtering are auditable; selected GOOD labels
  come from actual executed simulated base decisions, not oracle action selection.

### Before selecting the model

- At least 99% of a fixed representative validation sample passes syntax, schema,
  symbol and exact evidence resolution; record first-pass vs any bounded retry.
  This is a release target, not statistical proof from a small sample.
- All malformed/unresolved outputs are blocked before orders. HOLD remains a valid
  model decision but must not conceal parser failure.
- Finite training loss is necessary but selection also uses invalid rate, actions,
  costs, profit/loss, turnover and drawdown. No model-profit guarantee.
- Use the notebook's 20-session before/after comparison on declared validation
  dates; this is distinct from the roughly 60-session training collection and
  final five-session historical test.
- Same initial cash, universe, periods, costs, risk, tools, context budget and
  memory reset for base and SFT; prompts receive evolving account state.
- JSON syntax/schema/citation failures and cash/holdings/quantity/risk rejections
  have separate counts. Latency is measured from actual calls; no seven-hour
  silent inference stage.

### Before paper release

- Endpoint/client enforce paper-only mode. Read account/clock/open orders/fills
  before decisions and reconcile again before submission.
- Durable intent exists before POST. A timeout triggers lookup by client order ID,
  not a blind retry. Unknown orders retain reservations and block duplicate risk.
- Cash/shares include open-order reservations and all decisions in the batch.
  Partial fills, rejection, cancellation and restart do not double-count fills.
- Invalid output, stale execution data, closed market and unavailable broker state
  produce explicit skipped/blocked events. No manual manufacture of decisions.
- Ten consecutive iterations show tools, retrieved memory, decision/risk status
  and continuing operation through recoverable failures.
- Broker-confirmed paper order/status evidence is required. Perform an authorized
  rehearsal only when market conditions permit; simulated fills do not satisfy
  this gate. November 8 is Sunday, so an equity fill rehearsal cannot be left to
  the final Sunday window. Broker API/restart tests can still run when closed.
- LLM reflection names stock, executed action and measured gain/loss, matures
  after the configured delay, and persists with source/time/experiment lineage.

## Trainer corrections and experiment status

DPO: preserve the correct SFT reference, verify completion/template boundaries,
use a separate conservative learning rate, add an explicit likelihood-preserving
anchor if adopted, and gate checkpoint selection on generated decision validity.

PPO: correct the differentiable KL objective, behavior/current probability
consistency and dropout handling; seed value initialization, stabilize value/
reward scale, use meaningful rollout batches, restrict vocabulary statistics to
necessary positions/chunks, and add validity/KL early stops. Require numerical
gradient/ratio tests and one real-model smoke run before any new training claim.
This corrected PPO run is included in the plan. Numerical correctness and a
measured pilot are prerequisites; an unsuccessful pilot must be reported honestly.

## Freeze and submission contents

Release manifest records git commit, model revision, adapter/tokenizer checksums,
dependency lock, training/label/source identities, chosen checkpoint, prompt/tool
schemas, evidence budgets/aliases, universe/candidate rule, risk thresholds,
execution policy, loop schedule, cache/quota/staleness settings, memory/reflection
rules and validation/test identities. Credentials stay external.

Provide one start/verify/export workflow, restart instructions, requirement-to-log
mapping, before/after metrics and honest optional-method failure analysis. If a
notebook artifact is required, add thin wrappers to tested modules, not a second
implementation. Keep weights/config immutable during the live week; permit only
new market data, broker state and automatic reflection memory under frozen rules.

The notebook lists November 9–13 as the live paper week. October 26–30 remains a
separate historical test; do not equate the two. Log daily ET account balances,
order history and final equity from Alpaca for the actual graded live week.

## Decisions/information needed now

Confirm exact freeze cutoff, graded live dates and whether continued SFT from the
current adapter is acceptable when transparently reported. Verify full artifact
transfer and raw-news cache availability. Keep a GPU slot available and identify
the persistent machine used for the paper loop: a Slurm training allocation may
not be an always-on live runner. Do not send credentials; use the local ignored
environment file and test secret presence without printing values.

## Provider references

- Alpha Vantage free capacity/entitlements: https://www.alphavantage.co/support/
- Alpaca orders: https://docs.alpaca.markets/us/docs/working-with-orders
- Alpaca client-order-ID lookup: https://docs.alpaca.markets/us/v1.4.2/reference/getorderbyclientorderid
