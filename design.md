# Trading agent: assignment scope and simplification review

Scope updated November 8, 2026. The assignment notebook is the requirements
reference; it is not an implemented trading system. The supported additions are
SEC filings, market-news discovery and PPO. DPO, vector memory, confidence sizing,
stop-loss/take-profit and extra provider examples are outside this scope.

## One workflow

`cached market sources → Qwen research/tools → risk → replay or paper execution`

`actual base trajectories → GOOD-only SFT → optional PPO → matched validation`

The same policy/tool implementation is used for collection, comparison and paper
research. The simulator and Alpaca each provide execution; Qwen cannot place orders.
[project.json](project.json) owns paths and budgets; `python -m agent` owns stage
selection; one Slurm wrapper runs the same commands on Turing. There are no
stage-dependent shell exports or auto-installation on GPU nodes.

## Critical comparison with the starter

| Area | What was unnecessarily complex | Decision |
| --- | --- | --- |
| Workflow | Old counterfactual training and new genuine trajectories both looked current; many scripts had contradictory defaults | Delete superseded job scripts and phase guides; one supported workflow |
| Data | Fresh market research required five obsolete training-target files, which were copied through every repair step | Source is `source.json`, prices and evidence; old sources read in place without loading old targets |
| SFT | Repeated custom dataset validation, a separate preflight on every run, unused TRL requirement | One shared corpus validator; internal check before weights; standalone preflight only when requested |
| Checkpoints | Saving every two steps also forced full validation every two steps | Separate save/evaluation cadence; retain real optimizer/scheduler/RNG resumes |
| Model input | All 100 precise marks plus mandatory evidence overflowed the prompt before generation | Full whitelist stays; initial deep context is bounded; other stocks are researched through tools |
| Model output | Formatting alone invalidated otherwise valid JSON in fences or prose | Extract one unambiguous JSON response, then validate its actual content; never invent HOLDs or trades |
| Notebook facade | Validation-only backtest, incompatible collector, 100-iteration/60-second limits | Support train collection and notebook log/update_outcome; finite multi-hour loops; continue explicitly recoverable failures |
| Risk metadata | All 100 stocks shared an unknown sector and one 30% exposure cap | Use the grouping labels supplied by the starter, consistently in replay and paper |
| Optional features | Code volume obscured missing functionality | Add actual market-wide discovery; keep PPO optional and describe its different training task honestly |
| Releases | Immutable packaging and one-shot historical test were presented as prerequisites for any demo | Keep them as final assessment tools; basic research/replay/paper runs do not depend on packaging |

Do not equate a large test count with successful model behavior. The last real
smoke had six pre-generation failures; a native-tokenizer regression reproduced a
prompt-budget defect, but the corrected real-model smoke still has to run.

## Requirement mapping

- **Task 1:** pinned Qwen, multi-turn registered tools, four decision-return values,
  actual full conversations and raw completions. Invalid output is recorded, not
  silently converted to a successful trade. Explain HOLD examples in the report.
- **Task 2:** cached Alpaca news; yfinance OHLCV and local MACD; free Alpha Vantage
  RSI and GLOBAL_QUOTE live; clearly labeled local historical equivalents. A free
  daily quote is not executable real-time pricing. SEC is the extra source.
- **Task 3:** persistent, time-scoped BM25 memory and Qwen-generated lessons.
  Paper lessons currently use delayed confirmed **closed-lot** outcomes, not the
  starter's delayed portfolio mark. This is a deliberate attribution difference:
  a buy-and-hold run may produce no new lesson; show actual retrieval/reflection
  evidence and disclose this limitation in the report.
- **Task 4:** base-model collection with real tool use, actual simulated fills,
  GOOD/BAD outcome labels, GOOD-only final-completion SFT, measured LoRA fraction
  below 0.1%, finite loss and matched base/SFT comparison.
- **Task 5:** current account, news/context, memory, decisions, paper orders,
  reconciliation and delayed reflections. A finite multi-hour schedule and daily
  ET logs are supported. Show ten consecutive iterations and real paper order
  status, then November 9–13 daily logs/order history/final balance.
- **Market discovery:** Alpaca news without `symbols`, optional local keyword
  filter, mentioned whitelist tickers and article text. It never grows the
  whitelist automatically. Replay only searches available archived whitelist
  news; it does not claim to reconstruct an unrestricted historical market feed.
- **PPO:** optional final-decision training initialized from completed SFT. It
  uses one-stock cash/held-share scenarios with news/SEC context, not multi-turn
  portfolio/tool RL. Compare through the same full research validation loop;
  optimizer reward alone is not evidence of deployment performance.

## Keep these safeguards

Paper-only endpoint enforcement; durable intent before POST; client-order-ID
reconciliation after uncertainty; independent cash/share/risk limits; no future
replay tools; created/updated/acceptance-time availability; nominal historical
share units and separate split/dividend events; train/validation/test/memory
isolation; complete checkpoint resumes; exact model and data identity.

The starter's same-close fill, permissive fallback trades and unlimited retry
patterns are not copied. We decide at a completed close and execute at the next
open with costs. GOOD requires a researched executed trade and **positive net**
equity change; the starter permits zero profit. These differences must be stated,
not hidden as exact notebook equivalence.

Exact short evidence references remain a provenance extension. They are checked
against what the model actually saw, with no fuzzy matching or fabricated citation.
This adds complexity beyond the rubric, but avoids claiming support from an SEC
filing or article never shown to Qwen. The initial model-facing text is bounded;
large source IDs, full archives and future adjustment factors remain outside it.

## Data, splits and evaluation

Existing normalized/hydrated sources remain usable. New source-only bundles omit
obsolete targets. The actual new SFT corpus still has its own verified split
manifest, because that protects what is trained rather than merely loading data.

Use 60 eligible 2026 training sessions and optionally 20 October sessions from the
latest earlier midterm year (2022 for this archive), with independent portfolios.
The validation plan has 20 sessions, separate from training. Report this optional
cohort augmentation; it is not a notebook requirement. Starting cash is $100,000.

Validation reports all invalid outputs, holds, rejected orders, executed trades,
ending equity, dollar P&L, benchmark return, drawdown and inference latency.
Choose on validation only. The October 26–30 historical test is a separate,
one-shot assessment under frozen code/weights/settings, not the November live week.

The legacy counterfactual preparation, static prediction scorers and DPO Python
modules remain only to reproduce old experiments and existing tests. They are
not public stages or prerequisites. Deleting shared contracts/simulator code just
because it also serves old tests would create another untested trading system.
