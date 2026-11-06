# Training and paper-trading readiness audit — November 6, 2026

This is a read-only diagnosis of production code and downloaded
`artifacts/mixed-2026-test-v5` artifacts. No model training, broker orders, or
evaluation reruns were performed. Findings below distinguish observed behavior
from hypotheses requiring a controlled experiment.

This records the pre-fix state. Subsequent breakpoint 1 implementation addresses
the citation/compaction, preflight and diagnostic-accounting findings; see
[review-breakpoints.md](review-breakpoints.md) for the current review status.
The remaining PPO, tool integration and paper-runner findings still apply.

## Observed model failures

All four prediction files contain 3,700 readable records.

| Model | Parseable JSON responses | Evaluator-accepted decisions |
| --- | ---: | ---: |
| Base | 3,624 | 0 |
| SFT | 3,674 | 797 |
| DPO | 1 | 0 |
| PPO | 0 | 0 |

Base often uses `security`/`quantity` instead of `symbol`/`qty`, omits evidence
citations, or emits multiple contradictory decisions for one symbol. The system
prompt in `agent/datasets.py:HistoricalDatasetBuilder.SYSTEM` requests strict JSON
but does not specify the required schema. A JSON syntax check does not establish
decision validity.

SFT largely learned the JSON structure. On 1,724 readable local validation
examples, 1,519 outputs fail specifically on unavailable evidence citations, 193
pass, and 12 are malformed JSON. For AAPL, the generated SEC metadata hash differs
from the target metadata hash. Compaction removes filing metadata/older filings
but leaves the original completion unchanged. In this readable subset, 1,534
targets cite evidence absent from the policy-visible prompt. This is a confirmed
training/inference contract problem, not simply insufficient training time.

DPO responses are generally very short fenced JSON with required fields missing.
Removing fences would not restore symbol, quantity or evidence citations. Saved
training state shows chosen and rejected response likelihoods both deteriorating
while preference loss improves. Best loss checkpoint is step 250; lower DPO loss
does not establish preservation of executable decisions. The installed TRL clones
a reference adapter; `ref_model=None` alone is not evidence of a broken reference.
The shared 2e-4 learning rate, weak action-only preferences and absent SFT anchor
are plausible causes of degradation, to test with short checkpoint experiments.

PPO logs contain 100 updates, of which 99 have invalid proposals. Each replay
episode terminates after one decision, so this represents about 100 sampled
decisions, not eight decisions per update. The run does not justify more updates.
The update uses a signed sampled `mean(new_logp - reference_logp)` directly as a
loss. Its gradient on fixed sampled tokens does not implement the intended
distributional KL constraint. Collection uses temperature/top-p sampling, while
statistics use raw logits; collection is in eval mode and updates in train mode
with LoRA dropout. These inconsistencies require correction and numerical tests
before treating PPO as reliable.

## Generation budget and memory

Using the complete cached Qwen tokenizer, reference completions in the readable
validation subset have median length 385 tokens and maximum 569; 1,438 exceed the
256-token generation budget. SFT has only 26 non-JSON outputs, so truncation is a
secondary issue, not the explanation for its thousands of citation failures.
PPO's malformed text also cannot be repaired by just increasing output length.

`QwenPPOOptimizer._statistics` creates full-sequence float32 vocabulary logits,
log-probabilities, probabilities and entropy, even though the loss later selects
only generated action positions. At roughly 152k vocabulary entries and 4,096
tokens, each full float32 vocabulary tensor is about 2.5 GB. This is a concrete
source of excessive VRAM demand. Reducing context without a token-aware evidence
budget leads to the separate context-exhaustion exception seen on Turing.

## Evidence and evaluation limitations

`agent/prepare_cli.py:evidence_provider` adds only `aggregate_news(...)`; it does
not add raw Alpaca article headlines/summaries. The free provider has no sentiment
scores, so counts/age/null sentiment do not supply the intended news-language
evidence. Raw article storage must be checked before planning offline recovery.

Compaction copies trained outputs and evaluation directories into a new artifact
and changes prompts without a complete new model/data lineage contract. Existing
evaluation files may be stale, and the Slurm evaluation wrapper skips them when
present. New experiments need distinct evaluation identities/output directories.

Training prompts assume independent $100,000 accounts holding five shares of the
current symbol. Portfolio replay starts from cash and carries holdings. It does
not regenerate decisions using the replay's evolving account. A schema-valid
decision can therefore be rejected by portfolio checks. `invalid_decisions`
mixes parse failures and execution rejection; 3,595 does not mean 3,595 JSON
failures. The validation replay covers 37 sessions, not the five-session test.

Targets are next-session counterfactual buy/sell/hold labels with boilerplate
reasons. They are not tool-use trajectories or evidence-specific reasoning
supervision. Counterfactual labels can be used as supervised outcomes, but must
remain outside prompts and must not be described as established trading skill.

## Assignment and actual Alpaca execution

The model remains Qwen/Qwen2.5-1.5B-Instruct. Historical adapters, strict validation,
BM25 components, independent risk checks, and split guards are useful foundations.
However, `trading_agent.py:AlpacaPaperBroker.submit` raises NotImplementedError.
There is no executable paper runner connecting policy, live tools, account state,
risk approval, durable order intent, submission, reconciliation and fill status.

The assignment notebook requires concepts absent from the current evaluated path:

- Task 1: multi-turn tool calls, decision/prompt/completion/tool-count outputs.
- Task 2: registered data tools, cached Alpaca news, yfinance history/MACD and
  the specified Alpha Vantage RSI/GLOBAL_QUOTE functions (confirm any allowed
  substitution with the instructor).
- Task 3: persistent BM25 memory plus meaningful Qwen-generated trade reflections.
- Task 4: quality-labeled backtest trajectories, completion-only LoRA, comparable
  before/after evaluation and explanation of changed labels/sample counts.
- Task 5: live paper loop, delayed reflections, retrieved memory, order status,
  ten consecutive iterations and daily logs/order history/final balance.

DPO and PPO are optional challenges. The notebook lists November 9–13 as the live
week; the user selected October 26–30 as a historical test. These can coexist but
must be explicitly distinguished and the actual graded live dates confirmed.

## Recovery gates before another long training run

1. Recover complete audit/data files; record source commit, versions, byte counts
   and SHA-256 checksums. Preserve existing adapters and checkpoints.
2. Revalidate existing predictions without GPU, separating JSON/schema/symbol/
   citation errors from cash/holdings/quantity rejection. Preserve original scores.
3. Build one prompt/output contract: explicit schema, short stable evidence aliases,
   only visible citations, exact provenance mapping and tokenizer-aware input/output
   budgets. Update SFT/DPO targets and PPO allowed evidence consistently.
4. Restore actual article text and eligible SEC excerpts; make missing evidence
   explicit. Preserve source archives and point-in-time availability.
5. Try existing checkpoints on a fixed small validation sample before deciding
   whether SFT must be retrained. Do not alter test data for checkpoint selection.
6. Gate any DPO/PPO pilot on executable-output validity, unchanged pre-update PPO
   ratios, correct loss gradients and held-out action quality. Stop a degrading
   pilot before spending hours; publish progress and resumable predictions.
7. Implement and test the paper-only runner, including restart-safe outbox,
   reconciliation, broker holdings, pending exposure and reflection integration.
   Broker submission requires explicit operational authorization.

## Additional information needed

Local files are incomplete and some are growing during this audit, consistent
with a transfer still in progress: outcome-audit.jsonl truncates before validation;
at the time sampled, validation.jsonl had 1,724 readable rows followed by a
truncated record despite Turing reporting 3,700. The downloaded SFT tokenizer JSON
was also truncated at the time inspected.
Final adapter safetensors files have complete lengths according to their headers;
all prediction files are readable. Do not infer that the Turing originals are
corrupt from a damaged local transfer.

Needed from Turing: complete v4/v5 split data, DPO pairs, PPO rows, outcome audit,
evidence archive/cache containing article text, manifests, final trainer states,
PPO metrics/state JSON, exact submitted scripts/environment overrides and job
logs. Transfer checksums and byte counts before copying huge optimizer files.
No credentials or .env contents are needed. Confirm graded live dates and whether
the instructor permits own-system equivalents for notebook tool requirements.
