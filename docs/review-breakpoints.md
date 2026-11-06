# Implementation review breakpoints

This is the review ledger for the two-day recovery plan. Each breakpoint ends with
a focused diff, tests, an independent read-only audit, limitations and commands.
No long training job is part of breakpoint 1. The source artifacts and existing
adapters remain unchanged. Breakpoint 1 is committed in functional groups; later
breakpoint changes remain separate for review before committing.

| Breakpoint | Scope | Acceptance / review evidence |
| --- | --- | --- |
| 1 — evidence/data contract | Exact visible citations, short references, safe artifact repair, CPU preflight, diagnostic scoring/accounting | Reviewed and committed; details below |
| 2 — corrected PPO | KL loss/gradients, behavior probabilities, dropout, value seeding, action-only statistics, bounded rollout batches and stop conditions | Implementation started; numerical tests plus short real-model collect/update/save/reload smoke; no full run before that |
| 3 — assignment research/training | Required callable tools, actual news/SEC text, bounded context, GOOD/BAD trajectory recorder, SFT data masks | Provider/cutoff/quota tests; captured tool trajectory; auditable GOOD counts; one bounded SFT run after preflight |
| 4 — paper runner/RAG | Alpaca paper-only execution, outbox/reconciliation, pending exposure, BM25 and delayed LLM reflections | Failure/restart/partial-fill tests; ten iterations; broker-confirmed order/status when market permits |
| 5 — release/freeze | Stateful base/SFT/PPO comparison, small-to-large validation, daily evidence exports and release manifest | One selected adapter/loop, measured latency/validity/P&L, hashes and final held-out protocol |

## Breakpoint 1: implemented behavior

Functional commits: `d0f425a` evidence contracts; `fefeb77` training preflight;
`b628b9c` artifact repair; `ddc70ce` diagnostics and portfolio accounting.

- `agent/policy/evidence.py` extracts only explicit user/tool evidence records,
  checks cutoff/availability and conflicts, and ignores archive indexes/assistant
  claims. Short aliases require exact original-transcript binding before resolving
  back to full IDs. No guessed IDs or approximate matching.
- New prepared system prompts state the required JSON decision schema.
- `agent/compact_cli.py` exports only prepared files to a new directory. It rejects
  overwrites, nested output paths, incomplete files and sources changing during
  transformation. Models/checkpoints/evaluations are not copied.
- Repaired SFT/DPO targets cite the exact selected evidence. Originally invented
  citations fail rather than being erased. If evidence removal affects a target,
  replace its old rationale with a neutral historical-label/provenance statement
  and record that repair; do not invent evidence-based explanations.
- DPO candidate/state/audit identities are rebuilt. PPO allowlists and outcome
  audits use the selected visible IDs. The source's full provenance remains in
  sidecars with source/prompt/evidence/content hashes.
- Optional short-reference exports use E1/E2-style IDs consistently across all
  targets and prompts. Recompacting an already aliased source is rejected; retain
  the original full-ID source.
- SFT/DPO/PPO CLI checks occur before model allocation. SFT checks target semantics
  and verified dataset splits; DPO compares the serialized pair with independently
  reconstructed candidates and checks visible citations; PPO validates all replay
  rows, cutoffs, symbol support and exact visible allowlists before loading Qwen.
- Evaluation intersects prompt-visible IDs with the outcome audit, categorizes
  failures and records scoring version `visible-evidence-v2`.
- Portfolio diagnostics separate policy failures from execution rejections, require
  actual closing marks for holdings, and compute a daily symbol-matched benchmark
  using a fixed cohort available at entry. Static-prediction replay remains static;
  generating from evolving account state belongs to breakpoint 5.
- `agent.audit_cli` provides CPU-only target and saved-validation-prediction
  checks. Saved predictions receive a file SHA-256 and available evaluation
  metadata; mismatched dataset identity is rejected. Missing metadata is labeled
  unverified. Original evaluation outputs are not modified.

## Verification and independent audit

Full suite at handoff: 211 tests run, one skipped; no failures. Focused regression
coverage includes future/off-prompt citations, poisoned alias maps, hidden rationale
removal, both full/short-ID artifact export, valid rebuilt DPO audits, rejection
before model loading, unchanged source files and daily benchmark accounting.
Changed Python files pass Ruff; Python compilation and Slurm shell syntax pass.
The offline demo remains covered by its integration tests.

Independent review found the alias-binding and rationale-retention defects during
implementation; both were corrected and re-tested. The final read-only review
reported no remaining breakpoint-1 blockers and independently ran 44 focused
tests successfully.

The real local v5 artifact did not pass integrity loading: `invalid dataset row
3951` in examples.jsonl at the time checked. Several files had been growing during
the earlier audit. Finish/check the transfer; do not truncate the tail to force
the manifest to pass. This is not evidence of damage to the Turing originals.
Full production-artifact verification remains pending that complete input.

## Review commands

Locally, inspect the changed files and run:

```bash
python3 -m unittest discover -s tests
python3 -m compileall -q agent
bash -n scripts/compact_turing.slurm scripts/audit_turing.slurm
```

After these changes are available on Turing, audit the existing artifact using
the installed project environment (no uv or GPU required):

```bash
cd ~/llm_trading_agent
IN="$HOME/llm_trading_agent/artifacts/mixed-2026-test-v5"
.venv/bin/python -m agent.audit_cli --prepared "$IN"
```

Exit 0 means target contracts passed; exit 2 means the artifact loaded but targets
were unsupported; exit 1 means integrity/input validation failed. This does not
certify profitability, assignment-label compliance or live execution readiness.

Read-only diagnostic rescore of existing SFT generations:

```bash
.venv/bin/python -m agent.audit_cli \
  --prepared "$IN" \
  --predictions "$IN/eval-sft/predictions.jsonl"
```

The same audit can be queued as a CPU job:

```bash
sbatch --export=ALL,PREPARED_ROOT="$IN" scripts/audit_turing.slurm
```

If the source is complete, create a separate repaired artifact:

```bash
OUT="$HOME/llm_trading_agent/artifacts/mixed-2026-contract-v6"
sbatch --export=ALL,INPUT_ROOT="$IN",OUTPUT_ROOT="$OUT",SHORT_EVIDENCE_IDS=1,MAX_NEWS=4,MAX_FILINGS=1,MAX_TEXT_CHARS=400 \
  scripts/compact_turing.slurm
```

After that job completes:

```bash
.venv/bin/python -m agent.audit_cli --prepared "$OUT"
```

Do not run training on it yet. These are repaired counterfactual targets, not the
notebook's GOOD/BAD trajectory dataset. New news inputs, token-aware budgeting and
PPO corrections still need their breakpoints. Existing models are intentionally
absent from the new directory; their original paths remain intact.

## Next review: breakpoint 2

Implement corrected PPO objectives and probability accounting, reduce memory
allocations without silently discarding context, and test collect/update/restore.
Record pilot budget, starting/reference SFT identity, sampled decisions/update,
early stops and validation milestones. Run the short GPU smoke only after review;
do not resume the previous defective PPO optimizer state.
