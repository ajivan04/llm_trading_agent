# Breakpoint 2 — corrected PPO, ready for review

Current commands and final protocol: [final-retraining.md](final-retraining.md).
The material below records the earlier breakpoint; use the final runbook for retraining.

No production Qwen training, provider requests or Alpaca orders were run. Tests
use a tiny randomly initialized Qwen2 with real LoRA layers on CPU. The full-size
1.5B A30 memory/throughput and real SFT output-validity smoke remain pending.

## Optimizer changes

- Exact differentiable categorical KL(pi || frozen SFT reference) replaces the
  signed sampled log-probability loss. Analytic-gradient tests cover identical
  and nonidentical distributions.
- Collection uses an explicit neutral generation configuration: temperature/top-p
  1, top-k 0, no inherited vocabulary/repetition/length penalties. Old/current
  probabilities match this raw distribution. Nonraw sampling is rejected.
- LoRA and attention dropout are disabled consistently. Train mode retains
  nonreentrant activation checkpointing. A ratio discrepancy greater than 5%
  before the first optimizer step fails rather than silently updating.
- Qwen logits_to_keep projects only causal action positions through the LM head.
  A decoder hook captures final hidden states for values without all hidden layers.
  Reference forward has no gradients. Backward runs per transition instead of
  retaining an entire rollout graph. Float32 reductions are chunked over action
  positions; completion autograd intermediates still live until that backward.
- Seed and zero-initialize the value head. Actor/critic CLI learning rates default
  to 1e-5/1e-6, entropy bonus zero. These are pilot choices, not proven optima.
- Eight independent episodes per rollout by default. Explicit episode_count and
  terminal/truncated boundaries isolate GAE. Disjoint seed windows avoid replaying
  almost the same reset seeds on consecutive updates.

## Operations and recovery

- `--updates` is a total target: extending 2 to 10 performs eight more updates.
- Progress is flushed during full-dataset token preflight, collection and updates.
  Metrics/rollout journals are fsynced per record and include samples, action-token
  counts, seeds, KL, validity and categorical parse/citation/risk failures.
- Defaults stop after two consecutive batches above 50% invalid proposals, after
  an update exceeds exact reference KL 0.1, on nonfinite loss/gradients, or at a
  runtime budget checked between operations. Do not disable a failing gate simply
  to obtain a completed model.
- Stopped runs have training_status.json/recovery state but no completed
  training_manifest.json; CLI exits 2 so afterok dependents do not continue.
  Extending a healthy pilot archives its previous completion manifest. A
  safety-stopped run requires review/new experiment; budget stops can resume.
- Checkpoints bind implementation/model/tokenizer/data/prompt/settings, numeric
  mode, package versions and original SFT reference. Verify adapter/reference/
  tensor hashes, run fingerprint, policy version, update index and consumed IDs.
  Legacy defective optimizer states are rejected before allocating model weights.
- Fresh SFT tensor files are opened/validated on CPU; adapter directory hashes
  before and after loading reject an actively changing transfer.
- Construct all replay rows and size every prompt with the actual tokenizer before
  loading Qwen. Reserve the complete output allowance; never truncate/drop episodes
  to make the preflight pass. Recheck source hashes after preflight.

## Verification

The project-venv suite runs 230 tests, all passing without skips. Eight actual
Torch/Qwen2/PEFT tests verify KL gradients, chunk equivalence, causal logits and
LoRA-gradient parity with full forward, raw generation probabilities, unchanged
ratios, frozen reference, activation checkpointing, seeded zero-value factory,
separate optimizer groups and exact checkpoint recovery including the next Adam
update. Operational tests cover total-target resume, incremental progress,
runtime/validity/KL stops, no false completion and pre-GPU context rejection.

Independent experiment review verified the numerical path and ran the suite.
The final integrated review found no blocking issues; its last focused run also
passed all nine operational-runner tests.

```bash
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m unittest tests.test_qwen_ppo_numerics tests.test_ppo_runner
bash -n scripts/ppo_turing.slurm scripts/train_turing.slurm
```

## Turing smoke, pilot and continuation

Run only after review and input/adapter verification. The old SFT may fail on the
new short-reference prompt. Breakpoint 3 still supplies actual news, bounded
inputs and notebook GOOD/BAD trajectory training. This smoke does not establish
trading skill or assignment compliance.

The job uses the installed .venv directly, requires CUDA, and needs no uv. Choose
a new output directory; prepared artifacts intentionally contain no model copies.

```bash
cd ~/llm_trading_agent
PREP="$HOME/llm_trading_agent/artifacts/mixed-2026-contract-v6"
SFT="$HOME/llm_trading_agent/artifacts/mixed-2026-test-v5/sft"
PPO_OUT="$HOME/llm_trading_agent/artifacts/ppo-v2-pilot"

sbatch --export=ALL,PREPARED_ROOT="$PREP",SFT_ADAPTER="$SFT",PPO_OUTPUT_ROOT="$PPO_OUT",PPO_UPDATES=2 \
  scripts/ppo_turing.slurm
```

Review the actual smoke results before extending:

```bash
cat "$PPO_OUT/training_status.json"
tail -5 "$PPO_OUT/ppo_metrics.jsonl"
tail -5 "$PPO_OUT/ppo_rollouts.jsonl"
```

Require finite losses, stable initial ratios, acceptable output/risk failure
counts, bounded KL and measured peak_cuda_memory_bytes. Positive reward alone is
insufficient: initial holdings can move in price even when a proposal is invalid.

```bash
sbatch --export=ALL,PREPARED_ROOT="$PREP",SFT_ADAPTER="$SFT",PPO_OUTPUT_ROOT="$PPO_OUT",PPO_RESUME_FROM="$PPO_OUT",PPO_UPDATES=10 \
  scripts/ppo_turing.slurm
```

Only after the pilot and SFT comparison pass, extend to total 100:

```bash
sbatch --export=ALL,PREPARED_ROOT="$PREP",SFT_ADAPTER="$SFT",PPO_OUTPUT_ROOT="$PPO_OUT",PPO_RESUME_FROM="$PPO_OUT",PPO_UPDATES=100 \
  scripts/ppo_turing.slurm
```

Keep data, prompt, dtype, sampling, learning rates and PPO settings unchanged on
resume. Default budget is two hours checked between operations; set
PPO_MAX_SECONDS explicitly when allocating the reviewed four-hour cap. If root
state was torn mid-save, hashes reject it: select a verified checkpoint-N and a
new output directory, preserving newer journals. No automatic selection of
incomplete newest folders or overwriting later metrics.

Same-directory continuation restores the invalid streak from its matching metric
journal. A new-directory recovery branch does not inherit that journal
automatically; review its checkpoint/health history explicitly. Do not copy newer
unmatched records into an older checkpoint's new journal.

STAGE=ppo in train_turing.slurm delegates to this workflow and defaults to
OUTPUT_ROOT/ppo-v2, separate from legacy ppo results. Set SFT_ADAPTER explicitly
when the prepared artifact lacks sft. The new adapter is PPO_OUTPUT_ROOT/adapter.
The direct job needs no uv; train_turing.slurm still performs its uv setup first.
The legacy evaluation wrapper is not the planned stateful model comparison;
breakpoint 5 will use explicit selected adapter paths.

Later grammar-constrained PPO collection would require identical masks and
normalization in probability calculations. Do not attach constrained generation
to the current raw-distribution statistics.
