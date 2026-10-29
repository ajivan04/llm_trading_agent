# Training and experiment workflow

This repository trains only `Qwen/Qwen2.5-1.5B-Instruct` adapters and supports
research replay and Alpaca **paper trading only**. It has no real-money execution path.

## Environment

Create the locked environment after `uv.lock` has been generated:

```bash
uv sync --all-extras
```

For Colab, install from the repository root with:

```bash
pip install -e '.[data,train,notebook]'
```

Do not replace the pinned versions without rerunning unit tests and a GPU smoke test.
The base model and tokenizer default to revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`; changing it creates a different
experiment and must be recorded in the artifact manifest.

## Credentials

Training from an already-built local dataset requires no API secrets. Historical data
collection may use these environment variables:

| Name | Required for | Where to configure |
|---|---|---|
| `ALPACA_API_KEY` | Alpaca historical data and paper account | Local environment or Colab Secrets |
| `ALPACA_SECRET_KEY` | Alpaca historical data and paper account | Local environment or Colab Secrets |
| `ALPHA_VANTAGE_API_KEY` | Alpha Vantage news/data | Local environment or Colab Secrets |
| `FRED_API_KEY` | ALFRED/FRED macro vintages | Local environment or Colab Secrets |
| `HF_TOKEN` | Optional Hugging Face authentication/cache limits | Local environment or Colab Secrets |
| `SEC_USER_AGENT` | SEC request identification; not a secret | Environment or notebook configuration |

Never place values in a notebook, config file, CLI flag, git commit, dataset manifest,
experiment registry, or captured output. Locally, export them in the shell or use an
ignored `.env`; the project intentionally does not load `.env` automatically. In Colab,
open the key icon, create the names above, disable notebook access sharing, and read them
with `google.colab.userdata.get`. Do not print the returned values.

Alpaca credentials must come from a paper-trading account. The code must reject any
configuration that does not preserve `paper_trading_only=True`.

## Artifact flow

1. Fetch and archive raw point-in-time price sources with verified XNYS sessions.
2. Construct immutable snapshots and counterfactual labels from the next session.
3. Build a purged dataset artifact with `HistoricalDatasetBuilder` and verify its manifest.
4. Train SFT, use exported audited same-state DPO pairs, then optionally run fresh-rollout PPO.
5. Save adapters and manifests under an ignored artifact directory.
6. Select checkpoints on validation only.
7. Reserve the frozen release-test ID in the durable experiment registry before the one
   permitted release-test evaluation.

For the complete pre-event SFT → smoke → DPO → PPO → validation workflow, use the repository
script from a Colab L4 or another configured training machine:

```bash
chmod +x scripts/train_midterm.sh
SFT_STEPS=1000 DPO_STEPS=500 PPO_UPDATES=100 \
  SYMBOLS="AAPL MSFT NVDA" \
  ./scripts/train_midterm.sh
```

Omit `SYMBOLS` to use the full assignment universe. For a quick infrastructure rehearsal, set
`SFT_STEPS=2 DPO_STEPS=2 PPO_UPDATES=1`; use `SKIP_SMOKE=1` only when CUDA is unavailable.
The script does not run the held-out 2026 target-week test.

The preparation CLI now creates those inputs directly from daily yfinance bars. It also writes
the raw-bar archive, outcome audit, audited DPO pairs, and train-only PPO replay rows. Use a
subset during a rehearsal; omit `--symbols` for the assignment baseline universe. The requested
price range must include a warm-up period and one complete outcome session after `--test-end`.

For the 2026 midterm evaluation, use the event-regime mode. It keeps the 2026 target year out
of training, uses analogous Monday-through-Sunday event weeks from 2006, 2010, 2014, and 2018
for training, and uses 2022 as validation. The target window is November 2–8, with outcomes
allowed to mature through November 9. The regime is a market-timing split, not a political
prediction feature; election-related text is not added to policy prompts automatically.

```bash
uv run stock-agent-prepare \
  --start 2006-10-30 --end 2026-11-10 \
  --train-end 2018-11-13T23:59:59Z \
  --validation-end 2022-11-15T23:59:59Z \
  --test-end 2026-11-10T23:59:59Z \
  --output artifacts/dataset-v1 \
  --regime midterm-election \
  --symbols AAPL MSFT NVDA
```

Because the 2026 target week has not occurred at the time of this plan, create the frozen
training/validation artifact now with `--end 2022-11-15` and the same split boundaries; it will
contain no 2026 test rows. Record its `dataset_hash` and freeze it before November 2. After the
target week, build a separate evaluation-only artifact containing the 2026 rows and use the
frozen training hash in the release registry.

Freeze the training artifact, prompt, risk rules, seeds, and release-test ID no later than the
November 2 pre-event cutoff. Run the final target-week evaluation only after its market data has
matured; write it as a separate append-only evaluation artifact and do not rebuild the frozen
training artifact with target-week rows. Before then, use an earlier `--test-end` only for
development; do not treat that as the final election-week result.

This produces `artifacts/dataset-v1/dataset/` with verified `examples.jsonl`,
`train.jsonl`, `validation.jsonl`, `test.jsonl`, and `manifest.json`.

Start with a bounded SFT rehearsal and validate the saved adapter:

```bash
uv run stock-agent-train sft \
  --train artifacts/dataset-v1/dataset/train.jsonl \
  --validation artifacts/dataset-v1/dataset/validation.jsonl \
  --output artifacts/sft-smoke \
  --prompt-version v1 \
  --max-steps 2

uv run stock-agent-train smoke \
  --adapter-checkpoint artifacts/sft-smoke \
  --output artifacts/sft-smoke-check
```

Build DPO pairs only from independently simulated alternatives sharing the same state
hash, then initialize DPO from the selected SFT adapter:

```bash
uv run stock-agent-train dpo \
  --train artifacts/dataset-v1/dpo-train.jsonl \
  --validation artifacts/dataset-v1/dpo-validation.jsonl \
  --adapter-checkpoint artifacts/sft-smoke \
  --output artifacts/dpo-v1 \
  --prompt-version v1
```

PPO uses the included training-only historical environment. Its JSONL keeps future
execution and outcome prices private from the `messages` returned to the policy. Each
row must use `split: "train"` and provide `episode_id`, `messages`, `evidence_ids`,
`symbol`, `next_open`, `outcome_close`, and `starting_cash`. The environment configuration
contains only the path generated by preparation: `{"episodes_path":".../ppo-train.jsonl"}`.

```bash
uv run stock-agent-train ppo \
  --environment-factory agent.training.environment:create_training_environment \
  --environment-config artifacts/ppo-environment.json \
  --adapter-checkpoint artifacts/sft-smoke \
  --output artifacts/ppo-v1 \
  --prompt-version v1 \
  --updates 10
```

Compare a base model or frozen adapter on validation. The evaluator verifies the dataset
manifest, scores schema validity and counterfactual regret, and records every attempt. A test
run additionally requires a durable registry and consumes the one release-test reservation.

```bash
uv run stock-agent-evaluate \
  --prepared artifacts/dataset-v1 \
  --split validation \
  --adapter artifacts/sft-smoke \
  --output artifacts/eval-sft-validation
```

Run `python3 -m unittest discover -s tests -v` before training. The GPU smoke command in
the training CLI should be run before a long job; it verifies model load, optimizer steps,
adapter save/reload, and deterministic greedy generation. No training command is run by the
repository tests; run the smoke command yourself on the Colab L4 before increasing steps.
