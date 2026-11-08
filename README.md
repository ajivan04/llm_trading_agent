# Qwen trading agent

The assignment's research/tool loop, GOOD-only SFT, BM25 reflections and Alpaca
paper trading, extended with SEC filing research, market-news discovery and an
optional PPO experiment. Policy model: `Qwen/Qwen2.5-1.5B-Instruct`.

One configuration: [project.json](project.json). One run directory contains
collections, the GOOD corpus, adapters, comparisons and paper logs. Existing
`artifacts/final-nlp-source` is usable as-is; do not download or rebuild it again.

```bash
.venv/bin/python -m agent check
sbatch scripts/run_turing.slurm collect --smoke
# Inspect the smoke; then use the same fixed session plan:
sbatch scripts/run_turing.slurm collect --resume
sbatch scripts/run_turing.slurm sft --preflight
sbatch scripts/run_turing.slurm sft --smoke
sbatch scripts/run_turing.slurm sft --resume
sbatch scripts/run_turing.slurm evaluate --model all --smoke
sbatch scripts/run_turing.slurm evaluate --model all --resume
```

PPO is optional and uses the completed SFT adapter:

```bash
sbatch scripts/run_turing.slurm ppo --smoke
# Inspect validity/reward/KL before extending the same run:
sbatch scripts/run_turing.slurm ppo --resume --updates 10
sbatch scripts/run_turing.slurm ppo --resume
```

Read [the run instructions](docs/final-retraining.md) before paper submission.
The [assignment notebook](output/jupyter-notebook/trading-agent.ipynb) calls these
same modules. The provided `Assignment_2_Trading_Agent_update.ipynb` remains the
original assignment reference. [design.md](design.md) explains retained safeguards,
intentional differences from the starter and what was removed.

This repository is paper-only. Passing offline tests is not proof of profitable
trades or a successful Alpaca rehearsal. No training or orders are launched by
installation or by `check`.
