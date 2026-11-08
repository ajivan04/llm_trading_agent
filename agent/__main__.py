"""Assignment workflow: one configuration, one run directory, existing implementations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from agent.contracts import canonical_json


def configuration(path):
    path = Path(path).resolve()
    values = json.loads(path.read_text())
    defaults = dict(
        source="artifacts/final-nlp-source",
        run_dir="artifacts/assignment",
        env_file=".env",
        train_sessions=60,
        midterm_sessions=20,
        validation_sessions=20,
        context_tokens=4096,
        generation_tokens=512,
        sft_epochs=3,
        ppo_updates=100,
    )
    if set(values) - set(defaults):
        raise ValueError("unknown project setting; see project.json")
    cfg = {**defaults, **values}
    for key in ("source", "run_dir", "env_file"):
        cfg[key] = (path.parent / cfg[key]).resolve()
    if cfg["source"] == cfg["run_dir"] or cfg["source"].is_relative_to(cfg["run_dir"]):
        raise ValueError("run_dir must not contain or replace the source archive")
    for key in (
        "train_sessions",
        "validation_sessions",
        "context_tokens",
        "generation_tokens",
        "ppo_updates",
    ):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if type(cfg["midterm_sessions"]) is not int or cfg["midterm_sessions"] < 0:
        raise ValueError("midterm_sessions must be nonnegative")
    if (
        type(cfg["sft_epochs"]) not in (int, float)
        or not math.isfinite(cfg["sft_epochs"])
        or cfg["sft_epochs"] <= 0
    ):
        raise ValueError("sft_epochs must be positive and finite")
    return cfg


def _call(module, args):
    from importlib import import_module

    return import_module("agent." + module).main([str(arg) for arg in args])


def _budget(cfg):
    return [
        "--max-context-tokens",
        cfg["context_tokens"],
        "--max-new-tokens",
        cfg["generation_tokens"],
    ]


def _folds(cfg):
    return (
        [("train", "train", cfg["train_sessions"], [])]
        + (
            [("train-midterm", "train", cfg["midterm_sessions"], ["--cohort", "midterm"])]
            if cfg["midterm_sessions"]
            else []
        )
        + [("validation", "validation", cfg["validation_sessions"], [])]
    )


def _checkpoint(output):
    paths = [
        p
        for p in output.glob("checkpoint-*")
        if p.name.removeprefix("checkpoint-").isdigit() and (p / "sft_checkpoint.json").is_file()
    ]
    if not paths:
        raise ValueError("no complete checkpoint found; preserve this run and inspect its logs")
    return max(paths, key=lambda p: int(p.name.split("-")[-1]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("check", "collect", "sft", "ppo", "evaluate", "paper"))
    parser.add_argument("--config", type=Path, default=Path("project.json"))
    parser.add_argument(
        "--smoke", action="store_true", help="two-session/step/update pilot of the fixed full plan"
    )
    parser.add_argument("--resume", action="store_true", help="resume the same data/model/settings")
    parser.add_argument(
        "--preflight", action="store_true", help="SFT token/budget check without model weights"
    )
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument(
        "--steps", type=int, help="optional fixed SFT target; default three corpus passes"
    )
    parser.add_argument("--updates", type=int, help="PPO total target; default configured target")
    parser.add_argument("--model", choices=("base", "sft", "ppo", "all"), default="sft")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--interval", type=float, default=60)
    parser.add_argument("--cycle-id", help="stable paper session prefix; reuse on restart")
    parser.add_argument("--submit-paper", action="store_true")
    args = parser.parse_args(argv)
    cfg = configuration(args.config)
    root, source = cfg["run_dir"], cfg["source"]
    corpus = root / "corpus"
    cuda = ["--require-cuda"] if args.cuda else []
    if args.stage == "check":
        from agent.trajectory_cli import _plan, build_parser

        for name, split, count, cohort in _folds(cfg):
            plan_args = build_parser().parse_args(
                [
                    str(v)
                    for v in [
                        "plan",
                        "--prepared",
                        source,
                        "--split",
                        split,
                        "--sessions",
                        count,
                        *cohort,
                    ]
                ]
            )
            _, _, _, sessions, _, report = _plan(plan_args)
            print(
                canonical_json(
                    {
                        "cohort": name,
                        "sessions": len(sessions),
                        "first": sessions[0],
                        "last": sessions[-1],
                        "universe_size": report["universe_size"],
                        "raw_news_records": report["raw_news_records"],
                        "sec_text_records": report["sec_text_records"],
                    }
                )
            )
        return 0
    if args.stage == "collect":
        from agent.trajectories import load_collection
        from agent.trajectory_cli import review_smoke

        if not args.smoke:
            for name, *_ in _folds(cfg):
                review_smoke(load_collection(root / "collections" / name))
        for name, split, count, cohort in _folds(cfg):
            journal = root / "collections" / name
            call = [
                "collect",
                "--prepared",
                source,
                "--journal",
                journal,
                "--split",
                split,
                "--sessions",
                count,
                *cohort,
                *_budget(cfg),
                *cuda,
            ]
            if args.resume and (journal / "collection-manifest.json").exists():
                call += ["--resume"]
            if args.smoke:
                call += ["--max-sessions", 2]
            _call("trajectory_cli", call)
            if args.smoke:
                review_smoke(load_collection(journal))
        if not args.smoke:
            call = [
                "export",
                "--train-journal",
                root / "collections/train",
                "--validation-journal",
                root / "collections/validation",
                "--output",
                corpus,
            ]
            if cfg["midterm_sessions"]:
                call += ["--extra-train-journal", root / "collections/train-midterm"]
            _call("trajectory_cli", call)
        return 0
    if args.stage == "sft":
        manifest = json.loads((corpus / "dataset/manifest.json").read_text())
        count = len(manifest["split_example_ids"]["train"])
        if count == 0:
            raise ValueError(
                "no GOOD training examples; inspect collection rather than starting training"
            )
        steps = (
            args.steps
            if args.steps is not None
            else max(1, math.ceil(cfg["sft_epochs"] * count / 8))
        )
        if steps <= 0:
            raise ValueError("SFT steps must be positive")
        output = root / "sft"
        call = [
            "sft",
            "--train",
            corpus / "dataset/train.jsonl",
            "--validation",
            corpus / "dataset/validation.jsonl",
            "--output",
            output,
            "--prompt-version",
            "bounded-tool-research-v1",
            "--supervision",
            "trajectory-final",
            "--max-length",
            cfg["context_tokens"],
            "--max-steps",
            steps,
            "--gradient-accumulation-steps",
            8,
            "--save-steps",
            2,
            "--eval-steps",
            25,
            "--logging-steps",
            1,
            *cuda,
        ]
        print(
            canonical_json(
                {
                    "good_training_examples": count,
                    "fixed_steps": steps,
                    "approximate_corpus_passes": steps * 8 / count,
                }
            )
        )
        if args.preflight:
            call += ["--preflight-only"]
        else:
            if args.resume:
                call += ["--resume-from-checkpoint", _checkpoint(output)]
            if args.smoke:
                call += ["--stop-after-steps", min(2, steps)]
        return _call("training.cli", call)
    if args.stage == "ppo":
        prepared, output = root / "ppo-data", root / "ppo"
        if not (prepared / "ppo-environment.json").is_file():
            _call(
                "training.ppo_prepare",
                [
                    "--prepared",
                    source,
                    "--sft-dataset",
                    corpus / "dataset",
                    "--output",
                    prepared,
                    "--sessions",
                    cfg["train_sessions"],
                    "--max-length",
                    cfg["context_tokens"],
                    "--max-new-tokens",
                    cfg["generation_tokens"],
                ],
            )
        target = (
            args.updates if args.updates is not None else 2 if args.smoke else cfg["ppo_updates"]
        )
        call = [
            "ppo",
            "--require-assignment-sft",
            "--environment-factory",
            "agent.training.environment:create_training_environment",
            "--environment-config",
            prepared / "ppo-environment.json",
            "--output",
            output,
            "--prompt-version",
            "final-decision-nlp-v1",
            "--max-length",
            cfg["context_tokens"],
            "--max-new-tokens",
            cfg["generation_tokens"],
            "--updates",
            target,
            *cuda,
        ]
        call += (
            ["--resume-from-checkpoint", output]
            if args.resume
            else ["--adapter-checkpoint", root / "sft"]
        )
        return _call("training.cli", call)
    if args.stage == "evaluate":
        models = (
            ["base", "sft"] + (["ppo"] if (root / "ppo/adapter").is_dir() else [])
            if args.model == "all"
            else [args.model]
        )
        for model in models:
            call = [
                "--prepared",
                source,
                "--output",
                root / f"eval-{model}",
                "--sessions",
                cfg["validation_sessions"],
                *_budget(cfg),
                *cuda,
            ]
            if model != "base":
                call += ["--adapter", root / ("ppo/adapter" if model == "ppo" else "sft")]
            if args.smoke:
                call += ["--max-sessions", 2]
            if (
                args.resume
                and (root / f"eval-{model}" / "journal/collection-manifest.json").exists()
            ):
                call += ["--resume"]
            _call("research_eval_cli", call)
        return 0
    if args.model not in {"sft", "ppo"} or not args.cycle_id:
        raise ValueError("paper requires --model sft|ppo and a stable --cycle-id session prefix")
    from dataclasses import asdict

    from agent.artifacts import atomic_json
    from agent.config import Config
    from agent.source_bundle import load_source
    from agent.universe import BASELINE_UNIVERSE, assignment_sector_map

    universe = load_source(source)["universe"] or BASELINE_UNIVERSE
    paper = root / "paper" / args.model
    paper.mkdir(parents=True, exist_ok=True)
    config = paper / "config.json"
    if not config.exists():
        atomic_json(
            config,
            asdict(
                Config(
                    universe=universe,
                    sector_map=assignment_sector_map(universe)
                    if len(universe) == 100
                    else tuple((s, "unknown") for s in universe),
                )
            ),
        )
    return _call(
        "paper_cli",
        [
            "cycle",
            "--prepared",
            source,
            "--adapter",
            root / ("ppo/adapter" if args.model == "ppo" else "sft"),
            "--config",
            config,
            "--env-file",
            cfg["env_file"],
            "--outbox",
            paper / "outbox.sqlite",
            "--memory",
            paper / "memory.sqlite",
            "--run-id",
            f"{root.name}-{args.model}",
            "--cycle-id",
            args.cycle_id,
            "--max-context-tokens",
            cfg["context_tokens"],
            "--max-new-tokens",
            cfg["generation_tokens"],
            "--iterations",
            args.iterations,
            "--interval",
            args.interval,
            *(["--submit-paper"] if args.submit_paper else []),
        ],
    )


if __name__ == "__main__":
    raise SystemExit(main())
