"""Command-line entrypoint for reproducible SFT, DPO, and PPO training."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

from agent.contracts import content_hash

from .core import MODEL_REVISION, DPOPair, EpisodeExample
from .execution import (
    TrainerSettings,
    smoke_test_adapter,
    train_dpo,
    train_ppo,
    train_sft,
)
from .integrations import resolve_training_stack
from .qwen_ppo import QwenPPOOptimizer, QwenPPOSettings


def _jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected an object at {path}:{line_number}")
            records.append(row)
    return records


def _episode_example(row: dict[str, Any]) -> EpisodeExample:
    """Normalize JSON-compatible message lists into the tuple contract."""
    payload = dict(row)
    messages = payload.get("messages")
    if isinstance(messages, list):
        payload["messages"] = tuple(messages)
    return EpisodeExample(**payload)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _actual_data_hashes(args: argparse.Namespace) -> tuple[str, ...]:
    paths = []
    for name in ("train", "validation", "environment_config"):
        path = getattr(args, name, None)
        if path is not None:
            paths.append(Path(path))
    if args.method == "ppo" and args.environment_config is not None:
        config = json.loads(args.environment_config.read_text(encoding="utf-8"))
        episode_path = config.get("episodes_path") if isinstance(config, dict) else None
        if not isinstance(episode_path, str) or not episode_path:
            raise ValueError("PPO environment config must identify episodes_path")
        paths.append(Path(episode_path))
    return tuple(dict.fromkeys(_file_hash(path) for path in paths))


def _verify_sft_manifest(
    train_path: Path, validation_path: Path, train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
) -> str:
    if train_path.parent != validation_path.parent:
        raise ValueError("SFT train and validation files must share one dataset artifact")
    manifest_path = train_path.parent / "manifest.json"
    if not manifest_path.exists():
        raise ValueError("SFT requires the dataset artifact manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_hash = manifest.get("dataset_hash")
    if not isinstance(dataset_hash, str) or not dataset_hash:
        raise ValueError("SFT dataset manifest is missing dataset_hash")
    split_ids = manifest.get("split_example_ids", {})
    examples_path = train_path.parent / "examples.jsonl"
    if not examples_path.exists():
        raise ValueError("SFT dataset artifact is missing examples.jsonl")
    all_rows = _jsonl(examples_path)
    identity = {
        "examples": all_rows, "splits": split_ids,
        "purged": manifest.get("purged_episode_ids", []),
        "source_hashes": manifest.get("source_hashes", []),
        "boundaries": (
            manifest.get("train_end"), manifest.get("validation_end"),
            manifest.get("test_end"),
        ),
    }
    if "regime_id" in manifest:
        identity["regime_id"] = manifest["regime_id"]
    if "decision_session_filter_hash" in manifest:
        identity["decision_session_filter_hash"] = manifest["decision_session_filter_hash"]
    if content_hash(identity) != dataset_hash:
        raise ValueError("SFT dataset artifact hash verification failed")
    expected_train = [row for row in all_rows if row.get("split") == "train"]
    expected_validation = [row for row in all_rows if row.get("split") == "validation"]
    if train_rows != expected_train or validation_rows != expected_validation:
        raise ValueError("SFT split files differ from the verified dataset artifact")
    if [row.get("example_id") for row in train_rows] != list(split_ids.get("train", ())):
        raise ValueError("SFT train rows do not match the dataset manifest")
    if [row.get("example_id") for row in validation_rows] != list(
            split_ids.get("validation", ())):
        raise ValueError("SFT validation rows do not match the dataset manifest")
    return dataset_hash


def _audited_dpo_pair(row: dict[str, Any], expected_split: str) -> DPOPair:
    pair_row = row.get("pair")
    audit_hash = row.get("audit_hash")
    chosen = row.get("chosen_outcome")
    rejected = row.get("rejected_outcome")
    if not all(isinstance(item, dict) for item in (pair_row, chosen, rejected)):
        raise ValueError("DPO rows must contain the complete audited preference artifact")
    if not isinstance(audit_hash, str) or not audit_hash:
        raise ValueError("DPO row is missing audit_hash")
    if chosen.get("reset_id") == rejected.get("reset_id"):
        raise ValueError("DPO alternatives must use independent simulator resets")
    if (chosen.get("simulator_hash") != rejected.get("simulator_hash")
            or chosen.get("horizon_end") != rejected.get("horizon_end")):
        raise ValueError("DPO alternatives must share simulator and horizon")
    identity = {
        "state_hash": pair_row.get("state_hash"),
        "simulator_hash": chosen.get("simulator_hash"),
        "horizon_end": chosen.get("horizon_end"),
    }
    valid_hashes = {
        content_hash({**identity, "outcomes": (chosen, rejected)}),
        content_hash({**identity, "outcomes": (rejected, chosen)}),
    }
    if audit_hash not in valid_hashes:
        raise ValueError("DPO audit_hash does not match the recorded outcomes")
    pair_payload = dict(pair_row)
    pair_payload["split"] = expected_split
    pair_payload["audit_hash"] = audit_hash
    return DPOPair(**pair_payload)


def _settings(args: argparse.Namespace, hashes: tuple[str, ...]) -> TrainerSettings:
    return TrainerSettings(
        output_dir=args.output, model_revision=args.model_revision,
        tokenizer_revision=args.tokenizer_revision,
        prompt_version=args.prompt_version, data_hashes=hashes,
        seed=args.seed, learning_rate=args.learning_rate, max_steps=args.max_steps,
        max_length=args.max_length, train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_steps=args.save_steps, logging_steps=args.logging_steps,
        bf16=not args.no_bf16,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        resume_from_checkpoint=args.resume_from_checkpoint,
        adapter_checkpoint=args.adapter_checkpoint,
        reference_adapter_checkpoint=getattr(args, "reference_adapter_checkpoint", None),
    )


def _add_common(parser: argparse.ArgumentParser, *, datasets: bool = True) -> None:
    if datasets:
        parser.add_argument("--train", type=Path, required=True)
        parser.add_argument("--validation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--tokenizer-revision", default=MODEL_REVISION)
    parser.add_argument("--prompt-version", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument(
        "--adapter-checkpoint", type=Path,
        help="Initialize from an existing trainable LoRA adapter (for example SFT before DPO)",
    )
    parser.add_argument(
        "--reference-adapter-checkpoint", type=Path,
        help="Frozen SFT reference adapter for PPO (defaults to the initial adapter)",
    )
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="method", required=True)
    sft = commands.add_parser("sft", help="Train final-decision SFT")
    _add_common(sft)
    dpo = commands.add_parser("dpo", help="Train same-state DPO")
    _add_common(dpo)
    dpo.add_argument("--beta", type=float, default=0.1)
    ppo = commands.add_parser("ppo", help="Train with fresh simulator rollouts")
    _add_common(ppo, datasets=False)
    ppo.add_argument("--environment-factory", required=True, metavar="MODULE:CALLABLE")
    ppo.add_argument("--environment-config", type=Path)
    ppo.add_argument("--updates", type=int, required=True)
    ppo.add_argument("--max-episode-steps", type=int, default=8)
    ppo.add_argument("--max-new-tokens", type=int, default=256)
    ppo.add_argument("--temperature", type=float, default=0.8)
    ppo.add_argument("--top-p", type=float, default=0.95)
    ppo.add_argument("--ppo-epochs", type=int, default=4)
    smoke = commands.add_parser("smoke", help="GPU adapter save/reload generation check")
    smoke.add_argument("--adapter-checkpoint", type=Path, required=True)
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--prompt", default="Return a strict HOLD decision for AAPL.")
    smoke.add_argument("--model-revision", default=MODEL_REVISION)
    smoke.add_argument("--tokenizer-revision", default=MODEL_REVISION)
    smoke.add_argument("--max-new-tokens", type=int, default=32)
    return parser


def _environment(specification: str, config_path: Path | None):
    module_name, separator, attribute = specification.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("environment factory must use MODULE:CALLABLE syntax")
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError("environment factory must be callable")
    config = {}
    if config_path is not None:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("environment config must be a JSON object")
    environment = factory(config)
    if not callable(getattr(environment, "reset", None)) or not callable(
            getattr(environment, "step", None)):
        raise TypeError("environment must provide reset and step")
    return environment


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.method == "smoke":
        result = smoke_test_adapter(
            args.adapter_checkpoint, output_dir=args.output, prompt=args.prompt,
            model_revision=args.model_revision, tokenizer_revision=args.tokenizer_revision,
            max_new_tokens=args.max_new_tokens,
        )
        print(json.dumps(dict(result), sort_keys=True))
        return 0
    if args.method == "ppo":
        if args.resume_from_checkpoint is not None:
            checkpoint = Path(args.resume_from_checkpoint)
            args.adapter_checkpoint = checkpoint / "adapter"
            args.reference_adapter_checkpoint = checkpoint / "reference_adapter"
        elif args.adapter_checkpoint is None:
            raise ValueError("PPO requires --adapter-checkpoint from SFT")
        args.max_steps = args.updates
        settings = _settings(args, _actual_data_hashes(args))
        ppo_settings = QwenPPOSettings(
            max_episode_steps=args.max_episode_steps,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            top_p=args.top_p, epochs_per_rollout=args.ppo_epochs,
        )
        optimizer = QwenPPOOptimizer.from_pretrained(
            settings=settings, ppo=ppo_settings)
        result = train_ppo(
            _environment(args.environment_factory, args.environment_config), optimizer,
            settings=settings, updates=args.updates,
            package_versions=resolve_training_stack("ppo")["versions"],
        )
        print(json.dumps({
            "method": result.method, "output_dir": result.output_dir,
            "manifest_hash": result.manifest_hash, "metrics": dict(result.metrics),
            "elapsed_seconds": result.elapsed_seconds,
        }, sort_keys=True))
        return 0
    train_rows = _jsonl(args.train)
    validation_rows = _jsonl(args.validation) if args.validation else []
    data_hashes = _actual_data_hashes(args)
    if args.method == "sft":
        if args.validation is None:
            raise ValueError("SFT requires --validation")
        dataset_hash = _verify_sft_manifest(
            args.train, args.validation, train_rows, validation_rows)
        data_hashes = tuple(dict.fromkeys((dataset_hash, *data_hashes)))
    settings = _settings(args, data_hashes)
    if args.method == "sft":
        result = train_sft(
            [_episode_example(row) for row in train_rows],
            [_episode_example(row) for row in validation_rows], settings=settings,
        )
    else:
        result = train_dpo(
            [_audited_dpo_pair(row, "train") for row in train_rows],
            [_audited_dpo_pair(row, "validation") for row in validation_rows],
            settings=settings, beta=args.beta,
        )
    print(json.dumps({
        "method": result.method, "output_dir": result.output_dir,
        "manifest_hash": result.manifest_hash, "metrics": dict(result.metrics),
        "elapsed_seconds": result.elapsed_seconds,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
