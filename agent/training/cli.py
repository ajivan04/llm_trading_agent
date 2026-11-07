"""Command-line entrypoint for reproducible SFT, DPO, and PPO training."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from agent.contracts import canonical_json, content_hash

from .core import MODEL_NAME, MODEL_REVISION, DPOCandidate, DPOPair, EpisodeExample
from .execution import (
    PPOTrainingLimits,
    TrainerSettings,
    _path_hash,
    smoke_test_adapter,
    train_dpo,
    train_ppo,
    train_sft,
)
from .integrations import resolve_training_stack
from .qwen_ppo import PPO_IMPLEMENTATION_VERSION, QwenPPOOptimizer, QwenPPOSettings


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
    # The audit covers candidate outcomes, not the separately serialized pair.
    # Rebuild it so a changed prompt/response/split cannot retain a valid old hash.
    from agent.preferences import CandidateOutcome, build_audited_dpo_pair
    outcomes = []
    for payload in (chosen, rejected):
        item = dict(payload)
        item["candidate"] = DPOCandidate(**item["candidate"])
        outcomes.append(CandidateOutcome(**item))
    expected = build_audited_dpo_pair(*outcomes).pair
    recorded = {key: value for key, value in pair_row.items() if key != "audit_hash"}
    expected_fields = {key: value for key, value in asdict(expected).items()
                       if key != "audit_hash"}
    if recorded != expected_fields or expected.split != expected_split:
        raise ValueError("DPO pair does not match its audited candidates and split")
    if pair_row.get("audit_hash") not in (None, audit_hash):
        raise ValueError("DPO pair audit_hash differs from the outcome audit")
    # Prepared DPO prompts must preserve native chat data and supported citations.
    from agent.policy import parse_decisions
    from agent.policy.evidence import visible_evidence
    try:
        messages = json.loads(expected.prompt)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("DPO prompt must encode native chat messages") from exc
    visible = visible_evidence(messages)
    symbols = {item["symbol"] for item in visible.values() if item.get("symbol")}
    if not visible or not symbols:
        raise ValueError("DPO prompt requires visible symbol evidence")
    for response in (expected.chosen, expected.rejected):
        parse_decisions(response, allowed_symbols=symbols, allowed_evidence_ids=visible)
    pair_payload = dict(pair_row)
    pair_payload["split"] = expected_split
    pair_payload["audit_hash"] = audit_hash
    return DPOPair(**pair_payload)


def _settings(args: argparse.Namespace, hashes: tuple[str, ...]) -> TrainerSettings:
    return TrainerSettings(
        supervision_mode=getattr(args, "supervision", "final"),
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
    sft.add_argument("--supervision", choices=("final", "trajectory"), default="final")
    sft.add_argument("--preflight-only", action="store_true",
                     help="Validate all native token spans without loading policy weights")
    dpo = commands.add_parser("dpo", help="Train same-state DPO")
    _add_common(dpo)
    dpo.add_argument("--beta", type=float, default=0.1)
    ppo = commands.add_parser("ppo", help="Train with fresh simulator rollouts")
    _add_common(ppo, datasets=False)
    ppo.set_defaults(learning_rate=1e-5, save_steps=10, logging_steps=1)
    ppo.add_argument("--environment-factory", required=True, metavar="MODULE:CALLABLE")
    ppo.add_argument("--environment-config", type=Path)
    ppo.add_argument("--updates", type=int, required=True,
                     help="total update target, including updates restored from checkpoint")
    ppo.add_argument("--max-episode-steps", type=int, default=8)
    ppo.add_argument("--max-new-tokens", type=int, default=512)
    ppo.add_argument("--temperature", type=float, default=1.0)
    ppo.add_argument("--top-p", type=float, default=1.0)
    ppo.add_argument("--ppo-epochs", type=int, default=2)
    ppo.add_argument("--rollout-episodes", type=int, default=8)
    ppo.add_argument("--stats-chunk-size", type=int, default=32)
    ppo.add_argument("--target-kl", type=float, default=0.1)
    ppo.add_argument("--max-invalid-fraction", type=float, default=0.5)
    ppo.add_argument("--invalid-patience", type=int, default=2)
    ppo.add_argument("--max-seconds", type=float, default=7200)
    ppo.add_argument("--kl-coefficient", type=float, default=0.02)
    ppo.add_argument("--entropy-coefficient", type=float, default=0.0)
    ppo.add_argument("--value-learning-rate", type=float, default=1e-6)
    ppo.add_argument("--require-cuda", action="store_true",
                     help="fail before model loading if this GPU job has no CUDA device")
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
        status_path = args.output / "training_status.json"
        if status_path.is_file():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("status") == "stopped" and status.get("reason") in {
                "invalid_proposals", "reference_kl",
            }:
                raise ValueError("PPO pilot failed a safety gate; review before a new experiment")
        if args.resume_from_checkpoint is None and any(
            (args.output / name).exists()
            for name in ("ppo_state.json", "ppo_run.json", "training_manifest.json")
        ):
            raise ValueError(
                "PPO output contains an existing run; resume explicitly or use a new path")
        if args.resume_from_checkpoint is not None:
            checkpoint = Path(args.resume_from_checkpoint)
            metadata = json.loads((checkpoint / "ppo_state.json").read_text(encoding="utf-8"))
            if metadata.get("implementation_version") != PPO_IMPLEMENTATION_VERSION:
                raise ValueError(
                    "legacy PPO checkpoint cannot resume corrected PPO; start from SFT")
            if metadata.get("tensor_state_sha256") != _file_hash(checkpoint / "ppo_state.pt"):
                raise ValueError("PPO checkpoint tensor checksum failed before model loading")
            if (metadata.get("model_name") != MODEL_NAME
                    or metadata.get("model_revision") != args.model_revision
                    or metadata.get("tokenizer_revision") != args.tokenizer_revision):
                raise ValueError(
                    "PPO checkpoint model/tokenizer identity differs from requested policy")
            for name, key in (("adapter", "adapter_hash"),
                              ("reference_adapter", "reference_adapter_hash")):
                if (not (checkpoint / name).is_dir()
                        or _path_hash(checkpoint / name) != metadata.get(key)):
                    raise ValueError("PPO adapter checksum failed before model loading")
            restored = metadata.get("update_index")
            if type(restored) is not int or not 0 <= restored <= args.updates:
                raise ValueError("PPO checkpoint is beyond the requested total update target")
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
            rollout_episodes=args.rollout_episodes, stats_chunk_size=args.stats_chunk_size,
            target_kl=args.target_kl, max_invalid_fraction=args.max_invalid_fraction,
            kl_coefficient=args.kl_coefficient, entropy_coefficient=args.entropy_coefficient,
            value_learning_rate=args.value_learning_rate,
        )
        if args.resume_from_checkpoint is not None:
            stored = metadata.get("run_contract")
            if (not isinstance(stored, dict)
                    or content_hash(stored) != metadata.get("run_fingerprint")):
                raise ValueError(
                    "PPO checkpoint run-contract fingerprint failed before model loading")
            expected = {
                "model_revision": settings.model_revision,
                "tokenizer_revision": settings.tokenizer_revision,
                "prompt_version": settings.prompt_version, "data_hashes": settings.data_hashes,
                "seed": settings.seed, "learning_rate": settings.learning_rate,
                "max_length": settings.max_length, "bf16": settings.bf16,
                "gradient_checkpointing": settings.gradient_checkpointing,
                "ppo": asdict(ppo_settings),
            }
            if any(canonical_json(stored.get(key)) != canonical_json(value)
                   for key, value in expected.items()):
                raise ValueError("PPO resume contract changed; use the original data and settings")
        limits = PPOTrainingLimits(max_seconds=args.max_seconds,
                                   invalid_patience=args.invalid_patience)
        # Parse every replay row and verify visibility before allocating Qwen.
        environment = _environment(args.environment_factory, args.environment_config)
        from .validation import preflight_ppo_context
        stack = resolve_training_stack("ppo")
        if (args.resume_from_checkpoint is not None
                and stored.get("package_versions") != stack["versions"]):
            raise ValueError("PPO dependency versions changed; restore the original training stack")
        if args.require_cuda and not stack["modules"]["torch"].cuda.is_available():
            raise ValueError("CUDA is unavailable; refusing CPU training in the GPU job")
        tokenizer = stack["modules"]["transformers"].AutoTokenizer.from_pretrained(
            MODEL_NAME, revision=settings.tokenizer_revision,
            trust_remote_code=False,
        )
        preflight_ppo_context(environment, tokenizer, max_length=settings.max_length,
                              max_new_tokens=ppo_settings.max_new_tokens)
        if _actual_data_hashes(args) != settings.data_hashes:
            raise ValueError("PPO data changed during preflight; finish transfer before training")
        for adapter in (settings.adapter_checkpoint, settings.reference_adapter_checkpoint):
            if adapter is not None and not (
                (adapter / "adapter_config.json").is_file()
                and (adapter / "adapter_model.safetensors").is_file()
            ):
                raise ValueError("PPO requires a complete saved adapter before model loading")
            if adapter is not None:
                from safetensors import safe_open
                try:
                    with safe_open(str(adapter / "adapter_model.safetensors"),
                                   framework="pt", device="cpu") as weights:
                        if not list(weights.keys()):
                            raise ValueError("adapter has no tensors")
                except Exception as exc:
                    raise ValueError("PPO adapter tensors are incomplete or invalid") from exc
        if (args.resume_from_checkpoint is None
                and settings.reference_adapter_checkpoint is not None
                and _path_hash(settings.adapter_checkpoint)
                != _path_hash(settings.reference_adapter_checkpoint)):
            raise ValueError("fresh PPO policy and frozen reference must use the same SFT adapter")
        adapter_input_hashes = {
            path: _path_hash(path)
            for path in (settings.adapter_checkpoint, settings.reference_adapter_checkpoint)
            if path is not None
        }
        optimizer = QwenPPOOptimizer.from_pretrained(
            settings=settings, ppo=ppo_settings)
        if any(_path_hash(path) != digest for path, digest in adapter_input_hashes.items()):
            raise ValueError("PPO adapter changed during loading; finish transfer before training")
        result = train_ppo(
            environment, optimizer,
            settings=settings, updates=args.updates,
            package_versions=stack["versions"],
            limits=limits,
        )
        print(json.dumps({
            "method": result.method, "output_dir": result.output_dir,
            "manifest_hash": result.manifest_hash, "metrics": dict(result.metrics),
            "elapsed_seconds": result.elapsed_seconds,
        }, sort_keys=True))
        return 0 if result.metrics.get("status") == "completed" else 2
    train_rows = _jsonl(args.train)
    validation_rows = _jsonl(args.validation) if args.validation else []
    data_hashes = _actual_data_hashes(args)
    if args.method == "sft":
        if args.validation is None:
            raise ValueError("SFT requires --validation")
        dataset_hash = _verify_sft_manifest(
            args.train, args.validation, train_rows, validation_rows)
        from .validation import validate_sft_targets
        validate_sft_targets([
            _episode_example(row) for row in (*train_rows, *validation_rows)
        ])
        from agent.datasets import load_dataset_artifact
        verified = load_dataset_artifact(args.train.parent)
        if verified.manifest.dataset_hash != dataset_hash:
            raise ValueError("SFT artifact identity changed during preflight")
        has_trajectory = any(message.get("role") == "assistant"
                             for row in (*train_rows, *validation_rows)
                             for message in row["messages"])
        if has_trajectory != (args.supervision == "trajectory"):
            raise ValueError(
                "SFT supervision mode must match the dataset's actual tool trajectories")
        if args.supervision == "trajectory":
            trajectory_path = args.train.parent.parent / "trajectory-manifest.json"
            if not trajectory_path.exists():
                raise ValueError("trajectory SFT requires its collection/export manifest")
            trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
            if (trajectory.get("dataset_hash") != dataset_hash
                    or trajectory.get("supervision_mode") != "trajectory"
                    or trajectory.get("test_policy_calls") != 0):
                raise ValueError("trajectory export manifest does not match this dataset")
        data_hashes = tuple(dict.fromkeys((dataset_hash, *data_hashes)))
    settings = _settings(args, data_hashes)
    if args.method == "sft" and args.preflight_only:
        from agent.policy.loader import HuggingFaceQwenBackend

        from .core import build_sft_example, build_trajectory_sft_example
        backend = HuggingFaceQwenBackend(tokenizer_revision=args.tokenizer_revision)
        backend.load_tokenizer()
        builder = (build_trajectory_sft_example if args.supervision == "trajectory"
                   else build_sft_example)
        if not train_rows or not validation_rows:
            raise ValueError("SFT preflight requires nonempty train and validation")
        maximum = 0
        total = len(train_rows) + len(validation_rows)
        for row in (*train_rows, *validation_rows):
            item = _episode_example(row)
            built = builder(item, backend._tokenizer, max_length=args.max_length,
                            allowed_splits=(item.split,))
            maximum = max(maximum, len(built.input_ids))
        print(canonical_json({"event": "sft_preflight", "examples": total,
                              "max_tokens": maximum, "supervision": args.supervision}), flush=True)
        return 0
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
