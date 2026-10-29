"""Evaluate a base or LoRA Qwen policy on a verified decision-quality split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent.contracts import canonical_json
from agent.datasets import load_dataset_artifact
from agent.evaluation import (
    ExperimentRegistry,
    ExperimentRunner,
    artifact_hash,
    load_outcomes,
    score_generations,
)
from agent.training import MODEL_NAME, MODEL_REVISION


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--release-id")
    return parser


def _generate(examples, adapter: Path | None, max_new_tokens: int):
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - integration environment
        raise RuntimeError("install the train dependency group before evaluation") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, trust_remote_code=False,
    )
    kwargs = {"trust_remote_code": False}
    if torch.cuda.is_available():
        kwargs.update({"dtype": torch.bfloat16, "device_map": "auto"})
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, **kwargs,
    )
    if adapter is not None:
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    if not torch.cuda.is_available():
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        device = torch.device("mps" if mps is not None and mps.is_available() else "cpu")
        model = model.to(device)
    model.eval()
    results = {}
    max_context = int(getattr(model.config, "max_position_embeddings", 131072))
    for example in examples:
        encoded = tokenizer.apply_chat_template(
            list(example.messages), tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        )
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)
        if input_ids.shape[-1] + max_new_tokens > max_context:
            raise ValueError(
                f"{example.example_id} exceeds model context: "
                f"{input_ids.shape[-1]} + {max_new_tokens} > {max_context}"
            )
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        results[example.example_id] = tokenizer.decode(
            generated[0, input_ids.shape[-1]:], skip_special_tokens=True,
        )
    return results


def _write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_new_tokens <= 0 or (args.max_examples is not None and args.max_examples <= 0):
        raise ValueError("generation limits must be positive")
    if args.split == "test" and (args.registry is None or not args.release_id):
        raise ValueError("test evaluation requires --registry and --release-id")
    dataset = load_dataset_artifact(args.prepared / "dataset")
    examples = tuple(item for item in dataset.examples if item.split == args.split)
    if args.max_examples is not None:
        examples = examples[:args.max_examples]
    all_outcomes = load_outcomes(
        args.prepared / "outcome-audit.jsonl", split=args.split,
    )
    wanted = {item.example_id for item in examples}
    outcomes = tuple(item for item in all_outcomes if item.example_id in wanted)
    adapter_digest = artifact_hash(args.adapter) if args.adapter else "base-model"
    config = {
        "model_name": MODEL_NAME, "model_revision": MODEL_REVISION,
        "dataset_hash": dataset.manifest.dataset_hash,
        "adapter_hash": adapter_digest, "max_new_tokens": args.max_new_tokens,
        "max_examples": args.max_examples,
    }

    def evaluate(_config, _split):
        generations = _generate(examples, args.adapter, args.max_new_tokens)
        metrics, records = score_generations(examples, outcomes, generations)
        args.output.mkdir(parents=True, exist_ok=True)
        _write(args.output / "predictions.jsonl", "".join(
            canonical_json({**record, "generation": generations[record["example_id"]]}) + "\n"
            for record in records
        ))
        _write(args.output / "evaluation.json", canonical_json({
            "config": config, "split": args.split, "metrics": metrics,
        }) + "\n")
        return metrics

    registry = ExperimentRegistry(args.registry, release_id=args.release_id or "development")
    try:
        attempt = ExperimentRunner(evaluate, registry).run(config, split=args.split)
    finally:
        registry.close()
    print(json.dumps({
        "attempt_id": attempt.attempt_id, "metrics": dict(attempt.metrics),
        "output": str(args.output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
