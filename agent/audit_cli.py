"""Audit prepared targets and saved validation predictions without loading Qwen."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from agent.contracts import canonical_json
from agent.datasets import load_dataset_artifact
from agent.evaluation import load_outcomes, score_generations
from agent.evaluation.model import SCORING_VERSION, decision_failure_category
from agent.training.validation import validate_supervised_example


def audit_prepared(prepared: Path, *, predictions: Path | None = None) -> dict[str, Any]:
    """Verify all split identities/targets; score only saved validation generations.

    No model inference or test-set performance inspection occurs here. Aggregate
    target-contract checks include test rows solely to establish artifact integrity.
    """
    dataset = load_dataset_artifact(prepared / "dataset")
    report: dict[str, Any] = {
        "dataset_hash": dataset.manifest.dataset_hash,
        "scoring_version": SCORING_VERSION,
        "splits": {},
    }
    total_invalid = 0
    for split in ("train", "validation", "test"):
        examples = [item for item in dataset.examples if item.split == split]
        errors: Counter[str] = Counter()
        first_errors = []
        for example in examples:
            try:
                validate_supervised_example(example)
            except (TypeError, ValueError) as exc:
                category = decision_failure_category(str(exc))
                errors[category] += 1
                if len(first_errors) < 5:
                    first_errors.append({"example_id": example.example_id,
                                         "category": category})
        invalid = sum(errors.values())
        total_invalid += invalid
        report["splits"][split] = {
            "examples": len(examples), "valid_targets": len(examples) - invalid,
            "invalid_targets": invalid, "failure_counts": dict(errors),
            "first_errors": first_errors,
        }
    report["target_contract_passed"] = total_invalid == 0
    if predictions is not None:
        prediction_hash = hashlib.sha256(predictions.read_bytes()).hexdigest()
        provenance: dict[str, Any] = {"status": "unverified"}
        evaluation_path = predictions.parent / "evaluation.json"
        if evaluation_path.is_file():
            metadata_bytes = evaluation_path.read_bytes()
            metadata = json.loads(metadata_bytes)
            config = metadata.get("config", {})
            if not isinstance(config, dict):
                raise ValueError("evaluation metadata has an invalid config")
            if metadata.get("split") != "validation":
                raise ValueError("only validation predictions can be audited here")
            if config.get("dataset_hash") != dataset.manifest.dataset_hash:
                raise ValueError("saved predictions belong to a different dataset identity")
            provenance = {
                "status": "metadata_present",
                "evaluation_file_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
                "config": {key: config[key] for key in (
                    "model_name", "model_revision", "dataset_hash", "adapter_hash",
                    "max_new_tokens", "max_examples", "scoring_version",
                ) if key in config},
            }
        examples = tuple(item for item in dataset.examples if item.split == "validation")
        generations = {}
        for line_number, line in enumerate(predictions.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                example_id, generation = row["example_id"], row["generation"]
                if not isinstance(example_id, str) or not isinstance(generation, str):
                    raise ValueError("prediction fields must be text")
                if example_id in generations:
                    raise ValueError("duplicate prediction example ID")
                generations[example_id] = generation
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid prediction row {line_number}") from exc
        if set(generations) != {item.example_id for item in examples}:
            raise ValueError("saved predictions must exactly cover validation")
        if hashlib.sha256(predictions.read_bytes()).hexdigest() != prediction_hash:
            raise ValueError("predictions changed during audit; finish transfer first")
        outcomes = load_outcomes(prepared / "outcome-audit.jsonl", split="validation")
        metrics, _ = score_generations(examples, outcomes, generations)
        report["validation_predictions"] = {
            "path": str(predictions), "file_sha256": prediction_hash,
            "provenance": provenance, "metrics": metrics,
        }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--predictions", type=Path,
                        help="optional saved validation predictions; no inference")
    args = parser.parse_args(argv)
    try:
        report = audit_prepared(args.prepared, predictions=args.predictions)
    except (OSError, ValueError) as exc:
        print(canonical_json({
            "artifact_integrity_passed": False, "error_class": type(exc).__name__,
            "detail": str(exc),
        }), flush=True)
        return 1
    report["artifact_integrity_passed"] = True
    print(canonical_json(report), flush=True)
    return 0 if report["target_contract_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
