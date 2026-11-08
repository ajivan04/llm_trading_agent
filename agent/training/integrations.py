"""Lazy imports for only the training packages each implementation actually uses."""

from __future__ import annotations

import importlib
from importlib import metadata
from typing import Any


class TrainingStackUnavailable(RuntimeError):
    pass


_METHOD_PACKAGES = {
    "sft": ("torch", "transformers", "peft", "datasets"),
    "dpo": ("torch", "transformers", "peft", "datasets", "trl"),
    # The native PPO optimizer does not use TRL or Hugging Face datasets.
    "ppo": ("torch", "transformers", "peft"),
}
_COMMON_CAPABILITIES = (
    ("transformers", "AutoTokenizer"),
    ("transformers", "AutoModelForCausalLM"),
    ("peft", "LoraConfig"),
    ("peft", "get_peft_model"),
    ("peft", "PeftModel"),
)
_METHOD_CAPABILITIES = {
    "sft": (
        ("transformers", "Trainer"),
        ("transformers", "TrainingArguments"),
        ("transformers", "DataCollatorForSeq2Seq"),
        ("datasets", "Dataset"),
    ),
    "dpo": (("trl", "DPOTrainer"), ("trl", "DPOConfig"), ("datasets", "Dataset")),
    "ppo": (("torch", "optim"),),
}


def resolve_training_stack(
    method: str,
    *,
    expected_versions: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Import method-specific dependencies and verify their actual used surfaces."""
    if method not in _METHOD_PACKAGES:
        raise ValueError("method must be sft, dpo, or ppo")
    modules: dict[str, Any] = {}
    versions: dict[str, str] = {}
    required = _METHOD_PACKAGES[method]
    try:
        for package in required:
            modules[package] = importlib.import_module(package)
            versions[package] = metadata.version(package)
    except (ImportError, metadata.PackageNotFoundError) as exc:
        raise TrainingStackUnavailable(
            f"Install the pinned {', '.join(required)} packages for {method} training"
        ) from exc
    for package, expected in (expected_versions or {}).items():
        if versions.get(package) != expected:
            raise TrainingStackUnavailable(
                f"{package} version mismatch: expected {expected}, found {versions.get(package)}"
            )
    for module_name, attribute in (*_COMMON_CAPABILITIES, *_METHOD_CAPABILITIES[method]):
        if not hasattr(modules[module_name], attribute):
            raise TrainingStackUnavailable(
                f"installed {module_name} does not provide required {attribute}"
            )
    trainer = {"sft": "Trainer", "dpo": "DPOTrainer", "ppo": "optim"}[method]
    return {"modules": modules, "versions": versions, "trainer": trainer}
