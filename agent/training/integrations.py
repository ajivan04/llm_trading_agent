"""Lazy, version-visible integration with optional GPU training packages."""
from __future__ import annotations

import importlib
from importlib import metadata
from typing import Any


class TrainingStackUnavailable(RuntimeError):
    pass


_METHOD_CAPABILITIES = {
    "sft": ("trl", "SFTTrainer"),
    "dpo": ("trl", "DPOTrainer"),
    "ppo": ("trl", "PPOTrainer"),
}


def resolve_training_stack(
    method: str, *, expected_versions: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Import optional libraries and verify the exact trainer surface in use."""
    if method not in _METHOD_CAPABILITIES:
        raise ValueError("method must be sft, dpo, or ppo")
    required = ("transformers", "peft", "datasets", "trl")
    modules: dict[str, Any] = {}
    versions: dict[str, str] = {}
    try:
        for package in required:
            modules[package] = importlib.import_module(package)
            versions[package] = metadata.version(package)
    except (ImportError, metadata.PackageNotFoundError) as exc:
        raise TrainingStackUnavailable(
            "Install pinned transformers, peft, datasets, and trl packages before training"
        ) from exc
    for package, expected in (expected_versions or {}).items():
        if versions.get(package) != expected:
            raise TrainingStackUnavailable(
                f"{package} version mismatch: expected {expected}, found {versions.get(package)}"
            )
    for module_name, attribute in (
        ("transformers", "AutoTokenizer"), ("transformers", "AutoModelForCausalLM"),
        ("peft", "LoraConfig"), ("peft", "get_peft_model"),
        ("datasets", "Dataset"), _METHOD_CAPABILITIES[method],
    ):
        if not hasattr(modules[module_name], attribute):
            raise TrainingStackUnavailable(
                f"installed {module_name} does not provide required {attribute}"
            )
    return {"modules": modules, "versions": versions, "trainer": _METHOD_CAPABILITIES[method][1]}

