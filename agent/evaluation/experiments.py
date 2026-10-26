"""Frozen comparison plans and verified training/evaluation artifact lineage."""
from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from agent.contracts import FrozenDict, canonical_json, content_hash

from .registry import ExperimentAttempt, ExperimentRegistry, ExperimentRunner


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    method: str
    dataset_hash: str
    split_manifest_hash: str
    simulator_hash: str
    prompt_hash: str
    tool_schema_hash: str
    feature_version: str
    retrieval_version: str
    risk_config_hash: str
    execution_config_hash: str
    seed_budget: tuple[int, ...]
    evidence_token_budget: int
    sample_budget: int
    compute_budget_steps: int
    parent_checkpoint_hash: str | None = None

    def __post_init__(self) -> None:
        if self.method not in {"base", "sft", "dpo", "ppo"}:
            raise ValueError("method must be base, sft, dpo, or ppo")
        for name in (
            "name", "dataset_hash", "split_manifest_hash", "simulator_hash",
            "prompt_hash", "tool_schema_hash", "feature_version", "retrieval_version",
            "risk_config_hash", "execution_config_hash",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be nonempty")
        if not self.seed_budget or any(type(seed) is not int for seed in self.seed_budget):
            raise ValueError("seed_budget must contain integers")
        for name in ("evidence_token_budget", "sample_budget", "compute_budget_steps"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def spec_hash(self) -> str:
        return content_hash(self)


_FAIR_FIELDS = (
    "dataset_hash", "split_manifest_hash", "simulator_hash", "prompt_hash",
    "tool_schema_hash", "feature_version", "retrieval_version", "risk_config_hash",
    "execution_config_hash", "seed_budget", "evidence_token_budget", "sample_budget",
    "compute_budget_steps",
)


@dataclass(frozen=True)
class ComparisonPlan:
    comparison_id: str
    variants: tuple[ExperimentSpec, ...]

    def __post_init__(self) -> None:
        if not self.comparison_id.strip() or len(self.variants) < 2:
            raise ValueError("comparison requires an ID and at least two variants")
        if len({item.name for item in self.variants}) != len(self.variants):
            raise ValueError("variant names must be unique")
        reference = self.variants[0]
        for variant in self.variants[1:]:
            differences = [name for name in _FAIR_FIELDS
                           if getattr(variant, name) != getattr(reference, name)]
            if differences:
                raise ValueError(f"unfair comparison differs in: {', '.join(differences)}")


@dataclass(frozen=True)
class ArtifactManifest:
    spec_hash: str
    code_revision: str
    dependency_lock_hash: str
    model_revision: str
    tokenizer_revision: str
    adapter_hash: str
    checkpoint_hash: str
    parent_checkpoint_hash: str | None
    elapsed_seconds: float
    peak_gpu_memory_bytes: int | None
    metrics: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in (
            "spec_hash", "code_revision", "dependency_lock_hash", "model_revision",
            "tokenizer_revision", "adapter_hash", "checkpoint_hash",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be nonempty")
        if self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be nonnegative")
        if self.peak_gpu_memory_bytes is not None and self.peak_gpu_memory_bytes < 0:
            raise ValueError("peak GPU memory must be nonnegative")
        canonical_json(self.metrics)
        object.__setattr__(self, "metrics", FrozenDict(dict(self.metrics)))

    @property
    def manifest_hash(self) -> str:
        return content_hash(self)


def artifact_hash(path: Path) -> str:
    if not path.exists():
        raise ValueError("artifact path does not exist")
    files = sorted(item for item in path.rglob("*") if item.is_file()) if path.is_dir() else [path]
    digest = hashlib.sha256()
    for item in files:
        relative = str(item.relative_to(path)) if path.is_dir() else item.name
        digest.update(relative.encode())
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def verify_artifact(path: Path, manifest: ArtifactManifest) -> None:
    if artifact_hash(path) != manifest.checkpoint_hash:
        raise ValueError("checkpoint artifact hash mismatch")


class VariantTrainer(Protocol):
    def __call__(self, spec: ExperimentSpec) -> ArtifactManifest: ...


class ReplayEvaluator(Protocol):
    def __call__(
        self, spec: ExperimentSpec, artifact: ArtifactManifest, split: str,
    ) -> Mapping[str, Any]: ...


def run_variant(
    spec: ExperimentSpec, *, split: str, trainer: VariantTrainer,
    evaluator: ReplayEvaluator, registry: ExperimentRegistry,
) -> ExperimentAttempt:
    """Train and evaluate a development split.

    Release-test evaluation requires :func:`evaluate_variant` with the already
    selected artifact so test access cannot silently retrain the policy.
    """
    if split == "test":
        raise ValueError("release test requires a frozen artifact; use evaluate_variant")
    artifact = trainer(spec)
    return evaluate_variant(
        spec, artifact=artifact, split=split, evaluator=evaluator, registry=registry,
    )


def evaluate_variant(
    spec: ExperimentSpec, *, artifact: ArtifactManifest, split: str,
    evaluator: ReplayEvaluator, registry: ExperimentRegistry,
) -> ExperimentAttempt:
    """Evaluate an existing immutable artifact without invoking a trainer."""
    if artifact.spec_hash != spec.spec_hash:
        raise ValueError("training artifact does not match experiment specification")
    config = {
        "spec": spec, "artifact_manifest_hash": artifact.manifest_hash,
        "adapter_hash": artifact.adapter_hash,
    }
    return ExperimentRunner(
        lambda _config, selected: evaluator(spec, artifact, selected), registry,
    ).run(config, split=split)
