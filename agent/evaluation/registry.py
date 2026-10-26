"""Immutable experiment-attempt registry with one-shot final test control."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import Any, Mapping

from agent.contracts import FrozenDict, canonical_json, content_hash


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return FrozenDict({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    canonical_json(value)
    return value


@dataclass(frozen=True)
class ExperimentAttempt:
    attempt_id: str
    config_hash: str
    split: str
    metrics: Mapping[str, Any]


class ExperimentRegistry:
    def __init__(self) -> None:
        self._attempts: list[ExperimentAttempt] = []
        self._final_test_used = False

    @property
    def attempts(self) -> tuple[ExperimentAttempt, ...]:
        return tuple(self._attempts)

    def record(
        self, config: Mapping[str, Any], *, split: str, metrics: Mapping[str, Any],
    ) -> ExperimentAttempt:
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        canonical_json(config)
        canonical_json(metrics)
        if split == "test" and self._final_test_used:
            raise ValueError("final test has already been consumed")
        config_hash = content_hash(config)
        attempt_id = content_hash({
            "sequence": len(self._attempts), "config_hash": config_hash,
            "split": split, "metrics": metrics,
        })
        attempt = ExperimentAttempt(attempt_id, config_hash, split, _freeze(metrics))
        self._attempts.append(attempt)
        if split == "test":
            self._final_test_used = True
        return attempt


class ExperimentRunner:
    """Execute a supplied offline evaluator and record every completed attempt."""

    def __init__(
        self, evaluator: Callable[[Mapping[str, Any], str], Mapping[str, Any]],
        registry: ExperimentRegistry | None = None,
    ) -> None:
        if not callable(evaluator):
            raise TypeError("evaluator must be callable")
        self.evaluator = evaluator
        self.registry = registry or ExperimentRegistry()

    def run(self, config: Mapping[str, Any], *, split: str) -> ExperimentAttempt:
        metrics = self.evaluator(config, split)
        if not isinstance(metrics, Mapping):
            raise TypeError("evaluator must return a metrics mapping")
        return self.registry.record(config, split=split, metrics=metrics)

