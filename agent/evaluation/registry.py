"""Durable experiment attempts with an atomic, one-shot release-test guard."""
from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from agent.contracts import FrozenDict, canonical_json, content_hash


_SENSITIVE = {"api_key", "apikey", "token", "secret", "password", "authorization"}


def _is_sensitive(name: str) -> bool:
    return (name in _SENSITIVE
            or name.endswith(("_api_key", "_secret_key", "_access_token", "_password")))


def _reject_sensitive(value: Any, path: str = "config") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).casefold().replace("-", "_")
            if _is_sensitive(normalized):
                raise ValueError(f"sensitive field is not permitted in experiment records: {path}.{key}")
            _reject_sensitive(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_sensitive(item, f"{path}[{index}]")


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
    status: str = "completed"
    failure: str | None = None


class ExperimentRegistry:
    """Record every attempt and reserve a release test before data is exposed."""

    def __init__(self, path: Path | None = None, *, release_id: str = "default") -> None:
        if not isinstance(release_id, str) or not release_id.strip():
            raise ValueError("release_id must be nonempty")
        self.release_id = release_id.strip()
        self._memory: list[ExperimentAttempt] = []
        self._final_test_used = False
        self._db: sqlite3.Connection | None = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(path, timeout=30, isolation_level=None)
            self._db.row_factory = sqlite3.Row
            self._db.executescript(
                """CREATE TABLE IF NOT EXISTS experiment_attempts (
                       sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                       attempt_id TEXT UNIQUE NOT NULL,
                       release_id TEXT NOT NULL,
                       config_hash TEXT NOT NULL,
                       config_json TEXT NOT NULL,
                       split TEXT NOT NULL,
                       status TEXT NOT NULL,
                       metrics_json TEXT NOT NULL,
                       failure TEXT);
                   CREATE TABLE IF NOT EXISTS final_test_locks (
                       release_id TEXT PRIMARY KEY,
                       attempt_id TEXT UNIQUE NOT NULL);"""
            )

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None

    @property
    def attempts(self) -> tuple[ExperimentAttempt, ...]:
        if self._db is None:
            return tuple(self._memory)
        rows = self._db.execute(
            "SELECT attempt_id,config_hash,split,metrics_json,status,failure "
            "FROM experiment_attempts WHERE release_id=? ORDER BY sequence",
            (self.release_id,),
        ).fetchall()
        return tuple(ExperimentAttempt(
            row["attempt_id"], row["config_hash"], row["split"],
            _freeze(json.loads(row["metrics_json"])), row["status"], row["failure"],
        ) for row in rows)

    @staticmethod
    def _validate_split(split: str) -> None:
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")

    def begin(self, config: Mapping[str, Any], *, split: str) -> ExperimentAttempt:
        self._validate_split(split)
        _reject_sensitive(config)
        serialized = canonical_json(config)
        config_hash = content_hash(config)
        attempt_id = content_hash({
            "release_id": self.release_id, "nonce": uuid.uuid4().hex,
            "config_hash": config_hash, "split": split,
        })
        pending = ExperimentAttempt(attempt_id, config_hash, split, FrozenDict(), "pending")
        if self._db is None:
            if split == "test" and self._final_test_used:
                raise ValueError("final test has already been consumed")
            if split == "test":
                self._final_test_used = True
            self._memory.append(pending)
            return pending
        try:
            self._db.execute("BEGIN IMMEDIATE")
            if split == "test":
                self._db.execute(
                    "INSERT INTO final_test_locks(release_id,attempt_id) VALUES(?,?)",
                    (self.release_id, attempt_id),
                )
            self._db.execute(
                "INSERT INTO experiment_attempts"
                "(attempt_id,release_id,config_hash,config_json,split,status,metrics_json) "
                "VALUES(?,?,?,?,?,'pending','{}')",
                (attempt_id, self.release_id, config_hash, serialized, split),
            )
            self._db.commit()
        except sqlite3.IntegrityError as exc:
            self._db.rollback()
            if split == "test":
                raise ValueError("final test has already been consumed") from exc
            raise
        except Exception:
            self._db.rollback()
            raise
        return pending

    def finish(
        self, attempt_id: str, *, metrics: Mapping[str, Any] | None = None,
        failure: str | None = None,
    ) -> ExperimentAttempt:
        if (metrics is None) == (failure is None):
            raise ValueError("provide exactly one of metrics or failure")
        payload = metrics or {}
        canonical_json(payload)
        status = "completed" if failure is None else "failed"
        safe_failure = None if failure is None else str(failure).split(":", 1)[0]
        if self._db is None:
            for index, attempt in enumerate(self._memory):
                if attempt.attempt_id == attempt_id:
                    updated = ExperimentAttempt(
                        attempt.attempt_id, attempt.config_hash, attempt.split,
                        _freeze(payload), status, safe_failure,
                    )
                    self._memory[index] = updated
                    return updated
            raise ValueError("unknown attempt_id")
        cursor = self._db.execute(
            "UPDATE experiment_attempts SET status=?,metrics_json=?,failure=? "
            "WHERE attempt_id=? AND release_id=? AND status='pending'",
            (status, canonical_json(payload), safe_failure, attempt_id, self.release_id),
        )
        if cursor.rowcount != 1:
            raise ValueError("unknown or completed attempt_id")
        return next(item for item in self.attempts if item.attempt_id == attempt_id)

    def record(
        self, config: Mapping[str, Any], *, split: str, metrics: Mapping[str, Any],
    ) -> ExperimentAttempt:
        pending = self.begin(config, split=split)
        return self.finish(pending.attempt_id, metrics=metrics)


class ExperimentRunner:
    """Reserve, execute, and durably complete or fail an offline evaluation."""

    def __init__(
        self, evaluator: Callable[[Mapping[str, Any], str], Mapping[str, Any]],
        registry: ExperimentRegistry | None = None,
    ) -> None:
        if not callable(evaluator):
            raise TypeError("evaluator must be callable")
        self.evaluator = evaluator
        self.registry = registry or ExperimentRegistry()

    def run(self, config: Mapping[str, Any], *, split: str) -> ExperimentAttempt:
        pending = self.registry.begin(config, split=split)
        try:
            metrics = self.evaluator(config, split)
            if not isinstance(metrics, Mapping):
                raise TypeError("evaluator must return a metrics mapping")
        except Exception as exc:
            self.registry.finish(pending.attempt_id, failure=type(exc).__name__)
            raise
        return self.registry.finish(pending.attempt_id, metrics=metrics)
