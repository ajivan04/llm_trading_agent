"""Immutable, JSON-compatible contracts shared across agent components.

Timestamps are stored as normalized UTC ISO-8601 strings.  Provider adapters are
responsible for establishing truthful ``available_at`` values; retrieval time is
audit metadata and is deliberately not part of historical eligibility.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

UTC = timezone.utc


def normalize_timestamp(value: str | datetime) -> str:
    """Return an aware timestamp in UTC, rejecting naive or malformed values."""
    if isinstance(value, str):
        candidate = value.strip()
        if candidate.endswith(("Z", "z")):
            candidate = candidate[:-1] + "+00:00"
        try:
            value = datetime.fromisoformat(candidate)
        except ValueError as exc:
            raise ValueError("Invalid ISO-8601 timestamp") from exc
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("Timezone-aware timestamp required")
    if value.utcoffset() is None:
        raise ValueError("Timezone-aware timestamp required")
    return value.astimezone(UTC).isoformat()


def parse_timestamp(value: str | datetime) -> datetime:
    return datetime.fromisoformat(normalize_timestamp(value))


class FrozenDict(dict):
    """A small deeply-frozen dict that remains accepted by ``json.dumps``."""

    def _immutable(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("FrozenDict is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable

    def __deepcopy__(self, memo: dict[int, Any]) -> FrozenDict:
        return self


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return FrozenDict({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON numeric values must be finite")
        return value
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {item.name: _jsonable(getattr(value, item.name)) for item in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, datetime):
        return normalize_timestamp(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, float):
        raise ValueError("JSON numeric values must be finite")
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Serialize contracts/payloads deterministically for manifests and hashes."""
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


class JsonContract:
    def to_dict(self) -> dict[str, Any]:
        result = _jsonable(self)
        if not isinstance(result, dict):  # pragma: no cover - defensive
            raise TypeError("Contract did not serialize to an object")
        return result

    def to_json(self) -> str:
        return canonical_json(self)


def _required_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty")
    return value.strip()


def _symbol(value: str) -> str:
    return _required_text(value, "symbol").upper()


def _positive_int(value: int, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class Evidence(JsonContract):
    id: str
    source: str
    symbol: str | None
    observed_at: str
    available_at: str
    retrieved_at: str
    text: str
    values: Mapping[str, Any] = field(default_factory=FrozenDict)
    url: str | None = None
    revision_id: str | None = None
    accession_id: str | None = None
    vintage_id: str | None = None
    units: str | None = None
    entitlement: str | None = None
    raw_content_hash: str | None = None
    transformation_version: str = "v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _required_text(self.id, "id"))
        object.__setattr__(self, "source", _required_text(self.source, "source"))
        object.__setattr__(self, "symbol", _symbol(self.symbol) if self.symbol is not None else None)
        object.__setattr__(self, "text", _required_text(self.text, "text"))
        object.__setattr__(self, "transformation_version", _required_text(
            self.transformation_version, "transformation_version"))
        for name in ("observed_at", "available_at", "retrieved_at"):
            object.__setattr__(self, name, normalize_timestamp(getattr(self, name)))
        if parse_timestamp(self.observed_at) > parse_timestamp(self.available_at):
            raise ValueError("Evidence cannot be available before it was observed")
        if parse_timestamp(self.available_at) > parse_timestamp(self.retrieved_at):
            raise ValueError("Evidence cannot be retrieved before it was available")
        if not isinstance(self.values, Mapping):
            raise TypeError("values must be a mapping")
        object.__setattr__(self, "values", _freeze_json(self.values))
        if self.raw_content_hash is not None:
            digest = self.raw_content_hash.lower()
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise ValueError("raw_content_hash must be a SHA-256 hex digest")
            object.__setattr__(self, "raw_content_hash", digest)

    def eligible(self, as_of: str | datetime) -> bool:
        cutoff = parse_timestamp(as_of)
        return (parse_timestamp(self.observed_at) <= cutoff
                and parse_timestamp(self.available_at) <= cutoff)


@dataclass(frozen=True)
class Snapshot(JsonContract):
    snapshot_id: str
    as_of: str
    universe_version: str
    evidence_ids: tuple[str, ...]
    feature_version: str
    account_state: Mapping[str, Any]
    source_health: Mapping[str, Any]
    market_session_id: str
    compact_evidence: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        for name in ("snapshot_id", "universe_version", "feature_version", "market_session_id"):
            object.__setattr__(self, name, _required_text(getattr(self, name), name))
        object.__setattr__(self, "as_of", normalize_timestamp(self.as_of))
        ids = tuple(_required_text(item, "evidence_id") for item in self.evidence_ids)
        if len(ids) != len(set(ids)):
            raise ValueError("evidence_ids must be unique and ordered")
        object.__setattr__(self, "evidence_ids", ids)
        if not isinstance(self.account_state, Mapping) or not isinstance(self.source_health, Mapping):
            raise TypeError("account_state and source_health must be mappings")
        if not all(isinstance(item, Mapping) for item in self.compact_evidence):
            raise TypeError("compact_evidence must contain mappings")
        object.__setattr__(self, "account_state", _freeze_json(self.account_state))
        object.__setattr__(self, "source_health", _freeze_json(self.source_health))
        object.__setattr__(self, "compact_evidence", tuple(_freeze_json(x) for x in self.compact_evidence))


@dataclass(frozen=True)
class Decision(JsonContract):
    symbol: str
    action: str
    qty: int
    reason: str
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", _symbol(self.symbol))
        action = _required_text(self.action, "action").lower()
        if action not in {"buy", "sell", "hold"}:
            raise ValueError("action must be buy, sell, or hold")
        object.__setattr__(self, "action", action)
        if type(self.qty) is not int or self.qty < 0:
            raise ValueError("qty must be a nonnegative integer")
        if (action == "hold") != (self.qty == 0):
            raise ValueError("hold requires zero quantity; orders require positive quantity")
        object.__setattr__(self, "reason", _required_text(self.reason, "reason"))
        ids = tuple(_required_text(item, "evidence_id") for item in self.evidence_ids)
        if len(ids) != len(set(ids)):
            raise ValueError("evidence_ids must be unique")
        object.__setattr__(self, "evidence_ids", ids)


@dataclass(frozen=True)
class OrderIntent(JsonContract):
    intent_id: str
    run_id: str
    symbol: str
    action: str
    qty: int
    created_at: str
    client_order_id: str
    status: str = "intent"
    paper_only: bool = True

    def __post_init__(self) -> None:
        for name in ("intent_id", "run_id", "client_order_id"):
            object.__setattr__(self, name, _required_text(getattr(self, name), name))
        object.__setattr__(self, "symbol", _symbol(self.symbol))
        action = _required_text(self.action, "action").lower()
        if action not in {"buy", "sell"}:
            raise ValueError("Order intents must be buy or sell")
        object.__setattr__(self, "action", action)
        _positive_int(self.qty, "qty")
        object.__setattr__(self, "created_at", normalize_timestamp(self.created_at))
        if self.status not in {
            "intent", "submitting", "open", "unknown", "partial", "filled",
            "canceled", "expired", "rejected",
        }:
            raise ValueError("Invalid order intent status")
        if self.paper_only is not True:
            raise ValueError("Only paper-trading order intents are permitted")


@dataclass(frozen=True)
class Fill(JsonContract):
    order_id: str
    symbol: str
    action: str
    qty: int
    price: float
    filled_at: str
    broker_fill_id: str | None = None
    fee: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "order_id", _required_text(self.order_id, "order_id"))
        object.__setattr__(self, "symbol", _symbol(self.symbol))
        action = _required_text(self.action, "action").lower()
        if action not in {"buy", "sell"}:
            raise ValueError("Fill action must be buy or sell")
        object.__setattr__(self, "action", action)
        _positive_int(self.qty, "qty")
        if not isinstance(self.price, (int, float)) or isinstance(self.price, bool) or not math.isfinite(self.price) or self.price <= 0:
            raise ValueError("price must be positive and finite")
        if not isinstance(self.fee, (int, float)) or isinstance(self.fee, bool) or not math.isfinite(self.fee) or self.fee < 0:
            raise ValueError("fee must be nonnegative and finite")
        object.__setattr__(self, "filled_at", normalize_timestamp(self.filled_at))


@dataclass(frozen=True)
class Episode(JsonContract):
    episode_id: str
    run_id: str
    experiment_id: str
    fold: str
    snapshot_id: str
    decision_at: str
    outcome_at: str
    decisions: tuple[Decision, ...] = ()
    fills: tuple[Fill, ...] = ()
    state: Mapping[str, Any] = field(default_factory=FrozenDict)
    reward: float | None = None
    transcript: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        for name in ("episode_id", "run_id", "experiment_id", "snapshot_id"):
            object.__setattr__(self, name, _required_text(getattr(self, name), name))
        if self.fold not in {"train", "validation", "test", "paper", "demo"}:
            raise ValueError("Invalid episode fold")
        object.__setattr__(self, "decision_at", normalize_timestamp(self.decision_at))
        object.__setattr__(self, "outcome_at", normalize_timestamp(self.outcome_at))
        if parse_timestamp(self.outcome_at) < parse_timestamp(self.decision_at):
            raise ValueError("outcome_at cannot precede decision_at")
        object.__setattr__(self, "decisions", tuple(self.decisions))
        object.__setattr__(self, "fills", tuple(self.fills))
        if not all(isinstance(item, Decision) for item in self.decisions):
            raise TypeError("decisions must contain Decision contracts")
        if not all(isinstance(item, Fill) for item in self.fills):
            raise TypeError("fills must contain Fill contracts")
        if not isinstance(self.state, Mapping):
            raise TypeError("state must be a mapping")
        if not all(isinstance(item, Mapping) for item in self.transcript):
            raise TypeError("transcript must contain mappings")
        object.__setattr__(self, "state", _freeze_json(self.state))
        object.__setattr__(self, "transcript", tuple(_freeze_json(x) for x in self.transcript))
        if self.reward is not None and (isinstance(self.reward, bool)
                                        or not isinstance(self.reward, (int, float))
                                        or not math.isfinite(self.reward)):
            raise ValueError("reward must be finite")
