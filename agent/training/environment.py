"""Training-only historical PPO environment with private future execution data."""
from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.contracts import FrozenDict, canonical_json, normalize_timestamp, parse_timestamp
from agent.policy import parse_decisions

from .core import RewardComponents

_FORBIDDEN_OBSERVATION_KEYS = {
    "next_open", "next_close", "outcome", "outcome_close", "reward",
    "future_return", "realized_return", "preference_score",
}
_SENSITIVE_KEYS = {"api_key", "apikey", "token", "secret", "password", "authorization"}


def _audit_observation(value: Any, path: str = "messages") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).casefold().replace("-", "_")
            if normalized in _FORBIDDEN_OBSERVATION_KEYS:
                raise ValueError(
                    f"future outcome field is forbidden in PPO observation: {path}.{key}")
            if (normalized in _SENSITIVE_KEYS
                    or normalized.endswith(("_api_key", "_access_token", "_password"))):
                raise ValueError(f"sensitive field is forbidden in PPO observation: {path}.{key}")
            _audit_observation(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _audit_observation(item, f"{path}[{index}]")
    elif isinstance(value, str) and value.lstrip().startswith(("{", "[")):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return
        _audit_observation(decoded, path)


def _freeze_json(value: Any) -> Any:
    canonical_json(value)
    if isinstance(value, Mapping):
        return FrozenDict({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _positive(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return result


@dataclass(frozen=True)
class ReplayEpisode:
    episode_id: str
    split: str
    decision_at: str
    outcome_at: str
    messages: tuple[Mapping[str, Any], ...]
    evidence_ids: tuple[str, ...]
    symbol: str
    next_open: float
    outcome_close: float
    starting_cash: float
    starting_qty: int = 0
    max_qty: int = 50
    cost_bps: float = 5.0
    benchmark_return: float = 0.0

    def __post_init__(self) -> None:
        if self.split != "train":
            raise ValueError("PPO replay episodes must be train-only")
        if not self.episode_id.strip() or not self.messages:
            raise ValueError("episode_id and messages must be nonempty")
        decision_at = normalize_timestamp(self.decision_at)
        outcome_at = normalize_timestamp(self.outcome_at)
        if parse_timestamp(outcome_at) < parse_timestamp(decision_at):
            raise ValueError("PPO outcome_at cannot precede decision_at")
        object.__setattr__(self, "decision_at", decision_at)
        object.__setattr__(self, "outcome_at", outcome_at)
        if not all(isinstance(item, Mapping) for item in self.messages):
            raise TypeError("messages must contain mappings")
        _audit_observation(self.messages)
        object.__setattr__(self, "messages", tuple(_freeze_json(item) for item in self.messages))
        evidence_ids = tuple(str(item).strip() for item in self.evidence_ids)
        if not evidence_ids or any(not item for item in evidence_ids):
            raise ValueError("evidence_ids must be nonempty")
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("evidence_ids must be unique")
        object.__setattr__(self, "evidence_ids", evidence_ids)
        symbol = self.symbol.strip().upper()
        if not symbol:
            raise ValueError("symbol must be nonempty")
        object.__setattr__(self, "symbol", symbol)
        for name in ("next_open", "outcome_close", "starting_cash"):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        if type(self.starting_qty) is not int or self.starting_qty < 0:
            raise ValueError("starting_qty must be a nonnegative integer")
        if type(self.max_qty) is not int or self.max_qty <= 0:
            raise ValueError("max_qty must be positive")
        if not 0 <= self.cost_bps < 10_000:
            raise ValueError("cost_bps must be in [0, 10000)")
        if not math.isfinite(self.benchmark_return) or self.benchmark_return <= -1:
            raise ValueError("benchmark_return is invalid")


class HistoricalPPOEnvironment:
    """Independent one-step training episodes; future prices never enter observations."""

    def __init__(self, episodes: tuple[ReplayEpisode, ...]) -> None:
        if not episodes:
            raise ValueError("PPO environment requires training episodes")
        if len({item.episode_id for item in episodes}) != len(episodes):
            raise ValueError("PPO episode IDs must be unique")
        if episodes != tuple(sorted(
                episodes, key=lambda item: (item.decision_at, item.episode_id))):
            raise ValueError("PPO episodes must be chronological")
        self.episodes = episodes
        self._index = 0

    def _observation(self) -> Mapping[str, Any]:
        episode = self.episodes[self._index]
        return {"episode_id": episode.episode_id, "messages": episode.messages}

    def reset(self, *, seed: int) -> Mapping[str, Any]:
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        self._index = random.Random(seed).randrange(len(self.episodes))
        return self._observation()

    def step(self, action: str) -> tuple[Mapping[str, Any], float, bool, bool, Mapping[str, Any]]:
        episode = self.episodes[self._index]
        invalid = 0
        try:
            decisions = parse_decisions(
                action, allowed_symbols=(episode.symbol,),
                allowed_evidence_ids=episode.evidence_ids,
            )
            if len(decisions) != 1:
                raise ValueError("one decision is required")
            decision = decisions[0]
        except ValueError:
            invalid = 1
            decision = None
        cash = episode.starting_cash
        quantity = episode.starting_qty
        if decision is not None and decision.action != "hold":
            if decision.qty > episode.max_qty:
                invalid = 1
            elif decision.action == "buy":
                price = episode.next_open * (1 + episode.cost_bps / 10_000)
                if decision.qty * price > cash:
                    invalid = 1
                else:
                    cash -= decision.qty * price
                    quantity += decision.qty
            else:
                price = episode.next_open * (1 - episode.cost_bps / 10_000)
                if decision.qty > quantity:
                    invalid = 1
                else:
                    cash += decision.qty * price
                    quantity -= decision.qty
        equity_before = episode.starting_cash + episode.starting_qty * episode.next_open
        equity_after = cash + quantity * episode.outcome_close
        reward = RewardComponents(
            equity_before, equity_after, benchmark_return=episode.benchmark_return,
            invalid_proposals=invalid,
        ).total()
        # Rows carry independent starting portfolios. Crossing into another row would
        # falsely treat unrelated account state as a sequential trajectory.
        done = True
        next_observation = {"episode_id": "terminal", "messages": episode.messages}
        info = {
            "episode_id": episode.episode_id, "invalid_proposals": invalid,
            "equity_before": equity_before, "equity_after": equity_after,
            "reward_components": {
                "net_log_return": math.log(equity_after / equity_before),
                "benchmark_log_return": math.log1p(episode.benchmark_return),
                "invalid_proposals": float(invalid),
            },
        }
        return next_observation, reward, done, False, info


def create_training_environment(config: Mapping[str, Any]) -> HistoricalPPOEnvironment:
    """CLI factory; config must contain an episode JSONL path."""
    if set(config) - {"episodes_path"} or "episodes_path" not in config:
        raise ValueError("environment config must contain only episodes_path")
    path = Path(config["episodes_path"])
    episodes = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            row["messages"] = tuple(row["messages"])
            row["evidence_ids"] = tuple(row["evidence_ids"])
            episodes.append(ReplayEpisode(**row))
        except Exception as exc:
            raise ValueError(f"invalid PPO replay row {line_number}") from exc
    return HistoricalPPOEnvironment(tuple(episodes))
