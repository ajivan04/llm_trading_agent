"""Training-only historical PPO environment with private future execution data."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent.contracts import (
    FrozenDict,
    canonical_json,
    content_hash,
    normalize_timestamp,
    parse_timestamp,
)

from .core import RewardComponents

_FORBIDDEN_OBSERVATION_KEYS = {
    "next_open",
    "next_close",
    "outcome",
    "outcome_close",
    "reward",
    "future_return",
    "realized_return",
    "preference_score",
}
_SENSITIVE_KEYS = {"api_key", "apikey", "token", "secret", "password", "authorization"}


def _audit_observation(value: Any, path: str = "messages") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).casefold().replace("-", "_")
            if normalized in _FORBIDDEN_OBSERVATION_KEYS:
                raise ValueError(
                    f"future outcome field is forbidden in PPO observation: {path}.{key}"
                )
            if normalized in _SENSITIVE_KEYS or normalized.endswith(
                ("_api_key", "_access_token", "_password")
            ):
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
    decision_close: float | None = None
    enforce_risk: bool = False
    evidence_aliases: Mapping[str, str] = field(default_factory=dict)
    original_messages: tuple[Mapping[str, Any], ...] = ()

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
        from agent.policy.evidence import visible_evidence

        visible = visible_evidence(self.messages, expected_as_of=decision_at)
        if set(evidence_ids) != set(visible):
            raise ValueError("PPO evidence allowlist must exactly match visible prompt records")
        object.__setattr__(self, "evidence_ids", evidence_ids)
        symbol = self.symbol.strip().upper()
        if not symbol:
            raise ValueError("symbol must be nonempty")
        if symbol not in {item.get("symbol") for item in visible.values()}:
            raise ValueError("PPO symbol must be supported by visible prompt evidence")
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
        if type(self.enforce_risk) is not bool:
            raise ValueError("enforce_risk must be an explicit boolean")
        object.__setattr__(self, "evidence_aliases", _freeze_json(self.evidence_aliases))
        object.__setattr__(
            self, "original_messages", tuple(_freeze_json(x) for x in self.original_messages)
        )
        if self.evidence_aliases:
            from agent.policy.evidence import alias_messages

            projected, aliases = alias_messages(self.original_messages)
            if (
                canonical_json(projected) != canonical_json(self.messages)
                or aliases != self.evidence_aliases
            ):
                raise ValueError("PPO aliases do not match the original visible prompt")
        if self.enforce_risk:
            if self.decision_close is None:
                raise ValueError("risk-checked PPO requires a decision-time close")
            _positive(self.decision_close, "decision_close")
            states = []
            for message in self.messages:
                if message.get("role") == "user":
                    try:
                        payload = json.loads(message.get("content", ""))
                    except (TypeError, json.JSONDecodeError):
                        continue
                    if isinstance(payload, dict) and isinstance(payload.get("state"), dict):
                        states.append(payload["state"])
            expected_positions = {self.symbol: self.starting_qty} if self.starting_qty else {}
            if not states or any(
                state.get("cash") != self.starting_cash
                or state.get("positions") != expected_positions
                or state.get("equity")
                != self.starting_cash + self.starting_qty * self.decision_close
                or normalize_timestamp(state.get("as_of")) != self.decision_at
                for state in states
            ):
                raise ValueError("PPO starting state differs from actual prompt account")


class HistoricalPPOEnvironment:
    """Independent one-step training episodes; future prices never enter observations."""

    def __init__(self, episodes: tuple[ReplayEpisode, ...]) -> None:
        if not episodes:
            raise ValueError("PPO environment requires training episodes")
        if len({item.episode_id for item in episodes}) != len(episodes):
            raise ValueError("PPO episode IDs must be unique")
        if episodes != tuple(
            sorted(episodes, key=lambda item: (item.decision_at, item.episode_id))
        ):
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
        failure = None
        try:
            from agent.policy.evidence import resolve_decisions

            decisions = resolve_decisions(
                action,
                messages=episode.messages,
                allowed_symbols=(episode.symbol,),
                aliases=episode.evidence_aliases,
                original_messages=episode.original_messages if episode.evidence_aliases else None,
            )
            if len(decisions) != 1:
                raise ValueError("one decision is required")
            decision = decisions[0]
        except ValueError as exc:
            from agent.evaluation.model import decision_failure_category

            failure = decision_failure_category(str(exc))
            invalid = 1
            decision = None
        cash = episode.starting_cash
        quantity = episode.starting_qty
        if decision is not None and episode.enforce_risk:
            from agent.config import Config
            from trading_agent import Portfolio, RiskEngine

            config = Config(
                universe=(episode.symbol,),
                sector_map=((episode.symbol, "unknown"),),
                initial_cash=episode.starting_cash + episode.starting_qty * episode.decision_close,
                max_shares=episode.max_qty,
                slippage_bps=episode.cost_bps,
            )
            portfolio = Portfolio(
                episode.starting_cash,
                {episode.symbol: episode.starting_qty} if episode.starting_qty else {},
                peak_equity=config.initial_cash,
            )
            allowed = {episode.evidence_aliases.get(eid, eid) for eid in episode.evidence_ids}
            failure = RiskEngine(config).check(
                decision, portfolio, {episode.symbol: episode.decision_close}, allowed
            )
            if failure is None:
                failure = RiskEngine(config).check(
                    decision, portfolio, {episode.symbol: episode.next_open}, allowed
                )
            if failure is not None:
                invalid, decision = 1, None
                failure = "risk_" + failure
        if decision is not None and decision.action != "hold":
            if decision.qty > episode.max_qty:
                invalid = 1
                failure = "risk_quantity"
            elif decision.action == "buy":
                price = episode.next_open * (1 + episode.cost_bps / 10_000)
                if decision.qty * price > cash:
                    invalid = 1
                    failure = "risk_cash"
                else:
                    cash -= decision.qty * price
                    quantity += decision.qty
            else:
                price = episode.next_open * (1 - episode.cost_bps / 10_000)
                if decision.qty > quantity:
                    invalid = 1
                    failure = "risk_holdings"
                else:
                    cash += decision.qty * price
                    quantity -= decision.qty
        equity_before = episode.starting_cash + episode.starting_qty * episode.next_open
        equity_after = cash + quantity * episode.outcome_close
        reward = RewardComponents(
            equity_before,
            equity_after,
            benchmark_return=episode.benchmark_return,
            invalid_proposals=invalid,
        ).total()
        # Rows carry independent starting portfolios. Crossing into another row would
        # falsely treat unrelated account state as a sequential trajectory.
        done = True
        next_observation = {"episode_id": "terminal", "messages": episode.messages}
        info = {
            "episode_id": episode.episode_id,
            "invalid_proposals": invalid,
            "failure_category": failure,
            "equity_before": equity_before,
            "equity_after": equity_after,
            "reward_components": {
                "net_log_return": math.log(equity_after / equity_before),
                "benchmark_log_return": math.log1p(episode.benchmark_return),
                "invalid_proposals": float(invalid),
            },
        }
        if failure is not None:
            info["reward_components"][f"failure/{failure}"] = 1.0
        return next_observation, reward, done, False, info


def critical_ppo_code_hashes():
    """Bind reward, risk, prompts and optimizer semantics to compilation/resume."""
    import hashlib

    agent = Path(__file__).parent.parent
    names = (
        "training/ppo_prepare.py",
        "training/environment.py",
        "training/qwen_ppo.py",
        "training/execution.py",
        "training/core.py",
        "training/ppo_math.py",
        "training/validation.py",
        "context.py",
        "features.py",
        "price_basis.py",
        "contracts.py",
        "config.py",
        "policy/core.py",
        "policy/evidence.py",
        "../trading_agent.py",
    )
    return {name: hashlib.sha256((agent / name).read_bytes()).hexdigest() for name in names}


def create_training_environment(config: Mapping[str, Any]) -> HistoricalPPOEnvironment:
    """CLI factory; config must contain an episode JSONL path."""
    if set(config) - {"episodes_path", "protocol_path"} or "episodes_path" not in config:
        raise ValueError("environment config must contain episodes_path and optional protocol_path")
    path = Path(config["episodes_path"])
    encoded = path.read_bytes()
    protocol = None
    if "protocol_path" in config:
        import hashlib

        protocol = json.loads(Path(config["protocol_path"]).read_text())
        expected_hash = content_hash(
            {key: value for key, value in protocol.items() if key != "protocol_hash"}
        )
        if (
            protocol.get("protocol_hash") != expected_hash
            or protocol.get("version") != "final-decision-nlp-v1"
            or protocol.get("episodes_sha256") != hashlib.sha256(encoded).hexdigest()
            or protocol.get("critical_code_hashes") != critical_ppo_code_hashes()
        ):
            raise ValueError("PPO compiled protocol/episode integrity failed")
    episodes = []
    for line_number, line in enumerate(encoded.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            row["messages"] = tuple(row["messages"])
            row["evidence_ids"] = tuple(row["evidence_ids"])
            episodes.append(ReplayEpisode(**row))
        except Exception as exc:
            raise ValueError(f"invalid PPO replay row {line_number}") from exc
    if protocol is not None:
        train_end = parse_timestamp(protocol["train_end"])
        if (
            len(episodes) != protocol.get("episodes")
            or not all(item.enforce_risk for item in episodes)
            or any(parse_timestamp(item.outcome_at) > train_end for item in episodes)
        ):
            raise ValueError("PPO compiled episodes violate the train-only risk protocol")
    return HistoricalPPOEnvironment(tuple(episodes))
