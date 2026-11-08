"""Notebook-facing names for the configured research system, not raw Torch globals.

Pass a NotebookSession as ``model`` and ``tok=None`` (or that backend's loaded
tokenizer). The explicit session supplies sources, cutoff, replay or paper loop.
This preserves notebook concepts without duplicating trading/execution logic.
"""

from __future__ import annotations

import math
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from agent.contracts import canonical_json, normalize_timestamp, parse_timestamp
from agent.evaluation.research import evaluate_research_policy
from agent.execution import ExecutionPaused
from agent.memory import TradingMemory
from agent.policy import ToolRegistry
from agent.reflection import reflect_on_trade as _reflect_on_trade
from agent.research import ResearchPolicy
from agent.runtime import PaperTradingLoop
from agent.tools import AssignmentTools
from agent.trajectories import _result

_ALIASES = {"get_news": "get_news_for_symbols"}
_BUILTINS = frozenset(
    {
        "get_news_for_symbols",
        "get_price_history",
        "get_rsi",
        "get_macd",
        "get_global_quote",
        "get_sec_filings",
        "get_extra_context",
    }
)


@dataclass
class NotebookSession:
    policy: ResearchPolicy
    universe: tuple[str, ...]
    evidence_provider: Callable
    as_of: str | None = None
    paper_loop: PaperTradingLoop | None = None
    replay: Mapping[str, Any] | None = None
    last_result: Mapping[str, Any] | None = field(default=None, init=False)
    _tools: Any = field(default=None, init=False, repr=False)
    _registry: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.policy, ResearchPolicy) or not callable(self.evidence_provider):
            raise ValueError(
                "session requires a configured ResearchPolicy and explicit evidence source"
            )
        self.universe = tuple(s.strip().upper() for s in self.universe)
        if (
            not self.universe
            or any(not symbol for symbol in self.universe)
            or len(set(self.universe)) != len(self.universe)
        ):
            raise ValueError("session universe must be explicit and unique")
        if self.as_of is not None:
            self.as_of = normalize_timestamp(self.as_of)
        if self.paper_loop is not None and self.paper_loop.policy is not self.policy:
            raise ValueError("paper loop must use the same configured policy")

    def close(self):
        if self._tools is not None:
            self._tools.close()
        self._tools = self._registry = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def registry(self):
        if self._registry is None:
            if self.as_of is None:
                raise ValueError("bind an explicit session as_of before dispatching research tools")
            evidence = tuple(self.evidence_provider(self.as_of, self.universe))
            factory = (
                AssignmentTools
                if self.policy.mode == "historical"
                else AssignmentTools.from_environment
            )
            self._tools = factory(
                self.as_of, self.universe, evidence=evidence, cache_path=self.policy.cache_path
            )
            self._registry = ToolRegistry(self.as_of)
            self._tools.register_tools(self._registry)
        return self._registry


def _session(tok, model):
    if not isinstance(model, NotebookSession):
        raise TypeError(
            "model must be an explicit NotebookSession; raw notebook globals are unsupported"
        )
    tokenizer = getattr(model.policy.backend, "_tokenizer", None)
    if tok is not None and tok is not tokenizer:
        raise ValueError("tok must be the configured policy backend tokenizer, or None")
    return model


def register_tool(name, function=None, *, session: NotebookSession):
    """Expose a named built-in read-only tool; arbitrary callbacks are rejected."""
    actual = _ALIASES.get(name, name)
    if actual not in _BUILTINS:
        raise ValueError("only the configured read-only research built-ins may be registered")
    registry = session.registry()
    if function is not None and function != getattr(session._tools, actual):
        raise ValueError("custom callbacks cannot replace the tested cutoff-bound research tools")
    return next(schema for schema in registry.schemas if schema["name"] == actual)


def dispatch_tool(name, arguments, *, session: NotebookSession, as_of=None):
    registry = session.registry()
    if as_of is not None and normalize_timestamp(as_of) != registry.as_of:
        raise ValueError("dispatch cannot advance the bound decision cutoff")
    actual = _ALIASES.get(name, name)
    if actual not in _BUILTINS:
        raise ValueError("unsupported read-only research tool")
    return registry.dispatch(actual, arguments)


def llm_agent_decide(tok, model, state, universe=None, date=None, max_turns=6):
    """Return decisions, complete prompt, raw completion and actual tool-call count.

    Full prompt is the actual native message envelope serialized as JSON; it is
    not reconstructed from summary fields. The complete result/alias sidecar is
    retained as session.last_result. Failures raise, without fabricated HOLDs.
    """
    session = _session(tok, model)
    if (
        not isinstance(state, Mapping)
        or not {"cash", "equity", "positions"} <= set(state)
        or not isinstance(state["positions"], Mapping)
    ):
        raise ValueError("decision state requires actual cash, equity and position fields")
    if any(
        isinstance(state[key], bool)
        or not isinstance(state[key], (int, float))
        or not math.isfinite(state[key])
        or state[key] < 0
        for key in ("cash", "equity")
    ):
        raise ValueError("decision account amounts must be finite and nonnegative")
    if state["equity"] <= 0 or any(
        symbol not in session.universe or type(qty) is not int or qty < 0
        for symbol, qty in state["positions"].items()
    ):
        raise ValueError("decision state requires positive equity and supported long-only holdings")
    if type(max_turns) is not int or max_turns != session.policy.settings.max_turns:
        raise ValueError("max_turns must match the configured frozen research settings")
    if universe is not None and tuple(universe) != session.universe:
        raise ValueError("universe cannot silently differ from the configured session")
    cutoff = normalize_timestamp(state.get("as_of", session.as_of))
    if session.as_of is not None and cutoff != session.as_of:
        raise ValueError("account cutoff differs from the explicitly bound session")
    if date is not None:
        if isinstance(date, str) and len(date) == 10:
            if parse_timestamp(cutoff).date().isoformat() != date:
                raise ValueError("date cannot override the actual account cutoff")
        elif normalize_timestamp(date) != cutoff:
            raise ValueError("date cannot override the actual account cutoff")
    account = {**state, "as_of": cutoff}
    evidence = tuple(session.evidence_provider(cutoff, session.universe))
    response = session.policy(
        state=account, as_of=cutoff, evidence=evidence, universe=session.universe
    )
    session.last_result = response
    checked = _result(response, as_of=cutoff, universe=session.universe, state=account)
    return (
        [decision.to_dict() for decision in checked["decisions"]],
        canonical_json(response["messages"]),
        response["completion"],
        checked["tool_call_count"],
    )


def run_backtest(tok, model, collector=None, n_days=20, initial_cash=100000, verbose=True):
    """Use the dynamic validation runner; total_return is notebook PERCENT units."""
    del verbose  # The underlying runner emits durable flushed progress events.
    session = _session(tok, model)
    if session.replay is None:
        raise ValueError("run_backtest requires an explicit validation replay configuration")
    config = dict(session.replay)
    plan = tuple(config.get("decision_sessions", ()))
    if type(n_days) is not int or not 0 < n_days <= len(plan):
        raise ValueError("n_days must select a positive prefix of the declared validation plan")
    if config["settings"].initial_cash != initial_cash:
        raise ValueError("starting cash must match the frozen replay configuration")
    config["decision_sessions"] = plan
    config["max_sessions"] = n_days
    if config.get("split", "validation") != "validation":
        raise ValueError("notebook backtest exposes validation only, not held-out testing")
    config.pop("split", None)
    callback = getattr(collector, "on_replay_result", None) if collector is not None else None
    if collector is not None and not callable(callback):
        raise TypeError("collector must expose the explicit on_replay_result adapter")
    if "decide" in config:
        raise ValueError("the configured session policy owns backtest decisions")
    result = evaluate_research_policy(decide=session.policy, **config)
    collector_error = False
    if callback is not None:
        try:
            callback(result)
        except Exception:
            collector_error = True
    return {
        "equity_history": [point["equity"] for point in result.equity_curve],
        "trades": list(result.trades),
        "total_return": 100 * result.metrics["percentage_return"],
        "final_equity": result.metrics["ending_equity"],
        "metrics": dict(result.metrics),
        "return_units": "percent",
        "split": "validation",
        "collector_error": collector_error,
    }


def run_trading_loop(
    tc,
    tok,
    model,
    memory=None,
    collector=None,
    n_iterations=5,
    *,
    cycle_ids: Sequence[str] | None = None,
    submit=False,
    interval_seconds=0,
    sleep=time.sleep,
):
    """Bounded paper/dry cycles with stable IDs, generic errors and pause-on-uncertainty."""
    session = _session(tok, model)
    if not isinstance(tc, PaperTradingLoop) or tc is not session.paper_loop:
        raise TypeError("tc must be this session's configured PaperTradingLoop")
    if memory is not None and memory is not tc.memory:
        raise ValueError("memory cannot change the paper loop's isolated namespace")
    if type(n_iterations) is not int or not 1 <= n_iterations <= 100:
        raise ValueError("n_iterations must be bounded to 1..100")
    if type(submit) is not bool or not 0 <= interval_seconds <= 60:
        raise ValueError("submit must be explicit and interval_seconds must be 0..60")
    callback = getattr(collector, "on_cycle_result", None) if collector is not None else None
    if collector is not None and not callable(callback):
        raise TypeError("collector must expose the explicit on_cycle_result adapter")
    if isinstance(cycle_ids, (str, bytes)):
        raise ValueError("cycle_ids must be a sequence of complete IDs, not a string")
    if cycle_ids is None:
        if submit:
            raise ValueError("submitting paper cycles require explicit stable cycle_ids")
        prefix = "notebook-dry-" + uuid.uuid4().hex
        cycle_ids = [f"{prefix}-{index}" for index in range(n_iterations)]
    if (
        len(cycle_ids) != n_iterations
        or len(set(cycle_ids)) != len(cycle_ids)
        or any(not isinstance(key, str) or not key.strip() for key in cycle_ids)
    ):
        raise ValueError("cycle_ids must be unique explicit identifiers for every iteration")
    results = []
    for index, cycle_id in enumerate(cycle_ids):
        try:
            result = tc.run_once(cycle_id=cycle_id, submit=submit)
        except ExecutionPaused:
            results.append({"cycle_id": cycle_id, "status": "paused", "error": "execution_paused"})
            break
        except Exception:
            results.append({"cycle_id": cycle_id, "status": "error", "error": "paper_cycle_failed"})
            break
        item = {"cycle_id": cycle_id, "status": result["status"], "result": result}
        results.append(item)
        if callback is not None:
            try:
                callback(result)
            except Exception:
                item["collector_error"] = True  # A logging failure never retries a trade.
        if result["status"] != "completed":
            break
        if index + 1 < n_iterations and interval_seconds:
            try:
                sleep(interval_seconds)
            except Exception:
                item["interval_error"] = True
                break
    return results


def reflect_on_trade(
    tok, model, situation, decision, prev_equity, curr_equity, *, outcome=None, as_of=None
):
    session = _session(tok, model)
    return _reflect_on_trade(
        tok,
        session.policy.backend,
        situation,
        decision,
        prev_equity,
        curr_equity,
        outcome=outcome,
        as_of=as_of,
        backend=session.policy.backend,
    )


__all__ = [
    "NotebookSession",
    "TradingMemory",
    "register_tool",
    "dispatch_tool",
    "llm_agent_decide",
    "run_backtest",
    "run_trading_loop",
    "reflect_on_trade",
]
