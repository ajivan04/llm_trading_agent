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
from agent.trajectories import _result, collect_trajectories

_ALIASES = {"get_news": "get_news_for_symbols"}
_BUILTINS = frozenset(
    {
        "get_news_for_symbols",
        "search_market_news",
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


def list_tools(*, session: NotebookSession, verbose=True):
    """Return the configured tool schemas, optionally printing notebook status."""
    schemas = list(session.registry().schemas)
    if verbose:
        for schema in schemas:
            print(f"{schema['name']}: ready — {schema['description']}")
    return schemas


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
    """Replay an explicit training or validation plan with the same policy loop.

    Set ``session.replay['split'] = 'train'`` for the notebook's base collection
    step, or ``'validation'`` for before/after comparison. Held-out testing stays
    outside this convenience function. Returned ``collection`` contains actual
    prompts, raw completions, tool calls and outcomes even without a collector.
    """
    del verbose  # The underlying runner emits durable flushed progress events.
    session = _session(tok, model)
    if session.replay is None:
        raise ValueError("run_backtest requires an explicit replay configuration")
    config = dict(session.replay)
    plan = tuple(config.get("decision_sessions", ()))
    if type(n_days) is not int or not 0 < n_days <= len(plan):
        raise ValueError("n_days must select a positive prefix of the declared replay plan")
    if config["settings"].initial_cash != initial_cash:
        raise ValueError("starting cash must match the replay configuration")
    split = config.pop("split", "validation")
    if split not in {"train", "validation"}:
        raise ValueError("notebook backtest exposes train or validation, never held-out testing")
    if "decide" in config:
        raise ValueError("the configured session policy owns backtest decisions")
    config["decision_sessions"] = plan
    config["max_sessions"] = n_days
    callback = getattr(collector, "on_replay_result", None) if collector is not None else None
    notebook_collector = collector is not None and all(
        callable(getattr(collector, name, None)) for name in ("log", "update_outcome")
    )
    if collector is not None and not notebook_collector and not callable(callback):
        raise TypeError("collector must provide log/update_outcome or on_replay_result")
    if split == "train":
        # Training uses the actual base-policy collector, including every failed
        # or HOLD day. Its own boundary checks keep next-session labels private.
        if config.pop("benchmark_symbol", None) is not None:
            raise ValueError("training collection does not accept a benchmark override")
        collection = collect_trajectories(decide=session.policy, split="train", **config)
        equity = [initial_cash, *[row["equity_after"] for row in collection.records]]
        trades = [fill for row in collection.records for fill in row["fills"]]
        metrics = {
            "starting_cash": initial_cash,
            "ending_equity": equity[-1],
            "percentage_return": equity[-1] / initial_cash - 1,
            "dollar_profit": equity[-1] - initial_cash,
            "sessions": len(collection.records),
            **collection.counts,
        }
        result = collection
    else:
        result = evaluate_research_policy(decide=session.policy, **config)
        collection = getattr(result, "collection", None)
        equity = [point["equity"] for point in result.equity_curve]
        trades, metrics = list(result.trades), dict(result.metrics)
    collector_error = False
    if notebook_collector:
        if collection is None:
            raise ValueError("replay result lacks the actual collection required by the collector")
        try:
            for row in collection.records:
                if not row["messages"] or not row["completion"]:
                    continue  # Never create fallback model targets for failed setup/generation.
                index = collector.log(
                    canonical_json(row["messages"]),
                    row["completion"],
                    row["decisions"],
                    row["equity_before"],
                    tool_calls=row["tool_call_count"],
                )
                if index is not None:
                    collector.update_outcome(index, row["equity_after"])
            # The notebook may now call collector.auto_label() using these exact
            # matured outcomes. No manual GOOD decision is manufactured here.
        except Exception:
            collector_error = True
    elif callback is not None:
        try:
            callback(result)
        except Exception:
            collector_error = True
    return {
        "equity_history": equity,
        "trades": trades,
        "total_return": 100 * metrics["percentage_return"],
        "final_equity": metrics["ending_equity"],
        "metrics": metrics,
        "return_units": "percent",
        "split": split,
        "collector_error": collector_error,
        "collection": collection,
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
    if type(n_iterations) is not int or not 1 <= n_iterations <= 100000:
        raise ValueError("n_iterations must be a finite positive count, at most 100000")
    if (
        type(submit) is not bool
        or isinstance(interval_seconds, bool)
        or not isinstance(interval_seconds, (int, float))
        or not math.isfinite(interval_seconds)
        or not 0 <= interval_seconds <= 3600
    ):
        raise ValueError("submit must be explicit and interval_seconds must be finite in 0..3600")
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
            event = {"cycle_id": cycle_id, "status": "paused", "error": "execution_paused"}
            results.append(event)
            if callback is not None:
                try:
                    callback(event)
                except Exception:
                    event["collector_error"] = True
            break
        except Exception:
            event = {"cycle_id": cycle_id, "status": "error", "error": "paper_cycle_failed"}
            results.append(event)
            if callback is not None:
                try:
                    callback(event)
                except Exception:
                    event["collector_error"] = True
            break
        item = {"cycle_id": cycle_id, "status": result["status"], "result": result}
        results.append(item)
        if callback is not None:
            try:
                callback(result)
            except Exception:
                item["collector_error"] = True  # A logging failure never retries a trade.
        # Model/schema and source failures that explicitly returned without an
        # execution attempt should not end a multi-hour session. An unknown
        # status or exception may mean uncertain broker state, so stop instead.
        if result["status"] not in {
            "completed",
            "invalid_policy",
            "expired_signal",
            "data_unavailable",
            "provider_error",
        }:
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
    "list_tools",
    "dispatch_tool",
    "llm_agent_decide",
    "run_backtest",
    "run_trading_loop",
    "reflect_on_trade",
]
