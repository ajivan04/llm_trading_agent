"""Account-aware validation replay of the actual bounded research policy."""

from __future__ import annotations

import hashlib
import math
import statistics
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.contracts import canonical_json, content_hash, parse_timestamp
from agent.data.prices import DailyPriceBar
from agent.trajectories import TrajectoryCollection, TrajectorySettings, collect_trajectories

from .metrics import financial_metrics


@dataclass(frozen=True)
class ResearchReplayResult:
    metrics: Mapping[str, Any]
    equity_curve: tuple[Mapping[str, Any], ...]
    trades: tuple[Mapping[str, Any], ...]
    collection: TrajectoryCollection
    protocol: Mapping[str, Any]


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize_research_replay(
    collection: TrajectoryCollection,
    bars_by_symbol: Mapping[str, Sequence[DailyPriceBar]],
    *,
    benchmark_symbol: str | None = None,
) -> ResearchReplayResult:
    """Summarize every chronological decision, never GOOD-only filtered rows."""
    if collection.split not in {"validation", "test"}:
        raise ValueError("research model comparison requires validation or frozen test")
    records = collection.records
    if not records:
        raise ValueError("research evaluation needs at least one completed decision")
    initial = collection.settings.initial_cash
    benchmark_symbols = (
        (benchmark_symbol,) if benchmark_symbol is not None else collection.settings.universe
    )
    indexed = {
        symbol: {bar.session: bar for bar in bars_by_symbol.get(symbol, ())}
        for symbol in benchmark_symbols
    }
    first = str(records[0]["outcome_at"])[:10]
    entry = {symbol: rows.get(first) for symbol, rows in indexed.items()}
    if any(bar is None for bar in entry.values()):
        raise ValueError("benchmark requires every fixed-cohort opening mark")
    # Cohort and fractional benchmark shares are fixed at the first execution
    # open. Ex-date distributions on that entry date do not accrue to new buys.
    units = {
        symbol: initial / len(benchmark_symbols) / entry[symbol].open
        for symbol in benchmark_symbols
    }
    benchmark_cash = 0.0
    curve = [
        {
            "session": "initial",
            "cash": initial,
            "equity": initial,
            "benchmark_equity": initial,
            "gross_exposure": 0.0,
            "positions": {},
        }
    ]
    trades = []
    for index, row in enumerate(records):
        day = str(row["outcome_at"])[:10]
        for symbol in benchmark_symbols:
            bar = indexed[symbol].get(day)
            if bar is None:
                raise ValueError("benchmark cohort cannot change when a later mark is missing")
            if index:
                if bar.stock_splits not in {0, 1}:
                    units[symbol] *= bar.stock_splits
                benchmark_cash += units[symbol] * bar.dividends
        benchmark_equity = benchmark_cash + sum(
            units[symbol] * indexed[symbol][day].close for symbol in benchmark_symbols
        )
        after = row["portfolio_after"]
        equity = float(row["equity_after"])
        cash = float(after["cash"])
        if not math.isfinite(equity) or equity <= 0 or not math.isfinite(cash):
            raise ValueError("evaluation equity and cash must be finite with positive equity")
        curve.append(
            {
                "session": day,
                "decision_session": row["session"],
                "equity": equity,
                "cash": cash,
                "positions": after["positions"],
                "gross_exposure": equity - cash,
                "benchmark_equity": benchmark_equity,
                "policy_status": row.get("policy_status", row["status"]),
            }
        )
        trades.extend({**fill, "decision_session": row["session"]} for fill in row["fills"])
    turnover = sum(float(fill["qty"]) * float(fill["price"]) for fill in trades)
    financial = financial_metrics(
        [float(row["equity"]) for row in curve],
        benchmark_equities=[float(row["benchmark_equity"]) for row in curve],
        turnover=turnover,
        exposures=[
            {
                "gross": float(row["gross_exposure"]) / float(row["equity"]),
                "net": float(row["gross_exposure"]) / float(row["equity"]),
            }
            for row in curve[1:]
        ],
    )
    successful = sum(row["status"] == "success" for row in records)
    rejections = sum(len(row["execution_rejections"]) for row in records)
    proposed = sum(
        decision["action"] in {"buy", "sell"} for row in records for decision in row["decisions"]
    )
    holds = sum(decision["action"] == "hold" for row in records for decision in row["decisions"])
    latencies = [
        float(row["inference_seconds"])
        for row in records
        if row.get("inference_seconds") is not None
    ]
    if any(not math.isfinite(value) or value < 0 for value in latencies):
        raise ValueError("recorded decision latency must be finite and nonnegative")
    input_tokens = [
        row["initial_input_tokens"]
        for row in records
        if type(row.get("initial_input_tokens")) is int
    ]
    metrics = {
        **financial,
        "starting_cash": initial,
        "ending_equity": curve[-1]["equity"],
        "dollar_profit": float(curve[-1]["equity"]) - initial,
        "percentage_return": float(curve[-1]["equity"]) / initial - 1,
        "sessions": len(records),
        "planned_sessions": len(collection.planned_sessions),
        "complete_plan": len(records) == len(collection.planned_sessions),
        "valid_schema_rate": successful / len(records),
        "invalid_policy_decisions": len(records) - successful,
        "proposed_trades": proposed,
        "executed_trades": len(trades),
        "execution_rejections": rejections,
        "execution_rejection_rate": rejections / proposed if proposed else 0.0,
        "hold_decisions": holds,
        "tool_calls": sum(row["tool_call_count"] for row in records),
        "researched_decisions": sum(row["tool_call_count"] >= 1 for row in records),
        "researched_decision_rate": sum(row["tool_call_count"] >= 1 for row in records)
        / len(records),
        "failure_counts": dict(
            Counter(
                str(row.get("policy_failure") or row.get("error_class") or "invalid_policy_output")
                for row in records
                if row["status"] != "success"
            )
        ),
        "decision_cycle_seconds_mean": statistics.mean(latencies) if latencies else None,
        "decision_cycle_seconds_p50": _percentile(latencies, 0.5),
        "decision_cycle_seconds_p95": _percentile(latencies, 0.95),
        "initial_input_tokens_max": max(input_tokens) if input_tokens else None,
        "benchmark_symbols": len(benchmark_symbols),
        "test_policy_calls": len(records) if collection.split == "test" else 0,
    }
    protocol = {
        "version": (
            "account-aware-frozen-test-v1"
            if collection.split == "test"
            else "account-aware-validation-research-v1"
        ),
        "split": collection.split,
        "policy_identity": collection.policy_identity,
        "source_hash": collection.source_hash,
        "settings": collection.settings,
        "planned_sessions": collection.planned_sessions,
        "benchmark": {
            "type": "equal_weight_buy_hold",
            "symbols": benchmark_symbols,
            "entry": "first execution open",
            "cost_bps": 0,
            "corporate_actions": "split-adjusted units; dividends after entry date",
        },
        "execution": "decision close risk; next exchange-session open fill and close mark",
        "memory": "caller-owned scoped policy; not inherited from another run",
        "purpose": "validation only; not a held-out-week score",
    }
    return ResearchReplayResult(metrics, tuple(curve), tuple(trades), collection, protocol)


def _implementation_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    files = (
        "trading_agent.py",
        "agent/config.py",
        "agent/contracts.py",
        "agent/data/prices.py",
        "agent/price_basis.py",
        "agent/trajectories.py",
        "agent/evaluation/research.py",
        "agent/evaluation/metrics.py",
    )
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files}


def evaluate_research_policy(
    bars_by_symbol: Mapping[str, Sequence[DailyPriceBar]],
    *,
    decide: Callable[..., Any],
    settings: TrajectorySettings,
    decision_sessions: Sequence[str],
    next_session_resolver: Callable[[str], str],
    policy_identity: Mapping[str, Any] | None = None,
    evidence_provider: Callable | None = None,
    evidence_source_hash: str | None = None,
    journal: Path | None = None,
    resume: bool = False,
    max_sessions: int | None = None,
    benchmark_symbol: str | None = None,
) -> ResearchReplayResult:
    """Run the same bounded policy and independent risk path on current state.

    Dates must be consecutive exchange sessions so daily financial statistics do
    not disguise selected gaps as one-day returns. All failed/HOLD/risk-rejected
    decisions remain in the score. A total-prefix smoke can resume the same plan.
    """
    if not callable(next_session_resolver) or not decision_sessions:
        raise ValueError("evaluation requires an explicit contiguous exchange-session plan")
    if list(decision_sessions) != sorted(set(decision_sessions)):
        raise ValueError("evaluation decision sessions must be sorted and unique")
    if any(
        next_session_resolver(before) != after
        for before, after in zip(decision_sessions, decision_sessions[1:], strict=False)
    ):
        raise ValueError("daily validation metrics require contiguous exchange sessions")
    # The collector can skip incomplete/out-of-split training windows. A fixed
    # evaluation plan must instead fail before any model work or journal writes.
    if benchmark_symbol is not None and benchmark_symbol not in settings.universe:
        raise ValueError("benchmark symbol must belong to the source-bound evaluation universe")
    indexed = {
        symbol: {bar.session: bar for bar in bars_by_symbol.get(symbol, ())}
        for symbol in settings.universe
    }
    benchmark_symbols = (benchmark_symbol,) if benchmark_symbol is not None else settings.universe
    benchmark_bars = {
        symbol: {bar.session: bar for bar in bars_by_symbol.get(symbol, ())}
        for symbol in benchmark_symbols
    }
    outcomes = []
    for day in decision_sessions:
        try:
            following = next_session_resolver(day)
        except (KeyError, IndexError) as exc:
            raise ValueError("evaluation plan has no verified next exchange session") from exc
        if any(day not in rows or following not in rows for rows in indexed.values()):
            raise ValueError("every requested evaluation window requires complete universe marks")
        at = max(rows[day].close_at for rows in indexed.values())
        outcome_at = max(rows[following].close_at for rows in indexed.values())
        if not (
            parse_timestamp(settings.train_end)
            < parse_timestamp(at)
            < parse_timestamp(outcome_at)
            <= parse_timestamp(settings.validation_end)
        ):
            raise ValueError("every requested evaluation window must remain inside validation")
        if any(following not in rows for rows in benchmark_bars.values()):
            raise ValueError("benchmark requires all fixed-cohort execution marks before inference")
        outcomes.append(following)
    if any(
        next_session_resolver(before) != after
        for before, after in zip(outcomes, outcomes[1:], strict=False)
    ):
        raise ValueError("evaluation execution marks must follow consecutive exchange sessions")
    identity = dict(policy_identity or getattr(decide, "policy_identity", {}))
    code_hashes = _implementation_hashes()
    identity["evaluation_code_hashes"] = code_hashes
    identity["protocol_hash"] = content_hash(
        {
            "policy_protocol": identity.get("protocol_hash"),
            "implementation": code_hashes,
            "benchmark_symbols": benchmark_symbols,
            "benchmark_protocol": (
                "first-next-open equal-weight, zero costs, later splits/dividends-v1"
            ),
        }
    )
    collection = collect_trajectories(
        bars_by_symbol,
        decide=decide,
        settings=settings,
        evidence_provider=evidence_provider,
        evidence_source_hash=evidence_source_hash,
        split="validation",
        decision_sessions=decision_sessions,
        policy_identity=identity,
        next_session_resolver=next_session_resolver,
        journal=journal,
        resume=resume,
        max_sessions=max_sessions,
        evaluation_only=True,
    )
    return summarize_research_replay(collection, bars_by_symbol, benchmark_symbol=benchmark_symbol)


def write_research_result(result: ResearchReplayResult, output: Path) -> None:
    """Write summaries atomically; the checksum-verified journal remains source."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    documents = {
        "research_metrics.json": canonical_json(result.metrics) + "\n",
        "equity_curve.jsonl": "".join(canonical_json(row) + "\n" for row in result.equity_curve),
        "trades.jsonl": "".join(canonical_json(row) + "\n" for row in result.trades),
        "research_manifest.json": canonical_json(
            {
                "protocol": result.protocol,
                "protocol_hash": content_hash(result.protocol),
                "record_count": len(result.collection.records),
                "record_hash": content_hash(result.collection.records),
                "metrics_hash": content_hash(result.metrics),
            }
        )
        + "\n",
    }
    for name, text in documents.items():
        temporary = output / f".{name}.tmp"
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(output / name)
