"""Sequential portfolio replay for model-generated decision predictions."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.contracts import canonical_json
from agent.policy import parse_decisions

from .metrics import financial_metrics
from .model import load_outcomes


@dataclass(frozen=True)
class PortfolioReplayResult:
    metrics: dict[str, float | int | None]
    equity_curve: tuple[dict[str, Any], ...]
    trades: tuple[dict[str, Any], ...]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected an object at {path}:{line_number}")
        rows.append(value)
    return rows


def replay_predictions(
    prepared: Path, predictions: Path, *, split: str, initial_cash: float = 100_000.0,
    cost_bps: float = 5.0, max_qty: int = 50,
) -> PortfolioReplayResult:
    """Replay one decision per example as a chronological cash/share portfolio.

    Predictions are generated from point-in-time prompts. Future prices are read only
    from the prepared audit/source artifact after the action has been selected.
    """
    prepared, predictions = Path(prepared), Path(predictions)
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    if not math.isfinite(initial_cash) or initial_cash <= 0:
        raise ValueError("initial_cash must be positive and finite")
    if not 0 <= cost_bps < 10_000:
        raise ValueError("cost_bps must be in [0, 10000)")
    if type(max_qty) is not int or max_qty <= 0:
        raise ValueError("max_qty must be positive")

    # Import lazily because agent.datasets imports evaluation split helpers.
    from agent.datasets import load_dataset_artifact

    dataset = load_dataset_artifact(prepared / "dataset")
    examples = {item.example_id: item for item in dataset.examples if item.split == split}
    outcomes = {item.example_id: item for item in load_outcomes(
        prepared / "outcome-audit.jsonl", split=split)}
    prediction_rows = {row.get("example_id"): row for row in _load_jsonl(predictions)}
    if set(examples) != set(outcomes) or set(examples) != set(prediction_rows):
        raise ValueError("predictions, examples, and outcomes must cover the same split")

    bars = _load_jsonl(prepared / "price-bars.jsonl")
    bar_by_symbol_session = {
        (str(row["symbol"]).upper(), str(row["session"])): row for row in bars
    }
    events: list[dict[str, Any]] = []
    invalid_decisions = 0
    for example_id, example in examples.items():
        outcome = outcomes[example_id]
        decision_session = str(example.decision_at)[:10]
        execution_session = str(example.outcome_at)[:10]
        bar = bar_by_symbol_session.get((outcome.symbol, execution_session))
        if bar is None:
            raise ValueError(f"missing execution bar for {example_id}")
        raw = prediction_rows[example_id].get("generation", "")
        try:
            decisions = parse_decisions(
                raw, allowed_symbols=(outcome.symbol,),
                allowed_evidence_ids=outcome.evidence_ids,
            )
            if len(decisions) != 1:
                raise ValueError("exactly one decision is required")
            decision = decisions[0]
        except (TypeError, ValueError):
            decision = None
            invalid_decisions += 1
        events.append({
            "example_id": example_id, "decision_session": decision_session,
            "execution_session": execution_session, "symbol": outcome.symbol,
            "open": float(bar["open"]), "close": float(bar["close"]),
            "decision": decision,
        })
    events.sort(key=lambda row: (row["execution_session"], row["symbol"], row["example_id"]))

    cash = float(initial_cash)
    shares: dict[str, int] = defaultdict(int)
    trades: list[dict[str, Any]] = []
    curve = [{"session": "initial", "cash": cash, "equity": cash, "gross_exposure": 0.0}]
    total_turnover = 0.0
    rate = cost_bps / 10_000
    for session in sorted({row["execution_session"] for row in events}):
        session_events = [row for row in events if row["execution_session"] == session]
        for row in session_events:
            decision = row["decision"]
            if decision is None or decision.action == "hold":
                continue
            qty = int(decision.qty)
            symbol, price = row["symbol"], row["open"]
            valid = 0 < qty <= max_qty
            if decision.action == "buy":
                notional = qty * price * (1 + rate)
                valid = valid and notional <= cash
                if valid:
                    cash -= notional
                    shares[symbol] += qty
            elif decision.action == "sell":
                valid = valid and qty <= shares[symbol]
                if valid:
                    cash += qty * price * (1 - rate)
                    shares[symbol] -= qty
            else:
                valid = False
            if valid:
                turnover = qty * price
                total_turnover += turnover
                trades.append({
                    "example_id": row["example_id"], "session": session,
                    "symbol": symbol, "action": decision.action, "qty": qty,
                    "price": price, "notional": turnover,
                })
            else:
                invalid_decisions += 1
        close_values = {
            symbol: float(bar_by_symbol_session[(symbol, session)]["close"])
            for symbol in shares
            if (symbol, session) in bar_by_symbol_session
        }
        equity = cash + sum(shares[symbol] * close_values.get(symbol, 0.0)
                            for symbol in shares)
        gross = sum(shares[symbol] * close_values.get(symbol, 0.0) for symbol in shares)
        curve.append({"session": session, "cash": cash, "equity": equity,
                      "gross_exposure": gross})

    equities = [float(row["equity"]) for row in curve]
    benchmark = [float(initial_cash)]
    symbols = sorted({row["symbol"] for row in events})
    if symbols and len(curve) > 1:
        first_session, last_session = curve[1]["session"], curve[-1]["session"]
        first_prices = [bar_by_symbol_session[(symbol, first_session)]["open"] for symbol in symbols
                        if (symbol, first_session) in bar_by_symbol_session]
        last_prices = [bar_by_symbol_session[(symbol, last_session)]["close"] for symbol in symbols
                       if (symbol, last_session) in bar_by_symbol_session]
        if first_prices and last_prices:
            benchmark.extend(
                [float(initial_cash)] * (len(curve) - 1)
            )
            benchmark[-1] = initial_cash * (
                sum(float(last) / float(first) for first, last in zip(first_prices, last_prices))
                / len(first_prices)
            )
    metrics = financial_metrics(equities, benchmark_equities=benchmark, turnover=total_turnover)
    metrics.update({
        "starting_cash": initial_cash, "ending_equity": equities[-1],
        "dollar_profit": equities[-1] - initial_cash,
        "percentage_return": equities[-1] / initial_cash - 1,
        "invalid_decisions": invalid_decisions, "executed_trades": len(trades),
        "sessions": len(curve) - 1, "examples": len(events),
    })
    return PortfolioReplayResult(metrics, tuple(curve), tuple(trades))


def write_result(result: PortfolioReplayResult, output: Path) -> None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "portfolio_metrics.json").write_text(
        canonical_json(result.metrics) + "\n", encoding="utf-8")
    (output / "equity_curve.jsonl").write_text(
        "".join(canonical_json(row) + "\n" for row in result.equity_curve), encoding="utf-8")
    (output / "trades.jsonl").write_text(
        "".join(canonical_json(row) + "\n" for row in result.trades), encoding="utf-8")
