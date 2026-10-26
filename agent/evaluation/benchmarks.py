"""Comparable cash, buy-and-hold, basket, and momentum benchmark curves."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence


@dataclass(frozen=True)
class MarketSession:
    session: str
    opens: Mapping[str, float]
    closes: Mapping[str, float]

    def __post_init__(self) -> None:
        if not isinstance(self.session, str) or not self.session:
            raise ValueError("session must be nonempty")
        if set(self.opens) != set(self.closes) or not self.opens:
            raise ValueError("open/close symbols must match and be nonempty")
        for price in (*self.opens.values(), *self.closes.values()):
            if isinstance(price, bool) or not isinstance(price, (int, float)):
                raise ValueError("prices must be numeric")
            if not math.isfinite(float(price)) or price <= 0:
                raise ValueError("prices must be positive and finite")


@dataclass(frozen=True)
class BenchmarkCurve:
    name: str
    sessions: tuple[str, ...]
    equities: tuple[float, ...]
    turnover: float


Target = Callable[[int, Sequence[MarketSession]], Mapping[str, float] | None]


def _simulate(
    name: str, sessions: Sequence[MarketSession], initial_cash: float,
    cost_bps: float, target: Target,
) -> BenchmarkCurve:
    if not sessions or initial_cash <= 0 or not 0 <= cost_bps < 10_000:
        raise ValueError("invalid benchmark inputs")
    if tuple(item.session for item in sessions) != tuple(sorted(item.session for item in sessions)):
        raise ValueError("sessions must be chronological")
    if len({item.session for item in sessions}) != len(sessions):
        raise ValueError("sessions must be unique")
    universe = set(sessions[0].opens)
    if any(set(item.opens) != universe for item in sessions):
        raise ValueError("benchmark universe must be complete")
    cash, shares, turnover = float(initial_cash), {symbol: 0.0 for symbol in universe}, 0.0
    equities = [float(initial_cash)]
    rate = cost_bps / 10_000
    for index, item in enumerate(sessions):
        weights = target(index, sessions)
        if weights is not None:
            if set(weights) - universe or any(
                isinstance(weight, bool) or not isinstance(weight, (int, float))
                or not math.isfinite(float(weight)) or weight < 0
                for weight in weights.values()
            ):
                raise ValueError("invalid target weights")
            if sum(weights.values()) > 1 + 1e-12:
                raise ValueError("target weights cannot exceed one")
            equity_open = cash + sum(shares[s] * item.opens[s] for s in universe)
            current = {s: shares[s] * item.opens[s] for s in universe}
            desired = {s: equity_open * float(weights.get(s, 0.0)) for s in universe}
            sells = {s: max(0.0, current[s] - desired[s]) for s in universe}
            sold = sum(sells.values())
            cash += sold * (1 - rate)
            for symbol in universe:
                if sells[symbol]:
                    shares[symbol] -= sells[symbol] / item.opens[symbol]
            buys = {s: max(0.0, desired[s] - shares[s] * item.opens[s]) for s in universe}
            requested = sum(buys.values())
            scale = min(1.0, cash / (requested * (1 + rate))) if requested else 1.0
            bought = requested * scale
            for symbol in universe:
                amount = buys[symbol] * scale
                shares[symbol] += amount / item.opens[symbol]
            cash -= bought * (1 + rate)
            turnover += sold + bought
        equities.append(cash + sum(shares[s] * item.closes[s] for s in universe))
    return BenchmarkCurve(name, tuple(item.session for item in sessions), tuple(equities), turnover)


def benchmark_curves(
    sessions: Sequence[MarketSession], *, initial_cash: float,
    benchmark_symbol: str, momentum_lookback: int = 5, cost_bps: float = 0.0,
) -> dict[str, BenchmarkCurve]:
    if not sessions:
        raise ValueError("sessions must be nonempty")
    if type(momentum_lookback) is not int or momentum_lookback <= 0:
        raise ValueError("momentum_lookback must be positive")
    if benchmark_symbol not in sessions[0].opens:
        raise ValueError("benchmark symbol is absent")
    symbols = tuple(sorted(sessions[0].opens))

    def first_day(weights: Mapping[str, float]) -> Target:
        return lambda index, _sessions: weights if index == 0 else None

    equal = {symbol: 1 / len(symbols) for symbol in symbols}

    def momentum(index: int, history: Sequence[MarketSession]) -> Mapping[str, float]:
        # The trade at open[index] uses closes only through index-1.
        if index <= momentum_lookback:
            return {}
        returns = {
            symbol: history[index - 1].closes[symbol]
                    / history[index - 1 - momentum_lookback].closes[symbol] - 1
            for symbol in symbols
        }
        winner = max(symbols, key=lambda symbol: (returns[symbol], symbol))
        return {winner: 1.0} if returns[winner] > 0 else {}

    curves = {
        "cash": BenchmarkCurve(
            "cash", tuple(item.session for item in sessions),
            tuple([float(initial_cash)] * (len(sessions) + 1)), 0.0,
        ),
        "buy_and_hold": _simulate(
            "buy_and_hold", sessions, initial_cash, cost_bps,
            first_day({benchmark_symbol: 1.0}),
        ),
        "equal_weight": _simulate(
            "equal_weight", sessions, initial_cash, cost_bps, first_day(equal),
        ),
        "momentum": _simulate(
            "momentum", sessions, initial_cash, cost_bps, momentum,
        ),
    }
    return curves

