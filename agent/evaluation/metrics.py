"""Financial metrics and time-aware uncertainty helpers."""
from __future__ import annotations

import math
import random
import statistics
from typing import Mapping, Sequence


def _returns(equities: Sequence[float]) -> list[float]:
    if not equities or any(not math.isfinite(float(value)) or value <= 0 for value in equities):
        raise ValueError("equities must be positive and finite")
    return [new / old - 1 for old, new in zip(equities, equities[1:])]


def financial_metrics(
    equities: Sequence[float], *, benchmark_equities: Sequence[float] | None = None,
    turnover: float = 0.0, annual_risk_free_rate: float = 0.0,
    exposures: Sequence[Mapping[str, float]] | None = None,
) -> dict[str, float | int | None]:
    returns = _returns(equities)
    if (not math.isfinite(float(turnover)) or turnover < 0
            or not math.isfinite(float(annual_risk_free_rate))
            or annual_risk_free_rate <= -1):
        raise ValueError("turnover/risk-free inputs are invalid")
    if benchmark_equities is not None and len(benchmark_equities) != len(equities):
        raise ValueError("benchmark curve length must match")
    daily_rf = (1 + annual_risk_free_rate) ** (1 / 252) - 1
    excess_daily = [value - daily_rf for value in returns]
    volatility = statistics.stdev(excess_daily) if len(excess_daily) > 1 else 0.0
    downside = [min(value, 0.0) for value in excess_daily]
    downside_deviation = (
        math.sqrt(sum(value * value for value in downside) / len(downside)) * math.sqrt(252)
        if downside else None
    )
    peak, max_drawdown = float(equities[0]), 0.0
    for equity in equities:
        peak = max(peak, float(equity))
        max_drawdown = max(max_drawdown, 1 - float(equity) / peak)
    total_return = equities[-1] / equities[0] - 1
    benchmark_return = (
        benchmark_equities[-1] / benchmark_equities[0] - 1
        if benchmark_equities is not None else None
    )
    result: dict[str, float | int | None] = {
        "total_return": total_return,
        "excess_return": total_return - benchmark_return if benchmark_return is not None else None,
        "max_drawdown": max_drawdown,
        "annualized_sharpe": (
            statistics.mean(excess_daily) / volatility * math.sqrt(252)
            if volatility else None
        ),
        "annualized_downside_deviation": downside_deviation,
        "turnover_over_initial_equity": turnover / equities[0],
        "n_returns": len(returns),
        "annual_risk_free_rate": annual_risk_free_rate,
    }
    if exposures is not None:
        if len(exposures) != len(returns):
            raise ValueError("one exposure record is required per return")
        gross = [float(item["gross"]) for item in exposures]
        net = [float(item["net"]) for item in exposures]
        result["average_gross_exposure"] = statistics.mean(gross) if gross else None
        result["average_net_exposure"] = statistics.mean(net) if net else None
    return result


def block_bootstrap_return_ci(
    returns: Sequence[float], *, block_size: int, samples: int = 1000,
    seed: int = 0, confidence: float = 0.95,
) -> dict[str, float | int]:
    if not returns or any(not math.isfinite(float(value)) or value <= -1 for value in returns):
        raise ValueError("returns must be finite and greater than -1")
    if type(block_size) is not int or not 1 <= block_size <= len(returns):
        raise ValueError("invalid block_size")
    if type(samples) is not int or samples <= 0 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap settings")
    rng = random.Random(seed)
    estimates = []
    n = len(returns)
    for _ in range(samples):
        draw = []
        while len(draw) < n:
            start = rng.randrange(n)
            draw.extend(returns[(start + offset) % n] for offset in range(block_size))
        wealth = math.prod(1 + value for value in draw[:n]) - 1
        estimates.append(wealth)
    estimates.sort()
    alpha = (1 - confidence) / 2
    lower_index = max(0, min(samples - 1, int(alpha * samples)))
    upper_index = max(0, min(samples - 1, int((1 - alpha) * samples) - 1))
    return {
        "lower": estimates[lower_index],
        "median": estimates[samples // 2],
        "upper": estimates[upper_index],
        "samples": samples,
        "block_size": block_size,
        "seed": seed,
    }


def exposure_match_returns(
    returns: Sequence[float], gross_exposures: Sequence[float], *, target_gross: float,
) -> tuple[float, ...]:
    if len(returns) != len(gross_exposures) or target_gross < 0:
        raise ValueError("returns/exposures must align and target_gross must be nonnegative")
    matched = []
    for value, exposure in zip(returns, gross_exposures):
        if not math.isfinite(float(value)) or not math.isfinite(float(exposure)) or exposure < 0:
            raise ValueError("returns and exposures must be finite; exposures nonnegative")
        matched.append(0.0 if exposure == 0 else float(value) * target_gross / exposure)
    return tuple(matched)

