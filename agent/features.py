"""Deterministic, point-in-time price features.

Only historically eligible yfinance bars for the requested symbol are consumed.
Raw OHLC values remain in their source Evidence; return-like features use the
adjusted close explicitly recorded by the adapter.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from .contracts import (
    Evidence, canonical_json, content_hash, normalize_timestamp, parse_timestamp,
)


@dataclass(frozen=True)
class FeatureParameters:
    """All feature windows are numbers of exchange sessions."""

    return_window: int = 5
    sma_window: int = 20
    rsi_window: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    volatility_window: int = 20
    volume_window: int = 20
    annualization_sessions: int = 252
    version: str = "price-features-v1"

    def __post_init__(self) -> None:
        integer_fields = (
            "return_window", "sma_window", "rsi_window", "macd_fast",
            "macd_slow", "volatility_window", "volume_window",
            "annualization_sessions",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.volatility_window < 2:
            raise ValueError("volatility_window must be at least 2")
        if self.macd_fast >= self.macd_slow:
            raise ValueError("macd_fast must be less than macd_slow")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("version must be nonempty")

    def to_dict(self) -> dict[str, int | str]:
        return {
            "return_window": self.return_window,
            "sma_window": self.sma_window,
            "rsi_window": self.rsi_window,
            "macd_fast": self.macd_fast,
            "macd_slow": self.macd_slow,
            "volatility_window": self.volatility_window,
            "volume_window": self.volume_window,
            "annualization_sessions": self.annualization_sessions,
            "version": self.version,
        }


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"price bar {name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"price bar {name} must be finite")
    return result


def _validate_bar(evidence: Evidence) -> tuple[str, float, float, float, float, str]:
    if evidence.source != "yfinance":
        raise ValueError("price evidence source must be yfinance")
    required = (
        "session", "raw", "volume", "dividends", "stock_splits",
        "adjustment_policy",
    )
    missing = [name for name in required if name not in evidence.values]
    if missing:
        raise ValueError(f"price bar is missing fields: {', '.join(missing)}")
    session = evidence.values["session"]
    raw = evidence.values["raw"]
    policy = evidence.values["adjustment_policy"]
    if not isinstance(session, str) or not session.strip():
        raise ValueError("price bar session must be nonempty")
    if not isinstance(raw, dict) or any(name not in raw for name in ("open", "high", "low", "close")):
        raise ValueError("price bar raw field must contain open/high/low/close")
    if not isinstance(policy, dict) or not policy:
        raise ValueError("adjustment_policy must be a nonempty mapping")
    for name in ("open", "high", "low", "close"):
        if _number(raw[name], name) <= 0:
            raise ValueError(f"price bar {name} must be positive")
    volume = _number(evidence.values["volume"], "volume")
    if volume < 0:
        raise ValueError("price bar volume must be nonnegative")
    dividend = _number(evidence.values["dividends"], "dividends")
    split = _number(evidence.values["stock_splits"], "stock_splits")
    if dividend < 0 or split < 0:
        raise ValueError("dividends and stock_splits must be nonnegative")
    return (
        session.strip(), _number(raw["close"], "close"), volume,
        dividend, split, canonical_json(policy),
    )


def _ema_latest(values: list[float], period: int) -> float | None:
    if len(values) < period:
        return None
    ema = sum(values[:period]) / period
    alpha = 2.0 / (period + 1.0)
    for value in values[period:]:
        ema = alpha * value + (1.0 - alpha) * ema
    return ema


def _rsi(values: list[float], window: int) -> float | None:
    if len(values) < window + 1:
        return None
    changes = [new - old for old, new in zip(values, values[1:])]
    gains = [max(change, 0.0) for change in changes]
    losses = [max(-change, 0.0) for change in changes]
    average_gain = sum(gains[:window]) / window
    average_loss = sum(losses[:window]) / window
    for gain, loss in zip(gains[window:], losses[window:]):
        average_gain = ((window - 1) * average_gain + gain) / window
        average_loss = ((window - 1) * average_loss + loss) / window
    if average_gain == 0.0 and average_loss == 0.0:
        return 50.0
    if average_loss == 0.0:
        return 100.0
    return 100.0 - (100.0 / (1.0 + average_gain / average_loss))


class PriceFeatureBuilder:
    def __init__(self, parameters: FeatureParameters | None = None) -> None:
        self.parameters = parameters or FeatureParameters()

    def build(
        self,
        evidence: Iterable[Evidence],
        *,
        symbol: str,
        as_of: str | datetime,
    ) -> Evidence:
        """Build one feature Evidence without accessing a bar after ``as_of``."""
        normalized_symbol = symbol.strip().upper()
        if not normalized_symbol:
            raise ValueError("symbol must be nonempty")
        cutoff = normalize_timestamp(as_of)
        selected = [
            item for item in evidence
            if item.symbol == normalized_symbol and item.eligible(cutoff)
        ]

        bars: list[tuple[Evidence, str, float, float, float, float, str]] = []
        seen_sessions: set[str] = set()
        policies: set[str] = set()
        for item in selected:
            session, raw_close, volume, dividend, split, policy_key = _validate_bar(item)
            if session in seen_sessions:
                raise ValueError(f"duplicate price bar for {normalized_symbol}/{session}")
            seen_sessions.add(session)
            policies.add(policy_key)
            bars.append((item, session, raw_close, volume, dividend, split, policy_key))
        if not bars:
            raise ValueError(f"no eligible yfinance bars for {normalized_symbol}")
        if len(policies) != 1:
            raise ValueError("price bars must use one adjustment_policy")
        bars.sort(key=lambda row: (
            parse_timestamp(row[0].available_at), row[1], row[0].id,
        ))

        # Build a point-in-time total-return series. Each step uses only the raw
        # closes and corporate action reported for the later, already-eligible bar.
        # Provider adjusted_close remains only on the transient adapter bar for
        # separate audit/archive use; current Yahoo adjustments can be revised
        # by future actions and are excluded from historically eligible Evidence.
        closes = [bars[0][2]]
        for previous, current in zip(bars, bars[1:]):
            split_factor = current[5] if current[5] > 0 else 1.0
            comparable_previous = previous[2] / split_factor
            gross_return = (current[2] + current[4]) / comparable_previous
            if not math.isfinite(gross_return) or gross_return <= 0:
                raise ValueError("corporate-action-adjusted gross return must be positive")
            closes.append(closes[-1] * gross_return)
        volumes = [row[3] for row in bars]
        p = self.parameters
        trailing_return = None
        if len(closes) >= p.return_window + 1:
            trailing_return = closes[-1] / closes[-1 - p.return_window] - 1.0
        sma_ratio = None
        if len(closes) >= p.sma_window:
            sma_ratio = closes[-1] / (sum(closes[-p.sma_window:]) / p.sma_window)
        fast = _ema_latest(closes, p.macd_fast)
        slow = _ema_latest(closes, p.macd_slow)
        macd = fast - slow if fast is not None and slow is not None else None
        realized_volatility = None
        if len(closes) >= p.volatility_window + 1:
            returns = [
                new / old - 1.0
                for old, new in zip(
                    closes[-p.volatility_window - 1:-1],
                    closes[-p.volatility_window:],
                )
            ]
            mean = sum(returns) / len(returns)
            variance = sum((value - mean) ** 2 for value in returns) / (len(returns) - 1)
            realized_volatility = math.sqrt(variance * p.annualization_sessions)
        relative_volume = None
        if len(volumes) >= p.volume_window + 1:
            previous_average = sum(volumes[-p.volume_window - 1:-1]) / p.volume_window
            relative_volume = volumes[-1] / previous_average if previous_average > 0 else None

        source_ids = tuple(row[0].id for row in bars)
        source_fingerprints = tuple(content_hash(row[0]) for row in bars)
        features = {
            "trailing_return": trailing_return,
            "sma_ratio": sma_ratio,
            "rsi": _rsi(closes, p.rsi_window),
            "macd": macd,
            "realized_volatility": realized_volatility,
            "relative_volume": relative_volume,
        }
        values = {
            "features": features,
            "feature_version": p.version,
            "windows": p.to_dict(),
            "source_evidence_ids": source_ids,
            "source_evidence_fingerprints": source_fingerprints,
            "adjustment_policy": bars[-1][0].values["adjustment_policy"],
            "price_basis": "point_in_time_total_return_from_raw_close_actions",
            "provider_adjusted_close_usage": "audit_only",
            "latest_session": bars[-1][1],
        }
        identity = {
            "symbol": normalized_symbol,
            "as_of": cutoff,
            "values": values,
        }
        latest_available = max(row[0].available_at for row in bars)
        latest_observed = max(row[0].observed_at for row in bars)
        latest_retrieved = max(row[0].retrieved_at for row in bars)
        return Evidence(
            id=f"feature-{content_hash(identity)}",
            source="derived.price_features",
            symbol=normalized_symbol,
            observed_at=latest_observed,
            available_at=latest_available,
            retrieved_at=latest_retrieved,
            text=f"Point-in-time {p.version} features for {normalized_symbol}",
            values=values,
            raw_content_hash=content_hash(identity),
            transformation_version=p.version,
        )


def build_price_features(
    evidence: Iterable[Evidence],
    *,
    symbol: str,
    as_of: str | datetime,
    parameters: FeatureParameters | None = None,
) -> Evidence:
    return PriceFeatureBuilder(parameters).build(evidence, symbol=symbol, as_of=as_of)
