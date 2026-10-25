"""Daily price contracts and a historically safe yfinance adapter.

The adapter deliberately delegates exchange-session bounds to a caller-supplied
calendar resolver.  A date in a provider index is not enough information to
truthfully infer an exchange close, particularly around holidays and DST.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from agent.contracts import Evidence, content_hash, normalize_timestamp, parse_timestamp


SessionBoundsResolver = Callable[[str, date], tuple[datetime, datetime] | None]
Downloader = Callable[..., Any]


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result) or (positive and result <= 0):
        qualifier = "positive and finite" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _session(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value.strip())
        except ValueError as exc:
            raise ValueError("session must be an ISO-8601 date") from exc
    raise ValueError("session must be an ISO-8601 date")


@dataclass(frozen=True)
class DailyPriceBar:
    """An immutable raw daily bar whose availability is verified externally."""

    symbol: str
    session: str
    open: float
    high: float
    low: float
    close: float
    adjusted_close: float
    volume: int
    dividends: float
    stock_splits: float
    open_at: str | datetime
    close_at: str | datetime
    retrieved_at: str | datetime

    def __post_init__(self) -> None:
        symbol = self.symbol.strip().upper() if isinstance(self.symbol, str) else ""
        if not symbol:
            raise ValueError("symbol must be nonempty")
        object.__setattr__(self, "symbol", symbol)
        session = _session(self.session)
        object.__setattr__(self, "session", session.isoformat())

        for name in ("open", "high", "low", "close", "adjusted_close"):
            object.__setattr__(self, name, _number(getattr(self, name), name, positive=True))
        if self.low > min(self.open, self.close) or self.high < max(self.open, self.close):
            raise ValueError("OHLC values must lie within the low/high range")
        if self.low > self.high:
            raise ValueError("low cannot exceed high")

        volume = _number(self.volume, "volume")
        if volume < 0 or not volume.is_integer():
            raise ValueError("volume must be a nonnegative integer")
        object.__setattr__(self, "volume", int(volume))
        for name in ("dividends", "stock_splits"):
            value = _number(getattr(self, name), name)
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, value)

        for name in ("open_at", "close_at", "retrieved_at"):
            object.__setattr__(self, name, normalize_timestamp(getattr(self, name)))
        if parse_timestamp(self.open_at) >= parse_timestamp(self.close_at):
            raise ValueError("session open must precede session close")
        if parse_timestamp(self.close_at) > parse_timestamp(self.retrieved_at):
            raise ValueError("bar cannot be retrieved before session close")

    @property
    def available_at(self) -> str:
        """A completed daily bar first becomes available at verified close."""
        return self.close_at

    def to_evidence(self) -> Evidence:
        values = {
            "session": self.session,
            "raw": {
                "open": self.open,
                "high": self.high,
                "low": self.low,
                "close": self.close,
            },
            "volume": self.volume,
            "dividends": self.dividends,
            "stock_splits": self.stock_splits,
            "adjustment_policy": {
                "download_auto_adjust": False,
                "execution_prices": "raw_unadjusted",
                "feature_reference": "point_in_time_from_raw_close_and_actions",
                "provider_adjusted_close": "excluded_from_historical_evidence",
                "corporate_actions": "retained_separately",
            },
        }
        record_hash = content_hash(values)
        return Evidence(
            id=f"yfinance:{self.symbol}:{self.session}:{record_hash[:16]}",
            source="yfinance",
            symbol=self.symbol,
            observed_at=self.close_at,
            available_at=self.close_at,
            retrieved_at=self.retrieved_at,
            text=f"{self.symbol} completed daily price bar for {self.session}",
            values=values,
            units="USD; volume=shares",
            raw_content_hash=record_hash,
            transformation_version="yfinance-daily-raw-v1",
        )


class YFinancePriceAdapter:
    """Download raw daily bars without deriving unverified session timestamps."""

    def __init__(
        self,
        session_bounds_resolver: SessionBoundsResolver | None,
        downloader: Downloader | None = None,
    ) -> None:
        self._session_bounds_resolver = session_bounds_resolver
        self._downloader = downloader or self._download

    @staticmethod
    def _download(symbol: str, **kwargs: Any) -> Any:
        try:
            import yfinance as yf  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("yfinance is required for the default downloader") from exc
        return yf.download(symbol, **kwargs)

    def fetch(
        self,
        symbol: str,
        *,
        start: str | date,
        end: str | date,
        retrieved_at: str | datetime,
    ) -> tuple[DailyPriceBar, ...]:
        symbol = symbol.strip().upper() if isinstance(symbol, str) else ""
        if not symbol:
            raise ValueError("symbol must be nonempty")
        start_date, end_date = _session(start), _session(end)
        if start_date >= end_date:
            raise ValueError("start must precede end")
        retrieved = normalize_timestamp(retrieved_at)
        result = self._downloader(
            symbol,
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            auto_adjust=False,
            actions=True,
            progress=False,
        )

        bars: dict[date, DailyPriceBar] = {}
        for session, row in self._iter_rows(result, symbol):
            if not start_date <= session < end_date:
                raise ValueError("provider returned a session outside requested coverage")
            row_symbol = self._get(row, "Symbol", "symbol", required=False)
            if row_symbol is not None and str(row_symbol).strip().upper() != symbol:
                raise ValueError("provider returned data for an unexpected symbol")
            if self._session_bounds_resolver is None:
                raise ValueError("verified exchange session bounds are required")
            bounds = self._session_bounds_resolver(symbol, session)
            if bounds is None or len(bounds) != 2:
                raise ValueError(f"missing exchange session bounds for {session.isoformat()}")
            bar = DailyPriceBar(
                symbol=symbol,
                session=session.isoformat(),
                open=self._get(row, "Open"),
                high=self._get(row, "High"),
                low=self._get(row, "Low"),
                close=self._get(row, "Close"),
                adjusted_close=self._get(row, "Adj Close", "Adj_Close", "adjusted_close"),
                volume=self._get(row, "Volume"),
                dividends=self._get(row, "Dividends", required=False, default=0.0),
                stock_splits=self._get(row, "Stock Splits", "Stock_Splits", required=False, default=0.0),
                open_at=bounds[0],
                close_at=bounds[1],
                retrieved_at=retrieved,
            )
            previous = bars.get(session)
            if previous is not None and previous != bar:
                raise ValueError(f"conflicting duplicate price bar for {session.isoformat()}")
            bars[session] = bar
        return tuple(bars[key] for key in sorted(bars))

    @staticmethod
    def _get(
        row: Mapping[Any, Any],
        *names: str,
        required: bool = True,
        default: Any = None,
    ) -> Any:
        for name in names:
            if name in row:
                return row[name]
        if required:
            raise ValueError(f"provider row is missing {names[0]}")
        return default

    @classmethod
    def _iter_rows(
        cls, result: Any, expected_symbol: str,
    ) -> Iterable[tuple[date, Mapping[Any, Any]]]:
        if result is None:
            return
        if isinstance(result, Mapping):
            source = result.items()
        elif hasattr(result, "iterrows"):
            source = result.iterrows()
        else:
            source = enumerate(result)
        for index, raw_row in source:
            if not isinstance(raw_row, Mapping):
                if hasattr(raw_row, "to_dict"):
                    raw_row = raw_row.to_dict()
                else:
                    raise TypeError("provider rows must be mappings")
            row: dict[Any, Any] = {}
            tuple_tickers: set[str] = set()
            for key, value in raw_row.items():
                normalized_key = key
                if isinstance(key, tuple):
                    if not key:
                        raise ValueError("provider returned an empty tuple column")
                    normalized_key = key[0]
                    if len(key) > 1 and key[1] not in (None, ""):
                        tuple_tickers.add(str(key[1]).strip().upper())
                if normalized_key in row and row[normalized_key] != value:
                    raise ValueError(f"ambiguous provider column {normalized_key!r}")
                row[normalized_key] = value
            if tuple_tickers and tuple_tickers != {expected_symbol}:
                raise ValueError(
                    f"provider tuple columns do not identify only {expected_symbol}"
                )
            session_value = cls._get(row, "Date", "date", "session", required=False)
            if session_value is None:
                session_value = index
            yield _session(session_value), row
