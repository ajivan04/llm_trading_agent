"""Research tools for the assignment; no tool can place an order.

Live AV values are timestamped at receipt rather than backdated from a daily
label. A newly received value after the immutable decision cutoff is cached for
the next snapshot and is not shown to the current policy.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from datetime import date as date_type
from html import unescape
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

from agent.contracts import (
    Evidence,
    canonical_json,
    content_hash,
    normalize_timestamp,
    parse_timestamp,
)
from agent.data.cache import CacheIdentity, SqliteResponseCache
from agent.policy import ToolRegistry, ToolSpec

_PERIODS = {"1d": 1, "5d": 5, "1mo": 23, "3mo": 66, "6mo": 132, "1y": 264}
_NEWS = {"alpaca.news", "alpha_vantage.news"}
_SENSITIVE = {"api_key", "apikey", "authorization", "secret", "token", "password"}


class _QuotaExceeded(RuntimeError):
    pass


def _number(value: Any, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError("expected numeric data")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("expected numeric data") from exc
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError("expected finite numeric data")
    return value


def _ema(values: Sequence[float], span: int) -> list[float]:
    result = [float(values[0])]
    alpha = 2 / (span + 1)
    for value in values[1:]:
        result.append(alpha * value + (1 - alpha) * result[-1])
    return result


class AssignmentTools:
    """Bound tool implementation with injectable transports and durable quotas.

    Injected av_request accepts public request parameters WITHOUT an API key.
    price_request(symbol, period) returns normalized Evidence records.
    news_request follows AlpacaNewsAdapter.fetch's keyword interface.
    sec_request(symbol, as_of) returns normalized eligible Evidence records.
    """

    def __init__(
        self,
        as_of: str,
        universe: Sequence[str],
        *,
        cache_path: Path | str = ":memory:",
        cache: SqliteResponseCache | None = None,
        mode: str = "historical",
        evidence: Sequence[Evidence] = (),
        av_request: Callable | None = None,
        price_request: Callable | None = None,
        news_request: Callable | None = None,
        sec_request: Callable | None = None,
        clock: Callable = lambda: datetime.now(UTC),
        av_api_key: str = "",
        alpaca_api_key: str = "",
        alpaca_secret_key: str = "",
        sec_user_agent: str = "",
        av_daily_limit: int = 25,
        news_lookback_days: int = 7,
        session_close: Callable[[str], str] | None = None,
    ) -> None:
        self._as_of = normalize_timestamp(as_of)
        if isinstance(universe, (str, bytes)) or not universe:
            raise ValueError("universe must contain unique symbols")
        self._universe = tuple(str(item).strip().upper() for item in universe)
        if any(not item for item in self._universe) or len(set(self._universe)) != len(
            self._universe
        ):
            raise ValueError("universe must contain unique symbols")
        if mode not in {"historical", "live"}:
            raise ValueError("mode must be historical or live")
        if type(av_daily_limit) is not int or not 1 <= av_daily_limit <= 25:
            raise ValueError("free Alpha Vantage daily budget must be 1..25")
        if type(news_lookback_days) is not int or news_lookback_days <= 0:
            raise ValueError("news lookback must be positive")
        if any(not isinstance(item, Evidence) for item in evidence):
            raise TypeError("tools require normalized Evidence records")
        self.mode, self.evidence, self.clock = mode, tuple(evidence), clock
        self._owns_cache = cache is None
        self.cache = cache or SqliteResponseCache(Path(cache_path))
        self.cache.db.execute(
            "CREATE TABLE IF NOT EXISTS assignment_av_requests (requested_at TEXT NOT NULL)"
        )
        self.cache.db.commit()
        self.av_daily_limit, self.news_lookback_days = av_daily_limit, news_lookback_days
        self._av_key, self._alpaca_key = av_api_key, alpaca_api_key
        self._alpaca_secret, self._sec_user_agent = alpaca_secret_key, sec_user_agent
        self._av_request = av_request or self._default_av_request
        self._price_request = price_request or self._default_price_request
        self._news_request = news_request or self._default_news_request
        self._sec_request = sec_request or self._default_sec_request
        self._session_close = session_close
        self._calendar = None
        self._secrets = tuple(
            value
            for value in (
                av_api_key,
                alpaca_api_key,
                alpaca_secret_key,
                sec_user_agent,
            )
            if value
        )

    @property
    def as_of(self) -> str:
        return self._as_of

    @property
    def universe(self) -> tuple[str, ...]:
        return self._universe

    @classmethod
    def from_environment(cls, as_of: str, universe: Sequence[str], **kwargs):
        """Read credentials only when explicitly constructing live tools."""
        return cls(
            as_of,
            universe,
            mode="live",
            av_api_key=os.environ.get("ALPHA_VANTAGE_API_KEY", ""),
            alpaca_api_key=os.environ.get("ALPACA_API_KEY", ""),
            alpaca_secret_key=os.environ.get("ALPACA_SECRET_KEY", ""),
            sec_user_agent=os.environ.get("SEC_USER_AGENT", ""),
            **kwargs,
        )

    def close(self) -> None:
        if self._owns_cache:
            self.cache.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def _now(self) -> datetime:
        return parse_timestamp(self.clock())

    def _redact(self, value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                str(key): self._redact(item)
                for key, item in value.items()
                if str(key).lower() not in _SENSITIVE
                and not str(key).lower().endswith(("_api_key", "_secret_key", "_token"))
            }
        if isinstance(value, (tuple, list)):
            return [self._redact(item) for item in value]
        if isinstance(value, str):
            for secret in self._secrets:
                value = value.replace(secret, "[REDACTED]")
        return value

    def _symbol(self, symbol: str) -> str:
        if not isinstance(symbol, str) or symbol.strip().upper() not in self._universe:
            raise ValueError("symbol is outside the allowed universe")
        return symbol.strip().upper()

    def _symbols(self, symbols: Sequence[str]) -> tuple[str, ...]:
        if isinstance(symbols, (str, bytes)) or not symbols or len(symbols) > 100:
            raise ValueError("symbols must be a bounded nonempty sequence")
        result = tuple(self._symbol(item) for item in symbols)
        if len(result) != len(set(result)):
            raise ValueError("symbols must be unique")
        return result

    def _output(self, source: str, items=(), *, cache_hit=False, status=None, **metadata) -> str:
        visible = [item for item in items if item.eligible(self.as_of)]
        pending = bool(items) and not visible
        rows = []
        for item in visible:
            # Provider provenance arrays are archive data, not model context.
            record = self._redact(item.to_dict())
            record["text"] = unescape(record["text"])[:1600]
            record["values"] = {
                key: value
                for key, value in record["values"].items()
                if key not in {"source_evidence_ids", "source_evidence_fingerprints"}
            }
            rows.append(record)
        return canonical_json(
            {
                "status": status
                or ("ok" if rows else "pending_snapshot" if pending else "missing"),
                "source": source,
                "as_of": self.as_of,
                "cache_hit": cache_hit,
                "cache_message": "Cache hit" if cache_hit else "Cache miss",
                "evidence": rows,
                **self._redact(metadata),
            }
        )

    def _cached(self, operation, symbols, params, fetch, *, ttl=timedelta(hours=24)):
        params = dict(params)
        if self.mode == "historical":
            params["historical_evidence_hash"] = content_hash(
                tuple(
                    self._redact(item.to_dict())
                    for item in self.evidence
                    if item.symbol in symbols and item.eligible(self.as_of)
                )
            )
        identity = CacheIdentity(
            source=f"assignment.{self.mode}.{operation}",
            symbols=tuple(symbols),
            params=params,
            snapshot_window=(
                self.as_of
                if self.mode == "historical"
                else f"live:{parse_timestamp(self.as_of).date().isoformat()}"
            ),
            transformation_version="assignment-tools-v2-live-actions",
        )
        cached = self.cache.get(identity, now=self._now())
        if cached is not None:
            records = tuple(Evidence(**item) for item in cached)
            if any(item.symbol not in symbols for item in records):
                raise ValueError("cache returned evidence outside requested symbols")
            return records, True
        records = tuple(fetch())
        if any(not isinstance(item, Evidence) for item in records):
            raise ValueError("provider must return normalized evidence")
        if any(item.symbol not in symbols for item in records):
            raise ValueError("provider returned evidence outside the requested universe")
        # Never persist raw provider error messages or credentials.
        payload = [self._redact(item.to_dict()) for item in records]
        self.cache.put(identity, payload, fetched_at=self._now(), ttl=ttl)
        return tuple(Evidence(**item) for item in payload), False

    def _reserve_av(self) -> bool:
        current = self._now()
        oldest = normalize_timestamp(current - timedelta(hours=24))
        connection = self.cache.db
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "DELETE FROM assignment_av_requests WHERE requested_at<=?", (oldest,)
            )
            used = connection.execute("SELECT COUNT(*) FROM assignment_av_requests").fetchone()[0]
            permitted = used < self.av_daily_limit
            if permitted:
                connection.execute(
                    "INSERT INTO assignment_av_requests VALUES (?)", (normalize_timestamp(current),)
                )
            connection.commit()
            return permitted
        except Exception:
            connection.rollback()
            raise

    def _historical(self, symbol: str, source=None):
        return tuple(
            sorted(
                (
                    item
                    for item in self.evidence
                    if item.symbol == symbol
                    and item.eligible(self.as_of)
                    and (source is None or item.source == source)
                ),
                key=lambda item: (item.available_at, item.id),
            )
        )

    def _bars(self, symbol: str, period="6mo"):
        if self.mode == "historical":
            return self._historical(symbol, "yfinance"), False
        return self._cached(
            "price_history",
            (symbol,),
            {"period": period},
            lambda: self._price_request(symbol, period),
        )

    def _close(self, session: str) -> str:
        if self._session_close is not None:
            return normalize_timestamp(self._session_close(session))
        if self._calendar is None:
            from agent.data.calendar import XNYSCalendar

            self._calendar = XNYSCalendar()
        return self._calendar.bounds(session).close_at

    def _derived(self, symbol, source, values, inputs, text):
        values = {
            **values,
            "input_count": len(inputs),
            "input_evidence_hash": content_hash(tuple(item.id for item in inputs)),
        }
        return Evidence(
            id=f"{source}:{symbol}:{content_hash(values)[:20]}",
            source=source,
            symbol=symbol,
            observed_at=max((item.observed_at for item in inputs), key=parse_timestamp),
            available_at=max((item.available_at for item in inputs), key=parse_timestamp),
            retrieved_at=max((item.retrieved_at for item in inputs), key=parse_timestamp),
            text=text,
            values=values,
            transformation_version="assignment-tools-v1",
        )

    def get_news_for_symbols(self, symbols, limit=5, date=None) -> str:
        normalized = self._symbols(symbols)
        if type(limit) is not int or not 1 <= limit <= 50:
            raise ValueError("news limit must be 1..50")
        cutoff = parse_timestamp(self.as_of)
        if date is not None:
            parsed = date_type.fromisoformat(date)
            day = datetime(parsed.year, parsed.month, parsed.day, tzinfo=UTC)
            if day.date() > cutoff.date():
                raise ValueError("news date cannot advance the decision cutoff")
            start, end = day, min(cutoff, day + timedelta(days=1) - timedelta(microseconds=1))
        else:
            start, end = cutoff - timedelta(days=self.news_lookback_days), cutoff
        if start >= end:
            return self._output("alpaca.news")

        def fetch():
            if self.mode == "historical":
                return tuple(
                    item
                    for item in self.evidence
                    if item.source in _NEWS
                    and item.symbol in normalized
                    and item.eligible(self.as_of)
                    and start <= parse_timestamp(item.available_at) <= end
                )
            return self._news_request(
                normalized,
                time_from=start,
                time_to=end,
                as_of=cutoff,
                retrieved_at=self._now(),
                limit=limit,
            )

        try:
            records, hit = self._cached(
                "news",
                normalized,
                {"date": date, "lookback_days": self.news_lookback_days, "limit": limit},
                fetch,
            )
            records = sorted(records, key=lambda item: (item.available_at, item.id), reverse=True)
            return self._output(
                "alpaca.news",
                records[:limit],
                cache_hit=hit,
                sentiment="not supplied by Alpaca; infer only from article text",
                coverage=("bounded latest provider page" if self.mode == "live"
                          else "eligible archive records, output limited to requested count"),
            )
        except Exception:
            return self._output("alpaca.news", status="provider_error", error="news_unavailable")

    def get_price_history(self, symbol: str, period="5d") -> str:
        symbol = self._symbol(symbol)
        if not isinstance(period, str) or period not in _PERIODS:
            raise ValueError("unsupported bounded price period")
        try:
            rows, hit = self._bars(symbol, period)
            return self._output(
                "yfinance",
                rows[-_PERIODS[period] :],
                cache_hit=hit,
                price_basis="raw completed daily bars",
            )
        except Exception:
            return self._output("yfinance", status="provider_error", error="prices_unavailable")

    def get_macd(self, symbol: str, interval="daily") -> str:
        symbol = self._symbol(symbol)
        if interval != "daily":
            raise ValueError("only daily MACD is supported")
        try:
            bars, hit = self._bars(symbol)
            bars = tuple(item for item in bars if item.eligible(self.as_of))
            if len(bars) < 35:
                return self._output(
                    "derived.local_macd", status="missing", error="insufficient_history"
                )
            closes = [_number(item.values["raw"]["close"], positive=True) for item in bars]
            fast, slow = _ema(closes, 12), _ema(closes, 26)
            macd = [a - b for a, b in zip(fast, slow, strict=True)]
            signal = _ema(macd, 9)[-1]
            record = self._derived(
                symbol,
                "derived.local_macd",
                {
                    "macd": macd[-1],
                    "signal": signal,
                    "histogram": macd[-1] - signal,
                    "method": "EMA12-EMA26; signal EMA9; adjust=False",
                },
                bars,
                f"{symbol} daily MACD computed from eligible raw closing prices",
            )
            return self._output("derived.local_macd", (record,), cache_hit=hit)
        except Exception:
            return self._output(
                "derived.local_macd", status="provider_error", error="macd_unavailable"
            )

    def get_rsi(self, symbol: str, interval="daily", time_period=14) -> str:
        symbol = self._symbol(symbol)
        if interval != "daily" or type(time_period) is not int or not 2 <= time_period <= 100:
            raise ValueError("RSI requires daily interval and time_period 2..100")
        if self.mode == "live":
            return self._av(
                "RSI",
                symbol,
                {"interval": interval, "time_period": str(time_period), "series_type": "close"},
            )
        bars = self._historical(symbol, "yfinance")
        if len(bars) <= time_period:
            return self._output("derived.local_rsi", error="insufficient_history")
        closes = [_number(item.values["raw"]["close"], positive=True) for item in bars]
        changes = [b - a for a, b in zip(closes, closes[1:], strict=False)]
        gain = sum(max(value, 0) for value in changes[:time_period]) / time_period
        loss = sum(max(-value, 0) for value in changes[:time_period]) / time_period
        for value in changes[time_period:]:
            gain = (gain * (time_period - 1) + max(value, 0)) / time_period
            loss = (loss * (time_period - 1) + max(-value, 0)) / time_period
        value = 50.0 if gain == loss == 0 else 100.0 if loss == 0 else 100 - 100 / (1 + gain / loss)
        record = self._derived(
            symbol,
            "derived.local_rsi",
            {
                "rsi": value,
                "time_period": time_period,
                "method": "Wilder smoothing",
                "signal": "overbought" if value >= 70 else "oversold" if value <= 30 else "neutral",
                "historical_substitution": "local calculation; not an AV historical response",
            },
            bars,
            f"{symbol} historical RSI calculated from eligible closes",
        )
        return self._output("derived.local_rsi", (record,))

    def get_global_quote(self, symbol: str) -> str:
        symbol = self._symbol(symbol)
        if self.mode == "live":
            return self._av("GLOBAL_QUOTE", symbol, {})
        bars = self._historical(symbol, "yfinance")
        if not bars:
            return self._output("derived.historical_quote")
        latest = bars[-1]
        price = _number(latest.values["raw"]["close"], positive=True)
        previous = (
            _number(bars[-2].values["raw"]["close"], positive=True) if len(bars) > 1 else None
        )
        record = self._derived(
            symbol,
            "derived.historical_quote",
            {
                "price": price,
                "volume": latest.values.get("volume"),
                "change": None if previous is None else price - previous,
                "historical_substitution": "completed yfinance close; not live GLOBAL_QUOTE",
                "realtime": False,
            },
            bars[-2:],
            f"{symbol} historical completed-close quote",
        )
        return self._output("derived.historical_quote", (record,), realtime=False)

    def _av(self, function, symbol, params):
        source = f"alpha_vantage.{function.lower()}"

        def fetch():
            if not self._reserve_av():
                raise _QuotaExceeded
            payload = self._av_request({"function": function, "symbol": symbol, **params})
            if not isinstance(payload, Mapping) or any(
                key in payload for key in ("Note", "Information", "Error Message")
            ):
                raise ValueError("provider rejected request")
            received = self._now()
            if function == "RSI":
                table = payload.get("Technical Analysis: RSI")
                if not isinstance(table, Mapping) or not table:
                    return ()
                session = max(table)
                value = _number(table[session]["RSI"])
                if not 0 <= value <= 100:
                    raise ValueError("invalid RSI")
                values = {
                    "rsi": value,
                    "time_period": int(params["time_period"]),
                    "signal": "overbought"
                    if value >= 70
                    else "oversold"
                    if value <= 30
                    else "neutral",
                }
            else:
                quote = payload.get("Global Quote")
                if not isinstance(quote, Mapping) or not quote:
                    return ()
                if quote.get("01. symbol", symbol).upper() != symbol:
                    raise ValueError("quote symbol mismatch")
                session = quote["07. latest trading day"]
                values = {
                    "price": _number(quote["05. price"], positive=True),
                    "change": _number(quote["09. change"]),
                    "volume": _number(quote["06. volume"]),
                    "realtime": False,
                }
                if values["volume"] < 0:
                    raise ValueError("invalid quote volume")
            observed = self._close(session)
            if parse_timestamp(observed) > received:
                raise ValueError("daily value is not completed")
            values.update(
                {
                    "provider_session": session,
                    "received_at": normalize_timestamp(received),
                    "historical_replay_safe": False,
                    "entitlement": "free_daily",
                }
            )
            return (
                Evidence(
                    id=f"{source}:{symbol}:{content_hash(values)[:20]}",
                    source=source,
                    symbol=symbol,
                    observed_at=observed,
                    available_at=received,
                    retrieved_at=received,
                    text=(
                        f"{symbol} Alpha Vantage {function}; free daily data, "
                        "not a realtime execution price"
                    ),
                    values=values,
                    transformation_version="assignment-tools-v1",
                ),
            )

        try:
            records, hit = self._cached(function.lower(), (symbol,), params, fetch)
            return self._output(
                source, records, cache_hit=hit, realtime=False, historical_replay_safe=False
            )
        except _QuotaExceeded:
            return self._output(source, status="quota_exhausted", realtime=False)
        except Exception:
            return self._output(source, status="provider_error", error="provider_unavailable")

    def get_sec_filings(self, symbol: str) -> str:
        symbol = self._symbol(symbol)
        try:
            if self.mode == "historical":
                records = tuple(
                    item for item in self._historical(symbol) if item.source.startswith("sec.")
                )
                return self._output("sec", records[-3:])
            records, hit = self._cached(
                "sec", (symbol,), {}, lambda: self._sec_request(symbol, self.as_of)
            )
            return self._output("sec", records[-3:], cache_hit=hit)
        except Exception:
            return self._output("sec", status="provider_error", error="sec_unavailable")

    def get_extra_context(self, symbols, max_symbols=3) -> str:
        symbols = self._symbols(symbols)
        if type(max_symbols) is not int or not 1 <= max_symbols <= 10:
            raise ValueError("context candidate budget must be 1..10")
        selected = symbols[:max_symbols]
        results = []
        for symbol in selected:
            results.extend(
                json.loads(function(symbol))
                for function in (
                    self.get_global_quote,
                    self.get_rsi,
                    self.get_macd,
                )
            )
        records = {item["id"]: item for result in results for item in result["evidence"]}
        return canonical_json(
            {
                "status": "ok" if records else "missing",
                "source": "derived.extra_context",
                "as_of": self.as_of,
                "cache_hit": bool(results) and all(item["cache_hit"] for item in results),
                "evidence": list(records.values()),
                "candidate_symbols": selected,
                "excluded_deep_research_symbols": symbols[max_symbols:],
                "results": [
                    {key: value for key, value in item.items() if key != "evidence"}
                    for item in results
                ],
            }
        )

    def register_tools(self, registry: ToolRegistry) -> None:
        if registry.as_of != self.as_of:
            raise ValueError("tool registry cutoff differs from tools cutoff")
        specifications = (
            (
                "get_news_for_symbols",
                "Alpaca news text; no provider sentiment scores",
                {"symbols": "array", "limit": "integer", "date": "string"},
                ("symbols",),
            ),
            (
                "get_price_history",
                "Eligible completed yfinance OHLCV",
                {"symbol": "string", "period": "string"},
                ("symbol",),
            ),
            (
                "get_rsi",
                "Daily RSI; Alpha Vantage live, labeled local calculation in replay",
                {"symbol": "string", "interval": "string", "time_period": "integer"},
                ("symbol",),
            ),
            (
                "get_macd",
                "MACD, EMA9 signal and histogram from eligible closes",
                {"symbol": "string", "interval": "string"},
                ("symbol",),
            ),
            (
                "get_global_quote",
                "Free daily AV quote, not realtime execution pricing",
                {"symbol": "string"},
                ("symbol",),
            ),
            (
                "get_sec_filings",
                "Eligible SEC filing text and metadata",
                {"symbol": "string"},
                ("symbol",),
            ),
        )
        for name, description, properties, required in specifications:
            function = getattr(self, name)
            registry.register(
                ToolSpec(name, description, properties, required),
                lambda arguments, cutoff, fn=function: json.loads(fn(**arguments)),
            )

    @property
    def schemas(self):
        registry = ToolRegistry(self.as_of)
        self.register_tools(registry)
        return registry.schemas

    def _default_av_request(self, params):
        if not self._av_key:
            raise RuntimeError("Alpha Vantage credentials unavailable")
        url = "https://www.alphavantage.co/query?" + urlencode({**params, "apikey": self._av_key})
        with urlopen(url, timeout=30) as response:  # noqa: S310 - fixed HTTPS host
            return json.loads(response.read().decode("utf-8"))

    def _default_news_request(self, symbols, **kwargs):
        from agent.data.news import AlpacaNewsAdapter

        return AlpacaNewsAdapter(self._alpaca_key, self._alpaca_secret).fetch(
            symbols, latest_only=True, **kwargs)

    def _default_price_request(self, symbol, period):
        import yfinance

        frame = yfinance.Ticker(symbol).history(period=period, auto_adjust=False, actions=True)
        received = self._now()
        records = []
        for index, row in frame.iterrows():
            session = index.date().isoformat()
            close_at = self._close(session)
            if parse_timestamp(close_at) > received:
                continue  # Today's incomplete daily candle is not a historical bar.
            values = {
                "session": session,
                "raw": {
                    name: _number(row[column], positive=True)
                    for name, column in (
                        ("open", "Open"),
                        ("high", "High"),
                        ("low", "Low"),
                        ("close", "Close"),
                    )
                },
                "volume": _number(row["Volume"]),
                "price_basis": "raw",
                "dividends": _number(row.get("Dividends", 0)),
                "stock_splits": _number(row.get("Stock Splits", 0)),
                "adjustment_policy": {"auto_adjust": False,
                                      "basis": "raw prices with separately recorded actions"},
                "historical_replay_safe": False,
            }
            records.append(
                Evidence(
                    id=f"yfinance-live:{symbol}:{session}:{content_hash(values)[:16]}",
                    source="yfinance",
                    symbol=symbol,
                    observed_at=close_at,
                    available_at=received,
                    retrieved_at=received,
                    text=f"{symbol} completed daily OHLCV {session}; captured live",
                    values=values,
                    transformation_version="assignment-live-bars-v1",
                )
            )
        return tuple(records)

    def _default_sec_request(self, symbol, cutoff):
        from agent.data.sec import SecDataAdapter
        from agent.prepare_cli import _sec_request, _sec_text_request

        adapter = SecDataAdapter(
            self._sec_user_agent,
            lambda url, headers: _sec_request(url, headers=headers),
            text_request=lambda url, headers: _sec_text_request(url, headers=headers),
        )
        cik = adapter.fetch_ticker_map().get(symbol)
        if cik is None:
            return ()
        filings = adapter.fetch_submissions(
            cik, symbol=symbol, as_of=cutoff, retrieved_at=self._now(), include_historical=False
        )
        selected = filings[-2:]
        return (
            *selected,
            *(
                adapter.fetch_filing_text(item, as_of=cutoff, retrieved_at=self._now())
                for item in selected
            ),
        )
