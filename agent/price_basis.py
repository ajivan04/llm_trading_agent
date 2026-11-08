"""Restore Yahoo split-adjusted series to nominal historical share units.

Adjustment factors are private preprocessing metadata. Policies see only the
restored historical price and actions already effective at their own cutoff.
"""

import math
from dataclasses import replace
from datetime import timedelta

from agent.contracts import parse_timestamp


def restore_nominal(bars, tail_actions=()):
    bars = tuple(sorted(bars, key=lambda b: b.session))
    if not bars or all(b.price_basis == "nominal" for b in bars):
        return bars
    if any(b.price_basis != "yahoo-split-adjusted" for b in bars):
        raise ValueError("price basis must be consistent before normalization")
    # Detect a contradictory nominal series before applying any inverse factors.
    for previous, current in zip(bars, bars[1:], strict=False):
        ratio = current.stock_splits
        if ratio and (ratio >= 1.5 or ratio <= 2 / 3):
            gap = previous.close / current.open
            adjusted_error, nominal_error = abs(math.log(gap)), abs(math.log(gap / ratio))
            if nominal_error + math.log(1.2) < adjusted_error:
                raise ValueError("split calibration contradicts the declared Yahoo price basis")
    factor = math.prod(float(a["ratio"]) for a in tail_actions)
    restored = []
    for bar in reversed(bars):
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError("invalid cumulative split factor")
        restored.append(
            replace(
                bar,
                open=bar.open * factor,
                high=bar.high * factor,
                low=bar.low * factor,
                close=bar.close * factor,
                dividends=bar.dividends * factor,
                volume=int(round(bar.volume / factor)),
                price_basis="nominal",
            )
        )
        if bar.stock_splits:
            factor *= bar.stock_splits
    return tuple(reversed(restored))


def complete_tail_actions(bars_by_symbol, *, fetcher=None, calendar=None):
    """Fetch only the missing action tail through each original capture time."""
    from agent.trajectory_cli import _calendar

    calendar = calendar or _calendar()
    requirements = {}
    for symbol, bars in bars_by_symbol.items():
        if not bars or all(b.price_basis == "nominal" for b in bars):
            continue
        last = max(bars, key=lambda b: b.session)
        captured = max(parse_timestamp(b.retrieved_at) for b in bars)
        sessions = []
        candidate = calendar.next(last.session)
        while parse_timestamp(candidate.open_at) <= captured:
            sessions.append(candidate.session)
            candidate = calendar.next(candidate.session)
        if sessions:
            requirements[symbol] = (sessions, captured)
    if not requirements:
        return {s: () for s in bars_by_symbol}
    start = min(v[0][0] for v in requirements.values())
    end = (max(v[1] for v in requirements.values()).date() + timedelta(days=1)).isoformat()
    if fetcher is None:
        import yfinance

        from agent.data.prices import YFinancePriceAdapter

        frame = yfinance.download(
            list(requirements),
            start=start,
            end=end,
            actions=True,
            auto_adjust=False,
            progress=False,
        )
        returned = {}
        for symbol in requirements:
            # A batch download has separate Price/Ticker column levels. Select
            # the ticker before the single-symbol adapter flattens field names.
            # Otherwise AAPL's and MSFT's Adj Close collide in the same row.
            if frame is None:
                raise ValueError("provider omitted split-action frame")
            columns = frame.columns
            if columns.nlevels > 1:
                if columns.nlevels != 2 or list(columns.names).count("Ticker") != 1:
                    raise ValueError("provider action columns lack an unambiguous Ticker level")
                if symbol not in columns.get_level_values("Ticker"):
                    raise ValueError(f"provider omitted split-action ticker {symbol}")
                selected = frame.xs(symbol, axis=1, level="Ticker", drop_level=True)
            elif len(requirements) == 1:
                selected = frame
            else:
                raise ValueError("multi-symbol action download requires ticker-labeled columns")
            if not selected.columns.is_unique:
                raise ValueError("provider returned duplicate action columns")
            items = []
            for session, row in YFinancePriceAdapter._iter_rows(selected, symbol):
                if "Stock Splits" not in row:
                    raise ValueError("provider omitted action coverage")
                value = float(row["Stock Splits"])
                if not math.isfinite(value) or value < 0:
                    raise ValueError("provider action coverage is incomplete")
                items.append({"session": session.isoformat(), "ratio": value})
            returned[symbol] = items
    else:
        returned = fetcher(tuple(requirements), start, end)
    result = {s: () for s in bars_by_symbol}
    for symbol, (sessions, _) in requirements.items():
        rows = {}
        for row in returned.get(symbol, ()):
            if row["session"] in rows:
                raise ValueError("duplicate action-tail session")
            ratio = float(row["ratio"])
            if not math.isfinite(ratio) or ratio < 0:
                raise ValueError("invalid tail split action")
            rows[row["session"]] = {**row, "ratio": ratio}
        if set(sessions) - set(rows):
            raise ValueError(
                "missing split-action tail; cannot certify nominal price reconstruction"
            )
        result[symbol] = tuple(
            {"session": day, "ratio": float(rows[day]["ratio"])}
            for day in sessions
            if rows[day]["ratio"] not in (0, 1)
        )
        if any(not math.isfinite(a["ratio"]) or a["ratio"] <= 0 for a in result[symbol]):
            raise ValueError("invalid tail split action")
    return result


def normalize_bundle(bars_by_symbol, *, fetcher=None, calendar=None):
    tails = complete_tail_actions(bars_by_symbol, fetcher=fetcher, calendar=calendar)
    return {s: restore_nominal(rows, tails[s]) for s, rows in bars_by_symbol.items()}, tails


def split_adjusted_closes(evidence):
    """Indicator prices in cutoff-share units using only already-eligible actions."""
    by_session = {}
    for item in evidence:
        key = item.values.get("session", item.observed_at[:10])
        old = by_session.get(key)
        if old is None or parse_timestamp(item.retrieved_at) > parse_timestamp(old.retrieved_at):
            by_session[key] = item
    factor, result = 1.0, []
    for item in reversed(
        sorted(by_session.values(), key=lambda r: r.values.get("session", r.observed_at[:10]))
    ):
        result.append(float(item.values["raw"]["close"]) / factor)
        action = float(item.values.get("stock_splits", 0))
        if not math.isfinite(action) or action < 0:
            raise ValueError("invalid eligible split factor")
        if action:
            factor *= action
    return list(reversed(result))
