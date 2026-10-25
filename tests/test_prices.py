import dataclasses
import unittest
from datetime import date, datetime, timezone
from unittest.mock import Mock

from agent.data.prices import DailyPriceBar, YFinancePriceAdapter


UTC = timezone.utc


def bounds(symbol, session):
    assert symbol == "AAPL"
    return (
        datetime(session.year, session.month, session.day, 14, 30, tzinfo=UTC),
        datetime(session.year, session.month, session.day, 21, 0, tzinfo=UTC),
    )


def row(session="2026-01-05", **overrides):
    values = {
        "session": session, "Symbol": "AAPL", "Open": 100, "High": 110,
        "Low": 95, "Close": 108, "Adj Close": 54, "Volume": 1_000,
        "Dividends": 0.25, "Stock Splits": 2.0,
    }
    values.update(overrides)
    return values


class DailyPriceBarTests(unittest.TestCase):
    def make_bar(self, **overrides):
        values = dict(
            symbol="aapl", session="2026-01-05", open=100, high=110, low=95,
            close=108, adjusted_close=54, volume=1000, dividends=.25,
            stock_splits=2, open_at="2026-01-05T14:30:00Z",
            close_at="2026-01-05T21:00:00Z",
            retrieved_at="2026-01-06T00:00:00Z",
        )
        values.update(overrides)
        return DailyPriceBar(**values)

    def test_preserves_raw_adjusted_and_actions_in_evidence(self):
        bar = self.make_bar()
        evidence = bar.to_evidence()
        self.assertEqual(bar.close, 108)
        self.assertEqual(bar.adjusted_close, 54)
        self.assertNotIn("adjusted_close", evidence.values)
        self.assertEqual(evidence.values["raw"]["close"], 108)
        self.assertEqual(evidence.values["dividends"], .25)
        self.assertEqual(evidence.values["stock_splits"], 2)
        self.assertFalse(evidence.values["adjustment_policy"]["download_auto_adjust"])
        self.assertEqual(evidence.source, "yfinance")
        revised = self.make_bar(close=107).to_evidence()
        self.assertNotEqual(evidence.id, revised.id)
        adjusted_only_revision = self.make_bar(adjusted_close=27).to_evidence()
        self.assertEqual(evidence.id, adjusted_only_revision.id)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            bar.close = 2

    def test_cutoff_eligibility_begins_at_verified_close(self):
        evidence = self.make_bar().to_evidence()
        self.assertFalse(evidence.eligible("2026-01-05T20:59:59Z"))
        self.assertTrue(evidence.eligible("2026-01-05T21:00:00Z"))
        self.assertEqual(evidence.available_at, self.make_bar().close_at)

    def test_rejects_bad_ohlc_and_actions(self):
        for change in ({"low": 105}, {"high": 105}, {"open": 0},
                       {"volume": -1}, {"volume": 1.5},
                       {"dividends": -1}, {"stock_splits": -1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.make_bar(**change)

    def test_rejects_naive_impossible_or_early_timestamps(self):
        for change in (
            {"open_at": datetime(2026, 1, 5, 9, 30)},
            {"open_at": "2026-01-05T22:00:00Z"},
            {"retrieved_at": "2026-01-05T20:59:59Z"},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.make_bar(**change)


class YFinancePriceAdapterTests(unittest.TestCase):
    def test_uses_explicit_safe_download_arguments(self):
        downloader = Mock(return_value=[row()])
        bars = YFinancePriceAdapter(bounds, downloader).fetch(
            "aapl", start="2026-01-01", end="2026-01-10",
            retrieved_at="2026-01-10T22:00:00Z",
        )
        downloader.assert_called_once_with(
            "AAPL", start="2026-01-01", end="2026-01-10",
            auto_adjust=False, actions=True, progress=False,
        )
        self.assertEqual(len(bars), 1)

    def test_requires_verified_session_bounds(self):
        adapter = YFinancePriceAdapter(lambda _symbol, _session: None, lambda *a, **k: [row()])
        with self.assertRaisesRegex(ValueError, "missing exchange session bounds"):
            adapter.fetch("AAPL", start="2026-01-01", end="2026-01-10",
                          retrieved_at="2026-01-10T22:00:00Z")

    def test_deduplicates_identical_rows_and_sorts_sessions(self):
        downloader = lambda *a, **k: [row("2026-01-06"), row(), row()]
        bars = YFinancePriceAdapter(bounds, downloader).fetch(
            "AAPL", start="2026-01-01", end="2026-01-10",
            retrieved_at="2026-01-10T22:00:00Z",
        )
        self.assertEqual([bar.session for bar in bars], ["2026-01-05", "2026-01-06"])

    def test_rejects_conflicting_duplicates_symbol_and_coverage(self):
        cases = (
            [row(), row(Close=107)],
            [row(Symbol="MSFT")],
            [row("2026-01-12")],
        )
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                YFinancePriceAdapter(bounds, lambda *a, _rows=rows, **k: _rows).fetch(
                    "AAPL", start=date(2026, 1, 1), end=date(2026, 1, 10),
                    retrieved_at="2026-01-10T22:00:00Z",
                )

    def test_validates_yfinance_multiindex_ticker_level(self):
        valid = [{
            ("Open", "AAPL"): 100, ("High", "AAPL"): 110,
            ("Low", "AAPL"): 95, ("Close", "AAPL"): 108,
            ("Adj Close", "AAPL"): 54, ("Volume", "AAPL"): 1_000,
            ("Dividends", "AAPL"): 0.0, ("Stock Splits", "AAPL"): 0.0,
            "session": "2026-01-05",
        }]
        bars = YFinancePriceAdapter(bounds, lambda *a, **k: valid).fetch(
            "AAPL", start="2026-01-01", end="2026-01-10",
            retrieved_at="2026-01-10T22:00:00Z",
        )
        self.assertEqual(len(bars), 1)

        invalid = [dict(valid[0])]
        invalid[0][("Open", "MSFT")] = invalid[0].pop(("Open", "AAPL"))
        with self.assertRaisesRegex(ValueError, "tuple columns"):
            YFinancePriceAdapter(bounds, lambda *a, **k: invalid).fetch(
                "AAPL", start="2026-01-01", end="2026-01-10",
                retrieved_at="2026-01-10T22:00:00Z",
            )


if __name__ == "__main__":
    unittest.main()
