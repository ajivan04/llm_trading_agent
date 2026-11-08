"""Live prefetch is separate from the policy's later immutable cutoff."""

import json
import tempfile
import unittest
from dataclasses import asdict, replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

from agent.contracts import canonical_json
from agent.data.prices import DailyPriceBar
from agent.live_context import LiveResearchSeed
from tests.test_paper_execution import NOW


def bar(_symbol, day, opened, closed):
    return DailyPriceBar(
        "AAPL",
        day,
        opened,
        max(opened, closed) + 1,
        min(opened, closed) - 1,
        closed,
        closed,
        1000,
        0,
        0,
        day + "T14:30:00Z",
        day + "T21:00:00Z",
        NOW,
    )


class LiveContextTests(unittest.TestCase):
    def test_receipt_time_and_one_price_policy_are_preserved(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            old = [bar("AAPL", "2026-11-04", 100, 101), bar("AAPL", "2026-11-05", 101, 102)]
            (path / "price-bars.jsonl").write_text(
                "".join(canonical_json(asdict(b)) + "\n" for b in old)
            )
            (path / "evidence.jsonl").write_text("")
            values = dict(old[-1].to_evidence().values)
            values["adjustment_policy"] = {"basis": "raw live capture"}
            captured = NOW + timedelta(seconds=30)
            live = replace(
                old[-1].to_evidence(),
                id="live-current",
                available_at=captured,
                retrieved_at=captured,
                values=values,
            )

            class Tools:
                def __enter__(self):
                    return self

                def __exit__(self, *_):
                    pass

                def _bars(self, *_):
                    return (live,), False

                def search_market_news(self, **_kwargs):
                    return json.dumps(
                        {"status": "missing", "evidence": [], "candidate_symbols": []}
                    )

                def get_news_for_symbols(self, *_args, **_kwargs):
                    return json.dumps({"status": "missing", "evidence": []})

                def get_sec_filings(self, *_):
                    return json.dumps({"status": "missing", "evidence": []})

            source = LiveResearchSeed(
                path,
                cache_path=path / "cache",
                tools_factory=lambda *_a, **_k: Tools(),
                clock=lambda: NOW,
            )
            source.refresh(("AAPL",), SimpleNamespace(positions={}))
            before = source(NOW.isoformat(), ("AAPL",))
            self.assertNotIn("live-current", {r.id for r in before})
            after = source((NOW + timedelta(minutes=1)).isoformat(), ("AAPL",))
            self.assertEqual([r.id for r in after], ["live-current"])
            self.assertEqual(source.status["news"], "missing")

    def test_duplicate_price_session_is_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            line = canonical_json(asdict(bar("AAPL", "2026-11-05", 101, 102))) + "\n"
            (path / "price-bars.jsonl").write_text(line * 2)
            (path / "evidence.jsonl").write_text("")
            with self.assertRaisesRegex(ValueError, "duplicated"):
                LiveResearchSeed(path, cache_path=path / "cache", clock=lambda: NOW)

    def test_live_daily_bars_satisfy_corporate_action_feature_contract(self):
        from unittest.mock import patch

        from agent.features import PriceFeatureBuilder
        from agent.tools import AssignmentTools

        rows = [
            (
                NOW - timedelta(days=2),
                {
                    "Open": 100,
                    "High": 102,
                    "Low": 99,
                    "Close": 101,
                    "Volume": 1000,
                    "Dividends": 0,
                    "Stock Splits": 0,
                },
            ),
            (
                NOW - timedelta(days=1),
                {
                    "Open": 101,
                    "High": 103,
                    "Low": 100,
                    "Close": 102,
                    "Volume": 1200,
                    "Dividends": 0,
                    "Stock Splits": 0,
                },
            ),
        ]
        ticker = SimpleNamespace(history=lambda **_: SimpleNamespace(iterrows=lambda: iter(rows)))
        with patch.dict("sys.modules", {"yfinance": SimpleNamespace(Ticker=lambda _: ticker)}):
            with AssignmentTools(
                NOW.isoformat(),
                ("AAPL",),
                mode="live",
                clock=lambda: NOW,
                session_close=lambda d: d + "T21:00:00Z",
            ) as tools:
                records = tools._default_price_request("AAPL", "6mo")
        self.assertEqual(len(records), 2)
        features = PriceFeatureBuilder().build(records, symbol="AAPL", as_of=NOW.isoformat())
        self.assertEqual(features.symbol, "AAPL")
