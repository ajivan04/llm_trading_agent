"""Live research is bounded; historical preparation retains complete pagination."""

import unittest
from unittest.mock import Mock

from agent.data.news import AlpacaNewsAdapter
from agent.data.sec import SecDataAdapter
from tests.test_sec import recent_payload


class LiveProviderBoundsTests(unittest.TestCase):
    def test_latest_news_stops_at_declared_newest_page(self):
        request = Mock(return_value={"news": [], "next_page_token": "more-historical-pages"})
        result = AlpacaNewsAdapter("key", "secret", request).fetch(
            ("AAPL",),
            time_from="2026-11-01T00:00:00Z",
            time_to="2026-11-06T00:00:00Z",
            as_of="2026-11-06T00:00:00Z",
            retrieved_at="2026-11-06T01:00:00Z",
            latest_only=True,
        )
        self.assertEqual(result, ())
        request.assert_called_once()
        self.assertEqual(request.call_args.args[0]["sort"], "desc")

    def test_live_sec_skips_old_submission_shards(self):
        request = Mock(
            return_value={
                "cik": "0000320193",
                "filings": {
                    "recent": recent_payload(),
                    "files": [{"name": "old.json", "filingFrom": "2016-01-01"}],
                },
            }
        )
        adapter = SecDataAdapter("Research researcher@example.com", request)
        result = adapter.fetch_submissions(
            320193,
            symbol="AAPL",
            as_of="2026-11-06T00:00:00Z",
            retrieved_at="2026-11-06T01:00:00Z",
            include_historical=False,
        )
        self.assertTrue(result)
        request.assert_called_once()
