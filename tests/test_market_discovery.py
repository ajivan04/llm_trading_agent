"""Market discovery adds read-only context, never whitelist or execution changes."""

import json
import unittest
from unittest.mock import Mock

from agent.contracts import Evidence
from agent.data.news import AlpacaNewsAdapter
from agent.policy import ToolRegistry
from agent.tools import AssignmentTools

CUTOFF = "2026-11-06T20:00:00Z"


def article(symbols, created="2026-11-05T12:00:00Z", updated="2026-11-05T12:00:00Z"):
    return {
        "id": 123,
        "headline": "AI &amp; chip demand rises",
        "summary": "MSFT and NVDA earnings",
        "source": "publisher",
        "url": "https://example.com/story",
        "symbols": symbols,
        "created_at": created,
        "updated_at": updated,
    }


class MarketDiscoveryTests(unittest.TestCase):
    def test_provider_market_request_omits_symbols_and_retains_mentions(self):
        request = Mock(
            return_value={
                "news": [article(["NVDA", "MSFT", "OUTSIDE"])],
                "next_page_token": "older",
            }
        )
        rows = AlpacaNewsAdapter("fake-key", "fake-secret", request).fetch_market(
            time_from="2026-11-01T00:00:00Z",
            time_to=CUTOFF,
            as_of=CUTOFF,
            retrieved_at=CUTOFF,
            latest_only=True,
        )
        self.assertNotIn("symbols", request.call_args.args[0])
        self.assertEqual(request.call_count, 1)
        self.assertIsNone(rows[0].symbol)
        self.assertEqual(rows[0].values["symbols"], ("MSFT", "NVDA", "OUTSIDE"))
        self.assertIn("AI & chip", rows[0].text)

    def test_live_cache_topic_and_whitelist_filter(self):
        rows = (
            Evidence(
                "market",
                "alpaca.news",
                None,
                "2026-11-05T12:00:00Z",
                "2026-11-05T12:00:00Z",
                CUTOFF,
                "AI chip earnings",
                {"symbols": ["NVDA", "MSFT", "OUTSIDE"]},
            ),
        )
        request = Mock(return_value=rows)
        with AssignmentTools(
            CUTOFF, ["NVDA", "MSFT"], mode="live", market_news_request=request, clock=lambda: CUTOFF
        ) as tools:
            first = json.loads(tools.search_market_news("AI"))
            second = json.loads(tools.search_market_news("AI"))
            self.assertEqual(first["candidate_symbols"], ["MSFT", "NVDA"])
            self.assertEqual(first["coverage"], "market-wide")
            self.assertTrue(second["cache_hit"])
            self.assertEqual(request.call_count, 1)
            self.assertEqual(json.loads(tools.search_market_news("oil"))["status"], "missing")
            self.assertEqual(tools.universe, ("NVDA", "MSFT"))

    def test_replay_deduplicates_articles_and_never_queries_live_or_future(self):
        def item(symbol, when="2026-11-05T12:00:00Z"):
            return Evidence(
                symbol + when,
                "alpaca.news",
                symbol,
                when,
                when,
                "2026-11-08T00:00:00Z",
                "AI chip earnings",
                raw_content_hash="a" * 64,
            )

        rows = [item("NVDA"), item("MSFT"), item("NVDA", "2026-11-07T12:00:00Z")]
        request = Mock(side_effect=AssertionError("live forbidden"))
        with AssignmentTools(
            CUTOFF, ["NVDA", "MSFT"], evidence=rows, market_news_request=request
        ) as tools:
            registry = ToolRegistry(CUTOFF)
            tools.register_tools(registry)
            result = registry.dispatch("search_market_news", {"topic": "AI"})
            self.assertEqual(result["coverage"], "archived-whitelist-news")
            self.assertEqual(len(result["evidence"]), 1)
            self.assertEqual(result["candidate_symbols"], ["MSFT", "NVDA"])
            request.assert_not_called()

    def test_later_article_versions_and_provider_errors_cannot_leak(self):
        request = Mock(return_value={"news": [article(["NVDA"], updated="2026-11-07T00:00:00Z")]})
        rows = AlpacaNewsAdapter("fake-key", "fake-secret", request).fetch_market(
            time_from="2026-11-01T00:00:00Z", time_to=CUTOFF, as_of=CUTOFF, retrieved_at=CUTOFF
        )
        self.assertEqual(rows, ())
        request = Mock(side_effect=RuntimeError("SECRET-PROVIDER-ECHO"))
        with AssignmentTools(CUTOFF, ["NVDA"], mode="live", market_news_request=request) as tools:
            result = tools.search_market_news()
            self.assertNotIn("SECRET", result)
            self.assertEqual(json.loads(result)["status"], "provider_error")
