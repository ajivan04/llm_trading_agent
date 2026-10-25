import tempfile
import unittest
from pathlib import Path

from agent.data.cache import SqliteResponseCache
from agent.data.news import (
    AlphaVantageNewsAdapter, CachedNewsAdapter, aggregate_news, canonicalize_url,
)


def article(
    *, ticker="AAPL", published="20260105T140000", url="https://Example.com/a?utm_source=x",
    score="0.4", relevance="0.8", title="Apple update",
):
    return {
        "title": title,
        "url": url,
        "time_published": published,
        "summary": "Company news",
        "source": "Example Wire",
        "topics": [{"topic": "Earnings", "relevance_score": "0.75"}],
        "ticker_sentiment": [{
            "ticker": ticker,
            "relevance_score": relevance,
            "ticker_sentiment_score": score,
            "ticker_sentiment_label": "Somewhat-Bullish",
        }],
    }


class AlphaVantageNewsTests(unittest.TestCase):
    def fetch(self, payloads, symbols=("AAPL",), **changes):
        calls = []

        def request(params):
            calls.append(dict(params))
            return payloads[params["tickers"]]

        kwargs = dict(
            time_from="2026-01-01T00:00:00Z",
            time_to="2026-01-06T00:00:00Z",
            as_of="2026-01-06T00:00:00Z",
            retrieved_at="2026-01-06T01:00:00Z",
            limit=50,
        )
        kwargs.update(changes)
        result = AlphaVantageNewsAdapter("secret-key", request).fetch(symbols, **kwargs)
        return result, calls

    def test_queries_each_ticker_independently(self):
        items, calls = self.fetch(
            {"AAPL": {"feed": [article()]}, "MSFT": {"feed": [article(ticker="MSFT")]}},
            symbols=("AAPL", "MSFT"),
        )
        self.assertEqual([call["tickers"] for call in calls], ["AAPL", "MSFT"])
        self.assertTrue(all("," not in call["tickers"] for call in calls))
        self.assertEqual({item.symbol for item in items}, {"AAPL", "MSFT"})

    def test_cutoff_deduplication_and_ordering(self):
        duplicate = article(url="https://example.com/a#fragment")
        later = article(published="20260105T150000", url="https://example.com/b")
        future = article(published="20260107T000000", url="https://example.com/future")
        items, _ = self.fetch({"AAPL": {"feed": [later, future, duplicate, article()]}})
        self.assertEqual(len(items), 2)
        self.assertEqual([item.url for item in items], ["https://example.com/a", "https://example.com/b"])
        self.assertLess(items[0].available_at, items[1].available_at)

        outside, _ = self.fetch(
            {"AAPL": {"feed": [article(published="20260106T120000")]}},
            time_to="2026-01-06T00:00:00Z", as_of="2026-01-07T00:00:00Z",
            retrieved_at="2026-01-07T01:00:00Z",
        )
        self.assertEqual(outside, ())

    def test_content_sensitive_identity_and_preserved_fields(self):
        items, _ = self.fetch({"AAPL": {"feed": [article()]}})
        changed, _ = self.fetch({"AAPL": {"feed": [article(title="Corrected title")]}})
        self.assertNotEqual(items[0].id, changed[0].id)
        self.assertEqual(items[0].values["publisher"], "Example Wire")
        self.assertEqual(items[0].values["topics"][0]["topic"], "Earnings")
        self.assertEqual(items[0].values["relevance_score"], 0.8)

    def test_conflicting_same_url_revision_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "conflicting news revisions"):
            self.fetch({"AAPL": {"feed": [
                article(title="Original"), article(title="Corrected"),
            ]}})

    def test_identical_content_at_different_urls_is_deduplicated(self):
        items, _ = self.fetch({"AAPL": {"feed": [
            article(url="https://example.com/one"),
            article(url="https://mirror.example.com/two"),
        ]}})
        self.assertEqual(len(items), 1)

    def test_same_url_at_distinct_publication_times_is_preserved(self):
        items, _ = self.fetch({"AAPL": {"feed": [
            article(url="https://example.com/update", published="20260105T140000"),
            article(url="https://example.com/update", published="20260105T150000"),
        ]}})
        self.assertEqual(len(items), 2)

    def test_splits_full_windows_and_rejects_unprovable_truncation(self):
        calls = []

        def request(params):
            calls.append(dict(params))
            start, end = params["time_from"], params["time_to"]
            if start == "20260101T0000" and end == "20260106T0000":
                return {"feed": [
                    article(published="20260102T120000", url="https://e.com/1"),
                    article(published="20260105T120000", url="https://e.com/2"),
                ]}
            if end <= "20260103T1200":
                return {"feed": [article(
                    published="20260102T120000", url="https://e.com/1")]}
            return {"feed": [article(
                published="20260105T120000", url="https://e.com/2")]}

        items = AlphaVantageNewsAdapter("key", request).fetch(
            ("AAPL",), time_from="2026-01-01T00:00:00Z",
            time_to="2026-01-06T00:00:00Z", as_of="2026-01-06T00:00:00Z",
            retrieved_at="2026-01-06T01:00:00Z", limit=2,
        )
        self.assertEqual(len(items), 2)
        self.assertGreater(len(calls), 1)

        with self.assertRaisesRegex(RuntimeError, "coverage is incomplete"):
            AlphaVantageNewsAdapter(
                "key", lambda _params: {"feed": [article()]}
            ).fetch(
                ("AAPL",), time_from="2026-01-01T00:00:00Z",
                time_to="2026-01-01T00:01:00Z", as_of="2026-01-01T00:01:00Z",
                retrieved_at="2026-01-06T01:00:00Z", limit=1,
            )

    def test_rejected_payload_never_echoes_secret_or_provider_message(self):
        message = "secret-key quota exhausted"
        for payload in ({"Note": message}, {"Error Message": message}, {"feed": "bad"}):
            with self.subTest(payload=payload), self.assertRaises(RuntimeError) as raised:
                self.fetch({"AAPL": payload})
            self.assertNotIn("secret-key", str(raised.exception))
            self.assertNotIn("quota", str(raised.exception))

    def test_rejects_bad_item_shapes_and_scores(self):
        bad = article(relevance="2")
        with self.assertRaises(ValueError):
            self.fetch({"AAPL": {"feed": [bad]}})
        with self.assertRaises(ValueError):
            self.fetch({"AAPL": {"feed": [article(published="bad")]}})

    def test_url_canonicalization_removes_tracking_and_sorts_query(self):
        self.assertEqual(
            canonicalize_url("HTTPS://Example.COM/a?z=2&utm_campaign=x&a=1#part"),
            "https://example.com/a?a=1&z=2",
        )


class NewsAggregationTests(unittest.TestCase):
    def test_weighted_aggregate_and_explicit_missing_state(self):
        adapter = AlphaVantageNewsAdapter(
            "key", lambda params: {"feed": [
                article(score="1", relevance="0.75", url="https://e.com/1"),
                article(score="-1", relevance="0.25", url="https://e.com/2",
                        published="20260105T150000"),
            ]},
        )
        items = adapter.fetch(
            ("AAPL",), time_from="2026-01-01T00:00:00Z",
            time_to="2026-01-06T00:00:00Z", as_of="2026-01-06T00:00:00Z",
            retrieved_at="2026-01-06T01:00:00Z",
        )
        result = aggregate_news(
            items, symbol="AAPL", as_of="2026-01-06T00:00:00Z",
            retrieved_at="2026-01-06T01:00:00Z",
        )
        self.assertFalse(result.values["missing"])
        self.assertEqual(result.values["article_count"], 2)
        self.assertAlmostEqual(result.values["weighted_sentiment"], 0.5)
        self.assertGreater(result.values["sentiment_disagreement"], 0)

        missing = aggregate_news(
            (), symbol="AAPL", as_of="2026-01-06T00:00:00Z",
            retrieved_at="2026-01-06T01:00:00Z",
        )
        self.assertTrue(missing.values["missing"])
        self.assertIsNone(missing.values["weighted_sentiment"])


class CachedNewsTests(unittest.TestCase):
    def test_reuses_snapshot_for_24_hours_then_refreshes(self):
        calls = []

        def request(_params):
            calls.append(1)
            return {"feed": [article()]}

        with tempfile.TemporaryDirectory() as directory:
            cache = SqliteResponseCache(Path(directory) / "cache.sqlite")
            try:
                adapter = CachedNewsAdapter(AlphaVantageNewsAdapter("key", request), cache)
                arguments = {
                    "symbols": ("AAPL",),
                    "time_from": "2026-01-01T00:00:00Z",
                    "time_to": "2026-01-06T00:00:00Z",
                    "as_of": "2026-01-06T00:00:00Z",
                }
                first = adapter.fetch(
                    **arguments, retrieved_at="2026-01-06T01:00:00Z")
                reused = adapter.fetch(
                    **arguments, retrieved_at="2026-01-07T00:59:59Z")
                refreshed = adapter.fetch(
                    **arguments, retrieved_at="2026-01-07T01:00:00Z")
                self.assertEqual(first, reused)
                self.assertNotEqual(first[0].retrieved_at, refreshed[0].retrieved_at)
                self.assertEqual(len(calls), 2)
            finally:
                cache.close()

    def test_moving_live_timestamps_cannot_bypass_cache(self):
        calls = []

        def request(_params):
            calls.append(1)
            return {"feed": [article()]}

        with tempfile.TemporaryDirectory() as directory:
            cache = SqliteResponseCache(Path(directory) / "cache.sqlite")
            try:
                adapter = CachedNewsAdapter(AlphaVantageNewsAdapter("key", request), cache)
                first = adapter.fetch(
                    ("AAPL",), time_from="2026-01-01T00:00:00Z",
                    time_to="2026-01-06T00:00:00Z", as_of="2026-01-06T00:00:00Z",
                    retrieved_at="2026-01-06T01:00:00Z")
                second = adapter.fetch(
                    ("AAPL",), time_from="2026-01-01T00:00:01Z",
                    time_to="2026-01-06T00:00:01Z", as_of="2026-01-06T00:00:01Z",
                    retrieved_at="2026-01-06T01:00:01Z")
                self.assertEqual(first, second)
                self.assertEqual(len(calls), 1)
            finally:
                cache.close()

    def test_out_of_order_absolute_windows_do_not_alias(self):
        calls = []

        def request(params):
            calls.append(dict(params))
            published = ("20260109T120000" if params["time_to"] >= "20260110T0000"
                         else "20260104T120000")
            return {"feed": [article(published=published)]}

        with tempfile.TemporaryDirectory() as directory:
            cache = SqliteResponseCache(Path(directory) / "cache.sqlite")
            try:
                adapter = CachedNewsAdapter(AlphaVantageNewsAdapter("key", request), cache)
                adapter.fetch(
                    ("AAPL",), time_from="2026-01-05T00:00:00Z",
                    time_to="2026-01-10T00:00:00Z", as_of="2026-01-10T00:00:00Z",
                    retrieved_at="2026-01-10T01:00:00Z")
                earlier = adapter.fetch(
                    ("AAPL",), time_from="2025-12-31T00:00:00Z",
                    time_to="2026-01-05T00:00:00Z", as_of="2026-01-05T00:00:00Z",
                    retrieved_at="2026-01-10T01:00:01Z")
                self.assertEqual(len(calls), 2)
                self.assertLessEqual(
                    earlier[0].available_at, "2026-01-05T00:00:00+00:00")
            finally:
                cache.close()


if __name__ == "__main__":
    unittest.main()
