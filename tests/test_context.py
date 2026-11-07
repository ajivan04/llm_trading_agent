import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent.context import build_context, project_evidence_record
from agent.contracts import Evidence, canonical_json
from agent.policy.evidence import resolve_decisions, visible_evidence
from agent.prepare_cli import archive_news_evidence, main, select_news_evidence
from tests.test_compact_artifact import prepared_source
from tests.test_preparation import bar

AS_OF = "2026-01-05T21:00:00Z"


def record(eid, source, text, *, when="2026-01-02T12:00:00Z", values=None):
    return Evidence(eid, source, "AAPL", when, when, "2026-02-01T00:00:00Z",
                    text, values or {})


def context(evidence=(), **changes):
    options = dict(
        state={"as_of": AS_OF, "cash": 100000, "positions": {}}, evidence=evidence,
        token_counter=lambda messages: len(canonical_json(messages)),
        max_context_tokens=20000, reserved_final_tokens=512,
    )
    options.update(changes)
    return build_context(**options)


class ContextTests(unittest.TestCase):
    def test_actual_news_and_sec_text_survive_without_duplicate_unbounded_values(self):
        news = record("article", "alpaca.news", "AI product launch improves revenue guidance.",
                      values={"title": "AI product launch", "summary": "x" * 20000})
        sec = record("filing", "sec.filing_text", "Liquidity and capital spending risks.",
                     values={"form": "10-Q", "excerpt": "y" * 20000})
        price = record("price", "yfinance", "daily price", values={"close": 100})
        result = context((news, sec, price), required_evidence_ids=("price",))
        user = json.loads(result.packed.messages[1]["content"])
        text = result.packed.messages[1]["content"]
        self.assertIn("AI product launch improves", text)
        self.assertIn("Liquidity and capital", text)
        self.assertNotIn("x" * 100, text)
        self.assertNotIn("y" * 100, text)
        self.assertEqual(result.source_status, {"news": "raw_text", "sec": "text"})
        self.assertNotIn("sentiment_score", text)
        self.assertEqual(len(news.values["summary"]), 20000)
        self.assertEqual(set(result.aliases), {item["id"] for item in user["evidence"]})

    def test_projection_is_identical_for_later_tool_results(self):
        item = record("filing", "sec.filing_text", "SEC text", values={"excerpt": "SEC text"})
        result = context((item,))
        expected = visible_evidence(result.original_messages)["filing"]
        actual = project_evidence_record(item, as_of=AS_OF)
        self.assertEqual(actual, expected)

    def test_short_reference_is_bound_to_full_selected_source(self):
        item = record("article:opaque-hash", "alpaca.news", "Earnings release")
        result = context((item,))
        raw = canonical_json({"decisions": [{
            "symbol": "AAPL", "action": "hold", "qty": 0,
            "reason": "uncertain", "evidence_ids": ["E1"],
        }]})
        decisions = resolve_decisions(raw, messages=result.packed.messages,
                                     allowed_symbols=("AAPL",), aliases=result.aliases,
                                     original_messages=result.original_messages)
        self.assertEqual(decisions[0].evidence_ids, (item.id,))

    def test_real_counter_budget_excludes_optional_evidence_without_truncating_json(self):
        price = record("price", "yfinance", "daily price", values={"close": 100})
        news = record("news", "alpaca.news", "Long useful news " + "word " * 1000)
        base = context((price,), required_evidence_ids=("price",))
        result = context((price, news), required_evidence_ids=("price",),
                         max_context_tokens=base.packed.input_tokens + 612)
        self.assertLessEqual(result.packed.input_tokens + 512, base.packed.input_tokens + 612)
        self.assertEqual(tuple(item.id for item in result.available_evidence), ("price",))
        self.assertEqual(result.packed.excluded_evidence_ids, ("news",))
        self.assertEqual(result.source_status["news"], "omitted_by_budget")
        self.assertIsInstance(json.loads(result.packed.messages[1]["content"]), dict)

    def test_future_records_required_ids_and_minimum_budget_fail_closed(self):
        future = record("future", "alpaca.news", "future", when="2026-01-06T12:00:00Z")
        with self.assertRaisesRegex(ValueError, "future evidence"):
            context((future,))
        with self.assertRaisesRegex(ValueError, "required.*unavailable"):
            context((), required_evidence_ids=("missing",))
        with self.assertRaisesRegex(ValueError, "exceed.*budget"):
            context((), max_context_tokens=600)

    def test_missing_news_and_memory_availability_are_explicit(self):
        result = context()
        self.assertEqual(result.source_status["news"], "missing")
        self.assertNotIn("sentiment", result.packed.messages[1]["content"])
        with self.assertRaisesRegex(ValueError, "historically available"):
            context(memories=({"available_at": "2026-01-06T12:00:00Z", "reflection": "future"},))
        with self.assertRaisesRegex(ValueError, "historically available"):
            context(memories=({"reflection": "unverified"},))

    def test_news_selection_is_bounded_and_all_raw_articles_are_archived(self):
        articles = tuple(record(f"article-{i}", "alpaca.news", f"News {i}",
                                when=f"2026-01-02T12:00:{i:02d}Z") for i in range(10))
        selected = select_news_evidence(articles, as_of=AS_OF, lookback_days=60, max_articles=2)
        self.assertEqual([item.id for item in selected], ["article-9", "article-8"])
        with tempfile.TemporaryDirectory() as directory:
            original = prepared_source(Path(directory) / "source")
            archived = archive_news_evidence(original, articles)
            self.assertTrue({item.id for item in articles} <=
                            {row["id"] for row in archived.evidence_rows})
            self.assertFalse({item.id for item in articles} &
                             {row["id"] for row in original.evidence_rows})

    def test_prepare_cli_passes_raw_news_to_policy_and_archives_omitted_articles(self):
        sessions = [("2026-01-02", 100, 101), ("2026-01-05", 102, 103),
                    ("2026-01-06", 104, 102), ("2026-01-07", 101, 100),
                    ("2026-01-08", 99, 103), ("2026-01-09", 104, 105)]
        bars = tuple(bar(day, f"{day}T14:30:00Z", f"{day}T21:00:00Z", opened, closed)
                     for day, opened, closed in sessions)
        next_session = {first.session: second.session
                        for first, second in zip(bars, bars[1:], strict=False)}
        articles = tuple(record(f"article-{i}", "alpaca.news", f"AI earnings headline {i}",
                                when=f"2026-01-02T12:00:{i:02d}Z",
                                values={"relevance_score": 1.0, "sentiment_score": None})
                         for i in range(10))
        calendar = SimpleNamespace(next=lambda session:
                                   SimpleNamespace(session=next_session[session]))
        adapter = SimpleNamespace(fetch=lambda *args, **kwargs: bars)
        news_adapter = SimpleNamespace(fetch=lambda *args, **kwargs: articles)
        output = io.StringIO()
        with (tempfile.TemporaryDirectory() as directory,
              patch("agent.prepare_cli.XNYSCalendar", return_value=calendar),
              patch("agent.prepare_cli.YFinancePriceAdapter", return_value=adapter),
              patch("agent.prepare_cli.AlpacaNewsAdapter", return_value=news_adapter),
              patch.dict(os.environ, {"ALPACA_API_KEY": "TEST-DO-NOT-LOG",
                                     "ALPACA_SECRET_KEY": "TEST-SECRET-DO-NOT-LOG"}),
              contextlib.redirect_stdout(output)):
            main(["--start", "2026-01-02", "--end", "2026-01-10", "--symbols", "AAPL",
                  "--train-end", "2026-01-05T23:59:59Z", "--validation-end",
                  "2026-01-07T23:59:59Z", "--test-end", "2026-01-09T23:59:59Z",
                  "--output", directory, "--evidence-mode", "news", "--news-provider", "alpaca",
                  "--max-news-articles", "2"])
            first_line = (Path(directory) / "dataset/examples.jsonl").read_text().splitlines()[0]
            first = json.loads(first_line)
            user = json.loads(first["messages"][1]["content"])
            policy_news = [item for item in user["evidence"] if item["source"] == "alpaca.news"]
            self.assertEqual(len(policy_news), 2)
            self.assertIn("AI earnings headline", policy_news[0]["text"])
            archived = [json.loads(line) for line in
                        (Path(directory) / "evidence.jsonl").read_text().splitlines()]
            self.assertEqual(sum(row["source"] == "alpaca.news" for row in archived), 10)
        self.assertNotIn("TEST-DO-NOT-LOG", output.getvalue())
        self.assertNotIn("TEST-SECRET-DO-NOT-LOG", output.getvalue())


if __name__ == "__main__":
    unittest.main()
