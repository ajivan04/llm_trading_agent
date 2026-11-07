import json
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from agent.contracts import Evidence
from agent.policy import ToolRegistry
from agent.policy.evidence import visible_evidence
from agent.tools import AssignmentTools


def evidence(eid, source, when, text="record", values=None, symbol="AAPL"):
    return Evidence(eid, source, symbol, when, when, "2026-11-07T00:00:00Z", text, values or {})


def bars(n=45):
    return tuple(
        evidence(
            f"bar-{index}",
            "yfinance",
            (datetime(2026, 1, 1, 21, tzinfo=UTC) + timedelta(days=index)).isoformat(),
            values={
                "raw": {
                    "open": 100 + index,
                    "high": 101 + index,
                    "low": 99 + index,
                    "close": 100 + index,
                },
                "volume": 1000,
            },
        )
        for index in range(n)
    )


def av_payload(params):
    if params["function"] == "RSI":
        return {"Technical Analysis: RSI": {"2026-11-05": {"RSI": "72.5"}}}
    return {
        "Global Quote": {
            "01. symbol": params["symbol"],
            "05. price": "150.25",
            "06. volume": "1000",
            "07. latest trading day": "2026-11-05",
            "09. change": "1.25",
        }
    }


class AssignmentToolTests(unittest.TestCase):
    def test_historical_cache_is_bound_to_evidence_content(self):
        cutoff = "2026-11-06T22:00:00Z"
        first = evidence("first", "alpaca.news", "2026-11-05T20:00:00Z", "first headline")
        second = evidence("second", "alpaca.news", "2026-11-05T20:00:00Z", "second headline")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.sqlite"
            with AssignmentTools(cutoff, ["AAPL"], cache_path=path, evidence=[first]) as tools:
                self.assertFalse(json.loads(tools.get_news_for_symbols(["AAPL"]))["cache_hit"])
            with AssignmentTools(cutoff, ["AAPL"], cache_path=path, evidence=[second]) as tools:
                result = json.loads(tools.get_news_for_symbols(["AAPL"]))
                self.assertFalse(result["cache_hit"])
                self.assertEqual(result["evidence"][0]["id"], "second")

    def test_extra_context_exposes_evidence_without_duplicate_nested_records(self):
        with AssignmentTools("2026-02-20T22:00:00Z", ["AAPL", "MSFT"], evidence=bars()) as tools:
            result = json.loads(tools.get_extra_context(["AAPL", "MSFT"], max_symbols=1))
            self.assertEqual(result["candidate_symbols"], ["AAPL"])
            self.assertEqual(result["excluded_deep_research_symbols"], ["MSFT"])
            self.assertEqual(len(visible_evidence([{"role": "user", "content": result}])), 3)
            self.assertTrue(all("evidence" not in item for item in result["results"]))

    def test_quota_does_not_reset_at_an_undocumented_midnight_boundary(self):
        calls = []

        def request(params):
            calls.append(params)
            return av_payload(params)

        with tempfile.TemporaryDirectory() as directory:
            options = dict(
                cache_path=Path(directory) / "cache.sqlite",
                mode="live",
                av_request=request,
                av_daily_limit=1,
                session_close=lambda day: day + "T21:00:00Z",
            )
            with AssignmentTools(
                "2026-11-06T23:59:59Z",
                ["AAPL", "MSFT"],
                clock=lambda: "2026-11-06T23:59:59Z",
                **options,
            ) as tools:
                self.assertEqual(json.loads(tools.get_rsi("AAPL"))["status"], "ok")
            with AssignmentTools(
                "2026-11-07T00:00:01Z",
                ["AAPL", "MSFT"],
                clock=lambda: "2026-11-07T00:00:01Z",
                **options,
            ) as tools:
                self.assertEqual(json.loads(tools.get_rsi("MSFT"))["status"], "quota_exhausted")
            self.assertEqual(len(calls), 1)

    def test_historical_tools_never_call_live_transports_and_exclude_future(self):
        def forbidden(*_args, **_kwargs):
            self.fail("historical mode called a live transport")

        rows = (
            *bars(),
            evidence("article", "alpaca.news", "2026-02-14T20:00:00Z", "AI &amp; earnings"),
            evidence("future", "alpaca.news", "2026-02-16T20:00:00Z", "FUTURE"),
            evidence("filing", "sec.filing_text", "2026-02-14T20:00:00Z", "SEC risks"),
        )
        with AssignmentTools(
            "2026-02-15T22:00:00Z",
            ["AAPL"],
            evidence=rows,
            av_request=forbidden,
            price_request=forbidden,
            news_request=forbidden,
            sec_request=forbidden,
        ) as tools:
            news = json.loads(tools.get_news_for_symbols(["AAPL"]))
            self.assertEqual([x["id"] for x in news["evidence"]], ["article"])
            self.assertEqual(news["evidence"][0]["text"], "AI & earnings")
            self.assertTrue(json.loads(tools.get_news_for_symbols(["AAPL"]))["cache_hit"])
            quote = json.loads(tools.get_global_quote("AAPL"))
            self.assertEqual(quote["source"], "derived.historical_quote")
            self.assertFalse(quote["realtime"])
            self.assertEqual(json.loads(tools.get_rsi("AAPL"))["source"], "derived.local_rsi")
            self.assertEqual(json.loads(tools.get_rsi("AAPL"))["evidence"][0]["values"]["rsi"], 100)
            self.assertEqual(len(json.loads(tools.get_price_history("AAPL"))["evidence"]), 5)
            self.assertEqual(
                json.loads(tools.get_sec_filings("AAPL"))["evidence"][0]["id"], "filing"
            )

    def test_macd_signal_matches_closed_form_for_linear_prices(self):
        count = 45
        with AssignmentTools("2026-02-20T22:00:00Z", ["AAPL"], evidence=bars(count)) as tools:
            values = json.loads(tools.get_macd("AAPL"))["evidence"][0]["values"]

        def macd_at(index):
            fast = index + 1 - 5.5 * (1 - (11 / 13) ** index)
            slow = index + 1 - 12.5 * (1 - (25 / 27) ** index)
            return fast - slow

        expected = macd_at(count - 1)
        signal = sum(
            0.2 * (0.8 ** (count - 1 - index)) * macd_at(index) for index in range(1, count)
        )
        self.assertAlmostEqual(values["macd"], expected)
        self.assertAlmostEqual(values["signal"], signal)
        self.assertAlmostEqual(values["histogram"], expected - signal)

    def test_registry_implements_required_interfaces_with_immutable_cutoff(self):
        with AssignmentTools("2026-02-20T22:00:00Z", ["AAPL"], evidence=bars()) as tools:
            registry = ToolRegistry(tools.as_of)
            tools.register_tools(registry)
            names = {item["name"] for item in registry.schemas}
            self.assertTrue(
                {
                    "get_news_for_symbols",
                    "get_price_history",
                    "get_rsi",
                    "get_macd",
                    "get_global_quote",
                    "get_sec_filings",
                }
                <= names
            )
            result = registry.dispatch("get_rsi", {"symbol": "AAPL", "time_period": 14})
            self.assertIn("evidence", result)
            with self.assertRaises(AttributeError):
                tools.as_of = "2027-01-01T00:00:00Z"
            with self.assertRaisesRegex(ValueError, "cutoff differs"):
                tools.register_tools(ToolRegistry("2026-02-19T22:00:00Z"))
            with self.assertRaises(ValueError):
                registry.dispatch("get_rsi", {"symbol": "AAPL", "as_of": "future"})

    def test_shared_free_quota_and_cache_survive_restart(self):
        calls = []

        def request(params):
            self.assertNotIn("apikey", params)
            calls.append(params)
            return av_payload(params)

        cutoff = "2026-11-06T22:00:00Z"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tools.sqlite"
            options = dict(
                cache_path=path,
                mode="live",
                av_request=request,
                av_daily_limit=2,
                clock=lambda: cutoff,
                session_close=lambda day: day + "T21:00:00Z",
            )
            with AssignmentTools(cutoff, ["AAPL", "MSFT"], **options) as tools:
                self.assertEqual(json.loads(tools.get_rsi("AAPL"))["status"], "ok")
                self.assertTrue(json.loads(tools.get_rsi("AAPL"))["cache_hit"])
                quote = json.loads(tools.get_global_quote("AAPL"))
                self.assertFalse(quote["realtime"])
                self.assertEqual(quote["evidence"][0]["values"]["volume"], 1000)
            with AssignmentTools(cutoff, ["AAPL", "MSFT"], **options) as tools:
                self.assertTrue(json.loads(tools.get_rsi("AAPL"))["cache_hit"])
                self.assertEqual(json.loads(tools.get_rsi("MSFT"))["status"], "quota_exhausted")
                self.assertEqual(
                    json.loads(tools.get_global_quote("MSFT"))["status"], "quota_exhausted"
                )
            self.assertEqual(len(calls), 2)

    def test_new_live_receipt_waits_for_next_snapshot(self):
        calls = []

        def request(params):
            calls.append(params)
            return av_payload(params)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tools.sqlite"
            with AssignmentTools(
                "2026-11-06T22:00:00Z",
                ["AAPL"],
                cache_path=path,
                mode="live",
                av_request=request,
                clock=lambda: "2026-11-06T22:00:01Z",
                session_close=lambda day: day + "T21:00:00Z",
            ) as tools:
                response = json.loads(tools.get_rsi("AAPL"))
                self.assertEqual(response["status"], "pending_snapshot")
                self.assertFalse(response["evidence"])
            with AssignmentTools(
                "2026-11-06T22:00:02Z",
                ["AAPL"],
                cache_path=path,
                mode="live",
                av_request=request,
                clock=lambda: "2026-11-06T22:00:02Z",
                session_close=lambda day: day + "T21:00:00Z",
            ) as tools:
                response = json.loads(tools.get_rsi("AAPL"))
                self.assertEqual(response["status"], "ok")
                self.assertTrue(response["cache_hit"])
            self.assertEqual(len(calls), 1)

    def test_provider_errors_and_cache_never_echo_secrets(self):
        secret = "DO-NOT-EXPOSE-THIS-API-KEY"

        def request(_params):
            return {"Information": f"rejected request containing {secret}"}

        with AssignmentTools(
            "2026-11-06T22:00:00Z",
            ["AAPL"],
            mode="live",
            av_api_key=secret,
            av_request=request,
            clock=lambda: "2026-11-06T22:00:00Z",
        ) as tools:
            result = tools.get_rsi("AAPL")
            self.assertNotIn(secret, result)
            self.assertEqual(json.loads(result)["status"], "provider_error")
            self.assertEqual(
                tools.cache.db.execute("SELECT COUNT(*) FROM response_cache").fetchone()[0], 0
            )
        row = evidence(
            "article", "alpaca.news", "2026-11-05T22:00:00Z", secret, values={"api_key": secret}
        )
        with AssignmentTools(
            "2026-11-06T22:00:00Z", ["AAPL"], evidence=[row], av_api_key=secret
        ) as tools:
            self.assertNotIn(secret, tools.get_news_for_symbols(["AAPL"]))
            cached = tools.cache.db.execute("SELECT payload FROM response_cache").fetchone()[0]
            self.assertNotIn(secret, cached)
            self.assertNotIn('"api_key"', cached)

    def test_empty_news_is_valid_cached_missing_and_bad_arguments_rejected(self):
        with AssignmentTools("2026-11-06T22:00:00Z", ["AAPL"]) as tools:
            response = json.loads(tools.get_news_for_symbols(["AAPL"]))
            self.assertEqual(response["status"], "missing")
            self.assertEqual(response["evidence"], [])
            self.assertTrue(json.loads(tools.get_news_for_symbols(["AAPL"]))["cache_hit"])
            with self.assertRaisesRegex(ValueError, "cannot advance"):
                tools.get_news_for_symbols(["AAPL"], date="2026-11-07")
            for function, arguments in (
                (tools.get_rsi, {"symbol": "MSFT"}),
                (tools.get_rsi, {"symbol": "AAPL", "time_period": True}),
                (tools.get_price_history, {"symbol": "AAPL", "period": "max"}),
                (tools.get_news_for_symbols, {"symbols": ["AAPL"], "limit": 1000}),
            ):
                with self.subTest(function=function.__name__), self.assertRaises(ValueError):
                    function(**arguments)


if __name__ == "__main__":
    unittest.main()
