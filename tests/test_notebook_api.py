"""The notebook facade delegates to configured research/replay/paper boundaries."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent.contracts import parse_timestamp
from agent.execution import ExecutionPaused
from agent.notebook_api import (
    NotebookSession,
    dispatch_tool,
    list_tools,
    llm_agent_decide,
    register_tool,
    run_backtest,
    run_trading_loop,
)
from agent.policy.evidence import visible_evidence
from agent.research import ResearchPolicy, ResearchSettings
from agent.runtime import PaperTradingLoop
from agent.trajectories import TrajectorySettings
from tests.test_research_runtime import Backend, final
from tests.test_trajectories import DAYS, IDENTITY, bars

CUTOFF = DAYS[1] + "T21:00:00Z"


def session():
    def respond(messages):
        return final(next(iter(visible_evidence(messages))))

    backend = Backend(
        ['{"tool":"get_price_history","arguments":{"symbol":"AAPL","period":"5d"}}', respond]
    )
    policy = ResearchPolicy(
        backend,
        settings=ResearchSettings(
            max_context_tokens=20000, max_new_tokens=100, tool_context_reserve=2000
        ),
    )

    def source(cutoff, universe):
        return tuple(
            row.to_evidence()
            for row in bars()["AAPL"]
            if parse_timestamp(row.close_at) <= parse_timestamp(cutoff)
        )

    return NotebookSession(policy, ("AAPL",), source, as_of=CUTOFF)


class ScriptedLoop(PaperTradingLoop):
    def __init__(self, policy, events):
        self.policy, self.memory, self.events = policy, None, iter(events)
        self.calls = []

    def run_once(self, *, cycle_id, submit=False):
        self.calls.append((cycle_id, submit))
        event = next(self.events)
        if isinstance(event, Exception):
            raise event
        return event


class NotebookAPITests(unittest.TestCase):
    def test_decision_four_tuple_is_actual_research_not_fabricated_fallback(self):
        state = {"as_of": CUTOFF, "cash": 100000, "equity": 100000, "positions": {}}
        with session() as model:
            decisions, prompt, completion, count = llm_agent_decide(
                None, model, state, date=DAYS[1]
            )
            self.assertEqual(decisions[0]["symbol"], "AAPL")
            self.assertEqual(count, 1)
            self.assertEqual(json.loads(completion)["decisions"][0]["action"], "buy")
            messages = json.loads(prompt)
            self.assertTrue(any(m["role"] == "assistant" for m in messages))
            self.assertIn("UNTRUSTED_TOOL_RESULT", prompt)
            self.assertEqual(model.last_result["status"], "success")
        with self.assertRaises(TypeError):
            llm_agent_decide(None, object(), state)

    def test_bound_tools_alias_and_date_guards(self):
        with session() as model:
            self.assertEqual(
                register_tool("get_news", session=model)["name"], "get_news_for_symbols"
            )
            self.assertEqual(
                register_tool("search_market_news", session=model)["name"], "search_market_news"
            )
            self.assertEqual(
                dispatch_tool("search_market_news", {}, session=model)["coverage"],
                "archived-whitelist-news",
            )
            result = dispatch_tool("get_news", {"symbols": ["AAPL"]}, session=model)
            self.assertEqual(result["status"], "missing")
            with self.assertRaisesRegex(ValueError, "cutoff"):
                dispatch_tool(
                    "get_news", {"symbols": ["AAPL"]}, session=model, as_of="2026-02-01T00:00:00Z"
                )
            with self.assertRaises(ValueError):
                register_tool("submit_order", session=model)
            with self.assertRaises(ValueError):
                register_tool("get_news", lambda: "untrusted callback", session=model)
            state = {"as_of": CUTOFF, "cash": 100000, "equity": 100000, "positions": {}}
            with self.assertRaisesRegex(ValueError, "date"):
                llm_agent_decide(None, model, state, date="2026-02-01")
            with self.assertRaisesRegex(ValueError, "max_turns"):
                llm_agent_decide(None, model, state, max_turns=1)

    def test_backtest_percentage_units_and_fixed_plan_prefix(self):
        with session() as model:
            plan = ("2026-01-02", "2026-01-05", "2026-01-06")
            model.replay = {
                "decision_sessions": plan,
                "settings": SimpleNamespace(initial_cash=100000),
            }
            result = SimpleNamespace(
                equity_curve=[{"equity": 100000}, {"equity": 102000}],
                trades=[{"symbol": "AAPL"}],
                metrics={"percentage_return": 0.02, "ending_equity": 102000},
            )
            with patch(
                "agent.notebook_api.evaluate_research_policy", return_value=result
            ) as evaluate:
                output = run_backtest(None, model, n_days=2)
                self.assertEqual(output["total_return"], 2)
                self.assertEqual(output["final_equity"], 102000)
                self.assertEqual(evaluate.call_args.kwargs["decision_sessions"], plan)
                self.assertEqual(evaluate.call_args.kwargs["max_sessions"], 2)
                self.assertIs(evaluate.call_args.kwargs["decide"], model.policy)
            with patch("agent.notebook_api.evaluate_research_policy") as evaluate:
                with self.assertRaises(TypeError):
                    run_backtest(None, model, collector=object(), n_days=2)
                evaluate.assert_not_called()

    def test_training_backtest_collects_actual_prompts_and_outcomes_for_notebook_collector(self):
        import contextlib
        import io

        class Collector:
            def __init__(self):
                self.entries = []

            def log(self, prompt, completion, decisions, equity, tool_calls=0):
                self.entries.append(
                    {
                        "prompt": prompt,
                        "completion": completion,
                        "decisions": decisions,
                        "equity": equity,
                        "tool_calls": tool_calls,
                    }
                )
                return len(self.entries) - 1

            def update_outcome(self, index, equity):
                self.entries[index]["equity_after"] = equity

            def auto_label(self):
                for row in self.entries:
                    row["label"] = "good" if row["equity_after"] >= row["equity"] else "bad"

        with session() as model, contextlib.redirect_stdout(io.StringIO()):

            def respond(messages):
                return final(next(iter(visible_evidence(messages))))

            request = '{"tool":"get_price_history","arguments":{"symbol":"AAPL","period":"5d"}}'
            model.policy.backend.outputs = iter([request, respond] * 3)
            model.replay = {
                "split": "train",
                "bars_by_symbol": bars(),
                "settings": TrajectorySettings(
                    ("AAPL",),
                    "2026-01-07T23:59:59Z",
                    "2026-01-12T23:59:59Z",
                    "2026-01-14T23:59:59Z",
                ),
                "decision_sessions": DAYS[:3],
                "policy_identity": IDENTITY,
                "next_session_resolver": dict(zip(DAYS, DAYS[1:], strict=False)).__getitem__,
            }
            collector = Collector()
            output = run_backtest(None, model, collector, n_days=3)
            collector.auto_label()
            self.assertEqual(output["split"], "train")
            self.assertEqual(len(output["collection"].records), 3)
            self.assertEqual(len(collector.entries), 3)
            self.assertFalse(output["collector_error"])
            self.assertEqual(output["final_equity"], output["equity_history"][-1])
            for entry, record in zip(collector.entries, output["collection"].records, strict=True):
                self.assertEqual(json.loads(entry["prompt"]), record["messages"])
                self.assertEqual(entry["completion"], record["completion"])
                self.assertEqual(entry["equity_after"], record["equity_after"])
                self.assertEqual(entry["tool_calls"], 1)
                self.assertEqual(entry["label"], "good")
            model.replay["split"] = "test"
            with self.assertRaisesRegex(ValueError, "held-out"):
                run_backtest(None, model, n_days=3)

    def test_list_tools_exposes_actual_registered_interfaces(self):
        with session() as model:
            schemas = list_tools(session=model, verbose=False)
            names = {schema["name"] for schema in schemas}
            self.assertTrue(
                {"get_news_for_symbols", "get_rsi", "get_macd", "get_sec_filings"} <= names
            )
            self.assertNotIn("submit_order", names)

    def test_live_schedule_continues_recoverable_failures_and_supports_multi_hour_run(self):
        with session() as model:
            statuses = ["invalid_policy", "data_unavailable", "expired_signal"] + [
                "completed"
            ] * 117
            loop = ScriptedLoop(model.policy, [{"status": status} for status in statuses])
            model.paper_loop = loop
            sleeps = []
            output = run_trading_loop(
                loop, None, model, n_iterations=120, interval_seconds=120, sleep=sleeps.append
            )
            self.assertEqual(len(output), 120)
            self.assertEqual(len(loop.calls), 120)
            self.assertEqual(sleeps, [120] * 119)
            self.assertEqual(output[0]["status"], "invalid_policy")
            for bad in (float("inf"), float("nan"), -1, True):
                with self.assertRaises(ValueError):
                    run_trading_loop(loop, None, model, n_iterations=1, interval_seconds=bad)

    def test_unknown_execution_failure_stops_without_retrying_or_echoing_secret(self):
        with session() as model:
            loop = ScriptedLoop(
                model.policy, [OSError("private-broker-message"), {"status": "completed"}]
            )
            model.paper_loop = loop
            output = run_trading_loop(loop, None, model, n_iterations=2)
            self.assertEqual(len(loop.calls), 1)
            self.assertEqual(output[0]["status"], "error")
            self.assertNotIn("private-broker", json.dumps(output))

    def test_submitting_loop_needs_stable_ids_and_pauses_without_secret_echo(self):
        with session() as model:
            loop = ScriptedLoop(
                model.policy,
                [
                    {"status": "completed"},
                    ExecutionPaused("secret-api-key-should-not-leak"),
                    {"status": "completed"},
                ],
            )
            model.paper_loop = loop
            with self.assertRaisesRegex(ValueError, "cycle_ids"):
                run_trading_loop(loop, None, model, submit=True, n_iterations=3)
            self.assertEqual(loop.calls, [])
            result = run_trading_loop(
                loop,
                None,
                model,
                submit=True,
                n_iterations=3,
                cycle_ids=["first", "second", "third"],
            )
            self.assertEqual(len(loop.calls), 2)
            self.assertEqual(result[1]["status"], "paused")
            self.assertNotIn("secret-api-key", json.dumps(result))

    def test_logging_failure_never_retries_an_already_completed_cycle(self):
        with session() as model:
            loop = ScriptedLoop(model.policy, [{"status": "completed"}])
            model.paper_loop = loop

            def fail(_result):
                raise OSError("secret-provider-message")

            collector = SimpleNamespace(on_cycle_result=fail)
            result = run_trading_loop(
                loop,
                None,
                model,
                collector=collector,
                n_iterations=1,
                cycle_ids=["stable"],
                submit=True,
            )
            self.assertEqual(loop.calls, [("stable", True)])
            self.assertTrue(result[0]["collector_error"])
            self.assertNotIn("secret-provider-message", json.dumps(result))
            with self.assertRaises(ValueError):
                run_trading_loop(
                    loop, None, model, n_iterations=1, cycle_ids="letters", submit=True
                )


class PaperTerminalLoggingTests(unittest.TestCase):
    def test_pause_and_error_are_logged_without_retrying_or_exposing_exception_values(self):
        from agent.notebook_api import run_trading_loop

        for event, status in (
            (ExecutionPaused("private-broker-text"), "paused"),
            (RuntimeError("private-provider-text"), "error"),
        ):
            with self.subTest(status=status), session() as model:
                events = []
                model.paper_loop = ScriptedLoop(model.policy, [event])
                result = run_trading_loop(
                    model.paper_loop,
                    None,
                    model,
                    n_iterations=2,
                    collector=SimpleNamespace(on_cycle_result=events.append),
                )
                self.assertEqual(len(events), 1)
                self.assertEqual(events[0]["status"], status)
                self.assertEqual(len(model.paper_loop.calls), 1)
                self.assertNotIn("private-", json.dumps(result))
