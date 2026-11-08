import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from agent.evaluation.research import (
    evaluate_research_policy,
    summarize_research_replay,
    write_research_result,
)
from agent.trajectories import TrajectorySettings, export_good_dataset
from tests.test_trajectories import DAYS, IDENTITY, Policy, bars

SESSIONS = DAYS[3:6]
NEXT = dict(zip(DAYS, DAYS[1:], strict=False)).__getitem__


def settings():
    return TrajectorySettings(
        ("AAPL",), "2026-01-06T23:59:59Z", "2026-01-12T23:59:59Z", "2026-01-14T23:59:59Z"
    )


def evaluate(policy=None, rows=None, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return evaluate_research_policy(
            rows or bars(),
            decide=policy or Policy(),
            settings=settings(),
            decision_sessions=SESSIONS,
            next_session_resolver=NEXT,
            **kwargs,
        )


class ResearchEvaluationTests(unittest.TestCase):
    def test_stateful_research_uses_current_cash_positions_and_next_session_prices(self):
        policy = Policy()
        result = evaluate(policy)
        self.assertEqual(result.metrics["sessions"], 3)
        self.assertTrue(result.metrics["complete_plan"])
        self.assertEqual(result.metrics["executed_trades"], 3)
        self.assertEqual(policy.inputs[0]["state"]["positions"], {})
        self.assertEqual(policy.inputs[1]["state"]["positions"], {"AAPL": 1})
        self.assertLess(policy.inputs[1]["state"]["cash"], 100000)
        self.assertTrue(
            all(
                item.available_at <= policy.inputs[0]["as_of"]
                for item in policy.inputs[0]["evidence"]
            )
        )
        first = result.trades[0]
        next_bar = next(row for row in bars()["AAPL"] if row.session == DAYS[4])
        self.assertAlmostEqual(first["price"], next_bar.open * 1.0005)
        self.assertEqual(result.metrics["test_policy_calls"], 0)
        self.assertEqual(result.metrics["valid_schema_rate"], 1)
        self.assertGreater(result.metrics["dollar_profit"], 0)

    def test_hold_invalid_and_risk_rejected_decisions_all_remain_in_score(self):
        for policy, failure_count, rejections in (
            (Policy("hold"), 0, 0),
            (Policy(bad_citation=True), 3, 0),
            (Policy("sell"), 0, 3),
        ):
            with self.subTest(action=policy.action):
                result = evaluate(policy)
                self.assertEqual(result.metrics["sessions"], 3)
                self.assertEqual(len(result.equity_curve), 4)
                self.assertEqual(result.metrics["executed_trades"], 0)
                self.assertEqual(result.metrics["ending_equity"], 100000)
                self.assertEqual(result.metrics["invalid_policy_decisions"], failure_count)
                self.assertEqual(result.metrics["execution_rejections"], rejections)
        held = evaluate(Policy("hold"))
        self.assertEqual(held.metrics["hold_decisions"], 3)
        self.assertIsNone(held.metrics["annualized_sharpe"])
        self.assertEqual(held.metrics["max_drawdown"], 0)

    def test_fixed_benchmark_accounts_for_splits_and_dividends_after_entry(self):
        source = bars()
        source["AAPL"] = tuple(
            replace(
                row,
                open=75,
                high=76,
                low=74,
                close=75,
                adjusted_close=75,
                stock_splits=2,
                dividends=1,
            )
            if row.session == DAYS[5]
            else replace(row, open=80, high=81, low=79, close=80, adjusted_close=80)
            if row.session == DAYS[6]
            else row
            for row in source["AAPL"]
        )
        result = evaluate(Policy("hold"), source)
        entry = source["AAPL"][4].open
        self.assertAlmostEqual(result.equity_curve[2]["benchmark_equity"], 100000 / entry * 2 * 76)
        self.assertAlmostEqual(result.equity_curve[3]["benchmark_equity"], 100000 / entry * 2 * 81)
        self.assertEqual(result.metrics["ending_equity"], 100000)

    def test_context_progress_latency_and_resume_are_recorded_without_repeating_calls(self):
        with tempfile.TemporaryDirectory() as root:
            journal = Path(root) / "journal"
            initial = evaluate(journal=journal, max_sessions=1)
            self.assertFalse(initial.metrics["complete_plan"])
            policy = Policy()
            resumed = evaluate(policy, journal=journal, resume=True)
            self.assertEqual(len(policy.inputs), 2)
            self.assertTrue(resumed.metrics["complete_plan"])
            measured = replace(
                resumed.collection,
                records=tuple(
                    {**row, "inference_seconds": seconds, "initial_input_tokens": 100 + index}
                    for index, (row, seconds) in enumerate(
                        zip(resumed.collection.records, (1.0, 2.0, 3.0), strict=True)
                    )
                ),
            )
            scored = summarize_research_replay(measured, bars())
            self.assertEqual(scored.metrics["decision_cycle_seconds_mean"], 2)
            self.assertAlmostEqual(scored.metrics["decision_cycle_seconds_p95"], 2.9)
            self.assertEqual(scored.metrics["initial_input_tokens_max"], 102)
            output = Path(root) / "result"
            write_research_result(scored, output)
            self.assertEqual(
                json.loads((output / "research_metrics.json").read_text())["sessions"], 3
            )
            self.assertEqual(len((output / "equity_curve.jsonl").read_text().splitlines()), 4)
            manifest = json.loads((output / "research_manifest.json").read_text())
            self.assertEqual(
                manifest["protocol"]["purpose"], "validation only; not a held-out-week score"
            )

    def test_trained_adapter_is_allowed_but_evaluation_cannot_export_sft(self):
        policy = Policy()
        policy.policy_identity = {**IDENTITY, "adapter_hash": "chosen-sft-adapter"}
        result = evaluate(policy)
        self.assertEqual(result.collection.policy_identity["adapter_hash"], "chosen-sft-adapter")
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                export_good_dataset(result.collection, result.collection, Path(root) / "bad-export")

    def test_sparse_unsorted_or_test_windows_fail_before_policy(self):
        policy = Policy()
        for sessions in (SESSIONS[::-1], (SESSIONS[0], SESSIONS[2]), DAYS[6:]):
            with self.subTest(sessions=sessions), self.assertRaises(ValueError):
                evaluate_research_policy(
                    bars(),
                    decide=policy,
                    settings=settings(),
                    decision_sessions=sessions,
                    next_session_resolver=NEXT,
                )
        self.assertEqual(policy.inputs, [])
        with self.assertRaisesRegex(ValueError, "benchmark"):
            evaluate(benchmark_symbol="SPY")

    def test_mixed_invalid_window_and_late_missing_marks_fail_before_any_policy_call(self):
        policy = Policy()
        with self.assertRaisesRegex(ValueError, "inside validation"):
            evaluate_research_policy(
                bars(),
                decide=policy,
                settings=settings(),
                decision_sessions=DAYS[3:7],
                next_session_resolver=NEXT,
            )
        incomplete = bars()
        incomplete["AAPL"] = tuple(row for row in incomplete["AAPL"] if row.session != DAYS[5])
        with self.assertRaisesRegex(ValueError, "complete universe"):
            evaluate_research_policy(
                incomplete,
                decide=policy,
                settings=settings(),
                decision_sessions=SESSIONS,
                next_session_resolver=NEXT,
            )
        with self.assertRaisesRegex(ValueError, "benchmark"):
            evaluate_research_policy(
                bars(),
                decide=policy,
                settings=settings(),
                decision_sessions=SESSIONS,
                next_session_resolver=NEXT,
                benchmark_symbol="SPY",
            )
        self.assertEqual(policy.inputs, [])

    def test_external_benchmark_and_execution_code_change_cannot_escape_source_contract(self):
        from unittest.mock import patch

        policy = Policy()
        extra = bars()
        extra["SPY"] = tuple(replace(row, symbol="SPY") for row in extra["AAPL"])
        with self.assertRaisesRegex(ValueError, "source-bound"):
            evaluate(policy, extra, benchmark_symbol="SPY")
        self.assertEqual(policy.inputs, [])
        with tempfile.TemporaryDirectory() as directory:
            journal = Path(directory) / "journal"
            result = evaluate(journal=journal, max_sessions=1)
            hashes = result.collection.policy_identity["evaluation_code_hashes"]
            self.assertIn("trading_agent.py", hashes)
            self.assertIn("agent/config.py", hashes)
            with patch(
                "agent.evaluation.research._implementation_hashes",
                return_value={**hashes, "agent/config.py": "changed"},
            ):
                with self.assertRaisesRegex(ValueError, "resume contract changed"):
                    evaluate(policy, journal=journal, resume=True)
            self.assertEqual(policy.inputs, [])


class ResearchEvaluationCLITests(unittest.TestCase):
    def test_plan_only_does_not_create_or_load_a_backend(self):
        from unittest.mock import patch

        from agent.research_eval_cli import main
        from tests.test_trajectory_cli import Calendar, prepared

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            prepared(source)
            output = Path(directory) / "evaluation"
            with (
                patch("agent.trajectory_cli._calendar", return_value=Calendar()),
                patch("agent.research_eval_cli.HuggingFaceQwenBackend") as create,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(
                    main(
                        [
                            "--prepared",
                            str(source),
                            "--output",
                            str(output),
                            "--symbols",
                            "AAPL",
                            "--sessions",
                            "2",
                            "--plan-only",
                        ]
                    ),
                    0,
                )
                create.assert_not_called()
                self.assertFalse(output.exists())

    def test_cli_resumes_current_state_and_refuses_missing_adapter(self):
        from unittest.mock import patch

        from agent.research_eval_cli import main
        from tests.test_trajectory_cli import Calendar, prepared

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            prepared(source)
            output = Path(directory) / "evaluation"
            common = [
                "--prepared",
                str(source),
                "--output",
                str(output),
                "--symbols",
                "AAPL",
                "--sessions",
                "2",
            ]
            first, second = Policy(), Policy()
            from agent.research import ResearchSettings
            first.settings = second.settings = ResearchSettings()
            with (
                patch("agent.trajectory_cli._calendar", return_value=Calendar()),
                patch("agent.research_eval_cli.HuggingFaceQwenBackend"),
                patch("agent.research_eval_cli.ResearchPolicy", side_effect=(first, second)),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(main([*common, "--max-sessions", "1"]), 0)
                self.assertEqual(main([*common, "--max-sessions", "2", "--resume"]), 0)
                self.assertEqual(len(first.inputs), 1)
                self.assertEqual(len(second.inputs), 1)
                self.assertEqual(second.inputs[0]["state"]["positions"], {"AAPL": 1})
                metrics = json.loads((output / "research_metrics.json").read_text())
                self.assertTrue(metrics["complete_plan"])
                with self.assertRaisesRegex(ValueError, "complete saved PEFT"):
                    main([*common, "--adapter", str(source / "missing-adapter")])
