import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent.evaluation.portfolio import replay_predictions
from agent.training import EpisodeExample


def replay_example(symbol, decision_day, outcome_day):
    decision_at = f"{decision_day}T21:00:00Z"
    return EpisodeExample(
        f"{symbol}:{decision_day}", "validation", decision_at, f"{outcome_day}T21:00:00Z",
        "snapshot", "account", "evidence",
        ({"role": "user", "content": json.dumps({
            "as_of": decision_at, "evidence": [{
                "id": "price", "source": "test", "symbol": symbol, "text": "price",
                "observed_at": decision_at, "available_at": decision_at,
            }],
        })},), "completion",
    )


def run_case(examples, bars, *, buy_first=False):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        outcomes, predictions = [], []
        for index, example in enumerate(examples):
            symbol = example.example_id.split(":", 1)[0]
            outcomes.append({
                "example_id": example.example_id, "split": "validation", "symbol": symbol,
                "evidence_ids": ["price"], "objectives": {"buy": 0.1, "hold": 0, "sell": -0.1},
                "chosen": "buy",
            })
            buy = buy_first and index == 0
            predictions.append({"example_id": example.example_id, "generation": json.dumps({
                "decisions": [{"symbol": symbol, "action": "buy" if buy else "hold",
                               "qty": 1 if buy else 0, "reason": "test",
                               "evidence_ids": ["price"]}],
            })})
        for name, rows in (("outcome-audit", outcomes), ("price-bars", bars),
                           ("predictions", predictions)):
            (root / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        with patch("agent.datasets.load_dataset_artifact",
                   return_value=SimpleNamespace(examples=tuple(examples))):
            return replay_predictions(root, root / "predictions.jsonl", split="validation")


class PortfolioReplayTests(unittest.TestCase):
    def test_replays_cash_shares_costs_and_profit(self):
        example = EpisodeExample(
            "AAPL:2026-01-05", "validation", "2026-01-05T21:00:00Z",
            "2026-01-06T21:00:00Z", "snapshot", "account", "evidence",
            ({"role": "user", "content": json.dumps({
                "as_of": "2026-01-05T21:00:00Z",
                "evidence": [{"id": "evidence", "source": "test", "symbol": "AAPL",
                              "observed_at": "2026-01-05T20:00:00Z",
                              "available_at": "2026-01-05T20:00:00Z", "text": "price"}],
            })},), "completion",
        )
        dataset = SimpleNamespace(examples=(example,))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "dataset").mkdir()
            (root / "outcome-audit.jsonl").write_text(json.dumps({
                "example_id": example.example_id, "split": "validation", "symbol": "AAPL",
                "evidence_ids": ["evidence"],
                "objectives": {"buy": 0.1, "hold": 0.0, "sell": -0.1},
                "chosen": "buy",
            }) + "\n")
            (root / "price-bars.jsonl").write_text("\n".join([
                json.dumps({"symbol": "AAPL", "session": "2026-01-05", "open": 100, "close": 100}),
                json.dumps({"symbol": "AAPL", "session": "2026-01-06", "open": 100, "close": 110}),
            ]) + "\n")
            predictions = root / "predictions.jsonl"
            predictions.write_text(json.dumps({
                "example_id": example.example_id,
                "generation": json.dumps({"decisions": [{
                    "symbol": "AAPL", "action": "buy", "qty": 10,
                    "reason": "test", "evidence_ids": ["evidence"],
                }]})
            }) + "\n")
            with patch("agent.datasets.load_dataset_artifact", return_value=dataset):
                result = replay_predictions(root, predictions, split="validation")
        self.assertEqual(result.metrics["executed_trades"], 1)
        self.assertAlmostEqual(result.metrics["ending_equity"], 100099.5)
        self.assertAlmostEqual(result.metrics["dollar_profit"], 99.5)
        self.assertEqual(result.metrics["invalid_policy_decisions"], 0)
        self.assertEqual(result.metrics["execution_rejections"], 0)

    def test_benchmark_has_daily_marks_and_fixed_entry_cohort(self):
        examples = [replay_example("AAA", "2026-01-05", "2026-01-06"),
                    replay_example("BBB", "2026-01-06", "2026-01-07")]
        bars = [
            {"symbol": "AAA", "session": "2026-01-06", "open": 100, "close": 110},
            {"symbol": "AAA", "session": "2026-01-07", "open": 115, "close": 120},
            {"symbol": "BBB", "session": "2026-01-07", "open": 1000, "close": 4000},
        ]
        result = run_case(examples, bars)
        self.assertEqual(result.metrics["benchmark_symbols"], 1)
        for row, expected in zip(result.equity_curve, (100000, 110000, 120000), strict=True):
            self.assertAlmostEqual(row["benchmark_equity"], expected)

    def test_missing_benchmark_symbol_is_not_paired_with_another_symbol(self):
        examples = [replay_example("AAA", "2026-01-05", "2026-01-06"),
                    replay_example("BBB", "2026-01-06", "2026-01-07")]
        bars = [
            {"symbol": "AAA", "session": "2026-01-06", "open": 100, "close": 110},
            {"symbol": "BBB", "session": "2026-01-07", "open": 1000, "close": 4000},
        ]
        with self.assertRaisesRegex(ValueError, "missing benchmark closing mark for AAA"):
            run_case(examples, bars)

    def test_missing_mark_for_held_shares_fails_instead_of_valuing_at_zero(self):
        examples = [replay_example("AAA", "2026-01-05", "2026-01-06"),
                    replay_example("BBB", "2026-01-06", "2026-01-07")]
        bars = [
            {"symbol": "AAA", "session": "2026-01-06", "open": 100, "close": 110},
            {"symbol": "BBB", "session": "2026-01-07", "open": 1000, "close": 4000},
        ]
        with self.assertRaisesRegex(ValueError, "missing closing mark for held symbol AAA"):
            run_case(examples, bars, buy_first=True)


if __name__ == "__main__":
    unittest.main()
