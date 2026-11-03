import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent.evaluation.portfolio import replay_predictions
from agent.training import EpisodeExample


class PortfolioReplayTests(unittest.TestCase):
    def test_replays_cash_shares_costs_and_profit(self):
        example = EpisodeExample(
            "AAPL:2026-01-05", "validation", "2026-01-05T21:00:00Z",
            "2026-01-06T21:00:00Z", "snapshot", "account", "evidence",
            ({"role": "user", "content": "state"},), "completion",
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


if __name__ == "__main__":
    unittest.main()
