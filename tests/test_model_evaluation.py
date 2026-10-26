import unittest

from agent.contracts import canonical_json
from agent.evaluation import EvaluationOutcome, score_generations
from agent.training import EpisodeExample


class ModelEvaluationTests(unittest.TestCase):
    def test_scores_valid_invalid_accuracy_and_regret(self):
        examples = tuple(EpisodeExample(
            f"e{index}", "validation", "2026-01-05T21:00:00Z",
            "2026-01-06T21:00:00Z", f"s{index}", "a", "v",
            ({"role": "user", "content": "state"},), "expected",
        ) for index in range(2))
        outcomes = tuple(EvaluationOutcome(
            f"e{index}", "validation", "AAPL", ("price",),
            {"buy": 0.02, "hold": 0.0, "sell": -0.02}, "buy",
        ) for index in range(2))
        valid = canonical_json({"decisions": [{
            "symbol": "AAPL", "action": "buy", "qty": 1,
            "reason": "price", "evidence_ids": ["price"],
        }]})
        metrics, records = score_generations(
            examples, outcomes, {"e0": valid, "e1": "not-json"},
        )
        self.assertEqual(metrics["valid_schema_rate"], 0.5)
        self.assertEqual(metrics["expert_action_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["mean_regret"], 0.02)
        self.assertEqual(records[1]["error_class"], "ValueError")

    def test_requires_exact_outcome_coverage(self):
        example = EpisodeExample(
            "e", "validation", "2026-01-05T21:00:00Z", "2026-01-06T21:00:00Z",
            "s", "a", "v", ({"role": "user", "content": "state"},), "expected",
        )
        with self.assertRaisesRegex(ValueError, "exactly cover"):
            score_generations((example,), (), {})


if __name__ == "__main__":
    unittest.main()
