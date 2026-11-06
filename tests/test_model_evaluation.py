import unittest

from agent.contracts import canonical_json
from agent.evaluation import EvaluationOutcome, score_generations
from agent.training import EpisodeExample


class ModelEvaluationTests(unittest.TestCase):
    def test_scores_valid_invalid_accuracy_and_regret(self):
        examples = tuple(EpisodeExample(
            f"e{index}", "validation", "2026-01-05T21:00:00Z",
            "2026-01-06T21:00:00Z", f"s{index}", "a", "v",
            ({"role": "user", "content": canonical_json({
                "as_of": "2026-01-05T21:00:00Z",
                "evidence": [{"id": "price", "source": "test", "symbol": "AAPL",
                              "observed_at": "2026-01-05T20:00:00Z",
                              "available_at": "2026-01-05T20:00:00Z", "text": "price"}],
            })},), "expected",
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
        self.assertEqual(records[1]["failure_category"], "invalid_json")
        self.assertEqual(metrics["failure_counts"], {"invalid_json": 1})

    def test_archive_ids_do_not_authorize_off_prompt_citations(self):
        example = EpisodeExample(
            "e", "validation", "2026-01-05T21:00:00Z", "2026-01-06T21:00:00Z",
            "s", "a", "v", ({"role": "user", "content": canonical_json({
                "as_of": "2026-01-05T21:00:00Z", "evidence": [],
                "evidence_index": [{"id": "hidden"}],
            })},), "expected",
        )
        outcome = EvaluationOutcome("e", "validation", "AAPL", ("hidden",),
                                    {"buy": 0.02, "hold": 0, "sell": -0.02}, "buy")
        raw = canonical_json({"decisions": [{"symbol": "AAPL", "action": "buy",
            "qty": 1, "reason": "test", "evidence_ids": ["hidden"]}]})
        metrics, records = score_generations((example,), (outcome,), {"e": raw})
        self.assertEqual(metrics["json_valid_rate"], 1)
        self.assertEqual(metrics["valid_schema_rate"], 0)
        self.assertEqual(records[0]["failure_category"], "unavailable_evidence")

    def test_requires_exact_outcome_coverage(self):
        example = EpisodeExample(
            "e", "validation", "2026-01-05T21:00:00Z", "2026-01-06T21:00:00Z",
            "s", "a", "v", ({"role": "user", "content": "state"},), "expected",
        )
        with self.assertRaisesRegex(ValueError, "exactly cover"):
            score_generations((example,), (), {})


if __name__ == "__main__":
    unittest.main()
