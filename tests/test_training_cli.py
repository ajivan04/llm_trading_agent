import unittest

from agent.training.cli import _episode_example


class TrainingCliTests(unittest.TestCase):
    def test_json_messages_are_normalized_to_tuple(self):
        row = {
            "example_id": "example-1",
            "split": "train",
            "decision_at": "2026-01-01T00:00:00Z",
            "outcome_at": "2026-01-02T00:00:00Z",
            "snapshot_id": "snapshot",
            "account_state_hash": "account",
            "evidence_hash": "evidence",
            "messages": [
                {"role": "system", "content": "rules"},
                {"role": "user", "content": "state"},
            ],
            "completion": "decision",
        }
        example = _episode_example(row)
        self.assertIsInstance(example.messages, tuple)
        self.assertEqual(len(example.messages), 2)
