import unittest

from agent.preferences import CandidateOutcome, build_audited_dpo_pair
from agent.training import DPOCandidate


def outcome(name, response, net_return, *, reset_id=None, simulator="sim", horizon=None):
    candidate = DPOCandidate(
        name, "train", "same prompt", response, "snapshot", "account", "evidence",
        net_return - 0.01,
    )
    return CandidateOutcome(
        candidate, reset_id or f"reset-{name}", simulator,
        horizon or "2026-01-05T21:00:00Z", net_return, 0.001, 0.01,
    )


class PreferenceTests(unittest.TestCase):
    def test_audited_pair_preserves_outcomes_and_independent_resets(self):
        pair = build_audited_dpo_pair(
            outcome("buy", "buy", 0.10), outcome("hold", "hold", 0.02),
        )
        self.assertEqual(pair.pair.chosen_id, "buy")
        self.assertEqual(pair.chosen_outcome.net_return, 0.10)
        self.assertTrue(pair.audit_hash)

    def test_rejects_shared_reset_simulator_or_horizon_mismatch(self):
        first = outcome("buy", "buy", 0.10, reset_id="same")
        with self.assertRaisesRegex(ValueError, "reset"):
            build_audited_dpo_pair(first, outcome("hold", "hold", 0.02, reset_id="same"))
        with self.assertRaisesRegex(ValueError, "simulator"):
            build_audited_dpo_pair(first, outcome("hold", "hold", 0.02, simulator="other"))
        with self.assertRaisesRegex(ValueError, "horizon"):
            build_audited_dpo_pair(
                first, outcome("hold", "hold", 0.02, horizon="2026-01-06T21:00:00Z"))

    def test_objective_cannot_hide_double_counted_costs(self):
        candidate = DPOCandidate(
            "bad", "train", "prompt", "buy", "snapshot", "account", "evidence", 0.08)
        with self.assertRaisesRegex(ValueError, "objective"):
            CandidateOutcome(
                candidate, "reset", "sim", "2026-01-05T21:00:00Z",
                net_return=0.10, execution_costs=0.01, risk_penalty=0.01,
            )


if __name__ == "__main__":
    unittest.main()
