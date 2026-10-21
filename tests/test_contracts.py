import dataclasses
import json
import unittest
from datetime import datetime

from agent.config import Config, POLICY_MODEL
from agent.contracts import (
    Decision, Episode, Evidence, Fill, OrderIntent, Snapshot, content_hash,
    normalize_timestamp,
)


class TimestampTests(unittest.TestCase):
    def test_normalizes_aware_iso_timestamp_to_utc(self):
        self.assertEqual(normalize_timestamp("2026-01-02T09:30:00-05:00"),
                         "2026-01-02T14:30:00+00:00")
        self.assertEqual(normalize_timestamp("2026-01-02T14:30:00Z"),
                         "2026-01-02T14:30:00+00:00")

    def test_rejects_naive_timestamp(self):
        with self.assertRaises(ValueError):
            normalize_timestamp("2026-01-02T09:30:00")
        with self.assertRaises(ValueError):
            normalize_timestamp(datetime(2026, 1, 2))


class ContractTests(unittest.TestCase):
    def evidence(self, retrieved_at="2026-01-10T00:00:00+00:00"):
        return Evidence(
            "fred:gdp:2025q4", "alfred", None,
            "2025-12-31T23:59:00+00:00", "2026-01-05T13:30:00+00:00",
            retrieved_at, "GDP release", {"value": 2.3, "nested": [1, 2]},
        )

    def test_evidence_eligibility_uses_observed_and_available_not_retrieved(self):
        evidence = self.evidence(retrieved_at="2026-10-20T12:00:00+00:00")
        self.assertFalse(evidence.eligible("2026-01-05T13:29:59+00:00"))
        self.assertTrue(evidence.eligible("2026-01-05T13:30:00+00:00"))
        # Eligibility remains true even though the historical archive was retrieved later.
        self.assertTrue(evidence.eligible("2026-01-06T00:00:00+00:00"))

    def test_evidence_rejects_impossible_timestamp_order(self):
        with self.assertRaises(ValueError):
            Evidence("x", "source", "AAPL", "2026-01-03T00:00:00Z",
                     "2026-01-02T00:00:00Z", "2026-01-04T00:00:00Z", "text")
        with self.assertRaises(ValueError):
            self.evidence(retrieved_at="2026-01-01T00:00:00+00:00")

    def test_nested_payloads_and_dataclasses_are_immutable(self):
        evidence = self.evidence()
        with self.assertRaises((TypeError, dataclasses.FrozenInstanceError)):
            evidence.values["value"] = 5
        with self.assertRaises((TypeError, dataclasses.FrozenInstanceError)):
            evidence.values["nested"][0] = 5
        with self.assertRaises(dataclasses.FrozenInstanceError):
            evidence.text = "changed"

    def test_snapshot_preserves_order_and_serializes(self):
        snapshot = Snapshot(
            "snap-1", "2026-01-05T09:30:00-05:00", "u1", ("e2", "e1"),
            "f1", {"cash": 100000.0}, {"sec": "healthy"}, "2026-01-05",
            ({"id": "e2", "text": "compact"},),
        )
        self.assertEqual(snapshot.evidence_ids, ("e2", "e1"))
        self.assertEqual(json.loads(snapshot.to_json())["account_state"]["cash"], 100000.0)
        with self.assertRaises(ValueError):
            Snapshot("x", "2026-01-01T00:00:00Z", "u", ("e", "e"), "f", {}, {}, "s")

    def test_decision_order_fill_and_episode_validation(self):
        decision = Decision("aapl", "BUY", 2, "supported", ("e1",))
        fill = Fill("o1", "aapl", "buy", 2, 101.25, "2026-01-06T14:30:00Z")
        intent = OrderIntent("i1", "r1", "aapl", "buy", 2,
                             "2026-01-05T21:00:00Z", "client-1")
        episode = Episode("ep1", "r1", "experiment-1", "train", "snap-1",
                          "2026-01-05T21:00:00Z", "2026-01-06T21:00:00Z",
                          (decision,), (fill,), {"cash": 100000}, 0.01)
        self.assertTrue(intent.paper_only)
        self.assertEqual(json.loads(episode.to_json())["decisions"][0]["symbol"], "AAPL")
        with self.assertRaises(ValueError):
            Decision("AAPL", "hold", 1, "invalid")
        with self.assertRaises(ValueError):
            OrderIntent("i", "r", "AAPL", "buy", 1, "2026-01-01T00:00:00Z", "c",
                        paper_only=False)
        with self.assertRaises(ValueError):
            Episode("e", "r", "x", "test", "s", "2026-01-02T00:00:00Z",
                    "2026-01-01T00:00:00Z")

    def test_order_intent_represents_every_durable_state(self):
        states = {
            "intent", "submitting", "open", "unknown", "partial", "filled",
            "canceled", "expired", "rejected",
        }
        for status in states:
            with self.subTest(status=status):
                intent = OrderIntent(
                    f"intent-{status}", "run", "AAPL", "buy", 1,
                    "2026-01-05T21:00:00Z", f"client-{status}", status=status,
                )
                self.assertEqual(intent.status, status)
        with self.assertRaises(ValueError):
            OrderIntent("i", "r", "AAPL", "buy", 1, "2026-01-05T21:00:00Z",
                        "client", status="submitted")

    def test_content_hash_is_deterministic(self):
        left = {"b": [2, 1], "a": 3}
        right = {"a": 3, "b": [2, 1]}
        self.assertEqual(content_hash(left), content_hash(right))


class ConfigTests(unittest.TestCase):
    def test_defaults_preserve_model_and_operating_settings(self):
        config = Config()
        self.assertEqual(config.model_name, POLICY_MODEL)
        self.assertEqual(config.news_ttl_hours, 24)
        self.assertEqual(config.max_tool_turns, 6)
        self.assertEqual(config.max_position_weight, 0.10)
        self.assertTrue(config.paper_trading_only)

    def test_rejects_model_real_money_and_invalid_limits(self):
        with self.assertRaises(ValueError):
            Config(model_name="larger/model")
        with self.assertRaises(ValueError):
            Config(paper_trading_only=False)
        with self.assertRaises(ValueError):
            Config(max_position_weight=0.9)
        with self.assertRaises(ValueError):
            Config(universe=("AAPL",), sector_map=())

    def test_dict_round_trip_and_stable_hash(self):
        original = Config()
        restored = Config.from_dict(original.to_dict())
        self.assertEqual(original, restored)
        self.assertEqual(original.config_hash, restored.config_hash)
        self.assertEqual(json.loads(original.to_json())["model_name"], POLICY_MODEL)

    def test_unknown_config_is_rejected(self):
        with self.assertRaises(ValueError):
            Config.from_dict({"unknown": True})


if __name__ == "__main__":
    unittest.main()
