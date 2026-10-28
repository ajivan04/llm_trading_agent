import json
import tempfile
import unittest
from pathlib import Path

from agent.contracts import Decision, Evidence, content_hash
from agent.datasets import (
    HistoricalDatasetBuilder,
    HistoricalEpisode,
    load_dataset_artifact,
    load_examples,
)
from agent.snapshots import SnapshotBuilder


def evidence(name, available):
    return Evidence(
        name, "test", "AAPL", available, available, "2026-02-01T00:00:00Z",
        f"evidence {name}", values={"signal": 1},
    )


def episode(name, decision_at, outcome_at, *, state=None, evidence_item=None):
    item = evidence_item or evidence(f"e-{name}", decision_at)
    snapshot = SnapshotBuilder().build(
        as_of=decision_at, universe=("AAPL",), universe_version="u1",
        evidence=(item,), compact_evidence=(), feature_version="f1",
        account_state=state or {"cash": 1000}, source_health={"test": "ok"},
        market_session_id=f"XNYS-{decision_at[:10]}",
    )
    return HistoricalEpisode(
        name, snapshot, (item,),
        (Decision("AAPL", "hold", 0, "wait", (item.id,)),), outcome_at,
    )


class HistoricalDatasetTests(unittest.TestCase):
    def build(self, records):
        return HistoricalDatasetBuilder().build(
            records, train_end="2026-01-03T00:00:00Z",
            validation_end="2026-01-06T00:00:00Z",
            test_end="2026-01-09T00:00:00Z",
        )

    def test_recomputes_hashes_splits_and_keeps_outcomes_out_of_prompts(self):
        artifact = self.build([
            episode("train", "2026-01-01T21:00:00Z", "2026-01-02T21:00:00Z"),
            episode("validation", "2026-01-04T21:00:00Z", "2026-01-05T21:00:00Z"),
            episode("test", "2026-01-07T21:00:00Z", "2026-01-08T21:00:00Z"),
        ])
        self.assertEqual([item.split for item in artifact.examples],
                         ["train", "validation", "test"])
        prompt = artifact.examples[0].messages[1]["content"]
        self.assertNotIn("outcome_at", prompt)
        self.assertNotIn("reward", prompt)
        self.assertNotIn("retrieved_at", prompt)
        evidence_payload = json.loads(prompt)["evidence"]
        self.assertEqual(artifact.examples[0].evidence_hash, content_hash(evidence_payload))

    def test_purges_overlap_and_is_deterministic_under_input_reordering(self):
        train = episode("train", "2026-01-01T21:00:00Z", "2026-01-02T21:00:00Z")
        cross = episode("cross", "2026-01-02T21:00:00Z", "2026-01-04T21:00:00Z")
        first = self.build([train, cross])
        second = self.build([cross, train])
        self.assertEqual(first.manifest.dataset_hash, second.manifest.dataset_hash)
        self.assertEqual(first.manifest.purged_episode_ids, ("cross",))

    def test_rejects_future_or_sensitive_policy_fields(self):
        with self.assertRaisesRegex(ValueError, "sensitive"):
            self.build([episode(
                "bad", "2026-01-01T21:00:00Z", "2026-01-02T21:00:00Z",
                state={"api_key": "do-not-store"},
            )])
        future = evidence("future", "2026-01-03T21:00:00Z")
        with self.assertRaisesRegex(ValueError, "eligible"):
            episode(
                "bad", "2026-01-01T21:00:00Z", "2026-01-02T21:00:00Z",
                evidence_item=future,
            )
        safe = self.build([episode(
            "tokens", "2026-01-01T21:00:00Z", "2026-01-02T21:00:00Z",
            state={"token_count": 12},
        )])
        self.assertEqual(len(safe.examples), 1)

    def test_writes_and_loads_content_addressed_examples(self):
        artifact = self.build([
            episode("train", "2026-01-01T21:00:00Z", "2026-01-02T21:00:00Z")])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            artifact.write(output)
            loaded = load_examples(output / "examples.jsonl")
            self.assertEqual(loaded[0].example_id, "train")
            self.assertEqual(load_examples(output / "train.jsonl")[0].split, "train")
            self.assertEqual((output / "validation.jsonl").read_text(), "")
            self.assertEqual(
                json.loads((output / "manifest.json").read_text())["dataset_hash"],
                artifact.manifest.dataset_hash,
            )
            self.assertEqual(load_dataset_artifact(output), artifact)
            (output / "train.jsonl").write_text(
                (output / "train.jsonl").read_text() + "{}\n"
            )
            with self.assertRaises(ValueError):
                load_dataset_artifact(output)

    def test_regime_metadata_changes_dataset_identity(self):
        records = [episode(
            "train", "2026-01-01T21:00:00Z", "2026-01-02T21:00:00Z")]
        default = self.build(records)
        regime = HistoricalDatasetBuilder().build(
            records, train_end="2026-01-03T00:00:00Z",
            validation_end="2026-01-06T00:00:00Z", test_end="2026-01-09T00:00:00Z",
            regime_id="us-midterm-election-week-v1",
            decision_session_filter_hash="filter-hash",
        )
        self.assertNotEqual(default.manifest.dataset_hash, regime.manifest.dataset_hash)


if __name__ == "__main__":
    unittest.main()
