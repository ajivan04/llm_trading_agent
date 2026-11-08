import json
import tempfile
import unittest
from pathlib import Path

from agent.contracts import Evidence, canonical_json, parse_timestamp
from agent.hydrate_news import hydrate, hydration_plan
from agent.source_bundle import load_source
from tests.test_compact_artifact import prepared_source


class Provider:
    def __init__(self, fail_at=None):
        self.calls = 0
        self.fail_at = fail_at

    def fetch(self, symbols, **kwargs):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("interrupted")
        at = parse_timestamp(kwargs["time_from"])
        return (
            Evidence(
                "article:" + at.isoformat(),
                "alpaca.news",
                symbols[0],
                at,
                at,
                kwargs["retrieved_at"],
                "Actual historical article text.",
            ),
        )


class HydrationTests(unittest.TestCase):
    def source(self, root):
        path = root / "source"
        prepared_source(path)
        (path / ".env").write_text("DO-NOT-COPY")
        (path / "dataset/unexpected-secret.txt").write_text("DO-NOT-COPY")
        return path

    def plan(self, source):
        return hydration_plan(source, train_sessions=1, validation_sessions=1, symbols=["AAPL"])

    def test_preserves_raw_sources_without_copying_obsolete_targets_or_secrets(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = self.source(root)
            before = (source / "price-bars.jsonl").read_bytes()
            targets = (source / "dataset/examples.jsonl").read_bytes()
            plan = self.plan(source)
            output = root / "hydrated"
            result = hydrate(source, output, adapter=Provider(), plan=plan)
            self.assertEqual((output / "price-bars.jsonl").read_bytes(), before)
            self.assertEqual((source / "dataset/examples.jsonl").read_bytes(), targets)
            self.assertFalse((output / "dataset").exists())
            self.assertEqual(load_source(output)["train_end"], load_source(source)["train_end"])
            self.assertFalse((output / ".env").exists())
            self.assertFalse((output / "dataset/unexpected-secret.txt").exists())
            self.assertFalse((output / "ppo-train.jsonl").exists())
            original = [json.loads(x) for x in (source / "evidence.jsonl").read_text().splitlines()]
            rows = [json.loads(x) for x in (output / "evidence.jsonl").read_text().splitlines()]
            self.assertTrue(all(record in rows for record in original))
            self.assertGreater(result["raw_news_evidence_records_added"], 0)
            self.assertEqual(result["policy_calls"], 0)
            self.assertFalse(result["heldout_news_requested"])
            self.assertEqual(load_source(output)["scope"], "research")
            self.assertFalse((output / "preparation-manifest.json").exists())

    def test_resume_reuses_chunks_and_rejects_swapped_request_or_changed_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = self.source(root)
            plan, output = self.plan(source), root / "hydrated"
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                hydrate(source, output, adapter=Provider(fail_at=2), plan=plan)
            chunk = root / "hydrated.hydrating/chunk-0000.json"
            saved = json.loads(chunk.read_text())
            saved["request"]["index"] = 999
            chunk.write_text(canonical_json(saved))
            provider = Provider()
            with self.assertRaisesRegex(ValueError, "binding"):
                hydrate(source, output, adapter=provider, plan=plan, resume=True)
            self.assertEqual(provider.calls, 0)
            saved["request"]["index"] = 0
            chunk.write_text(canonical_json(saved))
            result = hydrate(source, output, adapter=provider, plan=plan, resume=True)
            self.assertEqual(provider.calls, len(plan["windows"]) - 1)
            self.assertGreater(result["raw_news_evidence_records_added"], 0)

    def test_future_provider_data_and_unbound_work_fail_before_publication(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = self.source(root), root / "hydrated"
            plan = self.plan(source)

            class Future:
                def fetch(self, *_a, **_kw):
                    return (
                        Evidence(
                            "future",
                            "alpaca.news",
                            "AAPL",
                            "2027-01-01T00:00:00Z",
                            "2027-01-01T00:00:00Z",
                            "2027-01-01T00:00:00Z",
                            "future",
                        ),
                    )

            with self.assertRaisesRegex(ValueError, "outside"):
                hydrate(source, output, adapter=Future(), plan=plan)
            self.assertFalse(output.exists())
            (source / "evidence.jsonl").write_text("")
            with self.assertRaisesRegex(ValueError, "checksum"):
                hydrate(source, output, adapter=Provider(), plan=plan, resume=True)
