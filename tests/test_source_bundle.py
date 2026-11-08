"""Source-only archives must not depend on stale model-training targets."""

import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.artifacts import atomic_json
from agent.contracts import content_hash
from agent.source_bundle import load_source, write_source
from agent.trajectory_cli import main
from tests.test_trajectory_cli import Calendar, prepared


class SourceBundleTests(unittest.TestCase):
    def test_legacy_source_loads_without_old_target_files_and_keeps_boundaries(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            prepared(root)
            (root / "dataset/train.jsonl").write_text("deliberately invalid obsolete target")
            source = load_source(root)
            self.assertEqual(source["train_end"], "2026-01-07T23:59:59Z")
            self.assertNotIn("dataset/train.jsonl", source["file_hashes"])
            with (
                patch("agent.trajectory_cli._calendar", return_value=Calendar()),
                patch("agent.trajectory_cli._policy", side_effect=AssertionError("no weights")),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(
                    main(["plan", "--prepared", str(root), "--symbols", "AAPL", "--sessions", "2"]),
                    0,
                )

    def test_canonical_three_file_source_plans_and_detects_changed_data(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            prepared(root)
            old = load_source(root)
            write_source(root, old, universe=("AAPL",))
            shutil.rmtree(root / "dataset")
            (root / "preparation-manifest.json").unlink()
            self.assertEqual(
                set(p.name for p in root.iterdir()),
                {"source.json", "price-bars.jsonl", "evidence.jsonl"},
            )
            metadata = load_source(root)
            self.assertEqual(metadata["scope"], "research")
            with (
                patch("agent.trajectory_cli._calendar", return_value=Calendar()),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(
                    main(["plan", "--prepared", str(root), "--symbols", "AAPL", "--sessions", "2"]),
                    0,
                )
            (root / "evidence.jsonl").write_text("")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_source(root)

    def test_normalize_then_hydrate_needs_no_obsolete_dataset(self):
        from agent.hydrate_news import hydrate, hydration_plan
        from agent.normalize_prices import normalize_source
        from tests.test_compact_artifact import prepared_source
        from tests.test_hydrate_news import Provider

        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            original, nominal, hydrated = root / "original", root / "nominal", root / "hydrated"
            prepared_source(original)
            normalize_source(original, nominal, tail_fetcher=lambda *_: self.fail("no tail needed"))
            plan = hydration_plan(
                nominal, train_sessions=1, validation_sessions=1, symbols=["AAPL"]
            )
            hydrate(nominal, hydrated, adapter=Provider(), plan=plan)
            self.assertFalse((nominal / "dataset").exists())
            self.assertFalse((hydrated / "dataset").exists())
            self.assertEqual(load_source(original)["train_end"], load_source(hydrated)["train_end"])
            self.assertIn("hydration-manifest.json", load_source(hydrated)["file_hashes"])

    def test_source_identity_boundaries_and_heldout_scope_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            prepared(root)
            old = load_source(root)
            write_source(root, old, universe=("AAPL",), scope="heldout", test_release_id="release")
            self.assertEqual(load_source(root)["test_release_id"], "release")
            with self.assertRaisesRegex(ValueError, "relabeled"):
                write_source(root, load_source(root), universe=("AAPL",), scope="research")
            with self.assertRaisesRegex(ValueError, "held-out source"):
                main(["plan", "--prepared", str(root), "--symbols", "AAPL", "--sessions", "2"])
            payload = json.loads((root / "source.json").read_text())
            payload["manifest"]["scope"] = "research"
            atomic_json(root / "source.json", payload)
            with self.assertRaisesRegex(ValueError, "identity"):
                load_source(root)
            payload["source_id"] = content_hash(payload["manifest"])
            payload["manifest"].pop("test_release_id")
            payload["manifest"]["boundaries"]["test_end"] = old["train_end"]
            payload["source_id"] = content_hash(payload["manifest"])
            atomic_json(root / "source.json", payload)
            with self.assertRaisesRegex(ValueError, "chronological"):
                load_source(root)

    def test_legacy_missing_integrity_fields_are_not_treated_as_canonical(self):
        for missing in ("source_hash", "evidence_count"):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                prepared(root)
                path = root / "preparation-manifest.json"
                meta = json.loads(path.read_text())
                if missing == "source_hash":
                    meta.pop("source_hash")
                else:
                    meta["counts"].pop("evidence")
                path.write_text(json.dumps(meta))
                with self.assertRaisesRegex(ValueError, "legacy source requires"):
                    load_source(root)
