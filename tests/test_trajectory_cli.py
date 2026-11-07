import contextlib
import io
import json
import shutil
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent.contracts import Evidence, canonical_json, content_hash
from agent.trajectories import TrajectorySettings, collect_trajectories, export_good_dataset
from agent.trajectory_cli import main
from tests.test_trajectories import DAYS, Policy, bars


class Calendar:
    def next(self, day):
        return SimpleNamespace(session=DAYS[DAYS.index(day) + 1])


def prepared(root):
    (root / "dataset").mkdir()
    settings = TrajectorySettings(
        ("AAPL",), "2026-01-07T23:59:59Z", "2026-01-12T23:59:59Z", "2026-01-14T23:59:59Z"
    )
    manifest = {**asdict(settings), "dataset_hash": "fixture"}
    (root / "dataset/manifest.json").write_text(canonical_json(manifest))
    source = [asdict(row) for row in bars()["AAPL"]]
    (root / "price-bars.jsonl").write_text("".join(canonical_json(row) + "\n" for row in source))
    evidence = [
        asdict(
            Evidence(
                id=source_name,
                source=source_name,
                symbol="AAPL",
                observed_at=f"{DAYS[0]}T12:00:00Z",
                available_at=f"{DAYS[0]}T12:00:00Z",
                retrieved_at="2026-02-01T00:00:00Z",
                text="Actual archived research text.",
            )
        )
        for source_name in ("alpaca.news", "sec.filing_text")
    ]
    (root / "evidence.jsonl").write_text("".join(canonical_json(row) + "\n" for row in evidence))
    (root / "preparation-manifest.json").write_text(
        canonical_json(
            {
                "source_hash": content_hash(tuple(source)),
                "dataset_hash": "fixture",
                "counts": {"evidence": len(evidence)},
            }
        )
    )


class TrajectoryCLITests(unittest.TestCase):
    def run_cli(self, arguments):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            result = main(arguments)
        self.assertEqual(result, 0)
        return [json.loads(line) for line in output.getvalue().splitlines()]

    def test_plan_is_cpu_only_and_reports_selected_window_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared(root)
            with (
                patch("agent.trajectory_cli._calendar", return_value=Calendar()),
                patch(
                    "agent.trajectory_cli._policy", side_effect=AssertionError("must not load Qwen")
                ),
            ):
                plan = self.run_cli(
                    ["plan", "--prepared", str(root), "--symbols", "AAPL", "--sessions", "2"]
                )[0]
                self.assertEqual(plan["sessions"], [DAYS[1], DAYS[2]])
                self.assertEqual(len(plan["decision_window_coverage"]), 2)
                self.assertEqual(plan["raw_news_records"], 2)
                with self.assertRaisesRegex(ValueError, "requested 60"):
                    main(["plan", "--prepared", str(root), "--symbols", "AAPL"])

    def test_truncated_archives_fail_before_model_and_future_news_does_not_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared(root)
            arguments = [
                "collect",
                "--prepared",
                str(root),
                "--symbols",
                "AAPL",
                "--sessions",
                "2",
                "--journal",
                str(root / "journal"),
            ]
            with (
                patch("agent.trajectory_cli._calendar", return_value=Calendar()),
                patch(
                    "agent.trajectory_cli._policy", side_effect=AssertionError("must not load Qwen")
                ),
            ):
                price = root / "price-bars.jsonl"
                original = price.read_text()
                price.write_text("\n".join(original.splitlines()[:-1]) + "\n")
                with self.assertRaisesRegex(ValueError, "source hash"):
                    main(arguments)
                price.write_text(original)
                archive = root / "evidence.jsonl"
                rows = [json.loads(line) for line in archive.read_text().splitlines()]
                for row in rows:
                    row["observed_at"] = row["available_at"] = "2026-01-13T12:00:00Z"
                archive.write_text("".join(canonical_json(row) + "\n" for row in rows))
                with self.assertRaisesRegex(ValueError, "raw articles"):
                    main(arguments)
                archive.write_text("")
                with self.assertRaisesRegex(ValueError, "archive count"):
                    main(arguments)

    def test_collect_resume_and_export_do_not_retrain_or_invoke_test_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared(root)
            with (
                patch("agent.trajectory_cli._calendar", return_value=Calendar()),
                patch("agent.trajectory_cli._policy", side_effect=lambda _: Policy()) as factory,
            ):
                for split in ("train", "validation"):
                    arguments = [
                        "collect",
                        "--prepared",
                        str(root),
                        "--symbols",
                        "AAPL",
                        "--sessions",
                        "2",
                        "--split",
                        split,
                        "--journal",
                        str(root / split),
                    ]
                    self.run_cli([*arguments, "--max-sessions", "1"])
                    self.run_cli([*arguments, "--resume"])
                calls = factory.call_count
                result = self.run_cli(
                    [
                        "export",
                        "--train-journal",
                        str(root / "train"),
                        "--validation-journal",
                        str(root / "validation"),
                        "--output",
                        str(root / "export"),
                    ]
                )[-1]
                self.assertEqual(factory.call_count, calls)
                self.assertEqual(result["train_counts"]["good_export_candidates"], 2)
                self.assertTrue((root / "export/dataset/manifest.json").is_file())

    def test_legacy_compaction_requires_matching_verified_target(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "prepared"
            root.mkdir()
            prepared(root)
            settings = TrajectorySettings(
                ("AAPL",),
                "2026-01-07T23:59:59Z",
                "2026-01-12T23:59:59Z",
                "2026-01-14T23:59:59Z",
            )
            with contextlib.redirect_stdout(io.StringIO()):
                train = collect_trajectories(bars(), decide=Policy(), settings=settings)
                validation = collect_trajectories(
                    bars(), decide=Policy(), settings=settings, split="validation"
                )
                artifact = export_good_dataset(train, validation, base / "actual")
            shutil.rmtree(root / "dataset")
            shutil.copytree(base / "actual/dataset", root / "dataset")
            compaction = {
                "dataset_hash": artifact.manifest.dataset_hash,
                "compaction_version": "evidence-compaction-v2",
            }
            (root / "compaction-manifest.json").write_text(canonical_json(compaction))
            arguments = ["plan", "--prepared", str(root), "--symbols", "AAPL", "--sessions", "2"]
            with patch("agent.trajectory_cli._calendar", return_value=Calendar()):
                result = self.run_cli(arguments)[0]
                self.assertTrue(
                    result["compaction_provenance"]["legacy_compaction_source_link_unverified"]
                )
                self.assertIn("compaction-manifest.json", result["source_file_hashes"])
                compaction["dataset_hash"] = "wrong-target"
                (root / "compaction-manifest.json").write_text(canonical_json(compaction))
                with self.assertRaisesRegex(ValueError, "compaction lineage"):
                    main(arguments)
                compaction["dataset_hash"] = artifact.manifest.dataset_hash
                (root / "compaction-manifest.json").write_text(canonical_json(compaction))
                (root / "dataset/train.jsonl").write_text("")
                with self.assertRaisesRegex(ValueError, "train dataset file"):
                    main(arguments)


if __name__ == "__main__":
    unittest.main()
