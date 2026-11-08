"""The public workflow must choose consistent source/corpus/model paths without shell state."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.__main__ import configuration, main


class AssignmentWorkflowTests(unittest.TestCase):
    def setup_run(self, root):
        root = root.resolve()
        config = root / "project.json"
        config.write_text(json.dumps({"source": "data", "run_dir": "experiment"}))
        corpus = root / "experiment/corpus/dataset"
        corpus.mkdir(parents=True)
        (corpus / "manifest.json").write_text(
            json.dumps({"split_example_ids": {"train": list(range(32))}})
        )
        return config, root / "experiment"

    def test_paths_resolve_from_config_and_bad_config_cannot_replace_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cfg, _ = self.setup_run(root)
            self.assertEqual(configuration(cfg)["source"], (root / "data").resolve())
            cfg.write_text(json.dumps({"source": "same", "run_dir": "same"}))
            with self.assertRaises(ValueError):
                configuration(cfg)
            cfg.write_text(json.dumps({"sft_epochs": float("nan")}))
            with self.assertRaises(ValueError):
                configuration(cfg)

    def test_sft_smoke_and_resume_keep_same_three_pass_total_and_separate_eval_cadence(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            config, run = self.setup_run(Path(folder))
            with patch("agent.__main__._call", return_value=0) as call:
                main(["sft", "--config", str(config), "--smoke"])
                first = call.call_args.args[1]
                self.assertEqual(first[first.index("--max-steps") + 1], 12)
                self.assertEqual(first[first.index("--stop-after-steps") + 1], 2)
                self.assertIn(run / "corpus/dataset/train.jsonl", first)
                self.assertIn("--eval-steps", first)
                for step in (2, 4):
                    checkpoint = run / f"sft/checkpoint-{step}"
                    checkpoint.mkdir(parents=True)
                    (checkpoint / "sft_checkpoint.json").write_text("{}")
                main(["sft", "--config", str(config), "--resume"])
                resumed = call.call_args.args[1]
                self.assertEqual(resumed[resumed.index("--max-steps") + 1], 12)
                self.assertNotIn("--stop-after-steps", resumed)
                self.assertEqual(
                    resumed[resumed.index("--resume-from-checkpoint") + 1], run / "sft/checkpoint-4"
                )

    def test_ppo_uses_compiled_train_data_and_completed_sft_parent(self):
        with tempfile.TemporaryDirectory() as folder:
            config, run = self.setup_run(Path(folder))
            with patch("agent.__main__._call", return_value=0) as call:
                main(["ppo", "--config", str(config), "--smoke"])
                compile_call, train_call = call.call_args_list
                self.assertEqual(compile_call.args[0], "training.ppo_prepare")
                self.assertIn(run / "corpus/dataset", compile_call.args[1])
                args = train_call.args[1]
                self.assertEqual(args[args.index("--updates") + 1], 2)
                self.assertIn(run / "sft", args)
                self.assertIn("--require-assignment-sft", args)
                self.assertIn(run / "ppo-data/ppo-environment.json", args)

    def test_full_collection_requires_successful_smoke_before_any_generation(self):
        with tempfile.TemporaryDirectory() as folder:
            config, _ = self.setup_run(Path(folder))
            with (
                patch("agent.trajectories.load_collection", return_value=object()),
                patch("agent.trajectory_cli.review_smoke", side_effect=ValueError("failed smoke")),
                patch("agent.__main__._call") as call,
            ):
                with self.assertRaisesRegex(ValueError, "failed smoke"):
                    main(["collect", "--config", str(config), "--resume"])
                call.assert_not_called()

    def test_collection_resume_does_not_resume_a_journal_that_was_never_started(self):
        with tempfile.TemporaryDirectory() as folder:
            config, run = self.setup_run(Path(folder))
            existing = run / "collections/train"
            existing.mkdir(parents=True)
            (existing / "collection-manifest.json").write_text("{}")
            with (
                patch("agent.trajectories.load_collection", return_value=object()),
                patch("agent.trajectory_cli.review_smoke"),
                patch("agent.__main__._call", return_value=0) as call,
            ):
                main(["collect", "--config", str(config), "--smoke", "--resume"])
                self.assertEqual(len(call.call_args_list), 3)
                self.assertIn("--resume", call.call_args_list[0].args[1])
                self.assertNotIn("--resume", call.call_args_list[1].args[1])
                self.assertNotIn("--resume", call.call_args_list[2].args[1])

    def test_paper_passes_evaluated_budgets_and_cycle_ids_are_stable_across_session_lengths(self):
        from agent.paper_cli import paper_cycle_ids

        with tempfile.TemporaryDirectory() as folder:
            config, run = self.setup_run(Path(folder))
            cfg = json.loads(config.read_text())
            cfg.update(context_tokens=3072, generation_tokens=384)
            config.write_text(json.dumps(cfg))
            with (
                patch("agent.source_bundle.load_source", return_value={"universe": ("AAPL",)}),
                patch("agent.__main__._call", return_value=0) as call,
            ):
                main(
                    ["paper", "--config", str(config), "--cycle-id", "session", "--iterations", "1"]
                )
                args = call.call_args.args[1]
                self.assertEqual(args[args.index("--max-context-tokens") + 1], 3072)
                self.assertEqual(args[args.index("--max-new-tokens") + 1], 384)
            self.assertEqual(paper_cycle_ids("session", 1)[0], paper_cycle_ids("session", 240)[0])
