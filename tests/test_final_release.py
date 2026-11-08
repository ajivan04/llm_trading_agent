import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from agent.artifacts import atomic_json
from agent.contracts import content_hash
from agent.hydrate_news import heldout_hydration_plan, hydrate
from agent.release import DECISION_WEEK, EXECUTION_WEEK, FrozenRelease, freeze_release
from agent.release_cli import run_heldout
from tests.final_fixtures import release_inputs


class EmptyProvider:
    def fetch(self, *_args, **_kwargs):
        return ()


class ReleaseTests(unittest.TestCase):
    def test_copy_assets_verify_tamper_and_ignore_mutable_checkpoint_state_in_adapter_identity(
        self,
    ):
        from agent.policy.loader import adapter_fingerprint

        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            args, _ = release_inputs(root)
            before = adapter_fingerprint(args["adapter"])
            (args["adapter"] / "optimizer.pt").write_bytes(b"not inference")
            (args["adapter"] / ".env").write_text("DO NOT PACKAGE")
            self.assertEqual(adapter_fingerprint(args["adapter"]), before)
            release = freeze_release(**args, output=root / "release")
            self.assertFalse((release.path / "adapter/optimizer.pt").exists())
            self.assertFalse((release.path / "adapter/.env").exists())
            release.verify_runtime()
            (release.path / "adapter/adapter_model.safetensors").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                FrozenRelease(release.path)

    def test_incomplete_validation_and_wrong_training_provenance_cannot_freeze(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            args, _ = release_inputs(root)
            path = args["validation"] / "research_metrics.json"
            metrics = json.loads(path.read_text())
            metrics["complete_plan"] = False
            atomic_json(path, metrics)
            report_path = args["validation"] / "research_manifest.json"
            report = json.loads(report_path.read_text())
            report["metrics_hash"] = content_hash(metrics)
            atomic_json(report_path, report)
            with self.assertRaisesRegex(ValueError, "complete NLP validation"):
                freeze_release(**args, output=root / "release")
            self.assertFalse((root / "release").exists())

    def test_freeze_rejects_unbound_weights_and_unlisted_packaged_code(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            args, _ = release_inputs(root)
            provenance = args["adapter"] / "training_manifest.json"
            original = json.loads(provenance.read_text())
            changed = json.loads(provenance.read_text())
            changed["metrics"]["inference_adapter_fingerprint"] = "different"
            atomic_json(provenance, changed)
            with self.assertRaisesRegex(ValueError, "final-only supervision"):
                freeze_release(**args, output=root / "release")
            atomic_json(provenance, original)
            release = freeze_release(**args, output=root / "release")
            (release.path / "code/agent/unlisted.py").write_text("# unexpected module")
            with self.assertRaisesRegex(ValueError, "module set"):
                FrozenRelease(release.path)

    def test_exact_five_day_test_and_pending_attempt_resume_without_new_calls(self):
        from tests.test_trajectories import Policy

        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            args, base_policy = release_inputs(root)
            release = freeze_release(**args, output=root / "release")
            plan = heldout_hydration_plan(args["prepared"], release)
            test_source = root / "test-source"
            hydrate(
                args["prepared"],
                test_source,
                adapter=EmptyProvider(),
                plan=plan,
                frozen_release=release,
            )

            class Crash(Policy):
                def __call__(self, **kwargs):
                    if len(self.inputs) == 1:
                        raise KeyboardInterrupt("simulate disconnect")
                    return super().__call__(**kwargs)

            policy = Crash()
            policy.policy_identity = base_policy.policy_identity
            output, registry = root / "test-result", root / "assessment.sqlite"
            with self.assertRaises(KeyboardInterrupt):
                run_heldout(release, test_source, output, registry, decide=policy)
            with self.assertRaisesRegex(ValueError, "identical pending"):
                run_heldout(
                    release,
                    test_source,
                    root / "other-result",
                    registry,
                    resume=True,
                    decide=policy,
                )
            resumed = Policy()
            resumed.policy_identity = base_policy.policy_identity
            result = run_heldout(
                release, test_source, output, registry, resume=True, decide=resumed
            )
            self.assertEqual(len(resumed.inputs), 4)
            self.assertEqual([r["session"] for r in result.collection.records], list(DECISION_WEEK))
            self.assertEqual(
                [r["outcome_at"][:10] for r in result.collection.records], list(EXECUTION_WEEK)
            )
            self.assertEqual(result.metrics["test_policy_calls"], 5)
            self.assertEqual(result.metrics["starting_cash"], 100000)
            with self.assertRaisesRegex(ValueError, "consumed"):
                run_heldout(release, test_source, root / "another", registry, decide=resumed)
            self.assertEqual(len(resumed.inputs), 4)
