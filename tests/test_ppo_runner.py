"""Operational PPO gates use deterministic CPU fakes; numerical tests use real Torch."""
import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent.training import PPORollout, PPOTransition, TrainerSettings, train_ppo
from agent.training.cli import build_parser, main
from agent.training.execution import PPOTrainingLimits
from agent.training.validation import preflight_ppo_context


class Optimizer:
    def __init__(self, output, *, invalid=0, kl=0, fail_at=None):
        self.output = Path(output)
        self.index = 0
        self.collected = 0
        self.invalid, self.kl, self.fail_at = invalid, kl, fail_at
        self.ppo = SimpleNamespace(target_kl=0.1, max_invalid_fraction=0.5)
        self._run_contract = {"implementation_version": "test"}

    @property
    def policy_version(self):
        return f"policy-{self.index}"

    def collect(self, _environment, *, rollout_id, seed):
        # A previous successful update must already be visible on disk.
        if self.collected:
            history = self.output / "ppo_metrics.jsonl"
            assert history.is_file()
            assert len(history.read_text().splitlines()) == self.index
        self.collected += 1
        transition = PPOTransition(
            rollout_id, self.policy_version, "state", (1, 2), (0, 1),
            (0, -0.5), (0, -0.5), 0, 0.01, 0, done=True,
            reward_components={"invalid_proposals": self.invalid},
        )
        return PPORollout(rollout_id, self.policy_version, "2026-01-01T00:00:00Z", (transition,))

    def update(self, rollout):
        if self.fail_at == self.index:
            raise ValueError("injected numerical failure")
        self.index += 1
        return {"loss": 0.1, "reference_kl": self.kl}

    def save(self, output):
        path = Path(output)
        path.mkdir(parents=True, exist_ok=True)
        (path / "ppo_state.json").write_text(json.dumps({"index": self.index}))

    def load(self, checkpoint):
        self.index = json.loads((Path(checkpoint) / "ppo_state.json").read_text())["index"]
        return self.index


class PPORunnerTests(unittest.TestCase):
    def settings(self, output, updates=2):
        return TrainerSettings(Path(output), "test", ("data",), max_steps=updates, save_steps=1)

    def run_training(self, optimizer, settings, updates, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return train_ppo(object(), optimizer, settings=settings, updates=updates,
                             package_versions={"torch": "test"}, clock=lambda: 1., **kwargs)

    def test_resume_targets_total_updates_and_preserves_incremental_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Optimizer(directory)
            self.run_training(first, self.settings(directory), 2)
            settings = replace(self.settings(directory, 3), resume_from_checkpoint=directory)
            resumed = Optimizer(directory)
            result = self.run_training(resumed, settings, 3)
            self.assertEqual(resumed.collected, 1)
            self.assertEqual(result.metrics["completed_updates"], 3)
            self.assertEqual(result.metrics["updates_this_invocation"], 1)
            records = [json.loads(line) for line in
                       (Path(directory) / "ppo_metrics.jsonl").read_text().splitlines()]
            self.assertEqual([item["update"] for item in records], [1, 2, 3])
            self.assertEqual(len(list(Path(directory).glob("completed_manifest-*.json"))), 1)

    def test_invalid_rollouts_stop_without_false_completion_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            optimizer = Optimizer(directory, invalid=1)
            result = self.run_training(optimizer, self.settings(directory, 10), 10)
            self.assertEqual(result.metrics["status"], "stopped")
            self.assertEqual(result.metrics["stop_reason"], "invalid_proposals")
            self.assertEqual(result.metrics["completed_updates"], 1)
            self.assertEqual(optimizer.collected, 2)
            self.assertFalse((Path(directory) / "training_manifest.json").exists())
            with self.assertRaisesRegex(ValueError, "failed a safety gate"):
                self.run_training(Optimizer(directory), replace(
                    self.settings(directory, 10), resume_from_checkpoint=directory), 10)

    def test_invalid_streak_survives_pilot_continuation(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_training(Optimizer(directory, invalid=1), self.settings(directory, 1), 1)
            resumed = Optimizer(directory, invalid=1)
            settings = replace(self.settings(directory, 3), resume_from_checkpoint=directory)
            result = self.run_training(resumed, settings, 3)
            self.assertEqual(result.metrics["stop_reason"], "invalid_proposals")
            self.assertEqual(result.metrics["updates_this_invocation"], 0)
            self.assertFalse((Path(directory) / "training_manifest.json").exists())

    def test_kl_stop_and_failure_preserve_status_without_completion(self):
        for failure in (False, True):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                optimizer = Optimizer(directory, kl=0.2, fail_at=0 if failure else None)
                if failure:
                    with self.assertRaisesRegex(ValueError, "injected"):
                        self.run_training(optimizer, self.settings(directory), 2)
                else:
                    result = self.run_training(optimizer, self.settings(directory), 2)
                    self.assertEqual(result.metrics["stop_reason"], "reference_kl")
                self.assertFalse((Path(directory) / "training_manifest.json").exists())
                status = json.loads((Path(directory) / "training_status.json").read_text())
                self.assertIn(status["status"], ("failed", "stopped"))

    def test_runtime_budget_is_checked_before_collection(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            optimizer = Optimizer(directory)
            ticks = iter([0., 2., 2.])
            result = train_ppo(object(), optimizer, settings=self.settings(directory), updates=2,
                               package_versions={}, limits=PPOTrainingLimits(max_seconds=1),
                               clock=ticks.__next__)
            self.assertEqual(optimizer.collected, 0)
            self.assertEqual(result.metrics["stop_reason"], "runtime_budget")

    def test_legacy_checkpoint_is_rejected_before_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "ppo_state.json").write_text('{"update_index": 1}')
            with patch("agent.training.cli.QwenPPOOptimizer.from_pretrained") as load:
                with self.assertRaisesRegex(ValueError, "legacy PPO checkpoint"):
                    main(["ppo", "--resume-from-checkpoint", directory, "--output", directory,
                          "--prompt-version", "test", "--updates", "10",
                          "--environment-factory",
                          "agent.training.environment:create_training_environment"])
                load.assert_not_called()

    def test_defaults_and_finite_learning_rate(self):
        args = build_parser().parse_args([
            "ppo", "--output", "unused", "--prompt-version", "test", "--updates", "10",
            "--environment-factory", "module:factory",
        ])
        self.assertEqual(args.learning_rate, 1e-5)
        self.assertEqual(args.rollout_episodes, 8)
        self.assertEqual(args.temperature, 1)
        for invalid in (float("nan"), float("inf"), 0):
            with self.subTest(value=invalid), self.assertRaisesRegex(ValueError, "learning_rate"):
                replace(self.settings("unused"), learning_rate=invalid)

    def test_context_preflight_checks_late_episodes_without_silent_truncation(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                self.kwargs = kwargs
                return {"input_ids": list(range(messages[0]["tokens"]))}

        tokenizer = Tokenizer()
        environment = SimpleNamespace(episodes=(
            SimpleNamespace(episode_id="short", messages=[{"tokens": 2}]),
            SimpleNamespace(episode_id="long", messages=[{"tokens": 8}]),
        ))
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "long.*8 input tokens"):
                preflight_ppo_context(environment, tokenizer, max_length=10, max_new_tokens=3)
        self.assertFalse(tokenizer.kwargs["truncation"])
        self.assertEqual(len(environment.episodes), 2)

    def test_oversized_replay_stops_cli_before_model_loading(self):
        tokenizer = SimpleNamespace(apply_chat_template=lambda *args, **kwargs:
                                    {"input_ids": [1, 2, 3]})
        stack = {"modules": {"transformers": SimpleNamespace(AutoTokenizer=SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: tokenizer,
        ))}, "versions": {}}
        environment = SimpleNamespace(episodes=(SimpleNamespace(episode_id="long", messages=[]),))
        with (patch("agent.training.cli._actual_data_hashes", return_value=("data",)),
              patch("agent.training.cli._environment", return_value=environment),
              patch("agent.training.cli.resolve_training_stack", return_value=stack),
              patch("agent.training.cli.QwenPPOOptimizer.from_pretrained") as load,
              contextlib.redirect_stdout(io.StringIO())):
            with self.assertRaisesRegex(ValueError, "before allocating model weights"):
                main(["ppo", "--adapter-checkpoint", "sft", "--output", "unused",
                      "--prompt-version", "test", "--updates", "2", "--max-length", "4",
                      "--max-new-tokens", "3", "--environment-factory", "module:factory"])
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
