"""CPU-only gates for final-only SFT, exact resumes and compiled NLP PPO."""

import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent.contracts import canonical_json, content_hash
from agent.training.core import build_final_trajectory_sft_example
from agent.training.environment import HistoricalPPOEnvironment, create_training_environment
from agent.training.execution import (
    LoRASettings,
    TrainerSettings,
    _parameter_metrics,
    _sft_callback,
    _sft_resume_contract,
    train_sft,
)
from agent.training.ppo_prepare import compile_ppo
from agent.trajectories import TrajectorySettings, collect_trajectories, export_good_dataset
from tests.test_training import episode
from tests.test_training_environment import replay_episode
from tests.test_trajectories import Policy, bars
from tests.test_trajectory_cli import Calendar, prepared
from tests.test_trajectory_masks import NativeTokenizer


class FinalTrainingTests(unittest.TestCase):
    def test_only_final_completion_learns_despite_actual_tool_history(self):
        item = replace(
            episode(),
            messages=(
                {"role": "system", "content": "RULES"},
                {"role": "user", "content": "STATE"},
                {
                    "role": "assistant",
                    "content": '{"name":"get_rsi","arguments":{"symbol":"AAPL"}}',
                },
                {"role": "user", "content": "PRIVATE_TOOL_NEWS"},
            ),
        )
        built = build_final_trajectory_sft_example(item, NativeTokenizer(), max_length=4096)
        learned = "".join(chr(token) for token in built.labels if token != -100)
        self.assertEqual(learned, item.completion + "~")
        self.assertIn("get_rsi", "".join(chr(token) for token in built.input_ids))
        self.assertNotIn("get_rsi", learned)
        self.assertNotIn("PRIVATE_TOOL_NEWS", learned)

    def test_smoke_and_full_resume_have_same_scheduler_contract(self):
        with tempfile.TemporaryDirectory() as root:
            settings = TrainerSettings(
                Path(root) / "sft",
                "v1",
                ("data",),
                max_steps=8,
                stop_after_steps=2,
                supervision_mode="trajectory-final",
            )
            contract = _sft_resume_contract(settings, LoRASettings(), {"torch": "fake"})
            checkpoint = settings.output_dir / "checkpoint-2"
            checkpoint.mkdir()
            for name in (
                "optimizer.pt",
                "scheduler.pt",
                "rng_state.pth",
                "adapter_config.json",
                "adapter_model.safetensors",
            ):
                (checkpoint / name).write_bytes(b"complete")
            (checkpoint / "trainer_state.json").write_text('{"global_step":2}')
            callback = _sft_callback(SimpleNamespace(), settings, contract)
            control = SimpleNamespace(should_save=False, should_training_stop=False)
            callback.on_step_end(None, SimpleNamespace(global_step=2), control)
            self.assertTrue(control.should_save)
            self.assertTrue(control.should_training_stop)
            callback.on_save(None, SimpleNamespace(global_step=2), control)
            resumed = replace(
                settings, resume_from_checkpoint=str(checkpoint), stop_after_steps=None
            )
            self.assertEqual(
                _sft_resume_contract(resumed, LoRASettings(), {"torch": "fake"}), contract
            )
            with self.assertRaisesRegex(ValueError, "contract changed"):
                _sft_resume_contract(
                    replace(resumed, max_steps=100), LoRASettings(), {"torch": "fake"}
                )
            (checkpoint / "optimizer.pt").write_bytes(b"bad transfer")
            with self.assertRaisesRegex(ValueError, "checksum"):
                _sft_resume_contract(resumed, LoRASettings(), {"torch": "fake"})

    def test_token_failure_does_not_allocate_weights_or_strand_a_run(self):
        from tests.test_training_execution import FakeModel, example, stack

        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "output"
            settings = TrainerSettings(output, "prompt", ("data",), max_length=1)
            with patch.object(FakeModel, "from_pretrained") as load:
                with self.assertRaisesRegex(ValueError, "max_length"):
                    train_sft(
                        [example("train")],
                        [example("validation")],
                        settings=settings,
                        stack_resolver=stack,
                    )
                load.assert_not_called()
                self.assertFalse(output.exists())

    def test_parameter_fraction_is_measured_and_base_cannot_be_trainable(self):
        model = SimpleNamespace(
            parameters=lambda: iter(
                [
                    SimpleNamespace(numel=lambda: 1_000_000, requires_grad=False),
                    SimpleNamespace(numel=lambda: 500, requires_grad=True),
                ]
            )
        )
        metrics = _parameter_metrics(model)
        self.assertLess(metrics["trainable_parameter_fraction"], 0.001)
        self.assertEqual(metrics["trainable_parameters"], 500)
        with self.assertRaisesRegex(ValueError, "LoRA fraction"):
            _parameter_metrics(
                SimpleNamespace(
                    parameters=lambda: iter([SimpleNamespace(numel=lambda: 10, requires_grad=True)])
                )
            )

    def test_compile_nlp_ppo_uses_training_only_without_oracle_targets(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source = root / "source"
            source.mkdir()
            prepared(source)
            settings = TrajectorySettings(
                ("AAPL",), "2026-01-07T23:59:59Z", "2026-01-12T23:59:59Z", "2026-01-14T23:59:59Z"
            )
            with contextlib.redirect_stdout(io.StringIO()):
                train = collect_trajectories(bars(), decide=Policy(), settings=settings)
                validation = collect_trajectories(
                    bars(), decide=Policy(), settings=settings, split="validation"
                )
                export_good_dataset(train, validation, root / "good")
                args = SimpleNamespace(
                    prepared=source,
                    output=root / "ppo",
                    sft_dataset=root / "good/dataset",
                    sessions=2,
                    symbols=["AAPL"],
                    max_length=4096,
                    max_new_tokens=512,
                )
                with patch("agent.trajectory_cli._calendar", return_value=Calendar()):
                    result = compile_ppo(args, backend=SimpleNamespace(token_count=lambda _: 1))
            self.assertEqual(result["episodes"], 4)
            self.assertEqual(result["test_policy_calls"], 0)
            config = json.loads((root / "ppo/ppo-environment.json").read_text())
            with patch("agent.training.environment.critical_ppo_code_hashes", return_value={}):
                with self.assertRaisesRegex(ValueError, "integrity"):
                    create_training_environment(config)
            env = create_training_environment(config)
            self.assertTrue(all(e.enforce_risk and e.split == "train" for e in env.episodes))
            for item in env.episodes:
                self.assertLessEqual(item.outcome_at, settings.train_end)
                observed = canonical_json(env._observation())
                self.assertNotIn("next_open", observed)
                self.assertIn("Actual archived research text.", canonical_json(item.messages))
                self.assertIn("final-decision-only", canonical_json(item.messages))
            cash_only = next(item for item in env.episodes if item.starting_qty == 0)
            env = HistoricalPPOEnvironment((cash_only,))
            alias = cash_only.evidence_ids[0]
            action = canonical_json(
                {
                    "decisions": [
                        {
                            "symbol": "AAPL",
                            "action": "buy",
                            "qty": 50,
                            "reason": "given evidence",
                            "evidence_ids": [alias],
                        }
                    ]
                }
            )
            env.reset(seed=1)
            _, _, _, _, info = env.step(action)
            self.assertEqual(info["invalid_proposals"], 0)
            with self.assertRaisesRegex(ValueError, "actual prompt account"):
                replace(cash_only, starting_cash=cash_only.starting_cash - 1)
            env = HistoricalPPOEnvironment((replace(cash_only, max_qty=500),))
            env.reset(seed=1)
            oversized = json.loads(action)
            oversized["decisions"][0]["qty"] = 500
            _, _, _, _, info = env.step(canonical_json(oversized))
            self.assertEqual(info["invalid_proposals"], 1)
            self.assertTrue(info["failure_category"].startswith("risk_"))

    def test_new_risk_protocol_rejects_changed_account_and_oversized_position(self):
        # An explicit state proof is required before full risk can run.
        with self.assertRaisesRegex(ValueError, "decision-time close"):
            replay_episode(enforce_risk=True)

    def test_assignment_ppo_rejects_legacy_or_partial_sft_before_model_loading(self):
        from agent.training.cli import _assignment_ppo_preflight
        from agent.training.core import MODEL_NAME, MODEL_REVISION

        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            protocol = {
                "version": "final-decision-nlp-v1",
                "sft_dataset_hash": "good-dataset",
                "max_length": 4096,
                "reserved_output_tokens": 512,
            }
            protocol["protocol_hash"] = content_hash(protocol)
            (root / "protocol.json").write_text(canonical_json(protocol))
            (root / "environment.json").write_text(
                canonical_json({"protocol_path": str(root / "protocol.json")})
            )
            args = SimpleNamespace(
                environment_factory="agent.training.environment:create_training_environment",
                environment_config=root / "environment.json",
                prompt_version="final-decision-nlp-v1",
                max_length=4096,
                max_new_tokens=512,
            )
            provenance = {
                "method": "sft",
                "model_name": MODEL_NAME,
                "model_revision": MODEL_REVISION,
                "tokenizer_revision": MODEL_REVISION,
                "supervision_mode": "trajectory-final",
                "status": "completed",
                "completed_steps": 8,
                "planned_steps": 8,
                "trainable_parameter_fraction": 0.0007,
                "data_hashes": ["good-dataset"],
                "inference_weights_verified": True,
            }
            _assignment_ppo_preflight(args, provenance)
            with self.assertRaisesRegex(ValueError, "complete matching"):
                _assignment_ppo_preflight(args, {**provenance, "status": "partial"})
            with self.assertRaisesRegex(ValueError, "complete matching"):
                _assignment_ppo_preflight(args, {**provenance, "supervision_mode": "trajectory"})


class FinalArtifactGuardTests(unittest.TestCase):
    def test_normalized_source_old_labels_rejected_before_loading_stack(self):
        from agent.training.cli import main

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "dataset").mkdir()
            (root / "preparation-manifest.json").write_text(
                json.dumps({"old_training_targets_usable": False})
            )
            with patch("agent.training.cli.resolve_training_stack") as load:
                with self.assertRaisesRegex(ValueError, "stale training labels"):
                    main(
                        [
                            "sft",
                            "--train",
                            str(root / "dataset/train.jsonl"),
                            "--validation",
                            str(root / "dataset/validation.jsonl"),
                            "--output",
                            str(root / "model"),
                            "--prompt-version",
                            "test",
                        ]
                    )
                load.assert_not_called()

    def test_assignment_parent_requires_exact_completed_inference_weights(self):
        from agent.policy.loader import adapter_fingerprint
        from agent.training.qwen_ppo import initial_sft_provenance

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "adapter_config.json").write_text("{}")
            weights = root / "adapter_model.safetensors"
            weights.write_bytes(b"completed-weights")
            manifest = {"adapter_config": {"supervision_mode": "trajectory-final"}}
            (root / "training_manifest.json").write_text(
                json.dumps(
                    {
                        "manifest": manifest,
                        "manifest_hash": content_hash(manifest),
                        "metrics": {"inference_adapter_fingerprint": adapter_fingerprint(root)},
                    }
                )
            )
            self.assertTrue(initial_sft_provenance(root)["inference_weights_verified"])
            weights.write_bytes(b"different-weights")
            with self.assertRaisesRegex(ValueError, "inference weights"):
                initial_sft_provenance(root)
