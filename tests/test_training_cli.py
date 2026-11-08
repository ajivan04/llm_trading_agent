import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent.training.cli import _episode_example


class TrainingCliTests(unittest.TestCase):
    def test_json_messages_are_normalized_to_tuple(self):
        row = {
            "example_id": "example-1",
            "split": "train",
            "decision_at": "2026-01-01T00:00:00Z",
            "outcome_at": "2026-01-02T00:00:00Z",
            "snapshot_id": "snapshot",
            "account_state_hash": "account",
            "evidence_hash": "evidence",
            "messages": [
                {"role": "system", "content": "rules"},
                {"role": "user", "content": "state"},
            ],
            "completion": "decision",
        }
        example = _episode_example(row)
        self.assertIsInstance(example.messages, tuple)
        self.assertEqual(len(example.messages), 2)


class MinimalTrainingStackTests(unittest.TestCase):
    def modules(self):
        return {
            "torch": SimpleNamespace(optim=object()),
            "transformers": SimpleNamespace(
                AutoTokenizer=object(),
                AutoModelForCausalLM=object(),
                Trainer=object(),
                TrainingArguments=object(),
                DataCollatorForSeq2Seq=object(),
            ),
            "peft": SimpleNamespace(
                LoraConfig=object(), get_peft_model=object(), PeftModel=object()
            ),
            "datasets": SimpleNamespace(Dataset=object()),
            "trl": SimpleNamespace(DPOTrainer=object(), DPOConfig=object()),
        }

    def test_each_method_imports_only_its_actual_dependencies(self):
        from agent.training.integrations import resolve_training_stack

        for method, packages, trainer in (
            ("sft", {"torch", "transformers", "peft", "datasets"}, "Trainer"),
            ("ppo", {"torch", "transformers", "peft"}, "optim"),
            ("dpo", {"torch", "transformers", "peft", "datasets", "trl"}, "DPOTrainer"),
        ):
            catalog = self.modules()
            with (
                self.subTest(method=method),
                patch("agent.training.integrations.metadata.version", return_value="x"),
                patch(
                    "agent.training.integrations.importlib.import_module",
                    side_effect=lambda name, modules=catalog: modules[name],
                ) as imported,
            ):
                result = resolve_training_stack(method, expected_versions={"torch": "x"})
                self.assertEqual(set(result["modules"]), packages)
                self.assertEqual(set(result["versions"]), packages)
                self.assertEqual(result["trainer"], trainer)
                self.assertEqual({call.args[0] for call in imported.call_args_list}, packages)

    def test_required_surface_and_pinned_versions_still_fail_closed(self):
        from agent.training.integrations import TrainingStackUnavailable, resolve_training_stack

        catalog = self.modules()
        del catalog["transformers"].Trainer
        with (
            patch("agent.training.integrations.metadata.version", return_value="x"),
            patch(
                "agent.training.integrations.importlib.import_module",
                side_effect=catalog.__getitem__,
            ),
        ):
            with self.assertRaisesRegex(TrainingStackUnavailable, "required Trainer"):
                resolve_training_stack("sft")
            with self.assertRaisesRegex(TrainingStackUnavailable, "version mismatch"):
                resolve_training_stack("ppo", expected_versions={"torch": "wrong"})

    def test_cli_separates_checkpoint_and_loss_evaluation_cadences(self):
        from agent.training.cli import _settings, build_parser
        from agent.training.execution import _argument_values

        args = build_parser().parse_args(
            [
                "sft",
                "--train",
                "train.jsonl",
                "--validation",
                "validation.jsonl",
                "--output",
                "out",
                "--prompt-version",
                "test",
                "--save-steps",
                "2",
                "--eval-steps",
                "30",
            ]
        )
        values = _argument_values(_settings(args, ("data",)), evaluate=True)
        self.assertEqual(values["save_steps"], 2)
        self.assertEqual(values["eval_steps"], 30)
        self.assertFalse(values["load_best_model_at_end"])
