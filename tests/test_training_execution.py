import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from agent.contracts import content_hash
from agent.training import (
    MODEL_REVISION,
    DPOPair,
    EpisodeExample,
    PPORollout,
    PPOTransition,
    QwenPPOOptimizer,
    QwenPPOSettings,
    TrainerSettings,
    train_dpo,
    train_ppo,
    train_sft,
)
from agent.training.cli import (
    _actual_data_hashes,
    _audited_dpo_pair,
    _verify_sft_manifest,
    build_parser,
)
from agent.training.qwen_ppo import _load_reference_policy


class FakeTokenizer:
    pad_token_id = None
    pad_token = None
    eos_token = "<eos>"

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.loaded = (args, kwargs)
        return cls()

    def apply_chat_template(self, messages, **kwargs):
        if kwargs.get("tokenize") is False:
            return "<prompt>"
        return {
            "input_ids": [1, 2, 3], "attention_mask": [1, 1, 1],
            "assistant_masks": [0, 1, 1],
        }

    def save_pretrained(self, output):
        Path(output).mkdir(parents=True, exist_ok=True)
        (Path(output) / "tokenizer.fake").write_text("ok", encoding="utf-8")


class FakeModel:
    config = SimpleNamespace(use_cache=True)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.loaded = (args, kwargs)
        return cls()


class FakeDataset:
    @classmethod
    def from_list(cls, rows):
        return list(rows)


class FakeArguments:
    def __init__(self, **kwargs):
        self.values = kwargs


class FakeCollator:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeTrainer:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.__class__.instances.append(self)

    def train(self, resume_from_checkpoint=None):
        self.resume = resume_from_checkpoint
        return SimpleNamespace(metrics={"loss": 0.25})

    def save_model(self, output):
        Path(output).mkdir(parents=True, exist_ok=True)
        (Path(output) / "adapter.fake").write_text("ok", encoding="utf-8")

    def save_state(self):
        self.state_saved = True


class FakeDPOConfig(FakeArguments):
    pass


class FakeTorch:
    seeds = []
    cuda = SimpleNamespace(
        manual_seed_all=lambda seed: FakeTorch.seeds.append(seed),
        is_available=lambda: False,
    )

    @classmethod
    def manual_seed(cls, seed):
        cls.seeds.append(seed)


def stack(_method, **_kwargs):
    transformers = SimpleNamespace(
        AutoTokenizer=FakeTokenizer, AutoModelForCausalLM=FakeModel,
        TrainingArguments=FakeArguments, Trainer=FakeTrainer,
        DataCollatorForSeq2Seq=FakeCollator,
    )
    peft = SimpleNamespace(
        LoraConfig=lambda **kwargs: kwargs,
        get_peft_model=lambda model, config: model,
        PeftModel=SimpleNamespace(from_pretrained=lambda model, path, is_trainable: model),
    )
    trl = SimpleNamespace(DPOConfig=FakeDPOConfig, DPOTrainer=FakeTrainer)
    return {
        "modules": {"torch": FakeTorch, "transformers": transformers, "peft": peft,
                    "datasets": SimpleNamespace(Dataset=FakeDataset), "trl": trl},
        "versions": {"torch": "x", "transformers": "x", "peft": "x",
                     "datasets": "x", "trl": "x"},
        "trainer": "fake",
    }


def example(split):
    return EpisodeExample(
        f"{split}-1", split, "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z",
        "snapshot", "account", "evidence", ({"role": "user", "content": "state"},),
        "decision",
    )


class ExecutionTests(unittest.TestCase):
    def settings(self, output):
        return TrainerSettings(
            Path(output), "prompt-v1", ("data-hash",), "model-rev", "token-rev",
            max_steps=2, max_length=8, save_steps=1, logging_steps=1,
        )

    def test_sft_trains_masked_rows_and_writes_manifest(self):
        with tempfile.TemporaryDirectory() as output:
            result = train_sft(
                [example("train")], [example("validation")],
                settings=self.settings(output), stack_resolver=stack,
            )
            self.assertEqual(result.method, "sft")
            row = FakeTrainer.instances[-1].kwargs["train_dataset"][0]
            self.assertEqual(row["labels"], [-100, 2, 3])
            manifest = json.loads((Path(output) / "training_manifest.json").read_text())
            self.assertEqual(manifest["manifest"]["model_name"],
                             "Qwen/Qwen2.5-1.5B-Instruct")
            self.assertNotIn("secret", json.dumps(manifest).lower())
            trainer = FakeTrainer.instances[-1]
            self.assertTrue(trainer.state_saved)
            self.assertEqual(
                trainer.kwargs["data_collator"].kwargs["label_pad_token_id"], -100)
            self.assertTrue(trainer.kwargs["args"].values["load_best_model_at_end"])

    def test_dpo_uses_same_state_pair_rows(self):
        pair = DPOPair(
            "prompt", "chosen", "rejected", "state", 0.2, "a", "b",
            audit_hash="audit")
        validation_pair = DPOPair(
            "prompt", "chosen", "rejected", "state", 0.2, "a", "b",
            "validation", "audit-validation")
        with tempfile.TemporaryDirectory() as output:
            result = train_dpo(
                [pair], [validation_pair], settings=self.settings(output),
                stack_resolver=stack,
            )
            self.assertEqual(result.method, "dpo")
            row = FakeTrainer.instances[-1].kwargs["train_dataset"][0]
            self.assertEqual(row, {"prompt": "prompt", "chosen": "chosen",
                                   "rejected": "rejected"})

    def test_dpo_renders_serialized_chat_prompt_for_trl(self):
        prompt = json.dumps([{"role": "user", "content": "state"}])
        pair = DPOPair(prompt, "chosen", "rejected", "state", 0.2, "a", "b",
                       audit_hash="audit")
        validation_pair = DPOPair(
            prompt, "chosen", "rejected", "state", 0.2, "a", "b",
            "validation", "audit-validation")
        with tempfile.TemporaryDirectory() as output:
            train_dpo(
                [pair], [validation_pair], settings=self.settings(output),
                stack_resolver=stack,
            )
            row = FakeTrainer.instances[-1].kwargs["train_dataset"][0]
            self.assertEqual(row["prompt"], "<prompt>")

    def test_dpo_split_is_enforced_at_trainer_boundary(self):
        train_pair = DPOPair(
            "prompt", "chosen", "rejected", "state", 0.2, "a", "b",
            audit_hash="audit")
        validation_pair = DPOPair(
            "prompt", "chosen", "rejected", "state", 0.2, "a", "b", "validation",
            "audit-validation")
        with tempfile.TemporaryDirectory() as output:
            with self.assertRaisesRegex(ValueError, "training data"):
                train_dpo(
                    [validation_pair], [], settings=self.settings(output),
                    stack_resolver=stack)
            with self.assertRaisesRegex(ValueError, "validation data"):
                train_dpo(
                    [train_pair], [train_pair], settings=self.settings(output),
                    stack_resolver=stack)

    def test_dpo_rejects_unaudited_pairs(self):
        pair = DPOPair("prompt", "chosen", "rejected", "state", 0.2, "a", "b")
        with tempfile.TemporaryDirectory() as output:
            with self.assertRaisesRegex(ValueError, "audited"):
                train_dpo([pair], [], settings=self.settings(output), stack_resolver=stack)

    def test_wrong_sft_split_is_rejected(self):
        with tempfile.TemporaryDirectory() as output:
            with self.assertRaisesRegex(ValueError, "training data"):
                train_sft(
                    [example("validation")], [example("validation")],
                    settings=self.settings(output),
                    stack_resolver=stack,
                )

    def test_training_defaults_pin_the_verified_qwen_revision(self):
        with tempfile.TemporaryDirectory() as output:
            settings = TrainerSettings(Path(output), "prompt", ("data",))
        self.assertEqual(settings.model_revision, MODEL_REVISION)
        self.assertEqual(settings.tokenizer_revision, MODEL_REVISION)
        with self.assertRaises(ValueError):
            QwenPPOSettings(temperature=0)

    def test_cli_exposes_ppo_and_gpu_smoke_commands(self):
        ppo = build_parser().parse_args([
            "ppo", "--output", "out", "--prompt-version", "v1",
            "--adapter-checkpoint", "adapter",
            "--environment-factory", "package.module:create", "--updates", "2",
        ])
        self.assertEqual(ppo.method, "ppo")
        smoke = build_parser().parse_args([
            "smoke", "--adapter-checkpoint", "adapter", "--output", "out",
        ])
        self.assertEqual(smoke.method, "smoke")

    def test_cli_hashes_actual_training_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            train = Path(directory, "train.jsonl")
            validation = Path(directory, "validation.jsonl")
            train.write_text("train", encoding="utf-8")
            validation.write_text("validation", encoding="utf-8")
            args = SimpleNamespace(
                method="sft", train=train, validation=validation,
                environment_config=None,
            )
            hashes = _actual_data_hashes(args)
            self.assertEqual(len(hashes), 2)

    def test_sft_manifest_is_recomputed_not_trusted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_rows = [{"example_id": "train", "split": "train"}]
            validation_rows = [{"example_id": "validation", "split": "validation"}]
            all_rows = [*train_rows, *validation_rows]
            split_ids = {"train": ["train"], "validation": ["validation"], "test": []}
            identity = {
                "examples": all_rows, "splits": split_ids, "purged": [],
                "source_hashes": ["source"],
                "boundaries": ("train-end", "validation-end", "test-end"),
            }
            dataset_hash = content_hash(identity)
            manifest = {
                "dataset_hash": dataset_hash, "split_example_ids": split_ids,
                "purged_episode_ids": [], "source_hashes": ["source"],
                "train_end": "train-end", "validation_end": "validation-end",
                "test_end": "test-end",
            }
            for name, rows in (("train", train_rows), ("validation", validation_rows),
                               ("examples", all_rows)):
                Path(root, f"{name}.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            Path(root, "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            _verify_sft_manifest(
                root / "train.jsonl", root / "validation.jsonl",
                train_rows, validation_rows)
            all_rows[0]["tampered"] = True
            Path(root, "examples.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in all_rows), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash verification"):
                _verify_sft_manifest(
                    root / "train.jsonl", root / "validation.jsonl",
                    train_rows, validation_rows)

    def test_cli_requires_recomputable_dpo_audit(self):
        pair = {
            "prompt": "prompt", "chosen": "buy", "rejected": "hold",
            "state_hash": "state", "preference_margin": 0.1,
            "chosen_id": "buy", "rejected_id": "hold", "split": "train",
            "audit_hash": None,
        }
        chosen = {
            "reset_id": "reset-buy", "simulator_hash": "sim",
            "horizon_end": "2026-01-05T00:00:00+00:00", "candidate": {"id": "buy"},
        }
        rejected = {
            "reset_id": "reset-hold", "simulator_hash": "sim",
            "horizon_end": "2026-01-05T00:00:00+00:00", "candidate": {"id": "hold"},
        }
        identity = {
            "state_hash": "state", "simulator_hash": "sim",
            "horizon_end": "2026-01-05T00:00:00+00:00",
            "outcomes": (chosen, rejected),
        }
        audit_hash = content_hash(identity)
        row = {"pair": pair, "audit_hash": audit_hash,
               "chosen_outcome": chosen, "rejected_outcome": rejected}
        self.assertEqual(_audited_dpo_pair(row, "train").audit_hash, audit_hash)
        with self.assertRaisesRegex(ValueError, "audit_hash"):
            _audited_dpo_pair({**row, "audit_hash": "forged"}, "train")

    def test_ppo_reference_loads_the_same_sft_adapter_frozen(self):
        calls = []
        peft = SimpleNamespace(PeftModel=SimpleNamespace(
            from_pretrained=lambda model, path, is_trainable:
                calls.append((path, is_trainable)) or model))
        modules = stack("ppo")["modules"]
        modules["peft"] = peft
        with tempfile.TemporaryDirectory() as adapter:
            settings = TrainerSettings(
                Path(adapter, "out"), "prompt", ("data",), adapter_checkpoint=Path(adapter))
            _load_reference_policy({"modules": modules}, settings)
        self.assertEqual(calls, [(adapter, False)])


class FakePPOOptimizer:
    def __init__(self):
        self.version = "policy-0"
        self.collected = []

    @property
    def policy_version(self):
        return self.version

    def collect(self, environment, *, rollout_id, seed):
        self.collected.append((rollout_id, seed))
        item = PPOTransition(
            rollout_id, self.version, "observation", (1, 2), (0, 1),
            (-0.5, -0.5), (-0.6, -0.6),
            0.0, 1.0, 0.0, done=True,
        )
        return PPORollout(rollout_id, self.version, "2026-01-01T00:00:00Z", (item,))

    def update(self, rollout):
        self.version = f"policy-{len(self.collected)}"
        return {"loss": 0.1}

    def save(self, output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "ppo.fake").write_text("ok", encoding="utf-8")

    def load(self, checkpoint):
        return 0


class PPOExecutionTests(unittest.TestCase):
    def test_ppo_collects_fresh_rollout_for_every_update(self):
        optimizer = FakePPOOptimizer()
        with tempfile.TemporaryDirectory() as output:
            settings = TrainerSettings(
                Path(output), "prompt", ("data",), "model", "token", max_steps=2,
            )
            result = train_ppo(
                object(), optimizer, settings=settings, updates=2,
                package_versions={"trl": "x"}, clock=iter([1.0, 2.5]).__next__,
            )
            self.assertEqual(result.method, "ppo")
            self.assertEqual(len({item[0] for item in optimizer.collected}), 2)
            payload = json.loads((Path(output) / "training_manifest.json").read_text())
            self.assertEqual(len(payload["rollout_ids"]), 2)


class TinyTensor:
    def __init__(self, values):
        self.values = values

    @property
    def ndim(self):
        return 2 if self.values and isinstance(self.values[0], list) else 1

    @property
    def shape(self):
        return ((len(self.values), len(self.values[0])) if self.ndim == 2
                else (len(self.values),))

    def to(self, _device):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values

    def __getitem__(self, index):
        if isinstance(index, tuple):
            row, column = index
            values = self.values[row]
            return TinyTensor(values[column]) if isinstance(column, slice) else values[column]
        value = self.values[index]
        return TinyTensor(value) if isinstance(value, list) else value


class NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class TinyTorch:
    cuda = SimpleNamespace(manual_seed_all=lambda _seed: None)

    @staticmethod
    def manual_seed(_seed):
        pass

    @staticmethod
    def no_grad():
        return NullContext()

    @staticmethod
    def ones_like(tensor):
        return TinyTensor([[1 for _ in tensor.values[0]]])

    @staticmethod
    def save(payload, path):
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @staticmethod
    def load(path, **_kwargs):
        return json.loads(Path(path).read_text(encoding="utf-8"))


class TinyPolicy:
    device = "cpu"

    def __init__(self):
        self.generate_kwargs = None

    def requires_grad_(self, _enabled):
        return self

    def eval(self):
        return self

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return TinyTensor([[1, 2, 3, 4]])

    def save_pretrained(self, output):
        Path(output).mkdir(parents=True, exist_ok=True)
        Path(output, "adapter.fake").write_text("updated", encoding="utf-8")


class TinyTokenizer:
    eos_token_id = 0

    def apply_chat_template(self, *_args, **_kwargs):
        return TinyTensor([[1, 2]])

    def decode(self, ids, **_kwargs):
        return "hold" if ids.tolist() == [3, 4] else "bad"

    def save_pretrained(self, output):
        Path(output).mkdir(parents=True, exist_ok=True)
        Path(output, "tokenizer.fake").write_text("tokenizer", encoding="utf-8")


class BatchEncodingTokenizer(TinyTokenizer):
    def apply_chat_template(self, *args, **kwargs):
        return {"input_ids": TinyTensor([[1, 2]])}


class TinyState:
    def __init__(self, payload):
        self.payload = payload

    def state_dict(self):
        return self.payload

    def load_state_dict(self, payload):
        self.payload = payload


class TinyQwenPPO(QwenPPOOptimizer):
    def _statistics(self, model, token_ids, attention_mask, *, with_values):
        width = token_ids.shape[-1]
        values = TinyTensor([[0.1] * width]) if with_values else None
        return TinyTensor([[-0.5] * width]), values, TinyTensor([[0.2] * width])

    def _state_value(self, observation):
        return 0.25


class OneStepEnvironment:
    def reset(self, *, seed):
        return {"messages": [{"role": "user", "content": "state"}]}

    def step(self, action):
        self.action = action
        return ({"messages": [{"role": "user", "content": "next"}]},
                1.0, True, False, {})


class ConcretePPOCollectionTests(unittest.TestCase):
    def test_qwen_collection_is_stochastic_and_reconstructible(self):
        with tempfile.TemporaryDirectory() as adapter:
            Path(adapter, "adapter.fake").write_text("sft", encoding="utf-8")
            settings = TrainerSettings(
                Path(adapter, "out"), "prompt", ("data",), adapter_checkpoint=Path(adapter))
            policy = TinyPolicy()
            optimizer = TinyQwenPPO(
                torch_module=TinyTorch(), tokenizer=TinyTokenizer(), policy_model=policy,
                reference_model=TinyPolicy(), value_head=object(), optimizer=object(),
                settings=settings, ppo=QwenPPOSettings(max_episode_steps=1),
            )
            environment = OneStepEnvironment()
            rollout = optimizer.collect(environment, rollout_id="rollout", seed=7)
            sample = rollout.transitions[0]
            self.assertTrue(policy.generate_kwargs["do_sample"])
            self.assertEqual(sample.prompt_token_ids, (1, 2))
            self.assertEqual(sample.action_token_ids, (3, 4))
            self.assertEqual(environment.action, "hold")
            self.assertTrue(sample.done)

    def test_qwen_accepts_batch_encoding_chat_template_result(self):
        with tempfile.TemporaryDirectory() as adapter:
            Path(adapter, "adapter.fake").write_text("sft", encoding="utf-8")
            settings = TrainerSettings(
                Path(adapter, "out"), "prompt", ("data",), adapter_checkpoint=Path(adapter))
            optimizer = TinyQwenPPO(
                torch_module=TinyTorch(), tokenizer=BatchEncodingTokenizer(),
                policy_model=TinyPolicy(), reference_model=TinyPolicy(),
                value_head=object(), optimizer=object(), settings=settings,
                ppo=QwenPPOSettings(max_episode_steps=1),
            )
            prompt, _ = optimizer._prompt({"messages": [{"role": "user", "content": "state"}]})
            self.assertEqual(prompt.shape[0], 1)

    def test_qwen_collection_rejects_exhausted_context(self):
        with tempfile.TemporaryDirectory() as adapter:
            Path(adapter, "adapter.fake").write_text("sft", encoding="utf-8")
            settings = TrainerSettings(
                Path(adapter, "out"), "prompt", ("data",), max_length=2,
                adapter_checkpoint=Path(adapter))
            optimizer = TinyQwenPPO(
                torch_module=TinyTorch(), tokenizer=TinyTokenizer(),
                policy_model=TinyPolicy(), reference_model=TinyPolicy(),
                value_head=object(), optimizer=object(), settings=settings,
                ppo=QwenPPOSettings(max_episode_steps=1),
            )
            with self.assertRaisesRegex(ValueError, "context window"):
                optimizer.collect(OneStepEnvironment(), rollout_id="rollout", seed=7)

    def test_consumed_rollouts_survive_checkpoint_reload(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary, "sft")
            source.mkdir()
            Path(source, "adapter.fake").write_text("sft", encoding="utf-8")
            output = Path(temporary, "checkpoint")
            settings = TrainerSettings(
                output, "prompt", ("data",), adapter_checkpoint=source)
            first = TinyQwenPPO(
                torch_module=TinyTorch(), tokenizer=TinyTokenizer(),
                policy_model=TinyPolicy(), reference_model=TinyPolicy(),
                value_head=TinyState({"value": 1}), optimizer=TinyState({"optimizer": 2}),
                settings=settings, ppo=QwenPPOSettings(),
            )
            first._consumed_rollout_ids.add("used-rollout")
            first._update_index = 3
            first.save(output)
            resumed_settings = TrainerSettings(
                output, "prompt", ("data",), adapter_checkpoint=output / "adapter",
                reference_adapter_checkpoint=output / "reference_adapter")
            second = TinyQwenPPO(
                torch_module=TinyTorch(), tokenizer=TinyTokenizer(),
                policy_model=TinyPolicy(), reference_model=TinyPolicy(),
                value_head=TinyState({}), optimizer=TinyState({}),
                settings=resumed_settings, ppo=QwenPPOSettings(),
            )
            self.assertEqual(second.load(output), 3)
            self.assertIn("used-rollout", second.consumed_rollout_ids)
            mismatched_settings = TrainerSettings(
                output, "different-prompt", ("data",),
                adapter_checkpoint=output / "adapter",
                reference_adapter_checkpoint=output / "reference_adapter")
            mismatched = TinyQwenPPO(
                torch_module=TinyTorch(), tokenizer=TinyTokenizer(),
                policy_model=TinyPolicy(), reference_model=TinyPolicy(),
                value_head=TinyState({}), optimizer=TinyState({}),
                settings=mismatched_settings, ppo=QwenPPOSettings(),
            )
            with self.assertRaisesRegex(ValueError, "run contract"):
                mismatched.load(output)


if __name__ == "__main__":
    unittest.main()
