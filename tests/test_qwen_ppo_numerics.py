"""Small real-Qwen/PEFT numerical PPO tests; never download model weights."""
import copy
import importlib.util
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from agent.training.execution import TrainerSettings
from agent.training.ppo_math import categorical_action_statistics
from agent.training.qwen_ppo import QwenPPOOptimizer, QwenPPOSettings

AVAILABLE = all(importlib.util.find_spec(name) for name in ("torch", "transformers", "peft"))


class Tokenizer:
    eos_token_id = 2

    def apply_chat_template(self, *_args, **_kwargs):
        import torch
        return torch.tensor([[3, 4, 5]])

    def decode(self, ids, **_kwargs):
        return json.dumps(ids.tolist())

    def save_pretrained(self, output):
        Path(output).mkdir(parents=True, exist_ok=True)
        (Path(output) / "tokenizer-test.json").write_text('{"test":true}')


class Environment:
    def __init__(self):
        self.seeds = []

    def reset(self, *, seed):
        self.seeds.append(seed)
        return {"messages": [{"role": "user", "content": "test state"}]}

    def step(self, _action):
        return ({"messages": [{"role": "user", "content": "terminal"}]},
                0.01 + len(self.seeds) * 0.001, True, False,
                {"reward_components": {"invalid_proposals": 0.0}})


@unittest.skipUnless(AVAILABLE, "optional Torch/Transformers/PEFT numerical stack")
class QwenPPONumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        cls.torch.set_num_threads(cls.threads)

    def base(self):
        from transformers import Qwen2Config, Qwen2ForCausalLM
        self.torch.manual_seed(77)
        return Qwen2ForCausalLM(Qwen2Config(
            vocab_size=24, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=128,
            attention_dropout=0.3, eos_token_id=2, bos_token_id=1, pad_token_id=2,
        ))

    def optimizer(self, root, *, resume=None, **ppo_changes):
        from peft import LoraConfig, PeftModel, get_peft_model
        torch = self.torch
        if resume is None:
            policy = get_peft_model(self.base(), LoraConfig(
                r=2, lora_alpha=4, lora_dropout=0.5, target_modules=["q_proj", "v_proj"],
                task_type="CAUSAL_LM",
            ))
            reference = copy.deepcopy(policy)
            source = root / "sft"
            policy.save_pretrained(source)
            settings = TrainerSettings(root / "run", "test", ("data",),
                                       adapter_checkpoint=source, learning_rate=1e-4,
                                       max_length=32, bf16=False)
        else:
            policy = PeftModel.from_pretrained(self.base(), resume / "adapter", is_trainable=True)
            reference = PeftModel.from_pretrained(
                self.base(), resume / "reference_adapter", is_trainable=False)
            settings = TrainerSettings(
                root / "run", "test", ("data",), learning_rate=1e-4, max_length=32,
                bf16=False, adapter_checkpoint=resume / "adapter",
                reference_adapter_checkpoint=resume / "reference_adapter",
            )
        value_head = torch.nn.Linear(16, 1)
        torch.nn.init.zeros_(value_head.weight)
        torch.nn.init.zeros_(value_head.bias)
        configuration = dict(rollout_episodes=2, max_new_tokens=3,
                             epochs_per_rollout=1, stats_chunk_size=1)
        configuration.update(ppo_changes)
        ppo = replace(QwenPPOSettings(), **configuration)
        optimizer = torch.optim.AdamW([
            {"params": [p for p in policy.parameters() if p.requires_grad],
             "lr": settings.learning_rate},
            {"params": list(value_head.parameters()), "lr": ppo.value_learning_rate},
        ])
        return QwenPPOOptimizer(
            torch_module=torch, tokenizer=Tokenizer(), policy_model=policy,
            reference_model=reference, value_head=value_head, optimizer=optimizer,
            settings=settings, ppo=ppo,
        )

    def test_exact_kl_gradient_matches_categorical_analytic_gradient(self):
        torch = self.torch
        logits = torch.tensor([[[0.4, -0.3, 0.8], [1.2, 0.2, -0.4]]], requires_grad=True)
        reference = torch.tensor([[[-0.2, 0.4, 0.7], [0.3, -0.1, 0.2]]])
        target = torch.tensor([[1, 2]])
        _, _, kl = categorical_action_statistics(
            torch, logits, target, reference_logits=reference, chunk_size=1)
        kl.sum().backward()
        logp = torch.log_softmax(logits.detach(), dim=-1)
        p = logp.exp()
        delta = logp - torch.log_softmax(reference, dim=-1)
        analytic = p * (delta - (p * delta).sum(dim=-1, keepdim=True))
        torch.testing.assert_close(logits.grad, analytic, atol=1e-6, rtol=1e-5)
        self.assertTrue(bool((kl >= -1e-7).all()))

    def test_identical_kl_and_gradient_are_zero_and_chunk_sizes_agree(self):
        torch = self.torch
        logits = torch.randn(1, 5, 7, requires_grad=True)
        target = torch.tensor([[1, 2, 3, 4, 5]])
        small = categorical_action_statistics(
            torch, logits, target, reference_logits=logits.detach(), chunk_size=1)
        large = categorical_action_statistics(
            torch, logits, target, reference_logits=logits.detach(), chunk_size=5)
        for first, second in zip(small, large, strict=True):
            torch.testing.assert_close(first, second)
        small[2].sum().backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits), atol=2e-7, rtol=0)

    def test_qwen_projects_only_causal_action_positions_with_gradient_parity(self):
        torch = self.torch
        with tempfile.TemporaryDirectory() as directory:
            optimizer = self.optimizer(Path(directory))
            ids = torch.tensor([[3, 4, 5, 6, 7, 8]])
            attention = torch.ones_like(ids)
            start = 4
            positions = torch.arange(start - 1, ids.shape[-1] - 1)
            seen = []
            head = optimizer.policy_model.get_output_embeddings()
            handle = head.register_forward_pre_hook(
                lambda _module, args: seen.append(args[0].shape))
            try:
                projected, _ = optimizer._forward_projected(
                    optimizer.policy_model, ids, attention, positions, with_values=True)
            finally:
                handle.remove()
            self.assertEqual(projected.shape, (1, 2, 24))
            self.assertEqual(tuple(seen[-1]), (1, 2, 16))
            selected, values, entropy = optimizer._statistics(
                optimizer.policy_model, ids, attention, with_values=True, action_start=start)
            self.assertEqual(tuple(values.shape), (1, 6))
            self.assertTrue(bool((selected[:, :start] == 0).all()))
            self.assertTrue(bool((entropy[:, :start] == 0).all()))
            optimizer.policy_model.zero_grad()
            selected.sum().backward()
            actual = {name: p.grad.clone() for name, p in optimizer.policy_model.named_parameters()
                      if p.requires_grad and p.grad is not None}
            optimizer.policy_model.zero_grad()
            full = optimizer.policy_model(input_ids=ids, attention_mask=attention,
                                          use_cache=False).logits
            expected = torch.log_softmax(full[:, start - 1:-1, :].float(), dim=-1).gather(
                -1, ids[:, start:].unsqueeze(-1)).squeeze(-1)
            expected.sum().backward()
            torch.testing.assert_close(selected[:, start:], expected)
            for name, p in optimizer.policy_model.named_parameters():
                if name in actual:
                    torch.testing.assert_close(actual[name], p.grad, atol=1e-6, rtol=1e-5)

    def test_raw_sampling_independent_episodes_ratio_and_frozen_reference(self):
        torch = self.torch
        with tempfile.TemporaryDirectory() as directory:
            optimizer = self.optimizer(Path(directory))
            optimizer.policy_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            environment = Environment()
            rollout = optimizer.collect(environment, rollout_id="fresh", seed=10)
            self.assertEqual(rollout.episode_count, 2)
            self.assertEqual(environment.seeds, [20, 21])
            self.assertEqual(len(rollout.transitions), 2)
            kwargs = optimizer._sampling_kwargs()
            self.assertEqual(kwargs["generation_config"].top_k, 0)
            self.assertEqual(kwargs["generation_config"].repetition_penalty, 1)
            optimizer.policy_model.train()
            for transition in rollout.transitions:
                ids = torch.tensor([transition.token_ids])
                start = transition.action_mask.index(1)
                logp, _, _, kl = optimizer._update_statistics(ids, torch.ones_like(ids), start)
                old = torch.tensor(transition.old_logprobs[start:])
                torch.testing.assert_close((logp - old).exp(), torch.ones_like(old),
                                           atol=1e-6, rtol=1e-5)
                torch.testing.assert_close(kl, torch.zeros_like(kl), atol=2e-7, rtol=0)
            reference_before = {name: value.clone()
                                for name, value in optimizer.reference_model.state_dict().items()}
            policy_before = {name: value.clone() for name, value
                             in optimizer.policy_model.named_parameters() if value.requires_grad}
            metrics = optimizer.update(rollout)
            self.assertLess(metrics["initial_ratio_max_error"], 1e-5)
            self.assertEqual(metrics["sampled_decisions"], 2)
            self.assertGreaterEqual(metrics["reference_kl"], -1e-6)
            self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in metrics.values()))
            for name, value in optimizer.reference_model.state_dict().items():
                torch.testing.assert_close(value, reference_before[name])
            self.assertTrue(any(not torch.equal(value, policy_before[name])
                                for name, value in optimizer.policy_model.named_parameters()
                                if name in policy_before))
            with self.assertRaisesRegex(ValueError, "already been consumed"):
                optimizer.update(rollout)

    def test_factory_seeds_initialization_and_zeroes_value_head_on_cpu(self):
        torch = self.torch
        with tempfile.TemporaryDirectory() as directory:
            initial = self.optimizer(Path(directory))
            with (patch("agent.training.qwen_ppo.resolve_training_stack",
                        return_value={"modules": {"torch": torch},
                                      "versions": {"torch": "test-version"}}),
                  patch("agent.training.qwen_ppo._load_policy",
                        return_value=(Tokenizer(), initial.policy_model)),
                  patch("agent.training.qwen_ppo._load_reference_policy",
                        return_value=initial.reference_model),
                  patch.object(torch.cuda, "is_available", return_value=False),
                  patch.object(torch.backends.mps, "is_available", return_value=False),
                  patch.object(torch, "manual_seed", wraps=torch.manual_seed) as seed):
                built = QwenPPOOptimizer.from_pretrained(settings=initial.settings)
                seed.assert_called_with(initial.settings.seed)
            self.assertTrue(built.policy_model.is_gradient_checkpointing)
            self.assertEqual(built._run_contract["package_versions"],
                             {"torch": "test-version"})
            self.assertEqual(built.optimizer.param_groups[0]["lr"], initial.settings.learning_rate)
            self.assertEqual(built.optimizer.param_groups[1]["lr"],
                             built.ppo.value_learning_rate)
            for value in built.value_head.parameters():
                torch.testing.assert_close(value, torch.zeros_like(value))

    def test_generation_scores_match_unwarped_action_probabilities(self):
        torch = self.torch
        with tempfile.TemporaryDirectory() as directory:
            optimizer = self.optimizer(Path(directory))
            optimizer.policy_model.eval()
            prompt = torch.tensor([[3, 4, 5]])
            config = optimizer._sampling_kwargs()["generation_config"]
            config.return_dict_in_generate = True
            config.output_scores = True
            with torch.no_grad():
                output = optimizer.policy_model.generate(
                    input_ids=prompt, attention_mask=torch.ones_like(prompt),
                    generation_config=config,
                )
                logp, _, _ = optimizer._statistics(
                    optimizer.policy_model, output.sequences,
                    torch.ones_like(output.sequences), with_values=False, action_start=3)
                for index, score in enumerate(output.scores):
                    self.assertTrue(bool(torch.isfinite(score).all()))
                    token = output.sequences[0, 3 + index]
                    sampled = torch.log_softmax(score.float(), dim=-1)[0, token]
                    torch.testing.assert_close(sampled, logp[0, 3 + index], atol=1e-6, rtol=1e-5)

    def test_checkpoint_restore_is_exact_and_rejects_tamper_or_legacy(self):
        torch = self.torch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            optimizer = self.optimizer(root)
            rollout = optimizer.collect(Environment(), rollout_id="once", seed=20)
            optimizer.update(rollout)
            checkpoint = optimizer.settings.output_dir
            restored = self.optimizer(root, resume=checkpoint)
            self.assertEqual(restored.load(checkpoint), 1)
            self.assertEqual(restored.policy_version, optimizer.policy_version)
            for first, second in zip(optimizer.value_head.parameters(),
                                     restored.value_head.parameters(), strict=True):
                torch.testing.assert_close(first, second)
            again = restored.collect(Environment(), rollout_id="again", seed=21)
            expected = optimizer.collect(Environment(), rollout_id="again", seed=21)
            self.assertEqual(again.transitions, expected.transitions)
            mismatch = self.optimizer(root, resume=checkpoint)
            mismatch._run_contract["bf16"] = True
            with self.assertRaisesRegex(ValueError, "run contract"):
                mismatch.load(checkpoint)
            mismatch._run_contract["bf16"] = False
            mismatch._run_contract["package_versions"] = {"torch": "different-version"}
            with self.assertRaisesRegex(ValueError, "run contract"):
                mismatch.load(checkpoint)
            # Equality of the next update verifies Adam moments were restored,
            # not merely the adapter/value weights used by collection.
            optimizer.update(expected)
            restored.update(again)
            self.assertEqual(restored.policy_version, optimizer.policy_version)
            for (name, first), (_, second) in zip(
                    optimizer.policy_model.named_parameters(),
                    restored.policy_model.named_parameters(), strict=True):
                if first.requires_grad:
                    with self.subTest(parameter=name):
                        torch.testing.assert_close(first, second, atol=1e-7, rtol=1e-6)
            for first, second in zip(optimizer.value_head.parameters(),
                                     restored.value_head.parameters(), strict=True):
                torch.testing.assert_close(first, second, atol=1e-7, rtol=1e-6)
            state_file = checkpoint / "ppo_state.json"
            saved = state_file.read_text()
            poisoned = json.loads(saved)
            poisoned["run_contract"]["initial_adapter_hash"] = "unbound-source"
            state_file.write_text(json.dumps(poisoned))
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                restored.load(checkpoint)
            for field, value in (("update_index", -1), ("update_index", True),
                                 ("consumed_rollout_ids", ["again", "again"]),
                                 ("consumed_rollout_ids", [""])):
                invalid = {**json.loads(saved), field: value}
                state_file.write_text(json.dumps(invalid))
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    restored.load(checkpoint)
            state = json.loads(saved)
            state.pop("implementation_version")
            state_file.write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError, "legacy PPO"):
                restored.load(checkpoint)
            state_file.write_text(saved)
            with (checkpoint / "ppo_state.pt").open("ab") as handle:
                handle.write(b"broken-transfer")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                restored.load(checkpoint)

    def test_context_reserves_complete_output_and_nonraw_sampling_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "temperature=1"):
            QwenPPOSettings(temperature=0.8)
        for invalid in (0, -1, float("nan"), float("inf")):
            with self.subTest(value_learning_rate=invalid), self.assertRaises(ValueError):
                QwenPPOSettings(value_learning_rate=invalid)
        with tempfile.TemporaryDirectory() as directory:
            optimizer = self.optimizer(Path(directory), max_new_tokens=30)
            with self.assertRaisesRegex(ValueError, "reserved completion"):
                optimizer.collect(Environment(), rollout_id="too-long", seed=1)


if __name__ == "__main__":
    unittest.main()
