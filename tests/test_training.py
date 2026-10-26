import math
import unittest

from agent.training import (
    MODEL_NAME,
    DPOCandidate,
    EpisodeExample,
    PPORollout,
    PPOTransition,
    RewardComponents,
    TrainingManifest,
    build_dpo_pair,
    build_sft_example,
    compute_gae,
    ppo_objective,
    validate_chronological_splits,
    validate_fresh_rollout,
)


def episode(split="train", decision="2026-01-01T00:00:00Z",
            outcome="2026-01-02T00:00:00Z"):
    return EpisodeExample(
        "example-1", split, decision, outcome, "snapshot", "account", "evidence",
        ({"role": "system", "content": "rules"},
         {"role": "user", "content": "state"}),
        '{"decisions":[{"symbol":"AAPL","action":"hold","qty":0,"reason":"x"}]}',
    )


class FakeTokenizer:
    def __init__(self, mask=(0, 0, 1, 1), tokens=(10, 11, 12, 13), attention=None):
        self.mask, self.tokens = mask, tokens
        self.attention = attention or (1,) * len(tokens)
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return {"input_ids": self.tokens, "attention_mask": self.attention,
                "assistant_masks": self.mask}


def transition(**changes):
    values = dict(
        rollout_id="rollout-1", policy_version="policy-1", observation_hash="obs",
        token_ids=(1, 2), action_mask=(0, 1), old_logprobs=(-1.0, -0.5),
        reference_logprobs=(-1.1, -0.6), value=0.5, reward=1.0,
        next_value=0.25, done=False, truncated=False,
    )
    values.update(changes)
    return PPOTransition(**values)


class SplitAndSFTTests(unittest.TestCase):
    def test_chronological_split_boundaries(self):
        examples = [
            episode(),
            episode("validation", "2026-01-03T00:00:00Z", "2026-01-04T00:00:00Z"),
            episode("test", "2026-01-05T00:00:00Z", "2026-01-06T00:00:00Z"),
        ]
        validate_chronological_splits(
            examples, train_end="2026-01-02T00:00:00Z",
            validation_end="2026-01-04T00:00:00Z", test_end="2026-01-06T00:00:00Z")
        with self.assertRaisesRegex(ValueError, "train boundary"):
            validate_chronological_splits(
                [episode(outcome="2026-01-03T00:00:00Z")],
                train_end="2026-01-02T00:00:00Z",
                validation_end="2026-01-04T00:00:00Z", test_end="2026-01-06T00:00:00Z")

    def test_native_template_masks_prompt_and_padding(self):
        tokenizer = FakeTokenizer(mask=(0, 1, 1, 1), attention=(1, 1, 1, 0))
        result = build_sft_example(episode(), tokenizer, max_length=4)
        self.assertEqual(result.labels, (-100, 11, 12, -100))
        self.assertEqual(result.supervised_tokens, 2)
        self.assertTrue(tokenizer.calls[0][1]["return_assistant_tokens_mask"])
        with self.assertRaisesRegex(ValueError, "max_length"):
            build_sft_example(episode(), tokenizer, max_length=3)
        with self.assertRaisesRegex(ValueError, "no supervised"):
            build_sft_example(episode(), FakeTokenizer(mask=(0, 0, 0, 0)), max_length=4)
        trajectory_prompt = EpisodeExample(
            "trajectory", "train", "2026-01-01T00:00:00Z",
            "2026-01-02T00:00:00Z", "snapshot", "account", "evidence",
            ({"role": "user", "content": "state"},
             {"role": "assistant", "content": "tool call"}),
            "final decision",
        )
        with self.assertRaisesRegex(ValueError, "trajectory builder"):
            build_sft_example(trajectory_prompt, tokenizer, max_length=4)


class DPOTests(unittest.TestCase):
    def candidate(self, candidate_id, response, objective, **changes):
        values = dict(
            candidate_id=candidate_id, split="train", prompt="prompt", response=response,
            snapshot_id="snapshot", account_state_hash="account", evidence_hash="evidence",
            objective=objective,
        )
        values.update(changes)
        return DPOCandidate(**values)

    def test_same_state_and_margin_are_required(self):
        pair = build_dpo_pair(
            self.candidate("good", "buy", 0.2), self.candidate("bad", "hold", 0.1),
            minimum_margin=0.05)
        self.assertEqual(pair.chosen, "buy")
        with self.assertRaisesRegex(ValueError, "same state"):
            build_dpo_pair(
                self.candidate("a", "buy", 1),
                self.candidate("b", "hold", 0, snapshot_id="other"))
        with self.assertRaisesRegex(ValueError, "minimum margin"):
            build_dpo_pair(
                self.candidate("a", "buy", .01), self.candidate("b", "hold", 0),
                minimum_margin=.1)


class PPOTests(unittest.TestCase):
    def test_rollout_freshness_and_masks(self):
        rollout = PPORollout(
            "rollout-1", "policy-1", "2026-01-01T00:00:00Z", (transition(),))
        validate_fresh_rollout(
            rollout, current_policy_version="policy-1", expected_rollout_id="rollout-1")
        with self.assertRaisesRegex(ValueError, "stale"):
            validate_fresh_rollout(
                rollout, current_policy_version="policy-2", expected_rollout_id="rollout-1")
        with self.assertRaisesRegex(ValueError, "action_mask"):
            transition(action_mask=(0, 0))
        sample = transition(token_ids=(10, 11, 12), action_mask=(0, 1, 1),
                            old_logprobs=(-1.0, -0.5, -0.4),
                            reference_logprobs=(-1.1, -0.6, -0.5))
        self.assertEqual(sample.prompt_token_ids, (10,))
        self.assertEqual(sample.action_token_ids, (11, 12))
        self.assertEqual(sample.attention_mask, (1, 1, 1))
        self.assertEqual(len(sample.old_values), 3)

    def test_gae_terminal_and_truncation_bootstrap(self):
        terminal = transition(value=0.5, reward=1.0, next_value=0.0, done=True)
        advantages, returns = compute_gae((terminal,), gamma=.9, gae_lambda=.95)
        self.assertAlmostEqual(advantages[0], .5)
        self.assertAlmostEqual(returns[0], 1.0)
        truncated = transition(value=.5, reward=1, next_value=.25, truncated=True)
        advantages, returns = compute_gae((truncated,), gamma=.9, gae_lambda=.95)
        self.assertAlmostEqual(advantages[0], .725)
        self.assertAlmostEqual(returns[0], 1.225)

    def test_clipped_objective_reports_diagnostics(self):
        result = ppo_objective(
            old_logprobs=(-1.0, -1.0), new_logprobs=(-1.0, math.log(2) - 1),
            advantages=(1.0, 1.0), action_mask=(1, 1),
            old_values=(0.0, 0.0), new_values=(0.0, 2.0), returns=(1.0, 1.0),
            entropies=(.5, .5), reference_logprobs=(-1.1, -1.1), clip_ratio=.2)
        self.assertGreater(result["clip_fraction"], 0)
        self.assertGreaterEqual(result["approx_kl"], 0)
        self.assertIn("reference_kl", result)


class RewardAndManifestTests(unittest.TestCase):
    def test_reward_does_not_double_subtract_execution_costs(self):
        reward = RewardComponents(100.0, 99.0).total()
        self.assertAlmostEqual(reward, math.log(.99))
        penalized = RewardComponents(
            100.0, 99.0, incremental_drawdown=.1, invalid_proposals=2).total()
        self.assertLess(penalized, reward)

    def test_manifest_is_fixed_model_and_deterministic(self):
        values = dict(
            model_name=MODEL_NAME, model_revision="rev", tokenizer_revision="tok",
            prompt_version="v1", package_versions={"trl": "1"},
            data_hashes=("abc",), seed=7, adapter_config={"r": 8, "targets": ["q_proj"]},
            learning_rate=1e-5, steps=10, method="sft",
        )
        first, second = TrainingManifest(**values), TrainingManifest(**values)
        self.assertEqual(first.manifest_hash, second.manifest_hash)
        with self.assertRaisesRegex(ValueError, "fixed"):
            TrainingManifest(**{**values, "model_name": "larger/model"})


if __name__ == "__main__":
    unittest.main()
