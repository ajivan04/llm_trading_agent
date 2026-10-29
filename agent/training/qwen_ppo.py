"""Concrete token-level PPO optimizer for the fixed Qwen policy.

Imports of Torch/Transformers remain behind ``from_pretrained`` so the package and
its pure contract tests work on machines without a compatible accelerator stack.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import chain
from pathlib import Path
from typing import Any

from agent.contracts import canonical_json, content_hash

from .core import MODEL_NAME, PPORollout, PPOTransition, compute_gae
from .execution import LoRASettings, PPOEnvironment, TrainerSettings, _load_policy, _path_hash
from .integrations import resolve_training_stack


def _load_reference_policy(stack: Mapping[str, Any], settings: TrainerSettings) -> Any:
    """Load the frozen reference from the same SFT adapter as the initial policy."""
    modules = stack["modules"]
    base = modules["transformers"].AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, revision=settings.model_revision, trust_remote_code=False,
        **({"dtype": modules["torch"].bfloat16}
           if settings.bf16 and modules["torch"].cuda.is_available() else {}),
    )
    reference_adapter = (
        settings.reference_adapter_checkpoint or settings.adapter_checkpoint)
    return modules["peft"].PeftModel.from_pretrained(
        base, str(reference_adapter), is_trainable=False,
    )


@dataclass(frozen=True)
class QwenPPOSettings:
    max_episode_steps: int = 8
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.95
    epochs_per_rollout: int = 4
    clip_ratio: float = 0.2
    value_clip: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    kl_coefficient: float = 0.02
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_grad_norm: float = 1.0

    def __post_init__(self) -> None:
        for name in ("max_episode_steps", "max_new_tokens", "epochs_per_rollout"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.temperature <= 0 or not 0 < self.top_p <= 1:
            raise ValueError("sampling temperature/top_p are invalid")
        if not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("gamma and gae_lambda must be in [0, 1]")
        for name in (
            "clip_ratio", "value_clip", "value_coefficient", "entropy_coefficient",
            "kl_coefficient", "max_grad_norm",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")


class QwenPPOOptimizer:
    """Sample from the current adapter and optimize token-level PPO objectives."""

    def __init__(
        self, *, torch_module: Any, tokenizer: Any, policy_model: Any,
        reference_model: Any, value_head: Any, optimizer: Any,
        settings: TrainerSettings, ppo: QwenPPOSettings,
    ) -> None:
        self.torch = torch_module
        self.tokenizer = tokenizer
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.value_head = value_head
        self.optimizer = optimizer
        self.settings = settings
        self.ppo = ppo
        self._update_index = 0
        self._consumed_rollout_ids: set[str] = set()
        self._run_contract = {
            "model_revision": settings.model_revision,
            "tokenizer_revision": settings.tokenizer_revision,
            "prompt_version": settings.prompt_version,
            "data_hashes": settings.data_hashes,
            "seed": settings.seed,
            "learning_rate": settings.learning_rate,
            "max_length": settings.max_length,
            "ppo": asdict(ppo),
            "initial_adapter_hash": _path_hash(
                settings.reference_adapter_checkpoint or settings.adapter_checkpoint),
        }
        self._run_fingerprint = content_hash(self._run_contract)
        self.reference_model.requires_grad_(False)
        self.reference_model.eval()

    @classmethod
    def from_pretrained(
        cls, *, settings: TrainerSettings, lora: LoRASettings | None = None,
        ppo: QwenPPOSettings | None = None,
        expected_versions: dict[str, str] | None = None,
    ) -> QwenPPOOptimizer:
        lora = lora or LoRASettings()
        ppo = ppo or QwenPPOSettings()
        if settings.adapter_checkpoint is None:
            raise ValueError("PPO must initialize from a trained SFT adapter_checkpoint")
        stack = resolve_training_stack("ppo", expected_versions=expected_versions)
        modules = stack["modules"]
        torch = modules["torch"]
        tokenizer, policy = _load_policy(stack, settings, lora)
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif mps is not None and mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        policy = policy.to(device)
        reference = _load_reference_policy(stack, settings)
        if settings.gradient_checkpointing and hasattr(policy, "gradient_checkpointing_enable"):
            policy.gradient_checkpointing_enable()
        hidden_size = getattr(policy.config, "hidden_size", None)
        if type(hidden_size) is not int or hidden_size <= 0:
            raise RuntimeError("Qwen policy config does not expose hidden_size")
        value_head = torch.nn.Linear(hidden_size, 1).to(device)
        parameters = chain(
            (item for item in policy.parameters() if item.requires_grad),
            value_head.parameters(),
        )
        optimizer = torch.optim.AdamW(parameters, lr=settings.learning_rate)
        return cls(
            torch_module=torch, tokenizer=tokenizer, policy_model=policy,
            reference_model=reference.to(device), value_head=value_head,
            optimizer=optimizer, settings=settings, ppo=ppo,
        )

    @property
    def policy_version(self) -> str:
        return content_hash({
            "run_fingerprint": self._run_fingerprint,
            "update_index": self._update_index,
        })

    @property
    def consumed_rollout_ids(self) -> frozenset[str]:
        return frozenset(self._consumed_rollout_ids)

    def _messages(self, observation: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
        messages = observation.get("messages")
        if (not isinstance(messages, (list, tuple)) or not messages
                or not all(isinstance(item, Mapping) for item in messages)):
            raise ValueError("PPO observations must contain nonempty chat messages")
        return messages

    def _prompt(self, observation: Mapping[str, Any]):
        messages = self._messages(observation)
        token_ids = self.tokenizer.apply_chat_template(
            list(messages), tokenize=True, add_generation_prompt=True,
            return_tensors="pt",
        )
        if getattr(token_ids, "ndim", None) != 2 or token_ids.shape[0] != 1:
            raise ValueError("chat template must return one rank-2 token batch")
        return token_ids.to(self.policy_model.device), content_hash(messages)

    def _statistics(self, model: Any, token_ids: Any, attention_mask: Any,
                    *, with_values: bool) -> tuple[Any, Any | None, Any]:
        output = model(
            input_ids=token_ids, attention_mask=attention_mask,
            output_hidden_states=with_values, use_cache=False,
        )
        logits = output.logits.float()
        log_distribution = self.torch.log_softmax(logits[:, :-1, :], dim=-1)
        predicted = log_distribution.gather(
            -1, token_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        leading = self.torch.zeros_like(predicted[:, :1])
        logprobs = self.torch.cat((leading, predicted), dim=1)
        probabilities = log_distribution.exp()
        entropy = -(probabilities * log_distribution).sum(dim=-1)
        entropy = self.torch.cat((self.torch.zeros_like(entropy[:, :1]), entropy), dim=1)
        values = None
        if with_values:
            values = self.value_head(output.hidden_states[-1].float()).squeeze(-1)
        return logprobs, values, entropy

    def _state_value(self, observation: Mapping[str, Any]) -> float:
        prompt, _ = self._prompt(observation)
        attention = self.torch.ones_like(prompt)
        with self.torch.no_grad():
            _, values, _ = self._statistics(
                self.policy_model, prompt, attention, with_values=True)
        return float(values[0, -1].detach().cpu())

    def collect(self, environment: PPOEnvironment, *, rollout_id: str,
                seed: int) -> PPORollout:
        if rollout_id in self._consumed_rollout_ids:
            raise ValueError("PPO rollout ID has already been consumed")
        self.torch.manual_seed(seed)
        if getattr(self.torch, "cuda", None) is not None:
            self.torch.cuda.manual_seed_all(seed)
        observation = environment.reset(seed=seed)
        policy_version = self.policy_version
        transitions = []
        self.policy_model.eval()
        for step_index in range(self.ppo.max_episode_steps):
            prompt, observation_hash = self._prompt(observation)
            prompt_length = int(prompt.shape[-1])
            remaining_context = self.settings.max_length - prompt_length
            if remaining_context <= 0:
                raise ValueError("PPO prompt exhausts the configured context window")
            attention = self.torch.ones_like(prompt)
            with self.torch.no_grad():
                generated = self.policy_model.generate(
                    input_ids=prompt, attention_mask=attention, do_sample=True,
                    temperature=self.ppo.temperature, top_p=self.ppo.top_p,
                    max_new_tokens=min(self.ppo.max_new_tokens, remaining_context),
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                if generated.shape[-1] <= prompt_length:
                    raise RuntimeError("sampled PPO action contains no tokens")
                full_attention = self.torch.ones_like(generated)
                old_logprobs, old_values, _ = self._statistics(
                    self.policy_model, generated, full_attention, with_values=True)
                reference_logprobs, _, _ = self._statistics(
                    self.reference_model, generated, full_attention, with_values=False)
            action_ids = generated[0, prompt_length:]
            action = self.tokenizer.decode(action_ids, skip_special_tokens=True)
            next_observation, reward, done, truncated, info = environment.step(action)
            if step_index + 1 == self.ppo.max_episode_steps and not done:
                truncated = True
            next_value = 0.0 if done else self._state_value(next_observation)
            token_values = tuple(float(item) for item in old_values[0].detach().cpu().tolist())
            action_mask = (0,) * prompt_length + (1,) * (generated.shape[-1] - prompt_length)
            transitions.append(PPOTransition(
                rollout_id=rollout_id, policy_version=policy_version,
                observation_hash=observation_hash,
                token_ids=tuple(int(item) for item in generated[0].detach().cpu().tolist()),
                attention_mask=tuple(1 for _ in range(generated.shape[-1])),
                action_mask=action_mask,
                old_logprobs=tuple(float(item) for item in old_logprobs[0].detach().cpu().tolist()),
                reference_logprobs=tuple(
                    float(item) for item in reference_logprobs[0].detach().cpu().tolist()),
                old_values=token_values, value=token_values[prompt_length - 1],
                reward=float(reward), next_value=next_value,
                done=bool(done), truncated=bool(truncated),
                reward_components=dict(info.get("reward_components", {})),
            ))
            observation = next_observation
            if done or truncated:
                break
        return PPORollout(
            rollout_id, policy_version, datetime.now(UTC), tuple(transitions))

    def update(self, rollout: PPORollout) -> Mapping[str, float]:
        if rollout.rollout_id in self._consumed_rollout_ids:
            raise ValueError("PPO rollout ID has already been consumed")
        if rollout.policy_version != self.policy_version:
            raise ValueError("cannot update from a stale PPO policy version")
        advantages, returns = compute_gae(
            rollout.transitions, gamma=self.ppo.gamma, gae_lambda=self.ppo.gae_lambda)
        totals: dict[str, float] = {}
        self.policy_model.train()
        for _epoch in range(self.ppo.epochs_per_rollout):
            self.optimizer.zero_grad()
            losses = []
            epoch_metrics: dict[str, list[float]] = {
                name: [] for name in (
                    "policy_loss", "value_loss", "entropy", "reference_kl",
                    "clip_fraction", "update_ratio")
            }
            value_predictions: list[float] = []
            value_targets: list[float] = []
            for transition, advantage, target_return in zip(
                    rollout.transitions, advantages, returns, strict=True):
                ids = self.torch.tensor(
                    [transition.token_ids], dtype=self.torch.long,
                    device=self.policy_model.device)
                attention = self.torch.tensor(
                    [transition.attention_mask], dtype=self.torch.long,
                    device=self.policy_model.device)
                selected = self.torch.tensor(
                    transition.action_mask, dtype=self.torch.bool,
                    device=self.policy_model.device)
                new_logprobs, new_values, entropy = self._statistics(
                    self.policy_model, ids, attention, with_values=True)
                new_logprobs = new_logprobs[0, selected]
                entropy = entropy[0, selected]
                old_logprobs = self.torch.tensor(
                    transition.old_logprobs, device=self.policy_model.device)[selected]
                reference = self.torch.tensor(
                    transition.reference_logprobs, device=self.policy_model.device)[selected]
                state_index = transition.action_mask.index(1) - 1
                new_state_value = new_values[0, state_index]
                old_state_value = self.torch.tensor(
                    transition.old_values[state_index], device=self.policy_model.device)
                ratio = (new_logprobs - old_logprobs).exp()
                clipped_ratio = ratio.clamp(1 - self.ppo.clip_ratio, 1 + self.ppo.clip_ratio)
                advantage_tensor = self.torch.full_like(ratio, float(advantage))
                policy_loss = -self.torch.minimum(
                    ratio * advantage_tensor, clipped_ratio * advantage_tensor).mean()
                target = self.torch.full_like(new_state_value, float(target_return))
                clipped_value = old_state_value + (new_state_value - old_state_value).clamp(
                    -self.ppo.value_clip, self.ppo.value_clip)
                value_loss = 0.5 * self.torch.maximum(
                    (new_state_value - target).pow(2), (clipped_value - target).pow(2))
                mean_entropy = entropy.mean()
                reference_kl = (new_logprobs - reference).mean()
                loss = (policy_loss + self.ppo.value_coefficient * value_loss
                        - self.ppo.entropy_coefficient * mean_entropy
                        + self.ppo.kl_coefficient * reference_kl)
                losses.append(loss)
                values = {
                    "policy_loss": policy_loss, "value_loss": value_loss,
                    "entropy": mean_entropy, "reference_kl": reference_kl,
                    "clip_fraction": ((ratio - 1).abs() > self.ppo.clip_ratio).float().mean(),
                    "update_ratio": ratio.mean(),
                }
                for name, value in values.items():
                    epoch_metrics[name].append(float(value.detach().cpu()))
                value_predictions.append(float(new_state_value.detach().cpu()))
                value_targets.append(float(target_return))
            total_loss = self.torch.stack(losses).mean()
            total_loss.backward()
            parameters = chain(
                (item for item in self.policy_model.parameters() if item.requires_grad),
                self.value_head.parameters(),
            )
            self.torch.nn.utils.clip_grad_norm_(parameters, self.ppo.max_grad_norm)
            self.optimizer.step()
            totals = {name: sum(values) / len(values)
                      for name, values in epoch_metrics.items()}
            totals["loss"] = float(total_loss.detach().cpu())
            target_mean = sum(value_targets) / len(value_targets)
            target_variance = sum(
                (value - target_mean) ** 2 for value in value_targets) / len(value_targets)
            error_mean = sum(
                target - predicted
                for target, predicted in zip(value_targets, value_predictions, strict=True)
            ) / len(value_targets)
            error_variance = sum(
                ((target - predicted) - error_mean) ** 2
                for target, predicted in zip(value_targets, value_predictions, strict=True)
            ) / len(value_targets)
            totals["explained_variance"] = (
                1.0 - error_variance / target_variance if target_variance else 0.0)
        totals["reward_mean"] = sum(
            item.reward for item in rollout.transitions) / len(rollout.transitions)
        component_names = sorted({
            name for item in rollout.transitions for name in item.reward_components
        })
        for name in component_names:
            totals[f"reward_component/{name}"] = sum(
                item.reward_components.get(name, 0.0) for item in rollout.transitions
            ) / len(rollout.transitions)
        self._consumed_rollout_ids.add(rollout.rollout_id)
        self._update_index += 1
        # Persist consumption together with the updated parameters before the caller
        # can request another rollout, preventing reuse after a process restart.
        self.save(self.settings.output_dir)
        return totals

    def save(self, output_dir: Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.policy_model.save_pretrained(str(output_dir / "adapter"))
        self.reference_model.save_pretrained(str(output_dir / "reference_adapter"))
        self.tokenizer.save_pretrained(str(output_dir / "adapter"))
        tensor_temporary = output_dir / "ppo_state.pt.tmp"
        self.torch.save({
            "value_head": self.value_head.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, tensor_temporary)
        tensor_temporary.replace(output_dir / "ppo_state.pt")
        adapter_hash = _path_hash(output_dir / "adapter")
        reference_adapter_hash = _path_hash(output_dir / "reference_adapter")
        state = {
            "model_name": MODEL_NAME, "model_revision": self.settings.model_revision,
            "tokenizer_revision": self.settings.tokenizer_revision,
            "update_index": self._update_index,
            "consumed_rollout_ids": sorted(self._consumed_rollout_ids),
            "policy_version": self.policy_version,
            "run_fingerprint": self._run_fingerprint,
            "run_contract": self._run_contract,
            "adapter_hash": adapter_hash,
            "reference_adapter_hash": reference_adapter_hash,
        }
        temporary = output_dir / "ppo_state.json.tmp"
        temporary.write_text(canonical_json(state) + "\n", encoding="utf-8")
        temporary.replace(output_dir / "ppo_state.json")

    def load(self, checkpoint: Path) -> int:
        checkpoint = Path(checkpoint)
        state = json.loads((checkpoint / "ppo_state.json").read_text(encoding="utf-8"))
        if state.get("model_name") != MODEL_NAME:
            raise ValueError("PPO checkpoint policy model mismatch")
        if state.get("model_revision") != self.settings.model_revision:
            raise ValueError("PPO checkpoint model revision mismatch")
        if state.get("tokenizer_revision") != self.settings.tokenizer_revision:
            raise ValueError("PPO checkpoint tokenizer revision mismatch")
        adapter_checkpoint = self.settings.adapter_checkpoint
        if (adapter_checkpoint is None
                or _path_hash(adapter_checkpoint) != state.get("adapter_hash")):
            raise ValueError(
                "resume optimizer must be initialized from the checkpoint adapter directory")
        reference_adapter = self.settings.reference_adapter_checkpoint
        if (reference_adapter is None
                or _path_hash(reference_adapter) != state.get("reference_adapter_hash")):
            raise ValueError(
                "resume optimizer must preserve the checkpoint reference adapter")
        stored_contract = state.get("run_contract")
        if not isinstance(stored_contract, Mapping):
            raise ValueError("PPO checkpoint is missing its run contract")
        current_contract = dict(self._run_contract)
        current_contract.pop("initial_adapter_hash", None)
        comparable_stored = dict(stored_contract)
        comparable_stored.pop("initial_adapter_hash", None)
        if canonical_json(comparable_stored) != canonical_json(current_contract):
            raise ValueError("PPO checkpoint run contract does not match current settings")
        tensors = self.torch.load(
            checkpoint / "ppo_state.pt", map_location=self.policy_model.device,
            weights_only=True,
        )
        self.value_head.load_state_dict(tensors["value_head"])
        self.optimizer.load_state_dict(tensors["optimizer"])
        self._run_contract = dict(stored_contract)
        self._run_fingerprint = str(state["run_fingerprint"])
        self._update_index = int(state["update_index"])
        self._consumed_rollout_ids = set(state["consumed_rollout_ids"])
        if state.get("policy_version") != self.policy_version:
            raise ValueError("PPO checkpoint policy version is inconsistent")
        return self._update_index
