"""Concrete token-level PPO optimizer for the fixed Qwen policy.

Imports of Torch/Transformers remain behind ``from_pretrained`` so the package and
its pure contract tests work on machines without a compatible accelerator stack.
"""

from __future__ import annotations

import hashlib
import json
import math
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
from .ppo_math import categorical_action_statistics

PPO_IMPLEMENTATION_VERSION = "ppo-v2-exact-categorical-kl"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def initial_sft_provenance(path: Path | None):
    if path is None or not (Path(path) / "training_manifest.json").is_file():
        return None
    payload = json.loads((Path(path) / "training_manifest.json").read_text())
    manifest, metrics = payload.get("manifest", {}), payload.get("metrics", {})
    if payload.get("manifest_hash") != content_hash(manifest):
        raise ValueError("initial SFT manifest identity is invalid")
    verified = False
    fingerprint = metrics.get("inference_adapter_fingerprint")
    if manifest.get("adapter_config", {}).get("supervision_mode") == "trajectory-final":
        from agent.policy.loader import adapter_fingerprint

        if not fingerprint or adapter_fingerprint(path) != fingerprint:
            raise ValueError("initial SFT inference weights differ from completed training")
        verified = True
    return {
        "inference_weights_verified": verified,
        "inference_adapter_fingerprint": fingerprint,
        "manifest_hash": payload["manifest_hash"],
        "method": manifest.get("method"),
        "model_name": manifest.get("model_name"),
        "model_revision": manifest.get("model_revision"),
        "tokenizer_revision": manifest.get("tokenizer_revision"),
        "data_hashes": manifest.get("data_hashes"),
        "supervision_mode": manifest.get("adapter_config", {}).get("supervision_mode"),
        "status": metrics.get("status"),
        "completed_steps": metrics.get("completed_steps"),
        "planned_steps": manifest.get("steps"),
        "trainable_parameter_fraction": metrics.get("trainable_parameter_fraction"),
    }


def _load_reference_policy(stack: Mapping[str, Any], settings: TrainerSettings) -> Any:
    """Load the frozen reference from the same SFT adapter as the initial policy."""
    modules = stack["modules"]
    base = modules["transformers"].AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        revision=settings.model_revision,
        trust_remote_code=False,
        **(
            {"dtype": modules["torch"].bfloat16}
            if settings.bf16 and modules["torch"].cuda.is_available()
            else {}
        ),
    )
    reference_adapter = settings.reference_adapter_checkpoint or settings.adapter_checkpoint
    return modules["peft"].PeftModel.from_pretrained(
        base,
        str(reference_adapter),
        is_trainable=False,
    )


@dataclass(frozen=True)
class QwenPPOSettings:
    max_episode_steps: int = 8
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    rollout_episodes: int = 8
    stats_chunk_size: int = 32
    target_kl: float = 0.1
    max_invalid_fraction: float = 0.5
    value_learning_rate: float = 1e-6
    epochs_per_rollout: int = 2
    clip_ratio: float = 0.2
    value_clip: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.0
    kl_coefficient: float = 0.02
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_grad_norm: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "max_episode_steps",
            "max_new_tokens",
            "epochs_per_rollout",
            "rollout_episodes",
            "stats_chunk_size",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.temperature != 1.0 or self.top_p != 1.0:
            raise ValueError("PPO sampling requires temperature=1 and top_p=1")
        if not math.isfinite(self.target_kl) or self.target_kl <= 0:
            raise ValueError("target_kl must be positive and finite")
        if not math.isfinite(self.max_invalid_fraction) or not 0 <= self.max_invalid_fraction <= 1:
            raise ValueError("max_invalid_fraction must be between zero and one")
        if not math.isfinite(self.value_learning_rate) or self.value_learning_rate <= 0:
            raise ValueError("value_learning_rate must be positive and finite")
        if not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("gamma and gae_lambda must be in [0, 1]")
        for name in (
            "clip_ratio",
            "value_clip",
            "value_coefficient",
            "entropy_coefficient",
            "kl_coefficient",
            "max_grad_norm",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")


class QwenPPOOptimizer:
    """Sample from the current adapter and optimize token-level PPO objectives."""

    def __init__(
        self,
        *,
        torch_module: Any,
        tokenizer: Any,
        policy_model: Any,
        reference_model: Any,
        value_head: Any,
        optimizer: Any,
        settings: TrainerSettings,
        ppo: QwenPPOSettings,
        package_versions: Mapping[str, str] | None = None,
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
        # LoRA dropout must not change the behavior/current density ratio. Keep
        # train() available for Qwen activation checkpointing, but disable all
        # dropout in both modes before collecting any actions.
        for model in (self.policy_model, self.reference_model):
            modules = getattr(model, "modules", None)
            if callable(modules):
                for module in modules():
                    if "Dropout" in type(module).__name__ and hasattr(module, "p"):
                        module.p = 0.0
                    if hasattr(module, "attention_dropout"):
                        module.attention_dropout = 0.0
            config = getattr(model, "config", None)
            if config is not None and hasattr(config, "attention_dropout"):
                config.attention_dropout = 0.0
        self._run_contract = {
            "implementation_version": PPO_IMPLEMENTATION_VERSION,
            "package_versions": dict(package_versions or {}),
            "model_revision": settings.model_revision,
            "tokenizer_revision": settings.tokenizer_revision,
            "prompt_version": settings.prompt_version,
            "data_hashes": settings.data_hashes,
            "seed": settings.seed,
            "learning_rate": settings.learning_rate,
            "max_length": settings.max_length,
            "bf16": settings.bf16,
            "gradient_checkpointing": settings.gradient_checkpointing,
            "ppo": asdict(ppo),
            "initial_sft": initial_sft_provenance(
                settings.reference_adapter_checkpoint or settings.adapter_checkpoint
            ),
            "initial_adapter_hash": _path_hash(
                settings.reference_adapter_checkpoint or settings.adapter_checkpoint
            ),
        }
        self._run_fingerprint = content_hash(self._run_contract)
        self.reference_model.requires_grad_(False)
        self.reference_model.eval()

    @classmethod
    def from_pretrained(
        cls,
        *,
        settings: TrainerSettings,
        lora: LoRASettings | None = None,
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
        torch.manual_seed(settings.seed)
        if getattr(torch, "cuda", None) is not None:
            torch.cuda.manual_seed_all(settings.seed)
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
            # Non-reentrant checkpointing supports a frozen embedding input;
            # reentrant mode can silently lose adapter gradients in this setup.
            policy.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        hidden_size = getattr(policy.config, "hidden_size", None)
        if type(hidden_size) is not int or hidden_size <= 0:
            raise RuntimeError("Qwen policy config does not expose hidden_size")
        value_head = torch.nn.Linear(hidden_size, 1).to(device)
        torch.nn.init.zeros_(value_head.weight)
        torch.nn.init.zeros_(value_head.bias)
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": [item for item in policy.parameters() if item.requires_grad],
                    "lr": settings.learning_rate,
                },
                {"params": list(value_head.parameters()), "lr": ppo.value_learning_rate},
            ]
        )
        return cls(
            torch_module=torch,
            tokenizer=tokenizer,
            policy_model=policy,
            reference_model=reference.to(device),
            value_head=value_head,
            optimizer=optimizer,
            settings=settings,
            ppo=ppo,
            package_versions=stack["versions"],
        )

    @property
    def policy_version(self) -> str:
        return content_hash(
            {
                "run_fingerprint": self._run_fingerprint,
                "update_index": self._update_index,
            }
        )

    @property
    def consumed_rollout_ids(self) -> frozenset[str]:
        return frozenset(self._consumed_rollout_ids)

    def _messages(self, observation: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
        messages = observation.get("messages")
        if (
            not isinstance(messages, (list, tuple))
            or not messages
            or not all(isinstance(item, Mapping) for item in messages)
        ):
            raise ValueError("PPO observations must contain nonempty chat messages")
        return messages

    def _prompt(self, observation: Mapping[str, Any]):
        messages = self._messages(observation)
        rendered = self.tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        # Transformers tokenizers may return either a tensor or a
        # BatchEncoding mapping containing `input_ids`.
        token_ids = rendered.get("input_ids") if isinstance(rendered, Mapping) else rendered
        if token_ids is None:
            raise ValueError("chat template did not return input_ids")
        if getattr(token_ids, "ndim", None) != 2 or token_ids.shape[0] != 1:
            raise ValueError("chat template must return one rank-2 token batch")
        return token_ids.to(self.policy_model.device), content_hash(messages)

    def _forward_projected(
        self, model: Any, token_ids: Any, attention_mask: Any, positions: Any, *, with_values: bool
    ) -> tuple[Any, Any | None]:
        """Use Qwen's pre-LM-head selection and capture only final decoder states.

        Calling the normal model/PEFT forward preserves adapter hooks. A hook on
        the decoder captures its final output without requesting all hidden layers.
        Unsupported model surfaces fail rather than reverting to full vocabulary
        logits for every prompt token.
        """
        captured = {}
        handle = None
        if with_values:
            get_base = getattr(model, "get_base_model", None)
            base = get_base() if callable(get_base) else model
            decoder = getattr(base, "model", None)
            if not callable(getattr(decoder, "register_forward_hook", None)):
                raise RuntimeError("PPO model must expose the Qwen decoder for value states")

            def capture(_module, _inputs, output):
                hidden = getattr(output, "last_hidden_state", None)
                if hidden is None and isinstance(output, tuple):
                    hidden = output[0]
                if hidden is None:
                    raise RuntimeError("Qwen decoder did not return final hidden states")
                captured["hidden"] = hidden

            handle = decoder.register_forward_hook(capture)
        try:
            output = model(
                input_ids=token_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                use_cache=False,
                return_dict=True,
                logits_to_keep=positions,
            )
        finally:
            if handle is not None:
                handle.remove()
        logits = output.logits
        if logits.ndim != 3 or logits.shape[:2] != (1, int(positions.numel())):
            raise RuntimeError("Qwen did not project only the requested action positions")
        return logits, captured.get("hidden")

    def _statistics(
        self,
        model: Any,
        token_ids: Any,
        attention_mask: Any,
        *,
        with_values: bool,
        action_start: int = 1,
    ) -> tuple[Any, Any | None, Any]:
        length = int(token_ids.shape[-1])
        if not 1 <= action_start < length:
            raise ValueError("action statistics require a nonempty prompt and completion")
        positions = self.torch.arange(action_start - 1, length - 1, device=token_ids.device)
        logits, hidden = self._forward_projected(
            model, token_ids, attention_mask, positions, with_values=with_values
        )
        sampled, entropy, _ = categorical_action_statistics(
            self.torch, logits, token_ids[:, action_start:], chunk_size=self.ppo.stats_chunk_size
        )
        zeros = self.torch.zeros((1, action_start), device=token_ids.device, dtype=sampled.dtype)
        values = None
        if with_values:
            values = self.value_head(hidden.float()).squeeze(-1)
        return (
            self.torch.cat((zeros, sampled), dim=1),
            values,
            self.torch.cat((zeros, entropy), dim=1),
        )

    def _update_statistics(
        self, token_ids: Any, attention_mask: Any, action_start: int
    ) -> tuple[Any, Any, Any, Any]:
        positions = self.torch.arange(
            action_start - 1, int(token_ids.shape[-1]) - 1, device=token_ids.device
        )
        logits, hidden = self._forward_projected(
            self.policy_model, token_ids, attention_mask, positions, with_values=True
        )
        with self.torch.no_grad():
            reference_logits, _ = self._forward_projected(
                self.reference_model, token_ids, attention_mask, positions, with_values=False
            )
        sampled, entropy, divergence = categorical_action_statistics(
            self.torch,
            logits,
            token_ids[:, action_start:],
            reference_logits=reference_logits,
            chunk_size=self.ppo.stats_chunk_size,
        )
        state_value = self.value_head(hidden[:, action_start - 1, :].float()).squeeze()
        return sampled[0], state_value, entropy[0], divergence[0]

    def _state_value(self, observation: Mapping[str, Any]) -> float:
        prompt, _ = self._prompt(observation)
        attention = self.torch.ones_like(prompt)
        with self.torch.no_grad():
            positions = self.torch.tensor([int(prompt.shape[-1]) - 1], device=prompt.device)
            _, hidden = self._forward_projected(
                self.policy_model, prompt, attention, positions, with_values=True
            )
            value = self.value_head(hidden[:, -1, :].float()).squeeze()
        return float(value.detach().cpu())

    def _sampling_kwargs(self) -> dict[str, Any]:
        options = {
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "typical_p": 1.0,
            "min_p": None,
            "epsilon_cutoff": 0.0,
            "eta_cutoff": 0.0,
            "repetition_penalty": 1.0,
            "encoder_repetition_penalty": 1.0,
            "no_repeat_ngram_size": 0,
            "bad_words_ids": None,
            "suppress_tokens": None,
            "begin_suppress_tokens": None,
            "forced_bos_token_id": None,
            "forced_eos_token_id": None,
            "renormalize_logits": False,
            "min_new_tokens": None,
            "min_length": 0,
            "use_cache": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "max_new_tokens": self.ppo.max_new_tokens,
        }
        if getattr(self.policy_model, "generation_config", None) is not None:
            # A fresh config prevents inherited Qwen generation defaults (e.g.
            # top_k=20) from changing the density used by PPO statistics.
            from transformers import GenerationConfig

            return {"generation_config": GenerationConfig(**options)}
        return options

    def collect(self, environment: PPOEnvironment, *, rollout_id: str, seed: int) -> PPORollout:
        if rollout_id in self._consumed_rollout_ids:
            raise ValueError("PPO rollout ID has already been consumed")
        self.torch.manual_seed(seed)
        if getattr(self.torch, "cuda", None) is not None:
            self.torch.cuda.manual_seed_all(seed)
        policy_version = self.policy_version
        transitions = []
        self.policy_model.eval()
        for episode_index in range(self.ppo.rollout_episodes):
            # Runner update seeds increase by one. Multiplying by episode count
            # prevents adjacent updates from reusing almost the entire window.
            observation = environment.reset(seed=seed * self.ppo.rollout_episodes + episode_index)
            for step_index in range(self.ppo.max_episode_steps):
                prompt, observation_hash = self._prompt(observation)
                prompt_length = int(prompt.shape[-1])
                if prompt_length + self.ppo.max_new_tokens > self.settings.max_length:
                    raise ValueError(
                        "PPO prompt exhausts the configured context window: "
                        f"{prompt_length} prompt + {self.ppo.max_new_tokens} reserved completion "
                        f"> {self.settings.max_length}; compact evidence before training"
                    )
                attention = self.torch.ones_like(prompt)
                with self.torch.no_grad():
                    generated = self.policy_model.generate(
                        input_ids=prompt,
                        attention_mask=attention,
                        **self._sampling_kwargs(),
                    )
                    if generated.shape[-1] <= prompt_length:
                        raise RuntimeError("sampled PPO action contains no tokens")
                    full_attention = self.torch.ones_like(generated)
                    old_logprobs, old_values, _ = self._statistics(
                        self.policy_model,
                        generated,
                        full_attention,
                        with_values=True,
                        action_start=prompt_length,
                    )
                    reference_logprobs, _, _ = self._statistics(
                        self.reference_model,
                        generated,
                        full_attention,
                        with_values=False,
                        action_start=prompt_length,
                    )
                action_ids = generated[0, prompt_length:]
                action = self.tokenizer.decode(action_ids, skip_special_tokens=True)
                next_observation, reward, done, truncated, info = environment.step(action)
                if step_index + 1 == self.ppo.max_episode_steps and not done:
                    truncated = True
                next_value = 0.0 if done else self._state_value(next_observation)
                token_values = tuple(float(item) for item in old_values[0].detach().cpu().tolist())
                action_mask = (0,) * prompt_length + (1,) * (generated.shape[-1] - prompt_length)
                transitions.append(
                    PPOTransition(
                        rollout_id=rollout_id,
                        policy_version=policy_version,
                        observation_hash=observation_hash,
                        token_ids=tuple(int(item) for item in generated[0].detach().cpu().tolist()),
                        attention_mask=tuple(1 for _ in range(generated.shape[-1])),
                        action_mask=action_mask,
                        old_logprobs=tuple(
                            float(item) for item in old_logprobs[0].detach().cpu().tolist()
                        ),
                        reference_logprobs=tuple(
                            float(item) for item in reference_logprobs[0].detach().cpu().tolist()
                        ),
                        old_values=token_values,
                        value=token_values[prompt_length - 1],
                        reward=float(reward),
                        next_value=next_value,
                        done=bool(done),
                        truncated=bool(truncated),
                        reward_components=dict(info.get("reward_components", {})),
                    )
                )
                observation = next_observation
                if done or truncated:
                    break
        return PPORollout(
            rollout_id,
            policy_version,
            datetime.now(UTC),
            tuple(transitions),
            episode_count=self.ppo.rollout_episodes,
        )

    def update(self, rollout: PPORollout) -> Mapping[str, float]:
        if rollout.rollout_id in self._consumed_rollout_ids:
            raise ValueError("PPO rollout ID has already been consumed")
        if rollout.policy_version != self.policy_version:
            raise ValueError("cannot update from a stale PPO policy version")
        advantages, returns = compute_gae(
            rollout.transitions, gamma=self.ppo.gamma, gae_lambda=self.ppo.gae_lambda
        )
        totals: dict[str, float] = {}
        # Dropout has been disabled explicitly. train() retains the supported
        # model's activation checkpointing while preserving rollout densities.
        self.policy_model.train()
        initial_ratio_max_error = 0.0
        for epoch in range(self.ppo.epochs_per_rollout):
            self.optimizer.zero_grad(set_to_none=True)
            epoch_loss = 0.0
            epoch_metrics: dict[str, list[float]] = {
                name: []
                for name in (
                    "policy_loss",
                    "value_loss",
                    "entropy",
                    "reference_kl",
                    "clip_fraction",
                    "update_ratio",
                )
            }
            value_predictions: list[float] = []
            value_targets: list[float] = []
            for transition, advantage, target_return in zip(
                rollout.transitions, advantages, returns, strict=True
            ):
                ids = self.torch.tensor(
                    [transition.token_ids], dtype=self.torch.long, device=self.policy_model.device
                )
                attention = self.torch.tensor(
                    [transition.attention_mask],
                    dtype=self.torch.long,
                    device=self.policy_model.device,
                )
                action_start = transition.action_mask.index(1)
                new_logprobs, new_state_value, entropy, divergence = self._update_statistics(
                    ids, attention, action_start
                )
                old_logprobs = self.torch.tensor(
                    transition.old_logprobs[action_start:], device=self.policy_model.device
                )
                state_index = action_start - 1
                old_state_value = self.torch.tensor(
                    transition.old_values[state_index], device=self.policy_model.device
                )
                ratio = (new_logprobs - old_logprobs).exp()
                if epoch == 0:
                    initial_ratio_max_error = max(
                        initial_ratio_max_error, float((ratio - 1).abs().max().detach().cpu())
                    )
                    if initial_ratio_max_error > 0.05:
                        raise ValueError(
                            "PPO behavior/current probability ratio changed before optimization"
                        )
                clipped_ratio = ratio.clamp(1 - self.ppo.clip_ratio, 1 + self.ppo.clip_ratio)
                advantage_tensor = self.torch.full_like(ratio, float(advantage))
                policy_loss = -self.torch.minimum(
                    ratio * advantage_tensor, clipped_ratio * advantage_tensor
                ).mean()
                target = self.torch.full_like(new_state_value, float(target_return))
                clipped_value = old_state_value + (new_state_value - old_state_value).clamp(
                    -self.ppo.value_clip, self.ppo.value_clip
                )
                value_loss = 0.5 * self.torch.maximum(
                    (new_state_value - target).pow(2), (clipped_value - target).pow(2)
                )
                mean_entropy = entropy.mean()
                reference_kl = divergence.mean()
                loss = (
                    policy_loss
                    + self.ppo.value_coefficient * value_loss
                    - self.ppo.entropy_coefficient * mean_entropy
                    + self.ppo.kl_coefficient * reference_kl
                )
                if not bool(self.torch.isfinite(loss).detach().cpu()):
                    raise ValueError("PPO loss is nonfinite; stop and inspect the rollout")
                # Free each transition's autograd graph immediately rather than
                # retaining a rollout-sized stack of vocabulary distributions.
                (loss / len(rollout.transitions)).backward()
                epoch_loss += float(loss.detach().cpu()) / len(rollout.transitions)
                values = {
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                    "entropy": mean_entropy,
                    "reference_kl": reference_kl,
                    "clip_fraction": ((ratio - 1).abs() > self.ppo.clip_ratio).float().mean(),
                    "update_ratio": ratio.mean(),
                }
                for name, value in values.items():
                    epoch_metrics[name].append(float(value.detach().cpu()))
                value_predictions.append(float(new_state_value.detach().cpu()))
                value_targets.append(float(target_return))
            parameters = chain(
                (item for item in self.policy_model.parameters() if item.requires_grad),
                self.value_head.parameters(),
            )
            gradient_norm = self.torch.nn.utils.clip_grad_norm_(
                parameters, self.ppo.max_grad_norm, error_if_nonfinite=True
            )
            self.optimizer.step()
            totals = {name: sum(values) / len(values) for name, values in epoch_metrics.items()}
            totals["loss"] = epoch_loss
            totals["gradient_norm"] = float(gradient_norm.detach().cpu())
            target_mean = sum(value_targets) / len(value_targets)
            target_variance = sum((value - target_mean) ** 2 for value in value_targets) / len(
                value_targets
            )
            error_mean = sum(
                target - predicted
                for target, predicted in zip(value_targets, value_predictions, strict=True)
            ) / len(value_targets)
            error_variance = sum(
                ((target - predicted) - error_mean) ** 2
                for target, predicted in zip(value_targets, value_predictions, strict=True)
            ) / len(value_targets)
            totals["explained_variance"] = (
                1.0 - error_variance / target_variance if target_variance else 0.0
            )
        self.policy_model.eval()
        post_kl = []
        with self.torch.no_grad():
            for transition in rollout.transitions:
                ids = self.torch.tensor(
                    [transition.token_ids], dtype=self.torch.long, device=self.policy_model.device
                )
                attention = self.torch.tensor(
                    [transition.attention_mask],
                    dtype=self.torch.long,
                    device=self.policy_model.device,
                )
                _, _, _, divergence = self._update_statistics(
                    ids, attention, transition.action_mask.index(1)
                )
                post_kl.append(float(divergence.mean().detach().cpu()))
        if any(not math.isfinite(value) for value in post_kl):
            raise ValueError("PPO post-update KL is nonfinite; stop before publishing a checkpoint")
        totals["reference_kl"] = sum(post_kl) / len(post_kl)
        totals["reference_kl_max"] = max(post_kl)
        totals["initial_ratio_max_error"] = initial_ratio_max_error
        totals["sampled_decisions"] = float(len(rollout.transitions))
        totals["sampled_episodes"] = float(rollout.episode_count)
        totals["reward_mean"] = sum(item.reward for item in rollout.transitions) / len(
            rollout.transitions
        )
        component_names = sorted(
            {name for item in rollout.transitions for name in item.reward_components}
        )
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
        self.torch.save(
            {
                "value_head": self.value_head.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            tensor_temporary,
        )
        tensor_temporary.replace(output_dir / "ppo_state.pt")
        adapter_hash = _path_hash(output_dir / "adapter")
        reference_adapter_hash = _path_hash(output_dir / "reference_adapter")
        state = {
            "implementation_version": PPO_IMPLEMENTATION_VERSION,
            "model_name": MODEL_NAME,
            "model_revision": self.settings.model_revision,
            "tokenizer_revision": self.settings.tokenizer_revision,
            "update_index": self._update_index,
            "consumed_rollout_ids": sorted(self._consumed_rollout_ids),
            "policy_version": self.policy_version,
            "run_fingerprint": self._run_fingerprint,
            "run_contract": self._run_contract,
            "adapter_hash": adapter_hash,
            "reference_adapter_hash": reference_adapter_hash,
            "tensor_state_sha256": _file_sha256(output_dir / "ppo_state.pt"),
        }
        temporary = output_dir / "ppo_state.json.tmp"
        temporary.write_text(canonical_json(state) + "\n", encoding="utf-8")
        temporary.replace(output_dir / "ppo_state.json")

    def load(self, checkpoint: Path) -> int:
        checkpoint = Path(checkpoint)
        state = json.loads((checkpoint / "ppo_state.json").read_text(encoding="utf-8"))
        if state.get("implementation_version") != PPO_IMPLEMENTATION_VERSION:
            raise ValueError("legacy PPO checkpoint uses a different implementation version")
        if state.get("tensor_state_sha256") != _file_sha256(checkpoint / "ppo_state.pt"):
            raise ValueError(
                "PPO tensor state checksum mismatch; checkpoint is incomplete or changed"
            )
        if state.get("model_name") != MODEL_NAME:
            raise ValueError("PPO checkpoint policy model mismatch")
        if state.get("model_revision") != self.settings.model_revision:
            raise ValueError("PPO checkpoint model revision mismatch")
        if state.get("tokenizer_revision") != self.settings.tokenizer_revision:
            raise ValueError("PPO checkpoint tokenizer revision mismatch")
        adapter_checkpoint = self.settings.adapter_checkpoint
        if adapter_checkpoint is None or _path_hash(adapter_checkpoint) != state.get(
            "adapter_hash"
        ):
            raise ValueError(
                "resume optimizer must be initialized from the checkpoint adapter directory"
            )
        reference_adapter = self.settings.reference_adapter_checkpoint
        if reference_adapter is None or _path_hash(reference_adapter) != state.get(
            "reference_adapter_hash"
        ):
            raise ValueError("resume optimizer must preserve the checkpoint reference adapter")
        stored_contract = state.get("run_contract")
        if not isinstance(stored_contract, Mapping):
            raise ValueError("PPO checkpoint is missing its run contract")
        fingerprint = content_hash(stored_contract)
        if state.get("run_fingerprint") != fingerprint:
            raise ValueError("PPO checkpoint run contract fingerprint mismatch")
        update_index = state.get("update_index")
        if type(update_index) is not int or update_index < 0:
            raise ValueError("PPO checkpoint update_index must be a nonnegative integer")
        consumed = state.get("consumed_rollout_ids")
        if (
            not isinstance(consumed, list)
            or any(not isinstance(item, str) or not item.strip() for item in consumed)
            or len(set(consumed)) != len(consumed)
        ):
            raise ValueError("PPO checkpoint consumed rollout IDs must be unique nonempty strings")
        expected_version = content_hash(
            {
                "run_fingerprint": fingerprint,
                "update_index": update_index,
            }
        )
        if state.get("policy_version") != expected_version:
            raise ValueError("PPO checkpoint policy version is inconsistent")
        current_contract = dict(self._run_contract)
        current_contract.pop("initial_adapter_hash", None)
        current_contract.pop("initial_sft", None)
        comparable_stored = dict(stored_contract)
        comparable_stored.pop("initial_adapter_hash", None)
        comparable_stored.pop("initial_sft", None)
        if canonical_json(comparable_stored) != canonical_json(current_contract):
            raise ValueError("PPO checkpoint run contract does not match current settings")
        tensors = self.torch.load(
            checkpoint / "ppo_state.pt",
            map_location=self.policy_model.device,
            weights_only=True,
        )
        self.value_head.load_state_dict(tensors["value_head"])
        self.optimizer.load_state_dict(tensors["optimizer"])
        self._run_contract = dict(stored_contract)
        self._run_fingerprint = fingerprint
        self._update_index = update_index
        self._consumed_rollout_ids = set(consumed)
        return self._update_index
