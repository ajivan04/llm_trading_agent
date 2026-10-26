"""Pure-Python training contracts and numerically testable objectives."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from agent.contracts import FrozenDict, canonical_json, content_hash, normalize_timestamp, parse_timestamp


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
_SPLITS = {"train", "validation", "test"}


def _required(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty")
    return value.strip()


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _frozen_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return FrozenDict({str(key): _frozen_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_frozen_json(item) for item in value)
    canonical_json(value)
    return value


@dataclass(frozen=True)
class EpisodeExample:
    example_id: str
    split: str
    decision_at: str | datetime
    outcome_at: str | datetime
    snapshot_id: str
    account_state_hash: str
    evidence_hash: str
    messages: tuple[Mapping[str, Any], ...]
    completion: str

    def __post_init__(self) -> None:
        for name in ("example_id", "snapshot_id", "account_state_hash", "evidence_hash"):
            object.__setattr__(self, name, _required(getattr(self, name), name))
        if self.split not in _SPLITS:
            raise ValueError("split must be train, validation, or test")
        decision = normalize_timestamp(self.decision_at)
        outcome = normalize_timestamp(self.outcome_at)
        if parse_timestamp(outcome) < parse_timestamp(decision):
            raise ValueError("outcome_at cannot precede decision_at")
        object.__setattr__(self, "decision_at", decision)
        object.__setattr__(self, "outcome_at", outcome)
        if not isinstance(self.messages, tuple) or not self.messages:
            raise ValueError("messages must be a nonempty tuple")
        if not all(isinstance(message, Mapping) for message in self.messages):
            raise TypeError("messages must contain mappings")
        object.__setattr__(self, "messages", tuple(_frozen_json(x) for x in self.messages))
        object.__setattr__(self, "completion", _required(self.completion, "completion"))


def validate_chronological_splits(
    examples: Sequence[EpisodeExample], *, train_end: str | datetime,
    validation_end: str | datetime, test_end: str | datetime,
) -> None:
    train_boundary = parse_timestamp(train_end)
    validation_boundary = parse_timestamp(validation_end)
    test_boundary = parse_timestamp(test_end)
    if not train_boundary < validation_boundary < test_boundary:
        raise ValueError("split boundaries must be ordered")
    for example in examples:
        decision = parse_timestamp(example.decision_at)
        outcome = parse_timestamp(example.outcome_at)
        if example.split == "train" and outcome > train_boundary:
            raise ValueError("train example crosses the train boundary")
        if example.split == "validation" and not (
            decision > train_boundary and outcome <= validation_boundary
        ):
            raise ValueError("validation example crosses a split boundary")
        if example.split == "test" and not (
            decision > validation_boundary and outcome <= test_boundary
        ):
            raise ValueError("test example crosses a split boundary")


class ChatTemplateTokenizer(Protocol):
    def apply_chat_template(self, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class SFTExample:
    example_id: str
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    labels: tuple[int, ...]
    supervised_tokens: int


def build_sft_example(
    example: EpisodeExample, tokenizer: ChatTemplateTokenizer, *, max_length: int,
) -> SFTExample:
    """Render native chat format and supervise assistant generation tokens only."""
    if example.split != "train":
        raise ValueError("SFT examples must come from the train split")
    if type(max_length) is not int or max_length <= 0:
        raise ValueError("max_length must be positive")
    if any(message.get("role") == "assistant" for message in example.messages):
        raise ValueError(
            "final-decision SFT prompts cannot contain prior assistant messages; "
            "use an explicit trajectory builder for selective tool supervision"
        )
    messages = (*example.messages, FrozenDict({"role": "assistant", "content": example.completion}))
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_assistant_tokens_mask=True,
        truncation=False,
    )
    if not isinstance(rendered, Mapping):
        raise TypeError("native chat template must return a mapping")
    input_ids = tuple(int(x) for x in rendered.get("input_ids", ()))
    attention = tuple(int(x) for x in rendered.get("attention_mask", (1,) * len(input_ids)))
    assistant_mask = rendered.get("assistant_masks", rendered.get("assistant_tokens_mask"))
    if assistant_mask is None:
        raise ValueError("tokenizer chat template did not return an assistant token mask")
    assistant = tuple(int(x) for x in assistant_mask)
    if not input_ids or not (len(input_ids) == len(attention) == len(assistant)):
        raise ValueError("tokenizer returned inconsistent token arrays")
    if len(input_ids) > max_length:
        raise ValueError("SFT example exceeds max_length; explicit packing is required")
    if any(value not in (0, 1) for value in attention + assistant):
        raise ValueError("attention and assistant masks must be binary")
    labels = tuple(
        token if is_attended and is_assistant else -100
        for token, is_attended, is_assistant in zip(input_ids, attention, assistant)
    )
    supervised = sum(label != -100 for label in labels)
    if supervised == 0:
        raise ValueError("SFT example has no supervised assistant tokens")
    return SFTExample(example.example_id, input_ids, attention, labels, supervised)


@dataclass(frozen=True)
class DPOCandidate:
    candidate_id: str
    split: str
    prompt: str
    response: str
    snapshot_id: str
    account_state_hash: str
    evidence_hash: str
    objective: float

    def __post_init__(self) -> None:
        for name in (
            "candidate_id", "prompt", "response", "snapshot_id",
            "account_state_hash", "evidence_hash",
        ):
            object.__setattr__(self, name, _required(getattr(self, name), name))
        if self.split not in _SPLITS:
            raise ValueError("invalid split")
        object.__setattr__(self, "objective", _finite(self.objective, "objective"))

    @property
    def state_hash(self) -> str:
        return content_hash({
            "prompt": self.prompt, "snapshot_id": self.snapshot_id,
            "account_state_hash": self.account_state_hash,
            "evidence_hash": self.evidence_hash,
        })


@dataclass(frozen=True)
class DPOPair:
    prompt: str
    chosen: str
    rejected: str
    state_hash: str
    preference_margin: float
    chosen_id: str
    rejected_id: str


def build_dpo_pair(
    first: DPOCandidate, second: DPOCandidate, *, minimum_margin: float = 0.0,
) -> DPOPair:
    if first.split != "train" or second.split != "train":
        raise ValueError("DPO pairs must use train-split candidates")
    if first.state_hash != second.state_hash:
        raise ValueError("DPO candidates must come from the same state")
    if first.response == second.response:
        raise ValueError("DPO candidate responses must differ")
    margin_floor = _finite(minimum_margin, "minimum_margin")
    if margin_floor < 0:
        raise ValueError("minimum_margin must be nonnegative")
    chosen, rejected = (
        (first, second) if first.objective > second.objective else (second, first)
    )
    margin = chosen.objective - rejected.objective
    if margin <= 0 or margin < margin_floor:
        raise ValueError("DPO preference is tied or below the minimum margin")
    return DPOPair(
        chosen.prompt, chosen.response, rejected.response, chosen.state_hash,
        margin, chosen.candidate_id, rejected.candidate_id,
    )


@dataclass(frozen=True)
class PPOTransition:
    rollout_id: str
    policy_version: str
    observation_hash: str
    token_ids: tuple[int, ...]
    action_mask: tuple[int, ...]
    old_logprobs: tuple[float, ...]
    reference_logprobs: tuple[float, ...]
    value: float
    reward: float
    next_value: float
    done: bool = False
    truncated: bool = False

    def __post_init__(self) -> None:
        for name in ("rollout_id", "policy_version", "observation_hash"):
            object.__setattr__(self, name, _required(getattr(self, name), name))
        lengths = {
            len(self.token_ids), len(self.action_mask), len(self.old_logprobs),
            len(self.reference_logprobs),
        }
        if len(lengths) != 1 or not self.token_ids:
            raise ValueError("PPO token arrays must be nonempty and aligned")
        if any(value not in (0, 1) for value in self.action_mask) or not any(self.action_mask):
            raise ValueError("PPO action_mask must be binary with an action token")
        for name in ("value", "reward", "next_value"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        for values, name in (
            (self.old_logprobs, "old_logprobs"),
            (self.reference_logprobs, "reference_logprobs"),
        ):
            if any(not math.isfinite(float(value)) for value in values):
                raise ValueError(f"{name} must be finite")
        if self.done and self.truncated:
            raise ValueError("a transition cannot be both terminal and truncated")
        if self.done and self.next_value != 0:
            raise ValueError("terminal transitions must have zero next_value")


@dataclass(frozen=True)
class PPORollout:
    rollout_id: str
    policy_version: str
    generated_at: str | datetime
    transitions: tuple[PPOTransition, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "rollout_id", _required(self.rollout_id, "rollout_id"))
        object.__setattr__(self, "policy_version", _required(self.policy_version, "policy_version"))
        object.__setattr__(self, "generated_at", normalize_timestamp(self.generated_at))
        if not self.transitions:
            raise ValueError("rollout must contain transitions")
        for index, transition in enumerate(self.transitions):
            if transition.rollout_id != self.rollout_id:
                raise ValueError("transition rollout_id mismatch")
            if transition.policy_version != self.policy_version:
                raise ValueError("transition policy_version mismatch")
            if (transition.done or transition.truncated) and index != len(self.transitions) - 1:
                raise ValueError("episode boundary must end the rollout")


def validate_fresh_rollout(
    rollout: PPORollout, *, current_policy_version: str, expected_rollout_id: str,
) -> None:
    if rollout.policy_version != current_policy_version:
        raise ValueError("stale PPO rollout policy version")
    if rollout.rollout_id != expected_rollout_id:
        raise ValueError("unexpected or reused PPO rollout ID")


def compute_gae(
    transitions: Sequence[PPOTransition], *, gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    gamma = _finite(gamma, "gamma")
    gae_lambda = _finite(gae_lambda, "gae_lambda")
    if not 0 <= gamma <= 1 or not 0 <= gae_lambda <= 1:
        raise ValueError("gamma and gae_lambda must be between zero and one")
    advantages = [0.0] * len(transitions)
    carried = 0.0
    for index in range(len(transitions) - 1, -1, -1):
        transition = transitions[index]
        bootstrap = 0.0 if transition.done else transition.next_value
        delta = transition.reward + gamma * bootstrap - transition.value
        continues = 0.0 if (transition.done or transition.truncated) else 1.0
        carried = delta + gamma * gae_lambda * continues * carried
        advantages[index] = carried
    returns = [advantage + transition.value for advantage, transition in zip(advantages, transitions)]
    return tuple(advantages), tuple(returns)


def ppo_objective(
    *, old_logprobs: Sequence[float], new_logprobs: Sequence[float],
    advantages: Sequence[float], action_mask: Sequence[int],
    old_values: Sequence[float], new_values: Sequence[float],
    returns: Sequence[float], entropies: Sequence[float],
    reference_logprobs: Sequence[float] | None = None,
    clip_ratio: float = 0.2, value_clip: float = 0.2,
    value_coefficient: float = 0.5, entropy_coefficient: float = 0.01,
    kl_coefficient: float = 0.0,
) -> Mapping[str, float]:
    arrays = (old_logprobs, new_logprobs, advantages, action_mask,
              old_values, new_values, returns, entropies)
    if not arrays[0] or len({len(values) for values in arrays}) != 1:
        raise ValueError("PPO objective arrays must be nonempty and aligned")
    if reference_logprobs is not None and len(reference_logprobs) != len(old_logprobs):
        raise ValueError("reference_logprobs must align")
    indices = [index for index, selected in enumerate(action_mask) if selected == 1]
    if any(selected not in (0, 1) for selected in action_mask) or not indices:
        raise ValueError("action_mask must be binary with selected tokens")
    numeric_arrays = (
        old_logprobs, new_logprobs, advantages, old_values, new_values,
        returns, entropies,
    ) + ((reference_logprobs,) if reference_logprobs is not None else ())
    for values in numeric_arrays:
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError("PPO objective inputs must be finite")
    clip_ratio = _finite(clip_ratio, "clip_ratio")
    value_clip = _finite(value_clip, "value_clip")
    if clip_ratio < 0 or value_clip < 0:
        raise ValueError("clip values must be nonnegative")
    surrogates, approx_kls, clipped_flags, value_losses = [], [], [], []
    for index in indices:
        log_ratio = float(new_logprobs[index]) - float(old_logprobs[index])
        try:
            ratio = math.exp(log_ratio)
        except OverflowError as exc:
            raise ValueError("PPO log-probability ratio overflow") from exc
        if not math.isfinite(ratio):
            raise ValueError("PPO log-probability ratio must be finite")
        clipped = min(max(ratio, 1 - clip_ratio), 1 + clip_ratio)
        advantage = float(advantages[index])
        surrogates.append(min(ratio * advantage, clipped * advantage))
        approx_kls.append((ratio - 1.0) - log_ratio)
        clipped_flags.append(float(abs(ratio - 1.0) > clip_ratio))
        value_delta = float(new_values[index]) - float(old_values[index])
        clipped_value = float(old_values[index]) + min(max(value_delta, -value_clip), value_clip)
        raw_error = (float(new_values[index]) - float(returns[index])) ** 2
        clipped_error = (clipped_value - float(returns[index])) ** 2
        value_losses.append(0.5 * max(raw_error, clipped_error))
    policy_loss = -sum(surrogates) / len(surrogates)
    value_loss = sum(value_losses) / len(value_losses)
    entropy = sum(float(entropies[index]) for index in indices) / len(indices)
    reference_kl = 0.0
    if reference_logprobs is not None:
        reference_kl = sum(
            float(new_logprobs[index]) - float(reference_logprobs[index])
            for index in indices
        ) / len(indices)
    total = (policy_loss + value_coefficient * value_loss
             - entropy_coefficient * entropy + kl_coefficient * reference_kl)
    return FrozenDict({
        "loss": total,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "approx_kl": sum(approx_kls) / len(approx_kls),
        "reference_kl": reference_kl,
        "clip_fraction": sum(clipped_flags) / len(clipped_flags),
    })


@dataclass(frozen=True)
class RewardComponents:
    equity_before: float
    equity_after: float
    benchmark_return: float = 0.0
    incremental_drawdown: float = 0.0
    invalid_proposals: int = 0
    drawdown_weight: float = 0.1
    invalid_weight: float = 0.001

    def total(self) -> float:
        before = _finite(self.equity_before, "equity_before")
        after = _finite(self.equity_after, "equity_after")
        benchmark = _finite(self.benchmark_return, "benchmark_return")
        if before <= 0 or after <= 0 or benchmark <= -1:
            raise ValueError("reward equity/benchmark inputs are invalid")
        if type(self.invalid_proposals) is not int or self.invalid_proposals < 0:
            raise ValueError("invalid_proposals must be a nonnegative integer")
        # Equity is net of execution costs. There is intentionally no second cost term.
        return (
            math.log(after / before) - math.log1p(benchmark)
            - _finite(self.drawdown_weight, "drawdown_weight") * max(0.0, self.incremental_drawdown)
            - _finite(self.invalid_weight, "invalid_weight") * self.invalid_proposals
        )


@dataclass(frozen=True)
class TrainingManifest:
    model_name: str
    model_revision: str
    tokenizer_revision: str
    prompt_version: str
    package_versions: Mapping[str, str]
    data_hashes: tuple[str, ...]
    seed: int
    adapter_config: Mapping[str, Any]
    learning_rate: float
    steps: int
    method: str

    def __post_init__(self) -> None:
        if self.model_name != MODEL_NAME:
            raise ValueError("training policy model is fixed by the assignment")
        for name in ("model_revision", "tokenizer_revision", "prompt_version", "method"):
            object.__setattr__(self, name, _required(getattr(self, name), name))
        if self.method not in {"sft", "dpo", "ppo"}:
            raise ValueError("method must be sft, dpo, or ppo")
        if type(self.seed) is not int or type(self.steps) is not int or self.steps <= 0:
            raise ValueError("seed must be an integer and steps must be positive")
        rate = _finite(self.learning_rate, "learning_rate")
        if rate <= 0:
            raise ValueError("learning_rate must be positive")
        object.__setattr__(self, "learning_rate", rate)
        packages = {str(key): _required(value, f"package_versions[{key}]")
                    for key, value in self.package_versions.items()}
        object.__setattr__(self, "package_versions", FrozenDict(packages))
        hashes = tuple(_required(value, "data_hash") for value in self.data_hashes)
        if not hashes:
            raise ValueError("data_hashes must be nonempty")
        object.__setattr__(self, "data_hashes", hashes)
        object.__setattr__(self, "adapter_config", _frozen_json(self.adapter_config))

    @property
    def manifest_hash(self) -> str:
        return content_hash(self)

