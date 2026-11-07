"""CPU-only semantic preflight before an expensive model load."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from agent.policy import parse_decisions
from agent.policy.evidence import visible_evidence

from .core import EpisodeExample


def validate_supervised_example(example: EpisodeExample) -> None:
    """Require targets to cite actual eligible prompt evidence, never an archive."""
    visible = visible_evidence(example.messages, expected_as_of=example.decision_at)
    symbols = {record["symbol"] for record in visible.values() if record.get("symbol")}
    if not visible or not symbols:
        raise ValueError("training prompt requires visible symbol evidence")
    decisions = parse_decisions(
        example.completion, allowed_symbols=symbols, allowed_evidence_ids=visible,
    )
    for decision in decisions:
        if decision.action in {"buy", "sell"} and not decision.evidence_ids:
            raise ValueError("trade targets require visible supporting evidence")
        if any(visible[eid].get("symbol") not in (None, decision.symbol)
               for eid in decision.evidence_ids):
            raise ValueError("target cites evidence for another symbol")


def validate_sft_targets(examples: Sequence[EpisodeExample]) -> None:
    for example in examples:
        try:
            validate_supervised_example(example)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid supervised target for {example.example_id}: {exc}; "
                "repair prompt/target citations offline before training"
            ) from exc


def preflight_ppo_context(environment: Any, tokenizer: Any, *, max_length: int,
                          max_new_tokens: int) -> dict[str, int]:
    """Size every replay prompt before loading policy/reference weights.

    This rejects over-budget data; it never silently truncates, removes episodes,
    or shortens a completion allowance to make an allocation succeed.
    """
    episodes = getattr(environment, "episodes", None)
    if not isinstance(episodes, Sequence) or not episodes:
        raise ValueError("PPO context preflight requires an auditable nonempty replay episode list")
    maximum = 0
    oversized = 0
    first_errors = []
    for index, episode in enumerate(episodes, 1):
        rendered = tokenizer.apply_chat_template(
            list(episode.messages), tokenize=True, add_generation_prompt=True,
            return_dict=True, truncation=False,
        )
        ids = rendered.get("input_ids") if isinstance(rendered, Mapping) else rendered
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if isinstance(ids, (tuple, list)) and ids and isinstance(ids[0], (tuple, list)):
            if len(ids) != 1:
                raise ValueError("PPO token preflight requires one prompt per episode")
            ids = ids[0]
        if not isinstance(ids, (tuple, list)) or not ids:
            raise ValueError("PPO token preflight received empty or invalid token IDs")
        tokens = len(ids)
        maximum = max(maximum, tokens)
        if tokens + max_new_tokens > max_length:
            oversized += 1
            if len(first_errors) < 3:
                first_errors.append(f"{episode.episode_id} ({tokens} input tokens)")
        if index % 500 == 0 or index == len(episodes):
            from agent.contracts import canonical_json
            print(canonical_json({
                "event": "ppo_context_preflight", "checked": index, "total": len(episodes),
                "max_input_tokens": maximum, "reserved_output_tokens": max_new_tokens,
                "oversized_prompts": oversized,
            }), flush=True)
    if oversized:
        raise ValueError(
            f"PPO preflight: {oversized} prompts exceed max_length={max_length} with "
            f"{max_new_tokens} reserved output tokens; first: {', '.join(first_errors)}. "
            "Repair the offline context budget before allocating model weights."
        )
    return {"episodes": len(episodes), "max_input_tokens": maximum,
            "reserved_output_tokens": max_new_tokens}
