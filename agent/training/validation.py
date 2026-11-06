"""CPU-only semantic preflight before an expensive model load."""
from __future__ import annotations

from collections.abc import Sequence

from agent.policy import parse_decisions
from agent.policy.evidence import visible_evidence

from .core import EpisodeExample


def validate_supervised_example(example: EpisodeExample) -> None:
    """Require targets to cite actual eligible prompt evidence, never an archive."""
    visible = visible_evidence(example.messages, expected_as_of=example.decision_at)
    symbols = {record["symbol"] for record in visible.values() if record.get("symbol")}
    if not visible or not symbols:
        raise ValueError("training prompt requires visible symbol evidence")
    parse_decisions(
        example.completion, allowed_symbols=symbols, allowed_evidence_ids=visible,
    )


def validate_sft_targets(examples: Sequence[EpisodeExample]) -> None:
    for example in examples:
        try:
            validate_supervised_example(example)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid supervised target for {example.example_id}: {exc}; "
                "repair prompt/target citations offline before training"
            ) from exc
