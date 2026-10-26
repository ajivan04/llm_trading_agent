"""Decision-quality evaluation for frozen Qwen adapters."""
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from agent.policy import parse_decisions
from agent.training import EpisodeExample


@dataclass(frozen=True)
class EvaluationOutcome:
    example_id: str
    split: str
    symbol: str
    evidence_ids: tuple[str, ...]
    objectives: Mapping[str, float]
    chosen: str

    def __post_init__(self) -> None:
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("invalid evaluation split")
        if set(self.objectives) != {"buy", "sell", "hold"}:
            raise ValueError("objectives must contain buy, sell, and hold")
        if any(not math.isfinite(float(value)) for value in self.objectives.values()):
            raise ValueError("objectives must be finite")
        if self.chosen not in self.objectives:
            raise ValueError("chosen action is invalid")


def load_outcomes(path, *, split: str) -> tuple[EvaluationOutcome, ...]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            if raw["split"] != split:
                continue
            rows.append(EvaluationOutcome(
                example_id=raw["example_id"], split=raw["split"], symbol=raw["symbol"],
                evidence_ids=tuple(raw["evidence_ids"]), objectives=raw["objectives"],
                chosen=raw["chosen"],
            ))
        except Exception as exc:
            raise ValueError(f"invalid outcome audit row {line_number}") from exc
    return tuple(rows)


def score_generations(
    examples: Sequence[EpisodeExample], outcomes: Sequence[EvaluationOutcome],
    generations: Mapping[str, str],
) -> tuple[dict[str, float | int], tuple[Mapping[str, Any], ...]]:
    """Score schema validity, expert agreement, objective, and regret."""
    outcome_by_id = {item.example_id: item for item in outcomes}
    if len(outcome_by_id) != len(outcomes):
        raise ValueError("outcome IDs must be unique")
    if set(outcome_by_id) != {item.example_id for item in examples}:
        raise ValueError("outcomes must exactly cover evaluation examples")
    records = []
    valid = correct = trades = 0
    objective_sum = regret_sum = 0.0
    for example in examples:
        outcome = outcome_by_id[example.example_id]
        raw = generations.get(example.example_id, "")
        error = None
        try:
            decisions = parse_decisions(
                raw, allowed_symbols=(outcome.symbol,),
                allowed_evidence_ids=outcome.evidence_ids,
            )
            if len(decisions) != 1:
                raise ValueError("exactly one decision is required")
            action = decisions[0].action
            valid += 1
        except ValueError as exc:
            action = None
            error = type(exc).__name__
        worst = min(float(value) for value in outcome.objectives.values())
        objective = float(outcome.objectives[action]) if action is not None else worst
        best = max(float(value) for value in outcome.objectives.values())
        objective_sum += objective
        regret_sum += best - objective
        correct += int(action == outcome.chosen)
        trades += int(action in {"buy", "sell"})
        records.append({
            "example_id": example.example_id, "split": example.split,
            "action": action, "expert_action": outcome.chosen,
            "objective": objective, "regret": best - objective,
            "valid": action is not None, "error_class": error,
        })
    count = len(examples)
    if count == 0:
        raise ValueError("evaluation split must be nonempty")
    return ({
        "examples": count,
        "valid_schema_rate": valid / count,
        "expert_action_accuracy": correct / count,
        "trade_rate": trades / count,
        "mean_incremental_return": objective_sum / count,
        "mean_regret": regret_sum / count,
    }, tuple(records))
