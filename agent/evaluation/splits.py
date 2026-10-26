"""Chronological purged splits and rolling-fold construction."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Mapping

from agent.contracts import normalize_timestamp, parse_timestamp


@dataclass(frozen=True)
class EpisodeWindow:
    episode_id: str
    decision_at: str | datetime
    outcome_at: str | datetime

    def __post_init__(self) -> None:
        if not isinstance(self.episode_id, str) or not self.episode_id.strip():
            raise ValueError("episode_id must be nonempty")
        decision = normalize_timestamp(self.decision_at)
        outcome = normalize_timestamp(self.outcome_at)
        if parse_timestamp(outcome) < parse_timestamp(decision):
            raise ValueError("outcome_at cannot precede decision_at")
        object.__setattr__(self, "episode_id", self.episode_id.strip())
        object.__setattr__(self, "decision_at", decision)
        object.__setattr__(self, "outcome_at", outcome)


def purged_split(
    episodes: Iterable[EpisodeWindow], *, train_end: str | datetime,
    validation_end: str | datetime, test_end: str | datetime,
) -> dict[str, tuple[EpisodeWindow, ...]]:
    train_cut = parse_timestamp(train_end)
    validation_cut = parse_timestamp(validation_end)
    test_cut = parse_timestamp(test_end)
    if not train_cut < validation_cut < test_cut:
        raise ValueError("split boundaries must be strictly increasing")
    result: dict[str, list[EpisodeWindow]] = {
        "train": [], "validation": [], "test": [], "purged": [],
    }
    seen: set[str] = set()
    ordered = sorted(episodes, key=lambda item: (
        parse_timestamp(item.decision_at), parse_timestamp(item.outcome_at), item.episode_id,
    ))
    for episode in ordered:
        if episode.episode_id in seen:
            raise ValueError("episode IDs must be unique")
        seen.add(episode.episode_id)
        start, end = parse_timestamp(episode.decision_at), parse_timestamp(episode.outcome_at)
        if end <= train_cut:
            result["train"].append(episode)
        elif train_cut < start and end <= validation_cut:
            result["validation"].append(episode)
        elif validation_cut < start and end <= test_cut:
            result["test"].append(episode)
        else:
            result["purged"].append(episode)
    return {name: tuple(items) for name, items in result.items()}


@dataclass(frozen=True)
class Fold:
    fold_id: str
    train_sessions: tuple[str, ...]
    validation_sessions: tuple[str, ...]
    test_sessions: tuple[str, ...]


def rolling_folds(
    sessions: Iterable[str], *, train_size: int, validation_size: int,
    test_size: int, step: int | None = None,
) -> tuple[Fold, ...]:
    for value, name in ((train_size, "train_size"), (validation_size, "validation_size"),
                        (test_size, "test_size")):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be positive")
    stride = test_size if step is None else step
    if type(stride) is not int or stride <= 0:
        raise ValueError("step must be positive")
    ordered = tuple(sorted(sessions))
    if not ordered or len(ordered) != len(set(ordered)):
        raise ValueError("sessions must be nonempty and unique")
    width = train_size + validation_size + test_size
    folds = []
    for offset in range(0, len(ordered) - width + 1, stride):
        train_end = offset + train_size
        validation_end = train_end + validation_size
        folds.append(Fold(
            fold_id=f"fold-{len(folds):02d}",
            train_sessions=ordered[offset:train_end],
            validation_sessions=ordered[train_end:validation_end],
            test_sessions=ordered[validation_end:validation_end + test_size],
        ))
    return tuple(folds)

