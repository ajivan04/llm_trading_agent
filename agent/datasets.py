"""Leakage-resistant historical training dataset construction and artifacts."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from agent.contracts import Decision, Evidence, Snapshot, canonical_json, content_hash
from agent.evaluation.splits import EpisodeWindow, purged_split
from agent.training.core import EpisodeExample, validate_chronological_splits


_FORBIDDEN_PROMPT_KEYS = {
    "next_open", "next_close", "outcome", "reward", "preference_score",
    "future_return", "realized_return",
}
_SENSITIVE = {"api_key", "apikey", "token", "secret", "password", "authorization"}


def _is_sensitive(name: str) -> bool:
    return (name in _SENSITIVE
            or name.endswith(("_api_key", "_secret_key", "_access_token", "_password")))


def _audit_payload(value: Any, path: str = "payload") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).casefold().replace("-", "_")
            if normalized in _FORBIDDEN_PROMPT_KEYS:
                raise ValueError(f"future outcome field is forbidden in policy input: {path}.{key}")
            if _is_sensitive(normalized):
                raise ValueError(f"sensitive field is forbidden in dataset artifacts: {path}.{key}")
            _audit_payload(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _audit_payload(item, f"{path}[{index}]")


def policy_evidence(item: Evidence) -> dict[str, Any]:
    """Project evidence into policy-visible, cutoff-valid fields.

    ``retrieved_at`` remains audit lineage because a historical record is often
    downloaded long after the decision being replayed.
    """
    return {
        "id": item.id,
        "source": item.source,
        "symbol": item.symbol,
        "observed_at": item.observed_at,
        "available_at": item.available_at,
        "text": item.text,
        "values": item.values,
        "url": item.url,
        "revision_id": item.revision_id,
        "accession_id": item.accession_id,
        "vintage_id": item.vintage_id,
        "units": item.units,
        "entitlement": item.entitlement,
        "raw_content_hash": item.raw_content_hash,
        "transformation_version": item.transformation_version,
    }


@dataclass(frozen=True)
class HistoricalEpisode:
    episode_id: str
    snapshot: Snapshot
    evidence: tuple[Evidence, ...]
    decisions: tuple[Decision, ...]
    outcome_at: str

    def __post_init__(self) -> None:
        if not isinstance(self.episode_id, str) or not self.episode_id.strip():
            raise ValueError("episode_id must be nonempty")
        if not self.evidence or not self.decisions:
            raise ValueError("historical episodes require evidence and decisions")
        ids = tuple(item.id for item in self.evidence)
        if ids != self.snapshot.evidence_ids:
            raise ValueError("snapshot evidence IDs must exactly match supplied evidence order")
        for item in self.evidence:
            if not item.eligible(self.snapshot.as_of):
                raise ValueError("future evidence cannot enter a historical episode")
        allowed = set(ids)
        if any(not set(decision.evidence_ids) <= allowed for decision in self.decisions):
            raise ValueError("decision cites evidence outside its snapshot")


@dataclass(frozen=True)
class DatasetManifest:
    dataset_hash: str
    split_example_ids: Mapping[str, tuple[str, ...]]
    purged_episode_ids: tuple[str, ...]
    source_hashes: tuple[str, ...]
    train_end: str
    validation_end: str
    test_end: str


@dataclass(frozen=True)
class DatasetArtifact:
    examples: tuple[EpisodeExample, ...]
    manifest: DatasetManifest

    def write(self, output: Path) -> None:
        output.mkdir(parents=True, exist_ok=True)
        rows = "".join(canonical_json(example) + "\n" for example in self.examples)
        manifest = canonical_json(self.manifest)
        examples_path = output / "examples.jsonl"
        manifest_path = output / "manifest.json"
        examples_tmp = output / ".examples.jsonl.tmp"
        manifest_tmp = output / ".manifest.json.tmp"
        examples_tmp.write_text(rows)
        manifest_tmp.write_text(manifest)
        examples_tmp.replace(examples_path)
        manifest_tmp.replace(manifest_path)
        for split in ("train", "validation", "test"):
            split_rows = "".join(
                canonical_json(example) + "\n"
                for example in self.examples if example.split == split
            )
            temporary = output / f".{split}.jsonl.tmp"
            temporary.write_text(split_rows)
            temporary.replace(output / f"{split}.jsonl")


class HistoricalDatasetBuilder:
    """Recompute identities and keep outcome-only values outside policy prompts."""

    SYSTEM = (
        "Produce strict JSON paper-trading decisions from the supplied point-in-time "
        "snapshot and evidence. Never infer access to later market data."
    )

    def build(
        self, episodes: Iterable[HistoricalEpisode], *,
        train_end: str, validation_end: str, test_end: str,
    ) -> DatasetArtifact:
        records = tuple(episodes)
        if not records:
            raise ValueError("at least one historical episode is required")
        by_id = {item.episode_id: item for item in records}
        if len(by_id) != len(records):
            raise ValueError("episode IDs must be unique")
        split_windows = purged_split(
            (EpisodeWindow(item.episode_id, item.snapshot.as_of, item.outcome_at)
             for item in records),
            train_end=train_end, validation_end=validation_end, test_end=test_end,
        )
        assignments = {
            item.episode_id: split
            for split in ("train", "validation", "test")
            for item in split_windows[split]
        }
        examples: list[EpisodeExample] = []
        source_hashes: set[str] = set()
        for episode_id in sorted(assignments, key=lambda key: by_id[key].snapshot.as_of):
            record = by_id[episode_id]
            evidence_payload = tuple(policy_evidence(item) for item in record.evidence)
            policy_payload = {
                "as_of": record.snapshot.as_of,
                "market_session_id": record.snapshot.market_session_id,
                "universe_version": record.snapshot.universe_version,
                "feature_version": record.snapshot.feature_version,
                "account_state": record.snapshot.account_state,
                "source_health": record.snapshot.source_health,
                "evidence": evidence_payload,
            }
            _audit_payload(policy_payload)
            evidence_hash = content_hash(evidence_payload)
            source_hashes.update(content_hash(item) for item in record.evidence)
            completion = canonical_json({
                "decisions": [decision.to_dict() for decision in record.decisions],
            })
            examples.append(EpisodeExample(
                example_id=record.episode_id,
                split=assignments[record.episode_id],
                decision_at=record.snapshot.as_of,
                outcome_at=record.outcome_at,
                snapshot_id=record.snapshot.snapshot_id,
                account_state_hash=content_hash(record.snapshot.account_state),
                evidence_hash=evidence_hash,
                messages=(
                    {"role": "system", "content": self.SYSTEM},
                    {"role": "user", "content": canonical_json(policy_payload)},
                ),
                completion=completion,
            ))
        validate_chronological_splits(
            examples, train_end=train_end, validation_end=validation_end, test_end=test_end,
        )
        split_ids = {
            split: tuple(item.example_id for item in examples if item.split == split)
            for split in ("train", "validation", "test")
        }
        identity = {
            "examples": examples, "splits": split_ids,
            "purged": tuple(item.episode_id for item in split_windows["purged"]),
            "source_hashes": tuple(sorted(source_hashes)),
            "boundaries": (train_end, validation_end, test_end),
        }
        manifest = DatasetManifest(
            dataset_hash=content_hash(identity),
            split_example_ids=split_ids,
            purged_episode_ids=tuple(item.episode_id for item in split_windows["purged"]),
            source_hashes=tuple(sorted(source_hashes)),
            train_end=train_end, validation_end=validation_end, test_end=test_end,
        )
        return DatasetArtifact(tuple(examples), manifest)


def load_examples(path: Path) -> tuple[EpisodeExample, ...]:
    examples = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            row["messages"] = tuple(row["messages"])
            examples.append(EpisodeExample(**row))
        except Exception as exc:
            raise ValueError(f"invalid dataset row {line_number}") from exc
    return tuple(examples)


def load_dataset_artifact(root: Path) -> DatasetArtifact:
    """Load and verify the manifest, combined rows, and every split file."""
    root = Path(root)
    try:
        raw = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        raw["split_example_ids"] = {
            name: tuple(values) for name, values in raw["split_example_ids"].items()
        }
        raw["purged_episode_ids"] = tuple(raw["purged_episode_ids"])
        raw["source_hashes"] = tuple(raw["source_hashes"])
        manifest = DatasetManifest(**raw)
    except Exception as exc:
        raise ValueError("invalid dataset manifest") from exc
    examples = load_examples(root / "examples.jsonl")
    split_ids = {
        split: tuple(item.example_id for item in examples if item.split == split)
        for split in ("train", "validation", "test")
    }
    if split_ids != dict(manifest.split_example_ids):
        raise ValueError("dataset split IDs do not match the manifest")
    validate_chronological_splits(
        examples, train_end=manifest.train_end,
        validation_end=manifest.validation_end, test_end=manifest.test_end,
    )
    for split in ("train", "validation", "test"):
        expected = tuple(item for item in examples if item.split == split)
        if load_examples(root / f"{split}.jsonl") != expected:
            raise ValueError(f"{split} dataset file does not match examples.jsonl")
    identity = {
        "examples": examples,
        "splits": split_ids,
        "purged": manifest.purged_episode_ids,
        "source_hashes": manifest.source_hashes,
        "boundaries": (manifest.train_end, manifest.validation_end, manifest.test_end),
    }
    if content_hash(identity) != manifest.dataset_hash:
        raise ValueError("dataset content hash does not match the manifest")
    return DatasetArtifact(examples, manifest)
