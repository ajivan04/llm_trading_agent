"""Build a purged, content-addressed training dataset from historical episodes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent.contracts import Decision, Evidence, Snapshot
from agent.datasets import HistoricalDatasetBuilder, HistoricalEpisode


def _load(path: Path) -> tuple[HistoricalEpisode, ...]:
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            snapshot_row = dict(row["snapshot"])
            snapshot_row["evidence_ids"] = tuple(snapshot_row["evidence_ids"])
            snapshot_row["compact_evidence"] = tuple(snapshot_row.get("compact_evidence", ()))
            records.append(HistoricalEpisode(
                episode_id=row["episode_id"],
                snapshot=Snapshot(**snapshot_row),
                evidence=tuple(Evidence(**item) for item in row["evidence"]),
                decisions=tuple(Decision(
                    item["symbol"], item["action"], item["qty"], item["reason"],
                    tuple(item.get("evidence_ids", ())),
                ) for item in row["decisions"]),
                outcome_at=row["outcome_at"],
            ))
        except Exception as exc:
            raise ValueError(f"invalid historical episode at {path}:{line_number}") from exc
    return tuple(records)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--validation-end", required=True)
    parser.add_argument("--test-end", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact = HistoricalDatasetBuilder().build(
        _load(args.input), train_end=args.train_end,
        validation_end=args.validation_end, test_end=args.test_end,
    )
    artifact.write(args.output)
    print(json.dumps({
        "dataset_hash": artifact.manifest.dataset_hash,
        "examples": len(artifact.examples),
        "purged": len(artifact.manifest.purged_episode_ids),
        "output": str(args.output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
