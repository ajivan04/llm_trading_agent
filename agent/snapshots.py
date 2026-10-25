"""Point-in-time snapshot assembly and canonical identity."""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from typing import Any

from .contracts import Evidence, Snapshot, content_hash, normalize_timestamp


class SnapshotBuilder:
    """Validate and freeze the exact evidence payload exposed to the policy."""

    def build(
        self,
        *,
        as_of: str | datetime,
        universe: Sequence[str],
        universe_version: str,
        evidence: Iterable[Evidence],
        compact_evidence: Sequence[Mapping[str, Any]],
        feature_version: str,
        account_state: Mapping[str, Any],
        source_health: Mapping[str, Any],
        market_session_id: str,
    ) -> Snapshot:
        cutoff = normalize_timestamp(as_of)
        if isinstance(universe, (str, bytes)):
            raise TypeError("universe must be a sequence of symbols")
        if not all(isinstance(symbol, str) for symbol in universe):
            raise TypeError("universe symbols must be strings")
        normalized_universe = tuple(symbol.strip().upper() for symbol in universe)
        if not normalized_universe:
            raise ValueError("universe must not be empty")
        if any(not symbol for symbol in normalized_universe):
            raise ValueError("universe symbols must be nonempty")
        if len(normalized_universe) != len(set(normalized_universe)):
            raise ValueError("universe symbols must be unique")
        universe_set = set(normalized_universe)
        ordered_evidence = tuple(evidence)
        ids = tuple(item.id for item in ordered_evidence)
        if len(ids) != len(set(ids)):
            raise ValueError("evidence IDs must be unique")
        for item in ordered_evidence:
            if not item.eligible(cutoff):
                raise ValueError(f"evidence {item.id!r} is not eligible at snapshot cutoff")
            if item.symbol is not None and item.symbol not in universe_set:
                raise ValueError(f"evidence symbol {item.symbol!r} is outside the universe")
            if item.source == "derived.price_features":
                evidence_version = item.values.get("feature_version")
                if (evidence_version != feature_version
                        or item.transformation_version != feature_version):
                    raise ValueError(
                        f"feature evidence {item.id!r} does not match feature_version"
                    )
        compact = tuple(compact_evidence)
        if not all(isinstance(item, Mapping) for item in compact):
            raise TypeError("compact_evidence must contain mappings")
        for item in compact:
            reference = item.get("evidence_id", item.get("id"))
            if reference is not None and reference not in ids:
                raise ValueError(
                    f"compact evidence references unknown evidence ID {reference!r}")

        identity = {
            "as_of": cutoff,
            "universe": normalized_universe,
            "universe_version": universe_version,
            "evidence_ids": ids,
            "evidence_fingerprints": tuple(content_hash(item) for item in ordered_evidence),
            "feature_version": feature_version,
            "account_state": account_state,
            "source_health": source_health,
            "market_session_id": market_session_id,
            "compact_evidence": compact,
        }
        return Snapshot(
            snapshot_id=f"snapshot-{content_hash(identity)}",
            as_of=cutoff,
            universe_version=universe_version,
            evidence_ids=ids,
            feature_version=feature_version,
            account_state=account_state,
            source_health=source_health,
            market_session_id=market_session_id,
            compact_evidence=compact,
        )


def build_snapshot(**kwargs: Any) -> Snapshot:
    return SnapshotBuilder().build(**kwargs)
