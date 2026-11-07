"""Durable, split-isolated BM25 memory for confirmed generated trade lessons."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent.contracts import canonical_json, content_hash, normalize_timestamp, parse_timestamp
from agent.reflection import ClosedTradeOutcome, ReflectionResult
from agent.retrieval.bm25 import MemoryRecord, rank_bm25


class TradingMemory:
    """Notebook-facing memory names with explicit scope, provenance and cutoff.

    A file may contain multiple namespaces, but retrieval never shares their
    corpus statistics. Existing records are immutable. Repeating the same
    confirmed outcome preserves its first successful lesson instead of retrying
    generation until a desired conclusion is obtained.
    """

    def __init__(
        self,
        memory_file: str | Path = "trading_memory.sqlite3",
        *,
        experiment_id: str = "paper",
        run_id: str = "paper",
        fold: str = "paper",
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if fold not in {"train", "validation", "test", "paper"}:
            raise ValueError("memory fold must be train, validation, test, or paper")
        if any(
            not isinstance(value, str) or not value.strip() for value in (experiment_id, run_id)
        ):
            raise ValueError("memory requires explicit nonempty experiment/run scope")
        self.memory_file = str(memory_file)
        self.experiment_id, self.run_id, self.fold = experiment_id, run_id, fold
        self.clock = clock
        if self.memory_file != ":memory:":
            Path(self.memory_file).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.memory_file, timeout=30)
        try:
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.execute("""CREATE TABLE IF NOT EXISTS reflection_memories (
                memory_id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL,
                run_id TEXT NOT NULL, fold TEXT NOT NULL, available_at TEXT NOT NULL,
                payload TEXT NOT NULL, payload_hash TEXT NOT NULL)""")
            self.db.commit()
        except Exception:
            self.db.close()
            raise ValueError("memory store is invalid; preserve it for investigation") from None

    def close(self) -> None:
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _rows(self) -> list[dict[str, Any]]:
        result = []
        for memory_id, available_at, raw, digest in self.db.execute(
            "SELECT memory_id,available_at,payload,payload_hash FROM reflection_memories "
            "WHERE experiment_id=? AND run_id=? AND fold=? ORDER BY memory_id",
            (self.experiment_id, self.run_id, self.fold),
        ):
            try:
                row = json.loads(raw)
                if (
                    content_hash(row) != digest
                    or row["id"] != memory_id
                    or row["available_at"] != available_at
                    or (row["experiment_id"], row["run_id"], row["fold"])
                    != (self.experiment_id, self.run_id, self.fold)
                ):
                    raise ValueError("memory integrity mismatch")
            except (TypeError, KeyError, ValueError):
                raise ValueError("memory store integrity check failed") from None
            result.append(row)
        return result

    @property
    def memories(self) -> list[dict[str, Any]]:
        """Diagnostic scoped archive; policy callers must use cutoff retrieval."""
        return self._rows()

    def add_memory(
        self,
        situation: str,
        reflection: str,
        *,
        outcome: ClosedTradeOutcome,
        result: ReflectionResult,
    ) -> str:
        """Persist a successful generated lesson for one matched matured lot."""
        if not isinstance(outcome, ClosedTradeOutcome) or not isinstance(result, ReflectionResult):
            raise TypeError("memory requires confirmed outcome and generated reflection provenance")
        if (
            result.status != "success"
            or result.outcome_id != outcome.trade_id
            or result.outcome_hash != outcome.outcome_hash
            or reflection != result.lesson
            or not reflection.strip()
            or not isinstance(situation, str)
            or not situation.strip()
            or len(situation) > 1200
            or result.situation_hash != content_hash(situation.strip())
        ):
            raise ValueError("memory reflection does not match its confirmed generated outcome")
        at = normalize_timestamp(result.available_at)
        if not (
            parse_timestamp(outcome.confirmed_at)
            <= parse_timestamp(at)
            <= parse_timestamp(self.clock())
        ):
            raise ValueError(
                "memory outcome must be confirmed and lesson available by current time"
            )
        expected_source = "alpaca.paper" if self.fold == "paper" else "historical.replay"
        if outcome.source != expected_source:
            raise ValueError("paper and replay reflection provenance must remain isolated")
        identity = {
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "fold": self.fold,
            "outcome_id": outcome.trade_id,
        }
        memory_id = content_hash(identity)
        payload = {
            "id": memory_id,
            **identity,
            "available_at": at,
            "situation": situation,
            "reflection": reflection,
            "uncertainty": result.uncertainty,
            "outcome": outcome.__dict__,
            "outcome_hash": outcome.outcome_hash,
            "policy_identity": result.policy_identity,
            "net_realized_pnl": outcome.net_realized_pnl,
            "reflection_prompt_hash": result.prompt_hash,
        }
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            previous = self.db.execute(
                "SELECT payload,payload_hash FROM reflection_memories WHERE memory_id=?",
                (memory_id,),
            ).fetchone()
            if previous:
                saved = json.loads(previous[0])
                if (
                    content_hash(saved) != previous[1]
                    or saved["outcome_hash"] != outcome.outcome_hash
                    or saved["policy_identity"] != result.policy_identity
                ):
                    raise ValueError("existing memory has changed outcome or policy identity")
                return memory_id
            self.db.execute(
                "INSERT INTO reflection_memories VALUES (?,?,?,?,?,?,?)",
                (
                    memory_id,
                    self.experiment_id,
                    self.run_id,
                    self.fold,
                    at,
                    canonical_json(payload),
                    content_hash(payload),
                ),
            )
        return memory_id

    def context_memories(self, query: str, *, as_of: str, n_matches: int = 2):
        """Return only eligible top-k metadata for the research prompt boundary."""
        at = parse_timestamp(as_of)
        rows = self._rows()
        by_id = {row["id"]: row for row in rows}
        ranked = rank_bm25(
            [
                MemoryRecord(
                    row["id"],
                    row["experiment_id"],
                    row["run_id"],
                    row["fold"],
                    parse_timestamp(row["available_at"]),
                    row["situation"],
                    row["reflection"],
                )
                for row in rows
            ],
            experiment_id=self.experiment_id,
            run_id=self.run_id,
            fold=self.fold,
            query=query,
            as_of=at,
            k=n_matches,
        )
        return [
            {
                "id": item.memory.id,
                "available_at": normalize_timestamp(item.memory.available_at),
                "situation": item.memory.situation,
                "reflection": item.memory.reflection,
                "uncertainty": by_id[item.memory.id]["uncertainty"],
                "score": item.score,
            }
            for item in ranked
        ]

    def get_memories(
        self, query: str, n_matches: int = 2, *, as_of: str | None = None
    ) -> list[str]:
        """Notebook list-of-strings interface; replay must supply its logical cutoff."""
        if as_of is None:
            if self.fold != "paper":
                raise ValueError("replay memory retrieval requires an explicit as_of")
            as_of = normalize_timestamp(self.clock())
        return [
            row["reflection"]
            for row in self.context_memories(query, as_of=as_of, n_matches=n_matches)
        ]

    @property
    def memory_digest(self) -> str:
        return content_hash(self._rows())
