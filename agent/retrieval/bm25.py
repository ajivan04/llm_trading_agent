r"""Deterministic, storage-independent BM25 retrieval for reflection memory.

Tokenization uses Unicode ``\w`` runs after :meth:`str.casefold`. Thus letters,
digits, and underscores are tokens, while punctuation is a separator. Query
terms are treated as a set; repeating a term does not increase its influence.

The caller owns persistence and supplies records directly, or supplies a callable
that loads records for a requested run. Only records for that run whose outcome
was available at ``as_of`` participate in corpus statistics or ranking.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
import math
import re


_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def _require_aware(value: datetime, name: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be a timezone-aware datetime")


def tokenize(text: str) -> tuple[str, ...]:
    """Return case-insensitive Unicode word tokens in their original order."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return tuple(_TOKEN_RE.findall(text.casefold()))


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    """The fields needed to rank one persisted reflection-memory entry."""

    id: str
    experiment_id: str
    run_id: str
    fold: str
    available_at: datetime
    situation: str
    reflection: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("memory id must be a non-empty string")
        for value, name in (
            (self.experiment_id, "experiment_id"),
            (self.run_id, "run_id"),
            (self.fold, "fold"),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        _require_aware(self.available_at, "available_at")
        if not isinstance(self.situation, str):
            raise TypeError("situation must be a string")
        if not isinstance(self.reflection, str):
            raise TypeError("reflection must be a string")


@dataclass(frozen=True, slots=True)
class RankedMemory:
    """A retrieved memory and its positive BM25 relevance score."""

    memory: MemoryRecord
    score: float


MemoryLoader = Callable[[str, str, str], Iterable[MemoryRecord]]


def rank_bm25(
    records: Iterable[MemoryRecord],
    *,
    experiment_id: str,
    run_id: str,
    fold: str,
    query: str,
    as_of: datetime,
    k: int = 2,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[RankedMemory]:
    """Rank eligible records with Okapi BM25.

    Ties are resolved by ascending memory ID. Records with no matching query
    term are omitted instead of returning arbitrary zero-score results.
    """

    for value, name in (
        (experiment_id, "experiment_id"), (run_id, "run_id"), (fold, "fold")
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string")
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    _require_aware(as_of, "as_of")
    if not isinstance(k1, (int, float)) or isinstance(k1, bool) or k1 <= 0:
        raise ValueError("k1 must be positive")
    if not isinstance(b, (int, float)) or isinstance(b, bool) or not 0 <= b <= 1:
        raise ValueError("b must be between 0 and 1")

    # Sorting makes floating-point accumulation and therefore ranking reproducible
    # across processes regardless of hash randomization.
    query_terms = tuple(sorted(set(tokenize(query))))
    if not query_terms:
        return []

    eligible: list[tuple[MemoryRecord, Counter[str]]] = []
    for record in records:
        if not isinstance(record, MemoryRecord):
            raise TypeError("records must contain MemoryRecord instances")
        # MemoryRecord validates awareness, making this comparison unambiguous.
        if (record.experiment_id == experiment_id and record.run_id == run_id
                and record.fold == fold and record.available_at <= as_of):
            eligible.append((record, Counter(tokenize(record.situation))))

    if not eligible:
        return []

    document_count = len(eligible)
    average_length = sum(sum(terms.values()) for _, terms in eligible) / document_count
    document_frequency = {
        term: sum(1 for _, terms in eligible if terms[term] > 0)
        for term in query_terms
    }

    ranked: list[RankedMemory] = []
    for record, terms in eligible:
        document_length = sum(terms.values())
        score = 0.0
        for term in query_terms:
            frequency = terms[term]
            if not frequency:
                continue
            frequency_in_documents = document_frequency[term]
            inverse_document_frequency = math.log(
                1.0 + (document_count - frequency_in_documents + 0.5)
                / (frequency_in_documents + 0.5)
            )
            length_ratio = document_length / average_length if average_length else 0.0
            denominator = frequency + k1 * (1.0 - b + b * length_ratio)
            score += inverse_document_frequency * frequency * (k1 + 1.0) / denominator
        if score > 0.0:
            ranked.append(RankedMemory(record, score))

    ranked.sort(key=lambda result: (-result.score, result.memory.id))
    return ranked[:k]


class BM25Retriever:
    """Thin adapter around a caller-provided run-scoped storage loader."""

    def __init__(self, load_memories: MemoryLoader):
        if not callable(load_memories):
            raise TypeError("load_memories must be callable")
        self._load_memories = load_memories

    def retrieve(
        self, experiment_id: str, run_id: str, fold: str, query: str,
        as_of: datetime, k: int = 2,
    ) -> list[RankedMemory]:
        """Load and rank one experiment/run/fold memory corpus."""

        return rank_bm25(
            self._load_memories(experiment_id, run_id, fold),
            experiment_id=experiment_id, run_id=run_id, fold=fold,
            query=query, as_of=as_of, k=k,
        )
