"""Deterministic, point-in-time evidence compaction for policy prompts."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Mapping

from .contracts import Evidence, canonical_json, content_hash, parse_timestamp

COMPACTION_VERSION = "evidence-compaction-v2"
_NEWS_SOURCES = {"alpaca.news", "alpha_vantage.news"}
_EVENT_TERMS = {
    "earnings": ("earnings", "revenue", "profit", "guidance", "forecast"),
    "ai_product": ("artificial intelligence", " ai ", "gpu", "semiconductor", "model"),
    "m_and_a": ("acquisition", "acquire", "merger", "takeover"),
    "regulation": ("regulator", "regulatory", "sec", "lawsuit", "litigation"),
    "capital": ("capex", "capital expenditure", "liquidity", "financing"),
}


@dataclass(frozen=True)
class CompactionConfig:
    max_news: int = 8
    max_filings: int = 3
    max_text_chars: int = 1200
    recency_half_life_days: float = 14.0

    def __post_init__(self) -> None:
        if any(type(value) is not int or value <= 0
               for value in (self.max_news, self.max_filings, self.max_text_chars)):
            raise ValueError("compaction limits must be positive integers")
        if self.recency_half_life_days <= 0:
            raise ValueError("recency_half_life_days must be positive")


def _event_types(text: str) -> tuple[str, ...]:
    lowered = f" {text.casefold()} "
    return tuple(name for name, terms in _EVENT_TERMS.items()
                 if any(term in lowered for term in terms))


def _news_score(item: Evidence, cutoff: datetime, config: CompactionConfig) -> float:
    age = max(0.0, (cutoff - parse_timestamp(item.available_at)).total_seconds() / 86400)
    recency = math.exp(-math.log(2) * age / config.recency_half_life_days)
    relevance = float(item.values.get("relevance_score", 1.0) or 1.0)
    event_bonus = 1.0 + 0.1 * len(_event_types(item.text))
    return recency * max(0.0, min(1.0, relevance)) * event_bonus


def _compact_item(item: Evidence, *, max_text_chars: int) -> dict[str, Any]:
    text = item.text.strip()[:max_text_chars]
    values: dict[str, Any] = {}
    if item.source in _NEWS_SOURCES:
        values = {
            key: item.values[key] for key in (
                "title", "summary", "publisher", "published_at", "relevance_score",
                "sentiment_score", "sentiment_label",
            ) if key in item.values
        }
        values["event_types"] = _event_types(text)
    elif item.source.startswith("sec."):
        values = {
            key: item.values[key] for key in (
                "form", "base_form", "filing_date", "report_date", "accession",
                "section",
            ) if key in item.values
        }
        values["excerpt"] = text
    else:
        values = dict(item.values)
    return {
        "id": item.id,
        "source": item.source,
        "symbol": item.symbol,
        "observed_at": item.observed_at,
        "available_at": item.available_at,
        "text": text,
        "values": values,
    }


def compact_evidence(
    evidence: Iterable[Evidence], *, as_of: str | datetime,
    config: CompactionConfig = CompactionConfig(),
) -> tuple[dict[str, Any], ...]:
    """Return a deterministic policy-visible subset of eligible evidence.

    Full records remain unchanged in the archive. Every returned record retains
    its source evidence ID and availability timestamp, and future evidence is
    rejected rather than silently included.
    """
    cutoff = parse_timestamp(as_of)
    items = tuple(evidence)
    if len({item.id for item in items}) != len(items):
        raise ValueError("evidence IDs must be unique")
    if any(not item.eligible(cutoff) for item in items):
        raise ValueError("future evidence cannot be compacted")
    news = [item for item in items if item.source in _NEWS_SOURCES]
    filings = [item for item in items if item.source.startswith("sec.")]
    other = [item for item in items if item not in news and item not in filings]
    # Deduplicate identical article content while keeping the earliest stable ID.
    seen_content: set[str] = set()
    ranked_news = []
    for item in news:
        digest = str(item.raw_content_hash or content_hash(item.text))
        if digest in seen_content:
            continue
        seen_content.add(digest)
        ranked_news.append(item)
    ranked_news.sort(key=lambda item: (-_news_score(item, cutoff, config), item.id))
    selected_news = ranked_news[:config.max_news]

    # Prefer filing text over metadata for the same accession and then rank by recency.
    by_filing: dict[str, Evidence] = {}
    for item in filings:
        key = item.accession_id or item.id
        current = by_filing.get(key)
        if current is None or (item.source == "sec.filing_text" and current.source != item.source):
            by_filing[key] = item
    selected_filings = sorted(
        by_filing.values(), key=lambda item: (item.available_at, item.id), reverse=True
    )[:config.max_filings]
    selected = [*other, *selected_news, *selected_filings]
    selected.sort(key=lambda item: (item.available_at, item.id))
    return tuple(_compact_item(item, max_text_chars=config.max_text_chars) for item in selected)


def compaction_manifest(records: Iterable[Mapping[str, Any]], *, as_of: str | datetime,
                        config: CompactionConfig = CompactionConfig()) -> dict[str, Any]:
    rows = tuple(records)
    counts = Counter(str(row.get("source", "unknown")) for row in rows)
    return {
        "compaction_version": COMPACTION_VERSION,
        "as_of": parse_timestamp(as_of).isoformat(),
        "config": {
            "max_news": config.max_news, "max_filings": config.max_filings,
            "max_text_chars": config.max_text_chars,
            "recency_half_life_days": config.recency_half_life_days,
        },
        "record_count": len(rows),
        "source_counts": dict(sorted(counts.items())),
        "compaction_hash": content_hash(rows),
    }
