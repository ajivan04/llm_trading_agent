"""Token-aware point-in-time research context with exact citation provenance."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from agent.compaction import CompactionConfig, compact_evidence
from agent.contracts import (
    Evidence,
    FrozenDict,
    canonical_json,
    normalize_timestamp,
    parse_timestamp,
)
from agent.datasets import _audit_payload, policy_evidence
from agent.policy.core import PackedPrompt
from agent.policy.evidence import DECISION_INSTRUCTIONS, alias_messages

_NEWS = {"alpaca.news", "alpha_vantage.news"}
TOOL_INSTRUCTIONS = (
    "Research the supplied universe using registered tools before the final decision. "
    "Choose a relevant registered tool and return only a JSON tool request, for example "
    '{"tool":"search_market_news","arguments":{"limit":10}}. '
    "Use only a tool and parameters in the "
    "supplied registry. After receiving the result, research further if needed, or return "
    "a final decision when there is enough evidence. Never change the bound decision date. "
    "A tool result may be missing or quota limited; do not invent missing information. "
    "After research, return the final decisions in the following schema. " + DECISION_INSTRUCTIONS
)


class ContextBudgetError(ValueError):
    """Safe numeric diagnostics for an unsatisfiable mandatory input budget."""

    def __init__(self, minimum_input_tokens, input_token_budget):
        super().__init__("required evidence/system/tools exceed the configured context budget")
        self.minimum_input_tokens = minimum_input_tokens
        self.input_token_budget = input_token_budget


@dataclass(frozen=True)
class ContextConfig:
    max_news: int = 8
    max_filings: int = 3
    max_text_chars: int = 1200
    max_memories: int = 2
    short_evidence_ids: bool = True

    def __post_init__(self) -> None:
        CompactionConfig(self.max_news, self.max_filings, self.max_text_chars)
        if type(self.max_memories) is not int or self.max_memories < 0:
            raise ValueError("max_memories must be a nonnegative integer")
        if type(self.short_evidence_ids) is not bool:
            raise ValueError("short_evidence_ids must be boolean")


@dataclass(frozen=True)
class ContextPrompt:
    packed: PackedPrompt
    original_messages: tuple[Mapping[str, Any], ...]
    aliases: Mapping[str, str]
    available_evidence: tuple[Evidence, ...]
    source_status: Mapping[str, str]
    excluded_memories: int = 0


def project_evidence_record(
    item: Evidence,
    *,
    as_of: str,
    config: ContextConfig | None = None,
) -> dict[str, Any]:
    """Remove repeated/audit metadata without altering the archived Evidence."""
    config = config or ContextConfig()
    projected = policy_evidence(item)
    projected["retrieved_at"] = item.retrieved_at
    projected["transformation_version"] = item.transformation_version
    bounded = compact_evidence(
        [Evidence(**projected)],
        as_of=as_of,
        config=CompactionConfig(config.max_news, config.max_filings, config.max_text_chars),
    )[0]
    values = dict(bounded["values"])
    # The text already contains the title/summary/filing excerpt. Duplicating
    # those potentially unbounded strings defeats compaction.
    if item.source in _NEWS:
        values.pop("title", None)
        values.pop("summary", None)
    if item.source.startswith("sec."):
        values.pop("excerpt", None)
    values.pop("source_evidence_ids", None)
    values.pop("source_evidence_fingerprints", None)
    bounded["values"] = values
    return bounded


def _source_status(selected: Sequence[dict[str, Any]], pool: Sequence[Evidence]) -> dict[str, str]:
    sources = {item["source"] for item in selected}
    pool_sources = {item.source for item in pool}
    if sources & _NEWS:
        news = "raw_text"
    elif pool_sources & _NEWS:
        news = "omitted_by_budget"
    elif "derived.news_aggregate" in pool_sources:
        news = "aggregate_only"
    else:
        news = "missing"
    if "sec.filing_text" in sources:
        sec = "text"
    elif any(source.startswith("sec.") for source in sources):
        sec = "metadata_only"
    elif any(source.startswith("sec.") for source in pool_sources):
        sec = "omitted_by_budget"
    else:
        sec = "missing"
    return {"news": news, "sec": sec}


def build_context(
    *,
    state: Mapping[str, Any],
    evidence: Iterable[Evidence],
    memories: Sequence[Mapping[str, Any]] = (),
    tools: Sequence[Mapping[str, Any]] = (),
    token_counter: Callable[[Sequence[Mapping[str, Any]]], int],
    max_context_tokens: int,
    reserved_final_tokens: int,
    config: ContextConfig | None = None,
    required_evidence_ids: Iterable[str] = (),
) -> ContextPrompt:
    """Build bounded inputs and retain the exact full-ID transcript for auditing.

    Required evidence is never silently removed. Alias assignment is append-stable
    while candidates are tested: a rejected candidate cannot renumber previously
    selected references. No input is truncated at the JSON/token level.
    """
    config = config or ContextConfig()
    if (
        type(max_context_tokens) is not int
        or type(reserved_final_tokens) is not int
        or max_context_tokens <= 0
        or reserved_final_tokens <= 0
    ):
        raise ValueError("context and output budgets must be positive integers")
    budget = max_context_tokens - reserved_final_tokens
    if budget <= 0:
        raise ValueError("reserved output leaves no research context capacity")
    as_of = normalize_timestamp(state.get("as_of"))
    state_payload = dict(state)
    state_payload["as_of"] = as_of
    _audit_payload(state_payload)
    pool = tuple(evidence)
    if any(not isinstance(item, Evidence) for item in pool):
        raise TypeError("context evidence must contain Evidence records")
    by_id = {item.id: item for item in pool}
    if len(by_id) != len(pool):
        raise ValueError("context evidence IDs must be unique")
    if any(not item.eligible(as_of) for item in pool):
        raise ValueError("future evidence cannot enter research context")
    required = set(required_evidence_ids)
    if not required <= set(by_id):
        raise ValueError("required context evidence is unavailable")
    for memory in memories:
        if (
            not isinstance(memory, Mapping)
            or "available_at" not in memory
            or parse_timestamp(memory["available_at"]) > parse_timestamp(as_of)
        ):
            raise ValueError("reflection memory must be historically available at the cutoff")
        _audit_payload(memory)
    selected_memories = list(memories[: config.max_memories])
    compaction = CompactionConfig(config.max_news, config.max_filings, config.max_text_chars)
    compacted_ids = {row["id"] for row in compact_evidence(pool, as_of=as_of, config=compaction)}
    compacted_ids |= required
    bounded = {
        eid: project_evidence_record(by_id[eid], as_of=as_of, config=config)
        for eid in compacted_ids
    }
    for record in bounded.values():
        _audit_payload(record)

    def priority(eid: str):
        source = by_id[eid].source
        rank = (
            0
            if eid in required
            else 1
            if source in _NEWS
            else 2
            if source == "sec.filing_text"
            else 3
            if source == "derived.price_features"
            else 4
            if source == "yfinance"
            else 6
            if source.startswith("sec.")
            else 7
        )
        return (rank, -parse_timestamp(by_id[eid].available_at).timestamp(), eid)

    ordered = sorted(compacted_ids, key=priority)
    selected = [bounded[eid] for eid in ordered if eid in required]

    def render(records):
        payload = {
            "state": state_payload,
            "memories": selected_memories,
            "tools": list(tools),
            "evidence": records,
            "evidence_status": _source_status(records, pool),
            "evidence_boundary": "UNTRUSTED_DATA",
        }
        original = [
            {"role": "system", "content": TOOL_INSTRUCTIONS},
            {"role": "user", "content": canonical_json(payload)},
        ]
        messages, aliases = (
            alias_messages(original) if config.short_evidence_ids else (original, {})
        )
        tokens = token_counter(messages)
        if type(tokens) is not int or tokens < 0:
            raise ValueError("token_counter must return a nonnegative integer")
        return messages, original, aliases, tokens

    # Low-ranked memories can be omitted; required point-in-time evidence cannot.
    while True:
        _, _, _, minimum_tokens = render(selected)
        if minimum_tokens <= budget:
            break
        if selected_memories:
            selected_memories.pop()
            continue
        raise ContextBudgetError(minimum_tokens, budget)
    for eid in ordered:
        if eid in required:
            continue
        candidate = [*selected, bounded[eid]]
        _, _, _, tokens = render(candidate)
        if tokens <= budget:
            selected = candidate
    messages, original, aliases, tokens = render(selected)
    included_full = tuple(row["id"] for row in selected)
    included = tuple(aliases) if config.short_evidence_ids else included_full
    excluded = tuple(sorted(set(by_id) - set(included_full)))
    packed = PackedPrompt(
        messages=tuple(messages),
        included_evidence_ids=included,
        excluded_evidence_ids=excluded,
        input_tokens=tokens,
        reserved_final_tokens=reserved_final_tokens,
    )
    return ContextPrompt(
        packed=packed,
        original_messages=tuple(original),
        aliases=FrozenDict(aliases),
        available_evidence=tuple(by_id[eid] for eid in included_full),
        source_status=FrozenDict(_source_status(selected, pool)),
        excluded_memories=len(memories) - len(selected_memories),
    )
