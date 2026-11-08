"""Bounded research decisions shared by historical collection and paper integration."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agent.context import ContextBudgetError, ContextConfig, build_context, project_evidence_record
from agent.contracts import Evidence, content_hash, normalize_timestamp
from agent.features import PriceFeatureBuilder
from agent.policy import QwenToolPolicy, ToolRegistry
from agent.price_basis import split_adjusted_closes
from agent.tools import AssignmentTools


@dataclass(frozen=True)
class ResearchSettings:
    max_context_tokens: int = 4096
    max_new_tokens: int = 512
    max_turns: int = 6
    min_tool_calls: int = 1
    max_research_symbols: int = 8
    tool_context_reserve: int = 768
    training_template_reserve: int = 8
    context: ContextConfig = field(
        default_factory=lambda: ContextConfig(
            max_news=4,
            max_filings=2,
            max_text_chars=800,
        )
    )

    def __post_init__(self) -> None:
        for name in ("max_context_tokens", "max_new_tokens", "max_turns", "max_research_symbols"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if (
            type(self.min_tool_calls) is not int
            or self.min_tool_calls < 0
            or self.min_tool_calls >= self.max_turns
        ):
            raise ValueError("required tool research must leave a final turn")
        if type(self.training_template_reserve) is not int or self.training_template_reserve < 0:
            raise ValueError("training template headroom must be nonnegative")
        if self.max_context_tokens <= self.max_new_tokens + self.training_template_reserve:
            raise ValueError("research context must reserve final output tokens")
        if (
            type(self.tool_context_reserve) is not int
            or self.tool_context_reserve < 0
            or self.max_context_tokens - self.tool_context_reserve
            <= self.max_new_tokens + self.training_template_reserve
        ):
            raise ValueError("tool headroom must leave a usable initial research budget")


class ResearchPolicy:
    """A policy cannot submit orders or inspect any private execution outcome.

    The deterministic shortlist budgets initial deep context, not the allowed
    trading universe. The model may research any allowed symbol through tools.
    """

    def __init__(
        self,
        backend: Any,
        *,
        settings: ResearchSettings | None = None,
        mode: str = "historical",
        cache_path: Path | str = ":memory:",
    ) -> None:
        if mode not in {"historical", "live"}:
            raise ValueError("research mode must be historical or live")
        self.backend = backend
        self.settings = settings or ResearchSettings()
        self.mode, self.cache_path = mode, cache_path
        self.feature_builder = PriceFeatureBuilder()

    @property
    def policy_identity(self) -> dict[str, Any]:
        identity = dict(self.backend.policy_identity)
        root = Path(__file__).parent
        sources = (
            "research.py",
            "context.py",
            "policy/core.py",
            "policy/evidence.py",
            "policy/loader.py",
            "tools/assignment.py",
            "features.py",
            "price_basis.py",
        )
        code = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources}
        identity["protocol_hash"] = content_hash(
            {
                "version": "bounded-tool-research-v1",
                "settings": asdict(self.settings),
                "mode": self.mode,
                "code": code,
                "tokenizer_revision": getattr(self.backend, "tokenizer_revision", None),
                "candidate_rule": (
                    "held symbols, market-news mentions, then absolute trailing "
                    "five-session split-aware return"
                ),
                "decoding": "greedy strict final JSON; no post-hoc schema repair",
            }
        )
        return identity

    def _initial_context(self, evidence, universe, state, as_of):
        prices: dict[str, list[Evidence]] = defaultdict(list)
        extras = []
        for item in evidence:
            if not item.eligible(as_of):
                raise ValueError("future evidence cannot enter research tools")
            if item.source == "yfinance" and item.symbol in universe:
                prices[item.symbol].append(item)
            elif item.source != "derived.price_features":
                extras.append(item)
        for items in prices.values():
            items.sort(key=lambda item: (item.available_at, item.id))

        def score(symbol):
            rows = prices[symbol]
            closes = split_adjusted_closes(rows)
            old, new = closes[-min(6, len(closes))], closes[-1]
            return (-abs(float(new) / float(old) - 1), symbol)

        ranked = sorted(prices, key=score)
        held = [symbol for symbol in ranked if state.get("positions", {}).get(symbol, 0) > 0]
        mentions = Counter(
            symbol
            for item in extras
            if item.source in {"alpaca.news", "alpha_vantage.news"}
            for symbol in set(item.values.get("symbols", ()))
            if symbol in prices
        )
        discovered = sorted(mentions, key=lambda symbol: (-mentions[symbol], symbol))
        candidates = list(dict.fromkeys([*held, *discovered, *ranked]))
        candidates = candidates[: self.settings.max_research_symbols]
        selected = []
        for symbol in candidates:
            selected.extend(
                (
                    prices[symbol][-1],
                    self.feature_builder.build(prices[symbol], symbol=symbol, as_of=as_of),
                )
            )
        selected.extend(item for item in extras if item.symbol in {None, *candidates})
        by_id = {item.id: item for item in selected}
        required = [prices[candidates[0]][-1].id] if candidates else []
        # Preserve one real language record from each available NLP source before
        # optional records compete for the remaining tokenizer budget.
        for source in ({"alpaca.news", "alpha_vantage.news"}, {"sec.filing_text"}):
            records = [item for item in by_id.values() if item.source in source]
            if records:
                required.append(max(records, key=lambda item: (item.available_at, item.id)).id)
        return (
            tuple(by_id.values()),
            tuple(required),
            candidates,
            {symbol: float(rows[-1].values["raw"]["close"]) for symbol, rows in prices.items()},
        )

    def __call__(
        self,
        *,
        state: Mapping[str, Any],
        as_of: str,
        evidence: Sequence[Evidence],
        universe: Sequence[str],
        memories: Sequence[Mapping[str, Any]] = (),
    ) -> dict[str, Any]:
        cutoff = normalize_timestamp(as_of)
        if normalize_timestamp(state.get("as_of")) != cutoff:
            raise ValueError("research account state and tool cutoff must agree")
        universe = tuple(universe)
        initial, required, candidates, marks = self._initial_context(
            evidence, universe, state, cutoff
        )
        account = {
            **state,
            "as_of": cutoff,
            "universe": universe,
            "research_candidates": candidates,
            # Other allowed symbols remain researchable through the tools. Their
            # precise float marks need not occupy the initial mandatory envelope.
            "price_marks": {
                symbol: marks[symbol]
                for symbol in marks
                if symbol in candidates or state.get("positions", {}).get(symbol, 0) > 0
            },
            "price_mark_scope": "research_candidates_and_holdings",
        }
        factory = AssignmentTools if self.mode == "historical" else AssignmentTools.from_environment
        with factory(cutoff, universe, cache_path=self.cache_path, evidence=evidence) as tools:
            registry = ToolRegistry(cutoff)
            tools.register_tools(registry)
            try:
                context = build_context(
                    state=account,
                    evidence=initial,
                    memories=memories,
                    tools=registry.schemas,
                    token_counter=self.backend.token_count,
                    max_context_tokens=(
                        self.settings.max_context_tokens
                        - self.settings.tool_context_reserve
                        - self.settings.training_template_reserve
                    ),
                    reserved_final_tokens=self.settings.max_new_tokens,
                    config=self.settings.context,
                    required_evidence_ids=required,
                )
            except ContextBudgetError as exc:
                return {
                    "status": "context_failure",
                    "failure": "required_context_exceeds_budget",
                    "messages": [],
                    "completion": "",
                    "tool_calls": [],
                    "context_diagnostics": {
                        "minimum_input_tokens": exc.minimum_input_tokens,
                        "input_token_budget": exc.input_token_budget,
                    },
                }
            policy = QwenToolPolicy(
                self.backend,
                registry,
                max_turns=self.settings.max_turns,
                max_new_tokens=self.settings.max_new_tokens,
                token_counter=self.backend.token_count,
                max_context_tokens=(
                    self.settings.max_context_tokens - self.settings.training_template_reserve
                ),
                evidence_projector=lambda item: project_evidence_record(
                    item, as_of=cutoff, config=self.settings.context
                ),
                continue_tool_errors=True,
                min_tool_calls=self.settings.min_tool_calls,
                secret_values=getattr(tools, "_secrets", ()),
            )
            result = policy.run(
                context.packed,
                allowed_symbols=universe,
                original_messages=context.original_messages
                if self.settings.context.short_evidence_ids
                else None,
            )
        transcript = result.transcript
        outputs = transcript["raw_outputs"]
        return {
            "status": result.status,
            "failure": result.failure,
            "decisions": result.decisions,
            "messages": transcript["generation_messages"],
            "completion": outputs[-1] if outputs else "",
            "tool_calls": transcript["tool_calls"],
            "original_messages": transcript["original_messages"],
            "evidence_aliases": transcript["evidence_aliases"],
            "source_status": dict(context.source_status),
            "transcript": transcript,
            "initial_input_tokens": context.packed.input_tokens,
            "initial_excluded_evidence": context.packed.excluded_evidence_ids,
        }
