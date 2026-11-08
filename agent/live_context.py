"""Archive bootstrap plus bounded live prefetch before the immutable decision cutoff."""

import hashlib
import json
from collections import defaultdict, deque
from datetime import UTC, datetime, timedelta
from pathlib import Path

from agent.contracts import Evidence, canonical_json, content_hash, parse_timestamp
from agent.data.prices import DailyPriceBar
from agent.price_basis import split_adjusted_closes
from agent.tools import AssignmentTools


class LiveResearchSeed:
    def __init__(self, prepared, *, cache_path, tools_factory=None, clock=None, max_symbols=8):
        self.prepared, self.cache_path = Path(prepared), Path(cache_path)
        self.clock = clock or (lambda: datetime.now(UTC))
        self.tools_factory = tools_factory or AssignmentTools.from_environment
        if type(max_symbols) is not int or not 1 <= max_symbols <= 10:
            raise ValueError("live source budget must be 1..10 symbols")
        self.max_symbols = max_symbols
        self.records, self.status = {}, {}
        paths = [self.prepared / "price-bars.jsonl", self.prepared / "evidence.jsonl"]
        before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        prices = defaultdict(lambda: deque(maxlen=130))
        now = self.clock()
        with paths[0].open() as handle:
            previous = {}
            for line in handle:
                row = json.loads(line)
                if row.get("price_basis") != "nominal":
                    raise ValueError("archive must declare normalized nominal share units")
                bar = DailyPriceBar(**row)
                if bar.session <= previous.get(bar.symbol, ""):
                    raise ValueError("archive prices are duplicated or unordered")
                previous[bar.symbol] = bar.session
                if parse_timestamp(bar.close_at) <= now:
                    prices[bar.symbol].append(bar.to_evidence())
        for rows in prices.values():
            for item in rows:
                self.records[item.id] = item
        with paths[1].open() as handle:
            for line in handle:
                item = Evidence(**json.loads(line))
                # Do not repurpose old aggregate-only news as recovered article text.
                if (
                    item.source
                    in {"alpaca.news", "alpha_vantage.news", "sec.filing_text", "sec.submissions"}
                    and item.eligible(now)
                    and now - parse_timestamp(item.available_at) <= timedelta(days=60)
                ):
                    self.records[item.id] = item
        after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        if before != after:
            raise ValueError("archive changed during bootstrap; finish the transfer first")
        self.source_hash = content_hash(
            {
                "archive_hashes": before,
                "live_symbols": max_symbols,
                "prefetch": "held first then absolute five-session split-aware return",
            }
        )

    def refresh(self, universe, state):
        prices = defaultdict(list)
        for item in self(self.clock().isoformat(), universe):
            if item.source == "yfinance" and item.symbol in universe:
                prices[item.symbol].append(item)

        def rank(symbol):
            rows = sorted(prices[symbol], key=lambda r: r.values["session"])
            closes = split_adjusted_closes(rows)
            return (-abs(closes[-1] / closes[-min(6, len(closes))] - 1), symbol)

        ranked = sorted(prices, key=rank)
        held = [s for s in universe if state.positions.get(s, 0)]
        cutoff = self.clock().isoformat()
        with self.tools_factory(cutoff, universe, cache_path=self.cache_path) as tools:
            market = json.loads(tools.search_market_news(limit=20))
            discovered = [s for s in market.get("candidate_symbols", ()) if s in universe]
            selected = tuple(dict.fromkeys([*held, *discovered, *ranked]))[: self.max_symbols]
            self.status = {
                "candidate_symbols": selected,
                "budget": self.max_symbols,
                "market_news": market["status"],
            }
            for row in market.get("evidence", ()):
                item = Evidence(**row)
                self.records[item.id] = item
            if selected:
                news = json.loads(tools.get_news_for_symbols(selected, limit=8))
                self.status["news"] = news["status"]
                for row in news["evidence"]:
                    item = Evidence(**row)
                    self.records[item.id] = item
            for symbol in selected:
                try:
                    bars, _ = tools._bars(symbol, "6mo")
                    # Receipt time may be after the prefetch cutoff. These enter
                    # only the LATER policy snapshot, never the earlier tool turn.
                    for item in bars:
                        self.records[item.id] = item
                    self.status[symbol + ":prices"] = "ok" if bars else "missing"
                except Exception:
                    self.status[symbol + ":prices"] = "provider_error"
                filings = json.loads(tools.get_sec_filings(symbol))
                self.status[symbol + ":sec"] = filings["status"]
                for row in filings["evidence"]:
                    item = Evidence(**row)
                    self.records[item.id] = item

    def __call__(self, cutoff, universe):
        now = parse_timestamp(cutoff)
        prices, extras = {}, []
        for item in self.records.values():
            if item.symbol not in {*universe, None} or not item.eligible(cutoff):
                continue
            if item.source == "yfinance":
                key = (item.symbol, item.values["session"])
                old = prices.get(key)
                if old is None or parse_timestamp(item.retrieved_at) > parse_timestamp(
                    old.retrieved_at
                ):
                    prices[key] = item
            elif now - parse_timestamp(item.available_at) <= timedelta(days=60):
                extras.append(item)
        groups = defaultdict(list)
        for item in prices.values():
            groups[(item.symbol, canonical_json(item.values["adjustment_policy"]))].append(item)
        selected = []
        for symbol in universe:
            variants = [rows for (stock, _), rows in groups.items() if stock == symbol]
            if variants:
                # A live capture and an archive must not silently mix adjustment policies.
                rows = max(
                    variants, key=lambda items: max(parse_timestamp(r.retrieved_at) for r in items)
                )
                selected.extend(sorted(rows, key=lambda r: r.values["session"])[-130:])
        return tuple(selected) + tuple(extras)
