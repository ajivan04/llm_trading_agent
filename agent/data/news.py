"""Point-in-time Alpha Vantage news ingestion and deterministic aggregation."""
from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import urlopen

from agent.contracts import Evidence, content_hash, normalize_timestamp, parse_timestamp
from agent.data.cache import CacheIdentity, SqliteResponseCache


Request = Callable[[Mapping[str, str]], Mapping[str, Any]]
_ENDPOINT = "https://www.alphavantage.co/query"


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"news item {name} must be nonempty")
    return value.strip()


def _finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"news item {name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"news item {name} must be finite")
    return result


def _provider_time(value: Any) -> str:
    """Parse Alpha Vantage's documented UTC ``YYYYMMDDTHHMM[SS]`` timestamp."""
    text = _required_text(value, "time_published")
    formats = ("%Y%m%dT%H%M%S", "%Y%m%dT%H%M")
    for pattern in formats:
        try:
            return normalize_timestamp(datetime.strptime(text, pattern).replace(tzinfo=timezone.utc))
        except ValueError:
            continue
    raise ValueError("news item time_published has an invalid format")


def _query_time(value: str | datetime) -> str:
    return parse_timestamp(value).strftime("%Y%m%dT%H%M")


def canonicalize_url(value: Any) -> str:
    raw = _required_text(value, "url")
    parts = urlsplit(raw)
    if parts.scheme.lower() not in {"http", "https"} or not parts.netloc:
        raise ValueError("news item url must be HTTP(S)")
    query = [
        (key, item) for key, item in parse_qsl(parts.query, keep_blank_values=True)
        if not key.lower().startswith("utm_")
    ]
    host = (parts.hostname or "").lower()
    if parts.port:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme.lower(), host, parts.path or "/", urlencode(sorted(query)), ""))


class AlphaVantageNewsAdapter:
    """Fetch each ticker independently to avoid undocumented multi-ticker semantics."""

    def __init__(self, api_key: str, request: Request | None = None) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("Alpha Vantage API key is required")
        self._api_key = api_key.strip()
        self._request = request or self._default_request

    @staticmethod
    def _default_request(params: Mapping[str, str]) -> Mapping[str, Any]:
        url = f"{_ENDPOINT}?{urlencode(params)}"
        try:
            with urlopen(url, timeout=30) as response:  # noqa: S310 - fixed HTTPS endpoint
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # pragma: no cover - network dependent
            raise RuntimeError("Alpha Vantage news request failed") from exc
        if not isinstance(payload, Mapping):
            raise RuntimeError("Alpha Vantage returned an invalid response")
        return payload

    def fetch(
        self,
        symbols: Sequence[str],
        *,
        time_from: str | datetime,
        time_to: str | datetime,
        as_of: str | datetime,
        retrieved_at: str | datetime,
        limit: int = 50,
    ) -> tuple[Evidence, ...]:
        if isinstance(symbols, (str, bytes)):
            raise TypeError("symbols must be a sequence")
        normalized = tuple(str(symbol).strip().upper() for symbol in symbols)
        if not normalized or any(not symbol for symbol in normalized):
            raise ValueError("symbols must be nonempty")
        if len(normalized) != len(set(normalized)):
            raise ValueError("symbols must be unique")
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        start, end = parse_timestamp(time_from), parse_timestamp(time_to)
        cutoff, retrieved = parse_timestamp(as_of), parse_timestamp(retrieved_at)
        if start >= end or end > cutoff:
            raise ValueError("news window must be ordered and end at or before as_of")

        candidates: list[Evidence] = []
        for symbol in normalized:
            feed = self._complete_feed(symbol, start, end, limit)
            for raw in feed:
                item = self._normalize_item(raw, symbol, start, end, cutoff, retrieved)
                if item is not None:
                    candidates.append(item)

        candidates.sort(key=lambda item: (item.available_at, item.url or "", item.id))
        seen_urls: set[tuple[str | None, str, str]] = set()
        seen_hashes: set[tuple[str | None, str]] = set()
        url_hashes: dict[tuple[str | None, str, str], str] = {}
        result: list[Evidence] = []
        for item in candidates:
            canonical_url = str(item.values["canonical_url"])
            digest = str(item.values["article_content_hash"])
            url_key = (item.symbol, canonical_url, str(item.values["published_at"]))
            hash_key = (item.symbol, digest)
            previous_hash = url_hashes.get(url_key)
            if previous_hash is not None and previous_hash != digest:
                raise ValueError("conflicting news revisions share a canonical URL")
            url_hashes[url_key] = digest
            if url_key in seen_urls or hash_key in seen_hashes:
                continue
            seen_urls.add(url_key)
            seen_hashes.add(hash_key)
            result.append(item)
        return tuple(result)

    def _complete_feed(
        self, symbol: str, start: datetime, end: datetime, limit: int,
        *, depth: int = 0,
    ) -> list[Mapping[str, Any]]:
        payload = self._request({
            "function": "NEWS_SENTIMENT",
            "tickers": symbol,
            "time_from": _query_time(start),
            "time_to": _query_time(end),
            "sort": "EARLIEST",
            "limit": str(limit),
            "apikey": self._api_key,
        })
        feed = list(self._validate_payload(payload))
        if len(feed) < limit:
            return feed
        # Alpha Vantage exposes a bounded feed rather than a page token. Split the
        # time window until each response proves it was not truncated. Adjacent
        # windows overlap at the midpoint; content/URL dedupe removes that overlap.
        if depth >= 32 or end - start <= timedelta(minutes=1):
            raise RuntimeError("Alpha Vantage news coverage is incomplete")
        midpoint = start + (end - start) / 2
        return (
            self._complete_feed(symbol, start, midpoint, limit, depth=depth + 1)
            + self._complete_feed(symbol, midpoint, end, limit, depth=depth + 1)
        )

    @staticmethod
    def _validate_payload(payload: Any) -> Sequence[Mapping[str, Any]]:
        if not isinstance(payload, Mapping):
            raise RuntimeError("Alpha Vantage returned an invalid response")
        # Never echo provider messages: they may include request details or secrets.
        if any(key in payload for key in ("Error Message", "Information", "Note")):
            raise RuntimeError("Alpha Vantage news request was rejected or rate limited")
        feed = payload.get("feed")
        if not isinstance(feed, list) or not all(isinstance(item, Mapping) for item in feed):
            raise RuntimeError("Alpha Vantage returned an invalid news feed")
        return feed


    @staticmethod
    def _normalize_item(
        raw: Mapping[str, Any], symbol: str, start: datetime, end: datetime,
        cutoff: datetime, retrieved: datetime,
    ) -> Evidence | None:
        published = parse_timestamp(_provider_time(raw.get("time_published")))
        if published < start or published > end or published > cutoff:
            return None
        if published > retrieved:
            raise ValueError("news retrieval time precedes publication")
        canonical_url = canonicalize_url(raw.get("url"))
        title = _required_text(raw.get("title"), "title")
        summary = str(raw.get("summary") or "").strip()
        source = _required_text(raw.get("source"), "source")

        ticker_items = raw.get("ticker_sentiment")
        if not isinstance(ticker_items, list):
            raise ValueError("news item ticker_sentiment must be a list")
        matching = [
            entry for entry in ticker_items
            if isinstance(entry, Mapping)
            and str(entry.get("ticker", "")).strip().upper() == symbol
        ]
        if not matching:
            return None
        if len(matching) != 1:
            raise ValueError("news item has duplicate ticker sentiment")
        sentiment = matching[0]
        relevance = _finite(sentiment.get("relevance_score"), "relevance_score")
        score = _finite(sentiment.get("ticker_sentiment_score"), "ticker_sentiment_score")
        if not 0 <= relevance <= 1 or not -1 <= score <= 1:
            raise ValueError("news relevance/sentiment score is outside its valid range")

        raw_topics = raw.get("topics", [])
        if not isinstance(raw_topics, list):
            raise ValueError("news item topics must be a list")
        topics = []
        for topic in raw_topics:
            if not isinstance(topic, Mapping):
                raise ValueError("news topic must be an object")
            topic_relevance = _finite(
                topic.get("relevance_score"), "topic relevance_score"
            )
            if not 0 <= topic_relevance <= 1:
                raise ValueError("news topic relevance is outside its valid range")
            topics.append({
                "topic": _required_text(topic.get("topic"), "topic"),
                "relevance_score": topic_relevance,
            })
        values = {
            "canonical_url": canonical_url,
            "published_at": normalize_timestamp(published),
            "title": title,
            "summary": summary,
            "publisher": source,
            "topics": topics,
            "relevance_score": relevance,
            "sentiment_score": score,
            "sentiment_label": str(sentiment.get("ticker_sentiment_label") or "").strip(),
        }
        values["article_content_hash"] = content_hash({
            key: value for key, value in values.items() if key != "canonical_url"
        })
        record_hash = content_hash(values)
        return Evidence(
            id=f"alpha-news:{symbol}:{published.strftime('%Y%m%dT%H%M%S')}:{record_hash[:16]}",
            source="alpha_vantage.news",
            symbol=symbol,
            observed_at=published,
            available_at=published,
            retrieved_at=retrieved,
            text=f"{title}\n{summary}".strip(),
            values=values,
            url=canonical_url,
            entitlement="alpha_vantage",
            raw_content_hash=record_hash,
            transformation_version="alpha-vantage-news-v1",
        )


class CachedNewsAdapter:
    """Enforce the assignment's 24-hour reuse policy on normalized snapshots."""

    def __init__(self, adapter: AlphaVantageNewsAdapter, cache: SqliteResponseCache) -> None:
        self.adapter = adapter
        self.cache = cache

    def fetch(
        self, symbols: Sequence[str], *, time_from: str | datetime,
        time_to: str | datetime, as_of: str | datetime,
        retrieved_at: str | datetime, limit: int = 50,
    ) -> tuple[Evidence, ...]:
        start = parse_timestamp(time_from)
        end = parse_timestamp(time_to)
        cutoff = parse_timestamp(as_of)
        if end != cutoff:
            raise ValueError("cached live news requires time_to to equal as_of")
        lookback_seconds = (end - start).total_seconds()
        if lookback_seconds <= 0:
            raise ValueError("cached live news requires a positive lookback window")
        identity = CacheIdentity(
            source="alpha_vantage.news",
            symbols=tuple(symbols),
            params={
                "lookback_seconds": lookback_seconds,
                "limit": limit,
                "query_mode": "one-request-lane-per-symbol-with-window-splitting",
            },
            snapshot_window=f"rolling-live-news:{cutoff.date().isoformat()}",
            transformation_version="alpha-vantage-news-v1",
        )
        cached = self.cache.get(identity, now=retrieved_at)
        if cached is not None:
            if not isinstance(cached, list) or not all(isinstance(item, Mapping) for item in cached):
                raise ValueError("cached news payload is malformed")
            reconstructed = tuple(Evidence(**dict(item)) for item in cached)
            if all(
                start <= parse_timestamp(item.available_at) <= end
                and item.eligible(cutoff)
                for item in reconstructed
            ):
                return reconstructed
        result = self.adapter.fetch(
            symbols, time_from=time_from, time_to=time_to, as_of=as_of,
            retrieved_at=retrieved_at, limit=limit,
        )
        self.cache.put(
            identity, [item.to_dict() for item in result], fetched_at=retrieved_at,
            ttl=timedelta(hours=24),
        )
        return result


def aggregate_news(
    evidence: Iterable[Evidence],
    *,
    symbol: str,
    as_of: str | datetime,
    retrieved_at: str | datetime,
) -> Evidence:
    """Aggregate eligible articles, returning explicit missing state when empty."""
    normalized_symbol = symbol.strip().upper()
    if not normalized_symbol:
        raise ValueError("symbol must be nonempty")
    cutoff, retrieved = parse_timestamp(as_of), parse_timestamp(retrieved_at)
    if retrieved < cutoff:
        raise ValueError("retrieved_at cannot precede as_of")
    items = sorted(
        (
            item for item in evidence
            if item.source == "alpha_vantage.news"
            and item.symbol == normalized_symbol
            and item.eligible(cutoff)
        ),
        key=lambda item: (item.available_at, item.id),
    )
    if not items:
        values = {
            "missing": True,
            "article_count": 0,
            "weighted_sentiment": None,
            "sentiment_disagreement": None,
            "newest_age_hours": None,
            "source_evidence_ids": (),
        }
        available = cutoff
    else:
        weighted = []
        for item in items:
            relevance = _finite(item.values.get("relevance_score"), "relevance_score")
            score = _finite(item.values.get("sentiment_score"), "sentiment_score")
            weighted.append((relevance, score))
        total_weight = sum(weight for weight, _ in weighted)
        mean = (sum(weight * score for weight, score in weighted) / total_weight
                if total_weight > 0 else None)
        disagreement = None
        if mean is not None:
            disagreement = math.sqrt(
                sum(weight * (score - mean) ** 2 for weight, score in weighted) / total_weight
            )
        newest = max(parse_timestamp(item.available_at) for item in items)
        values = {
            "missing": False,
            "article_count": len(items),
            "weighted_sentiment": mean,
            "sentiment_disagreement": disagreement,
            "newest_age_hours": (cutoff - newest).total_seconds() / 3600,
            "source_evidence_ids": tuple(item.id for item in items),
            "source_evidence_fingerprints": tuple(content_hash(item) for item in items),
        }
        available = newest
        retrieved = max(retrieved, *(parse_timestamp(item.retrieved_at) for item in items))
    identity = {"symbol": normalized_symbol, "as_of": normalize_timestamp(cutoff), "values": values}
    digest = content_hash(identity)
    return Evidence(
        id=f"news-aggregate:{normalized_symbol}:{digest[:20]}",
        source="derived.news_aggregate",
        symbol=normalized_symbol,
        observed_at=available,
        available_at=available,
        retrieved_at=retrieved,
        text=(f"No eligible news for {normalized_symbol}" if values["missing"]
              else f"Aggregated {values['article_count']} news articles for {normalized_symbol}"),
        values=values,
        raw_content_hash=digest,
        transformation_version="news-aggregate-v1",
    )
