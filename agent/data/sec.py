"""Point-in-time normalization for SEC EDGAR submissions and company facts.

Network access is injected so tests and replays remain deterministic. Date-only
provider records require an explicit conservative availability resolver; this
module never invents an intraday publication time.
"""
from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Mapping
from datetime import date, datetime
from html.parser import HTMLParser
from typing import Any

from agent.contracts import Evidence, content_hash, normalize_timestamp, parse_timestamp


JsonRequest = Callable[..., Mapping[str, Any]]
TextRequest = Callable[..., str]
AvailabilityResolver = Callable[[Mapping[str, Any]], str | datetime]
_ACCESSION_RE = re.compile(r"^\d{10}-\d{2}-\d{6}$")


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._hidden = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style", "noscript"}:
            self._hidden += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript"} and self._hidden:
            self._hidden -= 1

    def handle_data(self, data: str) -> None:
        if not self._hidden:
            text = " ".join(data.split())
            if text:
                self.parts.append(text)


def normalize_cik(value: str | int) -> str:
    """Return the SEC's ten-digit CIK representation."""
    if isinstance(value, bool):
        raise ValueError("CIK must contain digits")
    text = str(value).strip().upper()
    if text.startswith("CIK"):
        text = text[3:]
    if not text.isdigit() or len(text) > 10 or int(text) <= 0:
        raise ValueError("CIK must be a positive number with at most ten digits")
    return text.zfill(10)


def normalize_ticker_map(payload: Mapping[str, Any]) -> dict[str, str]:
    """Normalize either SEC company_tickers mapping or fields/data payload."""
    if not isinstance(payload, Mapping):
        raise TypeError("ticker payload must be a mapping")
    records: list[Mapping[str, Any]] = []
    if "fields" in payload or "data" in payload:
        fields, data = payload.get("fields"), payload.get("data")
        if not isinstance(fields, list) or not isinstance(data, list):
            raise ValueError("ticker fields/data payload is malformed")
        for row in data:
            if not isinstance(row, list) or len(row) != len(fields):
                raise ValueError("ticker row does not match fields")
            records.append(dict(zip(fields, row)))
    else:
        records = [row for row in payload.values() if isinstance(row, Mapping)]
        if len(records) != len(payload):
            raise ValueError("ticker payload contains a malformed record")
    result: dict[str, str] = {}
    for record in records:
        ticker = record.get("ticker")
        cik = record.get("cik_str", record.get("cik"))
        if not isinstance(ticker, str) or not ticker.strip() or cik is None:
            raise ValueError("ticker record is missing ticker or CIK")
        symbol = ticker.strip().upper()
        normalized = normalize_cik(cik)
        if symbol in result and result[symbol] != normalized:
            raise ValueError(f"conflicting CIK mappings for {symbol}")
        result[symbol] = normalized
    return dict(sorted(result.items()))


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty")
    return value.strip()


def _availability(
    record: Mapping[str, Any], resolver: AvailabilityResolver | None,
) -> str:
    value = record.get("acceptanceDateTime", record.get("accepted"))
    if isinstance(value, str) and "T" in value:
        return normalize_timestamp(value)
    if resolver is None:
        raise ValueError("date-only SEC record requires an availability resolver")
    return normalize_timestamp(resolver(record))


def _parallel_records(recent: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(recent, Mapping) or not recent:
        raise ValueError("submissions recent payload must be a nonempty mapping")
    lengths = {len(value) for value in recent.values() if isinstance(value, list)}
    if len(lengths) != 1 or any(not isinstance(value, list) for value in recent.values()):
        raise ValueError("submissions recent arrays must have equal lengths")
    count = next(iter(lengths))
    return [{key: values[index] for key, values in recent.items()} for index in range(count)]


def _period_class(start: str | None, end: str | None) -> str:
    if not start:
        return "instant"
    if not end:
        return "duration:unknown"
    try:
        days = (date.fromisoformat(end) - date.fromisoformat(start)).days
    except ValueError as exc:
        raise ValueError("fact periods must use ISO dates") from exc
    if 70 <= days <= 110:
        return "duration:quarter"
    if 150 <= days <= 210:
        return "duration:half-year-ytd"
    if 240 <= days <= 300:
        return "duration:nine-month-ytd"
    if 330 <= days <= 380:
        return "duration:annual"
    return f"duration:{days}-days"


class SecDataAdapter:
    """Normalize SEC public JSON APIs into historically eligible Evidence."""

    def __init__(
        self,
        user_agent: str,
        request: JsonRequest,
        availability_resolver: AvailabilityResolver | None = None,
        text_request: TextRequest | None = None,
    ) -> None:
        user_agent = _required_text(user_agent, "user_agent")
        if "@" not in user_agent or " " not in user_agent:
            raise ValueError("SEC User-Agent must identify an organization and email")
        if not callable(request):
            raise TypeError("request must be callable")
        self.user_agent = user_agent
        self._request = request
        self._availability_resolver = availability_resolver
        self._text_request = text_request

    def _get(self, url: str) -> Mapping[str, Any]:
        payload = self._request(
            url,
            headers={
                "User-Agent": self.user_agent,
                "Accept-Encoding": "gzip, deflate",
                "Accept": "application/json",
            },
        )
        if not isinstance(payload, Mapping):
            raise ValueError("SEC returned a malformed JSON payload")
        return payload

    def fetch_ticker_map(self) -> dict[str, str]:
        return normalize_ticker_map(
            self._get("https://www.sec.gov/files/company_tickers.json")
        )

    def fetch_submissions(
        self,
        cik: str | int,
        *,
        symbol: str,
        as_of: str | datetime,
        retrieved_at: str | datetime,
        forms: Iterable[str] = ("8-K", "10-Q", "10-K"),
        include_historical: bool = True,
    ) -> tuple[Evidence, ...]:
        if type(include_historical) is not bool:
            raise ValueError("include_historical must be an explicit boolean")
        normalized_cik = normalize_cik(cik)
        payload = self._get(
            f"https://data.sec.gov/submissions/CIK{normalized_cik}.json"
        )
        if "cik" not in payload or normalize_cik(payload["cik"]) != normalized_cik:
            raise ValueError("SEC submissions payload CIK does not match requested CIK")
        filings = payload.get("filings")
        if not isinstance(filings, Mapping) or not isinstance(filings.get("recent"), Mapping):
            raise ValueError("SEC submissions payload is missing filings.recent")
        evidence = list(self.normalize_submissions(
            filings["recent"], cik=normalized_cik, symbol=symbol, as_of=as_of,
            retrieved_at=retrieved_at, forms=forms,
        ))
        # The SEC keeps only the recent filings inline. Older periods are listed
        # in filings.files and must be fetched explicitly for historical replay.
        files = filings.get("files", ()) if include_historical else ()
        cutoff = parse_timestamp(as_of).date().isoformat()
        if isinstance(files, list):
            for descriptor in files:
                if not isinstance(descriptor, Mapping):
                    continue
                name = descriptor.get("name")
                filing_from = descriptor.get("filingFrom")
                if (not isinstance(name, str) or not name.endswith(".json")
                        or ".." in name or not isinstance(filing_from, str)
                        or filing_from > cutoff):
                    continue
                historical = self._get(f"https://data.sec.gov/submissions/{name}")
                evidence.extend(self.normalize_submissions(
                    historical, cik=normalized_cik, symbol=symbol, as_of=as_of,
                    retrieved_at=retrieved_at, forms=forms,
                ))
        unique = {item.id: item for item in evidence}
        return tuple(sorted(unique.values(), key=lambda item: (
            item.available_at, item.accession_id or "", item.id,
        )))

    def fetch_filing_text(
        self, filing: Evidence, *, as_of: str | datetime, retrieved_at: str | datetime,
        max_chars: int = 4000,
    ) -> Evidence:
        """Fetch a bounded visible-text excerpt from an already eligible filing."""
        if filing.source != "sec.submissions" or not filing.url or not filing.accession_id:
            raise ValueError("filing text requires a SEC submission evidence record")
        if not filing.eligible(as_of):
            raise ValueError("filing is not eligible at the requested cutoff")
        if type(max_chars) is not int or not 256 <= max_chars <= 20_000:
            raise ValueError("max_chars must be between 256 and 20000")
        if self._text_request is None:
            raise RuntimeError("SEC text request function is not configured")
        raw = self._text_request(
            filing.url,
            headers={"User-Agent": self.user_agent, "Accept": "text/html"},
        )
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("SEC filing text response was empty")
        parser = _VisibleTextParser()
        parser.feed(raw)
        excerpt = " ".join(parser.parts).strip()[:max_chars].strip()
        if not excerpt:
            raise ValueError("SEC filing contained no visible text")
        values = {
            "source_filing_id": filing.id,
            "accession": filing.accession_id,
            "form": filing.values.get("form"),
            "excerpt_chars": len(excerpt),
            "bounded": True,
        }
        digest = content_hash({"filing": filing.id, "excerpt": excerpt})
        return Evidence(
            id=f"sec:filing-text:{filing.accession_id}:{digest[:16]}",
            source="sec.filing_text", symbol=filing.symbol,
            observed_at=filing.observed_at, available_at=filing.available_at,
            retrieved_at=normalize_timestamp(retrieved_at),
            text=f"SEC {filing.values.get('form', 'filing')} excerpt: {excerpt}",
            values=values, url=filing.url, accession_id=filing.accession_id,
            raw_content_hash=digest, transformation_version="sec-filing-text-v1",
        )

    def normalize_submissions(
        self,
        recent: Mapping[str, Any],
        *,
        cik: str | int,
        symbol: str,
        as_of: str | datetime,
        retrieved_at: str | datetime,
        forms: Iterable[str] = ("8-K", "10-Q", "10-K"),
    ) -> tuple[Evidence, ...]:
        normalized_cik = normalize_cik(cik)
        symbol = _required_text(symbol, "symbol").upper()
        cutoff, retrieved = parse_timestamp(as_of), normalize_timestamp(retrieved_at)
        selected = {_required_text(form, "form").upper() for form in forms}
        evidence: list[Evidence] = []
        for record in _parallel_records(recent):
            form = _required_text(record.get("form"), "form").upper()
            base_form = form[:-2] if form.endswith("/A") else form
            if base_form not in selected:
                continue
            accession = _required_text(record.get("accessionNumber"), "accessionNumber")
            if not _ACCESSION_RE.fullmatch(accession):
                raise ValueError("invalid SEC accession number")
            available = _availability(record, self._availability_resolver)
            if parse_timestamp(available) > cutoff:
                continue
            primary_value = record.get("primaryDocument")
            if not isinstance(primary_value, str) or not primary_value.strip():
                # SEC historical archives can contain metadata-only rows. They
                # cannot yield a filing URL or bounded text excerpt.
                continue
            primary = primary_value.strip()
            accession_path = accession.replace("-", "")
            url = (
                f"https://www.sec.gov/Archives/edgar/data/{int(normalized_cik)}/"
                f"{accession_path}/{primary}"
            )
            values = {
                "cik": normalized_cik,
                "accession": accession,
                "form": form,
                "base_form": base_form,
                "is_amendment": form.endswith("/A"),
                "filing_date": record.get("filingDate"),
                "report_date": record.get("reportDate"),
                "primary_document": primary,
                "items": record.get("items"),
            }
            record_hash = content_hash(values)
            evidence.append(Evidence(
                id=f"sec:filing:{accession}:{record_hash[:16]}",
                source="sec.submissions",
                symbol=symbol,
                observed_at=available,
                available_at=available,
                retrieved_at=retrieved,
                text=f"SEC {form} filing {accession} for {symbol}",
                values=values,
                url=url,
                accession_id=accession,
                raw_content_hash=record_hash,
                transformation_version="sec-submissions-v1",
            ))
        evidence.sort(key=lambda item: (item.available_at, item.accession_id or "", item.id))
        return tuple(evidence)

    def fetch_companyfacts(
        self,
        cik: str | int,
        *,
        symbol: str,
        as_of: str | datetime,
        retrieved_at: str | datetime,
        accession_availability: Mapping[str, str | datetime] | None = None,
    ) -> tuple[Evidence, ...]:
        normalized_cik = normalize_cik(cik)
        payload = self._get(
            f"https://data.sec.gov/api/xbrl/companyfacts/CIK{normalized_cik}.json"
        )
        return self.normalize_companyfacts(
            payload, cik=normalized_cik, symbol=symbol, as_of=as_of,
            retrieved_at=retrieved_at,
            accession_availability=accession_availability,
        )

    def normalize_companyfacts(
        self,
        payload: Mapping[str, Any],
        *,
        cik: str | int,
        symbol: str,
        as_of: str | datetime,
        retrieved_at: str | datetime,
        accession_availability: Mapping[str, str | datetime] | None = None,
    ) -> tuple[Evidence, ...]:
        normalized_cik = normalize_cik(cik)
        symbol = _required_text(symbol, "symbol").upper()
        cutoff, retrieved = parse_timestamp(as_of), normalize_timestamp(retrieved_at)
        payload_cik = payload.get("cik") if isinstance(payload, Mapping) else None
        if payload_cik is not None and normalize_cik(payload_cik) != normalized_cik:
            raise ValueError("companyfacts payload CIK does not match requested CIK")
        facts = payload.get("facts") if isinstance(payload, Mapping) else None
        if not isinstance(facts, Mapping):
            raise ValueError("companyfacts payload is missing facts")
        availability_map = accession_availability or {}
        result: dict[str, Evidence] = {}
        logical_facts: dict[tuple[Any, ...], Evidence] = {}
        for taxonomy in sorted(facts):
            tags = facts[taxonomy]
            if not isinstance(tags, Mapping):
                raise ValueError("companyfacts taxonomy must contain fact mappings")
            for tag in sorted(tags):
                definition = tags[tag]
                if not isinstance(definition, Mapping) or not isinstance(definition.get("units"), Mapping):
                    raise ValueError("companyfacts definition is malformed")
                for unit in sorted(definition["units"]):
                    observations = definition["units"][unit]
                    if not isinstance(observations, list):
                        raise ValueError("companyfacts unit observations must be a list")
                    for observation in observations:
                        if not isinstance(observation, Mapping):
                            raise ValueError("companyfacts observation must be a mapping")
                        accession = _required_text(observation.get("accn"), "accn")
                        form = _required_text(observation.get("form"), "form").upper()
                        availability_value = availability_map.get(accession)
                        if availability_value is not None:
                            available = normalize_timestamp(availability_value)
                        else:
                            record = dict(observation)
                            record["accessionNumber"] = accession
                            available = _availability(record, self._availability_resolver)
                        if parse_timestamp(available) > cutoff:
                            continue
                        value = observation.get("val")
                        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                            continue
                        start = observation.get("start")
                        end = observation.get("end")
                        if start is not None and not isinstance(start, str):
                            raise ValueError("fact start must be an ISO date string")
                        if end is not None and not isinstance(end, str):
                            raise ValueError("fact end must be an ISO date string")
                        period_class = _period_class(start, end)
                        values = {
                            "cik": normalized_cik,
                            "taxonomy": taxonomy,
                            "tag": tag,
                            "label": definition.get("label"),
                            "description": definition.get("description"),
                            "unit": unit,
                            "value": float(value),
                            "period_start": start,
                            "period_end": end,
                            "period_class": period_class,
                            "accession": accession,
                            "form": form,
                            "is_amendment": form.endswith("/A"),
                            "filed_date": observation.get("filed"),
                            "fiscal_year": observation.get("fy"),
                            "fiscal_period": observation.get("fp"),
                            "frame": observation.get("frame"),
                        }
                        record_hash = content_hash(values)
                        evidence = Evidence(
                            id=f"sec:fact:{accession}:{taxonomy}:{tag}:{unit}:{record_hash[:16]}",
                            source="sec.companyfacts",
                            symbol=symbol,
                            observed_at=available,
                            available_at=available,
                            retrieved_at=retrieved,
                            text=f"SEC fact {taxonomy}:{tag} ({unit}) for {symbol}",
                            values=values,
                            accession_id=accession,
                            units=unit,
                            raw_content_hash=record_hash,
                            transformation_version="sec-companyfacts-v1",
                        )
                        logical_key = (
                            normalized_cik, taxonomy, tag, unit, start, end, accession,
                        )
                        previous = logical_facts.get(logical_key)
                        if previous is not None and previous.id != evidence.id:
                            raise ValueError("conflicting duplicate SEC company fact")
                        logical_facts[logical_key] = evidence
                        result[evidence.id] = evidence
        ordered = sorted(result.values(), key=lambda item: (
            item.available_at, str(item.values["period_end"]),
            item.accession_id or "", item.id,
        ))
        return tuple(ordered)


def numeric_fact_trend(
    facts: Iterable[Evidence], *, as_of: str | datetime,
) -> Mapping[str, Any] | None:
    """Compare the final two compatible fact periods without causal language."""
    cutoff = parse_timestamp(as_of)
    items = tuple(item for item in facts if item.eligible(cutoff))
    if len(items) < 2:
        return None
    for item in items:
        if item.source != "sec.companyfacts":
            raise ValueError("trend inputs must be SEC company facts")
    compatibility = {
        (
            item.symbol, item.values.get("cik"),
            item.values.get("taxonomy"), item.values.get("tag"),
            item.values.get("unit"), item.values.get("period_class"),
        )
        for item in items
    }
    if len(compatibility) != 1:
        raise ValueError(
            "trend facts must share company, taxonomy, tag, unit, and period class"
        )
    # A later amendment/restatement for the same accounting period replaces the
    # earlier eligible fact for trend purposes, while both Evidence records remain
    # preserved in the audit store.
    latest_by_period: dict[tuple[Any, Any], Evidence] = {}
    for item in items:
        period = (item.values.get("period_start"), item.values.get("period_end"))
        previous = latest_by_period.get(period)
        if previous is None or (
            item.available_at, item.accession_id or "", item.id
        ) > (
            previous.available_at, previous.accession_id or "", previous.id
        ):
            latest_by_period[period] = item
    if len(latest_by_period) < 2:
        return None
    ordered = sorted(latest_by_period.values(), key=lambda item: (
        str(item.values.get("period_end") or ""), item.available_at,
        item.accession_id or "", item.id,
    ))
    previous, current = ordered[-2:]
    previous_value = float(previous.values["value"])
    current_value = float(current.values["value"])
    change = current_value - previous_value
    return {
        "kind": "period_over_period_change",
        "taxonomy": current.values["taxonomy"],
        "tag": current.values["tag"],
        "unit": current.values["unit"],
        "period_class": current.values["period_class"],
        "previous_value": previous_value,
        "current_value": current_value,
        "absolute_change": change,
        "fractional_change": change / abs(previous_value) if previous_value != 0 else None,
        "source_evidence_ids": (previous.id, current.id),
    }
