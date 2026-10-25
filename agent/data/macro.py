"""Point-in-time ALFRED macroeconomic evidence.

ALFRED observation and vintage fields are date-valued.  This module never turns
those dates into an invented release instant: callers must provide a conservative
availability resolver backed by a release calendar or documented policy.
"""
from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from agent.contracts import Evidence, content_hash, normalize_timestamp, parse_timestamp


AvailabilityResolver = Callable[[str, date, date], datetime | None]
Requester = Callable[[str, Mapping[str, str]], Mapping[str, Any]]


def _date(value: Any, name: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value.strip())
        except ValueError as exc:
            raise ValueError(f"{name} must be an ISO-8601 date") from exc
    raise ValueError(f"{name} must be an ISO-8601 date")


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty")
    return value.strip()


def _value(value: Any) -> float | None:
    if value in (None, "", "."):
        return None
    if isinstance(value, bool):
        raise ValueError("macro value must be numeric or missing")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("macro value must be numeric or missing") from exc
    if not math.isfinite(parsed):
        raise ValueError("macro value must be finite")
    return parsed


@dataclass(frozen=True)
class MacroObservation:
    series_id: str
    observation_date: str | date
    realtime_start: str | date
    realtime_end: str | date
    available_at: str | datetime
    retrieved_at: str | datetime
    units: str
    transformation: str
    value: float | None
    source_url: str = "https://fred.stlouisfed.org"

    def __post_init__(self) -> None:
        object.__setattr__(self, "series_id", _text(self.series_id, "series_id").upper())
        for name in ("observation_date", "realtime_start", "realtime_end"):
            object.__setattr__(self, name, _date(getattr(self, name), name).isoformat())
        if _date(self.realtime_end, "realtime_end") < _date(self.realtime_start, "realtime_start"):
            raise ValueError("realtime_end cannot precede realtime_start")
        object.__setattr__(self, "available_at", normalize_timestamp(self.available_at))
        object.__setattr__(self, "retrieved_at", normalize_timestamp(self.retrieved_at))
        if parse_timestamp(self.available_at) > parse_timestamp(self.retrieved_at):
            raise ValueError("macro observation cannot be retrieved before availability")
        object.__setattr__(self, "units", _text(self.units, "units"))
        object.__setattr__(self, "transformation", _text(self.transformation, "transformation"))
        object.__setattr__(self, "value", _value(self.value))
        object.__setattr__(self, "source_url", _text(self.source_url, "source_url"))

    def to_evidence(self, *, as_of: str | datetime) -> Evidence:
        cutoff = parse_timestamp(as_of)
        available = parse_timestamp(self.available_at)
        if (cutoff - available).total_seconds() < 0:
            raise ValueError("macro observation is not available at as_of")
        values = {
            "series_id": self.series_id,
            "observation_date": self.observation_date,
            "realtime_start": self.realtime_start,
            "value": self.value,
            "missing": self.value is None,
            "units": self.units,
            "transformation": self.transformation,
            "provenance": "ALFRED vintage observation",
        }
        digest = content_hash(values)
        return Evidence(
            id=f"alfred:{self.series_id}:{self.observation_date}:{self.realtime_start}:{digest[:16]}",
            source="alfred",
            symbol=None,
            # Conservatively treat the economic observation as observed no earlier
            # than its defensible release timestamp.
            observed_at=self.available_at,
            available_at=self.available_at,
            retrieved_at=self.retrieved_at,
            text=(f"{self.series_id} observation for {self.observation_date} "
                  f"as known in vintage {self.realtime_start}"),
            values=values,
            url=self.source_url,
            # realtime_end can reveal when this vintage was superseded, which may
            # be after the replay cutoff. Keep it only on the transient raw record.
            vintage_id=self.realtime_start,
            units=self.units,
            raw_content_hash=digest,
            transformation_version=f"alfred-{self.transformation}-v1",
        )


class AlfredAdapter:
    ENDPOINT = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(
        self,
        *,
        api_key: str,
        availability_resolver: AvailabilityResolver,
        requester: Requester | None = None,
    ) -> None:
        self._api_key = _text(api_key, "api_key")
        if not callable(availability_resolver):
            raise TypeError("availability_resolver must be callable")
        self._availability_resolver = availability_resolver
        self._requester = requester or self._request

    @staticmethod
    def _request(endpoint: str, params: Mapping[str, str]) -> Mapping[str, Any]:
        url = f"{endpoint}?{urlencode(params)}"
        try:
            with urlopen(Request(url, headers={"Accept": "application/json"}), timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception:
            raise RuntimeError("ALFRED request failed") from None
        if not isinstance(payload, Mapping):
            raise RuntimeError("ALFRED returned an invalid response")
        return payload

    def fetch(
        self,
        series_id: str,
        *,
        observation_start: str | date,
        observation_end: str | date,
        realtime_start: str | date,
        realtime_end: str | date,
        as_of: str | datetime,
        retrieved_at: str | datetime,
        units: str,
        transformation: str = "lin",
        source_url: str = "https://fred.stlouisfed.org",
    ) -> tuple[Evidence, ...]:
        series = _text(series_id, "series_id").upper()
        obs_start, obs_end = _date(observation_start, "observation_start"), _date(observation_end, "observation_end")
        vintage_start, vintage_end = _date(realtime_start, "realtime_start"), _date(realtime_end, "realtime_end")
        if obs_start > obs_end or vintage_start > vintage_end:
            raise ValueError("start dates must not follow end dates")
        cutoff = parse_timestamp(as_of)
        retrieved = normalize_timestamp(retrieved_at)
        params = {
            "series_id": series,
            "observation_start": obs_start.isoformat(),
            "observation_end": obs_end.isoformat(),
            "realtime_start": vintage_start.isoformat(),
            "realtime_end": vintage_end.isoformat(),
            "units": _text(transformation, "transformation"),
            "api_key": self._api_key,
            "file_type": "json",
        }
        try:
            payload = self._requester(self.ENDPOINT, params)
        except Exception:
            raise RuntimeError("ALFRED request failed") from None
        if not isinstance(payload, Mapping) or "observations" not in payload:
            raise RuntimeError("ALFRED returned an invalid response")
        if any(key in payload for key in ("error_code", "error_message")):
            raise RuntimeError("ALFRED returned an error")
        rows = payload["observations"]
        if isinstance(rows, (str, bytes)) or not isinstance(rows, Iterable):
            raise RuntimeError("ALFRED returned invalid observations")

        results: dict[str, Evidence] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                raise RuntimeError("ALFRED returned an invalid observation")
            observation_date = _date(row.get("date"), "date")
            vintage = _date(row.get("realtime_start"), "realtime_start")
            vintage_until = _date(row.get("realtime_end"), "realtime_end")
            if not obs_start <= observation_date <= obs_end:
                raise ValueError("ALFRED returned observation outside requested coverage")
            if not vintage_start <= vintage <= vintage_end:
                raise ValueError("ALFRED returned vintage outside requested coverage")
            if observation_date > cutoff.date() or vintage > cutoff.date():
                continue
            available = self._availability_resolver(series, observation_date, vintage)
            if available is None:
                raise ValueError(
                    f"no defensible availability for {series}/{observation_date}/{vintage}"
                )
            available_text = normalize_timestamp(available)
            if parse_timestamp(available_text) > cutoff:
                continue
            observation = MacroObservation(
                series_id=series,
                observation_date=observation_date,
                realtime_start=vintage,
                realtime_end=vintage_until,
                available_at=available_text,
                retrieved_at=retrieved,
                units=units,
                transformation=transformation,
                value=_value(row.get("value")),
                source_url=source_url,
            )
            evidence = observation.to_evidence(as_of=cutoff)
            previous = results.get(evidence.id)
            if previous is not None and previous != evidence:
                raise ValueError("conflicting duplicate ALFRED observation")
            results[evidence.id] = evidence
        return tuple(sorted(
            results.values(),
            key=lambda item: (
                item.values["observation_date"], item.values["realtime_start"], item.id,
            ),
        ))


def point_in_time_values(
    evidence: Iterable[Evidence],
    *,
    as_of: str | datetime,
    series_specs: Mapping[str, tuple[str, str]],
) -> dict[str, dict[str, Any]]:
    """Forward-fill the latest available nonmissing compatible observation."""
    cutoff = parse_timestamp(as_of)
    result: dict[str, dict[str, Any]] = {}
    for requested_series, (units, transformation) in series_specs.items():
        series_id = _text(requested_series, "series_id").upper()
        candidates: list[Evidence] = []
        for item in evidence:
            if item.source != "alfred" or not item.eligible(cutoff):
                continue
            values = item.values
            if values.get("series_id") != series_id:
                continue
            if values.get("units") != units or values.get("transformation") != transformation:
                continue
            if values.get("value") is None:
                continue
            candidates.append(item)
        if not candidates:
            result[series_id] = {
                "value": None, "missing": True, "units": units,
                "transformation": transformation, "evidence_id": None,
                "age_seconds": None,
            }
            continue
        # Later observation wins; for a revised value of the same observation,
        # later vintage/availability wins, but only among already eligible records.
        selected = max(candidates, key=lambda item: (
            item.values["observation_date"], item.values["realtime_start"],
            item.available_at, item.id,
        ))
        result[series_id] = {
            "value": selected.values["value"],
            "missing": False,
            "units": units,
            "transformation": transformation,
            "observation_date": selected.values["observation_date"],
            "vintage_id": selected.vintage_id,
            "evidence_id": selected.id,
            "age_seconds": (cutoff - parse_timestamp(selected.available_at)).total_seconds(),
        }
    return result
