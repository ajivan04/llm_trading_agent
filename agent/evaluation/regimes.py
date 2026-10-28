"""Declared historical market-regime windows for event-period evaluation."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta


def midterm_election_date(year: int) -> date:
    """Return the federal general-election Tuesday for a midterm year."""
    if type(year) is not int or year < 1900 or year % 2 != 0 or year % 4 == 0:
        raise ValueError("year must be an even, non-presidential election year")
    november_first = date(year, 11, 1)
    days_until_tuesday = (1 - november_first.weekday()) % 7
    return november_first + timedelta(days=days_until_tuesday)


@dataclass(frozen=True)
class ElectionWindow:
    year: int
    election_date: date
    start: date
    end: date

    def __post_init__(self) -> None:
        if self.election_date != midterm_election_date(self.year):
            raise ValueError("election_date does not match the declared midterm year")
        if not self.start <= self.election_date < self.end:
            raise ValueError("election window must contain election day")

    @property
    def identifier(self) -> str:
        return f"midterm-{self.year}-{self.start.isoformat()}-{self.end.isoformat()}"


@dataclass(frozen=True)
class ElectionRegimePlan:
    """Comparable event-week windows and leakage-safe split intent."""

    target: ElectionWindow
    historical: tuple[ElectionWindow, ...]
    regime: str = "us-midterm-election-week-v1"

    def __post_init__(self) -> None:
        if not self.historical:
            raise ValueError("at least one historical comparison window is required")
        years = {item.year for item in self.historical}
        if self.target.year in years:
            raise ValueError("target election year cannot appear in training windows")
        if len(years) != len(self.historical):
            raise ValueError("historical election years must be unique")

    @property
    def training_years(self) -> tuple[int, ...]:
        return tuple(item.year for item in self.training_windows)

    @property
    def validation_window(self) -> ElectionWindow:
        return max(self.historical, key=lambda item: item.year)

    @property
    def training_windows(self) -> tuple[ElectionWindow, ...]:
        return tuple(item for item in self.historical if item.year != self.validation_window.year)

    def decision_sessions(self, *, include_validation: bool = True) -> tuple[str, ...]:
        windows = self.training_windows
        if include_validation:
            windows = (*windows, self.validation_window)
        windows = (*windows, self.target)
        return tuple(sorted({
            (window.start + timedelta(days=offset)).isoformat()
            for window in windows
            for offset in range((window.end - window.start).days)
        }))

    def to_dict(self) -> dict[str, object]:
        return {
            "regime": self.regime,
            "target": _window_dict(self.target),
            "historical": [_window_dict(item) for item in self.historical],
            "training_years": self.training_years,
            "validation_year": self.validation_window.year,
            "validation_window": _window_dict(self.validation_window),
            "leakage_rule": "target-year observations are excluded from training and validation",
        }


def _window_dict(window: ElectionWindow) -> dict[str, object]:
    return {
        "year": window.year,
        "election_date": window.election_date.isoformat(),
        "start": window.start.isoformat(),
        "end": window.end.isoformat(),
        "identifier": window.identifier,
    }


def build_midterm_regime_plan(
    *, target_year: int = 2026, historical_years: Sequence[int] = (2022, 2018, 2014, 2010, 2006),
) -> ElectionRegimePlan:
    """Build Monday-through-Sunday event weeks around comparable midterms."""
    def window(year: int) -> ElectionWindow:
        election = midterm_election_date(year)
        start = election - timedelta(days=election.weekday())
        return ElectionWindow(year, election, start, start + timedelta(days=7))

    historical = tuple(window(year) for year in historical_years)
    return ElectionRegimePlan(target=window(target_year), historical=historical)
