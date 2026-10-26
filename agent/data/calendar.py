"""Verified XNYS session bounds and next-session selection."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from agent.contracts import normalize_timestamp, parse_timestamp


@dataclass(frozen=True)
class SessionBounds:
    session: str
    open_at: str
    close_at: str

    def __post_init__(self) -> None:
        opened = normalize_timestamp(self.open_at)
        closed = normalize_timestamp(self.close_at)
        if parse_timestamp(opened) >= parse_timestamp(closed):
            raise ValueError("session open must precede close")
        object.__setattr__(self, "open_at", opened)
        object.__setattr__(self, "close_at", closed)


class XNYSCalendar:
    """Thin lazy boundary around exchange_calendars, including holidays and DST."""

    def __init__(self, calendar: Any | None = None) -> None:
        if calendar is None:
            try:
                import exchange_calendars
            except ImportError as exc:  # pragma: no cover - optional integration
                raise RuntimeError("install the data dependency group for XNYS sessions") from exc
            calendar = exchange_calendars.get_calendar("XNYS")
        self.calendar = calendar

    def bounds(self, session: str | date) -> SessionBounds:
        label = self.calendar.date_to_session(session, direction="none")
        return SessionBounds(
            session=str(label.date()),
            open_at=str(self.calendar.session_open(label).isoformat()),
            close_at=str(self.calendar.session_close(label).isoformat()),
        )

    def next(self, session: str | date) -> SessionBounds:
        label = self.calendar.date_to_session(session, direction="none")
        next_label = self.calendar.next_session(label)
        current = self.bounds(label)
        following = self.bounds(next_label)
        if parse_timestamp(following.open_at) <= parse_timestamp(current.close_at):
            raise ValueError("next session must open after the current close")
        return following
