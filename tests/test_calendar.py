import unittest
import importlib.util
from datetime import date, datetime, timezone

from agent.data.calendar import XNYSCalendar


class Label:
    def __init__(self, value): self.value = value
    def date(self): return date.fromisoformat(self.value)


class FakeCalendar:
    sessions = {
        "2026-07-02": (datetime(2026, 7, 2, 13, 30, tzinfo=timezone.utc),
                       datetime(2026, 7, 2, 20, 0, tzinfo=timezone.utc)),
        "2026-07-06": (datetime(2026, 7, 6, 13, 30, tzinfo=timezone.utc),
                       datetime(2026, 7, 6, 20, 0, tzinfo=timezone.utc)),
    }
    def date_to_session(self, value, direction="none"):
        key = value.value if isinstance(value, Label) else str(value)
        if key not in self.sessions: raise ValueError("not a session")
        return Label(key)
    def session_open(self, label): return self.sessions[label.value][0]
    def session_close(self, label): return self.sessions[label.value][1]
    def next_session(self, label): return Label("2026-07-06")


class CalendarTests(unittest.TestCase):
    def test_next_session_skips_holiday_and_weekend(self):
        calendar = XNYSCalendar(FakeCalendar())
        following = calendar.next("2026-07-02")
        self.assertEqual(following.session, "2026-07-06")
        self.assertEqual(following.open_at, "2026-07-06T13:30:00+00:00")

    @unittest.skipUnless(
        importlib.util.find_spec("exchange_calendars"), "data dependency is not installed",
    )
    def test_real_calendar_dst_holiday_and_early_close(self):
        calendar = XNYSCalendar()
        self.assertEqual(calendar.bounds("2026-03-06").close_at[11:16], "21:00")
        self.assertEqual(calendar.bounds("2026-03-09").close_at[11:16], "20:00")
        with self.assertRaises(ValueError):
            calendar.bounds("2026-11-26")  # Thanksgiving
        self.assertEqual(calendar.bounds("2026-11-27").close_at[11:16], "18:00")


if __name__ == "__main__":
    unittest.main()
