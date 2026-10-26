import unittest
from datetime import datetime, timezone

from agent.contracts import Evidence
from trading_agent import HistoricalSourceAdapter


class Source:
    def __init__(self, records): self.records = records
    def evidence(self, symbols, as_of): return list(self.records)


def record(name, available):
    return Evidence(
        name, "source", "AAPL", available, available,
        "2026-02-01T00:00:00Z", name,
    )


class HistoricalSourceAdapterTests(unittest.TestCase):
    def test_combines_sorts_and_enforces_cutoff(self):
        adapter = HistoricalSourceAdapter((
            Source((record("later", "2026-01-02T00:00:00Z"),)),
            Source((record("earlier", "2026-01-01T00:00:00Z"),)),
        ))
        result = adapter.evidence(
            ("AAPL",), datetime(2026, 1, 2, tzinfo=timezone.utc))
        self.assertEqual([item.id for item in result], ["earlier", "later"])
        with self.assertRaisesRegex(ValueError, "after"):
            adapter.evidence(
                ("AAPL",), datetime(2026, 1, 1, tzinfo=timezone.utc))


if __name__ == "__main__":
    unittest.main()
