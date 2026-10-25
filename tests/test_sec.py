import unittest
from unittest.mock import Mock

from agent.data.sec import (
    SecDataAdapter, normalize_cik, normalize_ticker_map, numeric_fact_trend,
)


def recent_payload(**changes):
    payload = {
        "accessionNumber": ["0000320193-26-000001", "0000320193-26-000002"],
        "filingDate": ["2026-01-02", "2026-02-02"],
        "reportDate": ["2025-12-31", "2025-12-31"],
        "acceptanceDateTime": ["2026-01-02T21:01:02Z", "2026-02-02T21:01:02Z"],
        "form": ["10-K", "10-K/A"],
        "primaryDocument": ["aapl-20251231.htm", "aapl-amend.htm"],
        "items": ["", ""],
    }
    payload.update(changes)
    return payload


def companyfacts_payload(unit="USD", second_start="2025-04-01"):
    return {
        "facts": {"us-gaap": {"Revenue": {
            "label": "Revenue", "description": "Revenue from contracts",
            "units": {unit: [
                {"start": "2025-01-01", "end": "2025-03-31", "val": 100,
                 "accn": "0000320193-26-000010", "form": "10-Q",
                 "filed": "2026-04-20", "fy": 2026, "fp": "Q1"},
                {"start": second_start, "end": "2025-06-30", "val": 120,
                 "accn": "0000320193-26-000011", "form": "10-Q/A",
                 "filed": "2026-07-20", "fy": 2026, "fp": "Q2"},
            ]},
        }}},
    }


class SecAdapterTests(unittest.TestCase):
    def setUp(self):
        self.request = Mock()
        self.adapter = SecDataAdapter(
            "Example University research@example.edu", self.request,
            availability_resolver=lambda record: f"{record['filed']}T23:59:59-04:00",
        )

    def test_cik_and_ticker_map_normalization(self):
        self.assertEqual(normalize_cik("CIK320193"), "0000320193")
        self.assertEqual(normalize_cik(320193), "0000320193")
        self.assertEqual(
            normalize_ticker_map({"0": {"ticker": "aapl", "cik_str": 320193}}),
            {"AAPL": "0000320193"},
        )
        with self.assertRaises(ValueError):
            normalize_cik("abc")

    def test_submissions_cutoff_keeps_amendment_distinct(self):
        evidence = self.adapter.normalize_submissions(
            recent_payload(), cik=320193, symbol="aapl",
            as_of="2026-03-01T00:00:00Z", retrieved_at="2026-03-02T00:00:00Z",
        )
        self.assertEqual([item.values["form"] for item in evidence], ["10-K", "10-K/A"])
        self.assertFalse(evidence[0].values["is_amendment"])
        self.assertTrue(evidence[1].values["is_amendment"])
        self.assertNotEqual(evidence[0].id, evidence[1].id)
        before_amendment = self.adapter.normalize_submissions(
            recent_payload(), cik=320193, symbol="AAPL",
            as_of="2026-01-20T00:00:00Z", retrieved_at="2026-03-02T00:00:00Z",
        )
        self.assertEqual(len(before_amendment), 1)

    def test_date_only_requires_explicit_policy(self):
        adapter = SecDataAdapter("Example research@example.edu", Mock())
        payload = recent_payload(acceptanceDateTime=["", ""])
        with self.assertRaisesRegex(ValueError, "availability resolver"):
            adapter.normalize_submissions(
                payload, cik=320193, symbol="AAPL", as_of="2026-03-01T00:00:00Z",
                retrieved_at="2026-03-02T00:00:00Z",
            )

    def test_fetch_uses_identifying_user_agent(self):
        self.request.return_value = {
            "cik": "0000320193", "filings": {"recent": recent_payload()}}
        self.adapter.fetch_submissions(
            320193, symbol="AAPL", as_of="2026-01-20T00:00:00Z",
            retrieved_at="2026-03-02T00:00:00Z",
        )
        _, kwargs = self.request.call_args
        self.assertEqual(
            kwargs["headers"]["User-Agent"], "Example University research@example.edu"
        )
        self.assertNotIn("api", str(kwargs).lower())

        self.request.return_value = {
            "cik": "0000789019", "filings": {"recent": recent_payload()}}
        with self.assertRaisesRegex(ValueError, "CIK does not match"):
            self.adapter.fetch_submissions(
                320193, symbol="AAPL", as_of="2026-01-20T00:00:00Z",
                retrieved_at="2026-03-02T00:00:00Z",
            )

    def test_companyfacts_preserve_units_periods_and_availability(self):
        availability = {
            "0000320193-26-000010": "2026-04-20T20:00:00Z",
            "0000320193-26-000011": "2026-07-20T20:00:00Z",
        }
        facts = self.adapter.normalize_companyfacts(
            companyfacts_payload(), cik=320193, symbol="AAPL",
            as_of="2026-08-01T00:00:00Z", retrieved_at="2026-08-02T00:00:00Z",
            accession_availability=availability,
        )
        self.assertEqual(len(facts), 2)
        self.assertEqual(facts[0].units, "USD")
        self.assertEqual(facts[0].values["period_class"], "duration:quarter")
        self.assertTrue(facts[1].values["is_amendment"])
        self.assertEqual(numeric_fact_trend(
            facts, as_of="2026-08-01T00:00:00Z")["fractional_change"], .2)

    def test_trend_rejects_incompatible_units_and_periods(self):
        availability = {
            "0000320193-26-000010": "2026-04-20T20:00:00Z",
            "0000320193-26-000011": "2026-07-20T20:00:00Z",
        }
        usd = self.adapter.normalize_companyfacts(
            companyfacts_payload(), cik=320193, symbol="AAPL",
            as_of="2026-08-01T00:00:00Z", retrieved_at="2026-08-02T00:00:00Z",
            accession_availability=availability,
        )[0]
        shares = self.adapter.normalize_companyfacts(
            companyfacts_payload(unit="shares"), cik=320193, symbol="AAPL",
            as_of="2026-08-01T00:00:00Z", retrieved_at="2026-08-02T00:00:00Z",
            accession_availability=availability,
        )[0]
        with self.assertRaisesRegex(ValueError, "share company"):
            numeric_fact_trend([usd, shares], as_of="2026-08-01T00:00:00Z")

    def test_trend_uses_latest_eligible_restatement_per_period(self):
        availability = {
            "0000320193-26-000010": "2026-04-20T20:00:00Z",
            "0000320193-26-000011": "2026-07-20T20:00:00Z",
        }
        facts = list(self.adapter.normalize_companyfacts(
            companyfacts_payload(), cik=320193, symbol="AAPL",
            as_of="2026-08-01T00:00:00Z", retrieved_at="2026-08-02T00:00:00Z",
            accession_availability=availability,
        ))
        original = facts[1]
        amended_payload = companyfacts_payload()
        amended_payload["facts"]["us-gaap"]["Revenue"]["units"]["USD"] = [{
            "start": "2025-04-01", "end": "2025-06-30", "val": 125,
            "accn": "0000320193-26-000012", "form": "10-Q/A",
            "filed": "2026-07-25", "fy": 2026, "fp": "Q2",
        }]
        amended = self.adapter.normalize_companyfacts(
            amended_payload, cik=320193, symbol="AAPL",
            as_of="2026-08-01T00:00:00Z", retrieved_at="2026-08-02T00:00:00Z",
            accession_availability={
                "0000320193-26-000012": "2026-07-25T20:00:00Z",
            },
        )[0]
        trend = numeric_fact_trend(
            [facts[0], original, amended], as_of="2026-08-01T00:00:00Z")
        self.assertEqual(trend["current_value"], 125.0)
        self.assertEqual(trend["source_evidence_ids"][-1], amended.id)

        before_amendment = numeric_fact_trend(
            [facts[0], original, amended], as_of="2026-07-21T00:00:00Z")
        self.assertEqual(before_amendment["current_value"], 120.0)

    def test_companyfacts_deduplicate_and_reject_conflicts_and_wrong_cik(self):
        availability = {
            "0000320193-26-000010": "2026-04-20T20:00:00Z",
            "0000320193-26-000011": "2026-07-20T20:00:00Z",
        }
        payload = companyfacts_payload()
        rows = payload["facts"]["us-gaap"]["Revenue"]["units"]["USD"]
        rows.append(dict(rows[0]))
        facts = self.adapter.normalize_companyfacts(
            payload, cik=320193, symbol="AAPL", as_of="2026-08-01T00:00:00Z",
            retrieved_at="2026-08-02T00:00:00Z",
            accession_availability=availability,
        )
        self.assertEqual(len(facts), 2)

        rows[-1]["val"] = 999
        with self.assertRaisesRegex(ValueError, "conflicting duplicate"):
            self.adapter.normalize_companyfacts(
                payload, cik=320193, symbol="AAPL", as_of="2026-08-01T00:00:00Z",
                retrieved_at="2026-08-02T00:00:00Z",
                accession_availability=availability,
            )
        wrong_cik = companyfacts_payload()
        wrong_cik["cik"] = 789019
        with self.assertRaisesRegex(ValueError, "CIK does not match"):
            self.adapter.normalize_companyfacts(
                wrong_cik, cik=320193, symbol="AAPL",
                as_of="2026-08-01T00:00:00Z",
                retrieved_at="2026-08-02T00:00:00Z",
                accession_availability=availability,
            )

    def test_malformed_payloads_and_user_agent_are_rejected(self):
        with self.assertRaises(ValueError):
            SecDataAdapter("anonymous", Mock())
        with self.assertRaises(ValueError):
            self.adapter.normalize_submissions(
                {"form": ["10-K"], "accessionNumber": []}, cik=320193,
                symbol="AAPL", as_of="2026-01-01T00:00:00Z",
                retrieved_at="2026-01-02T00:00:00Z",
            )


if __name__ == "__main__":
    unittest.main()
