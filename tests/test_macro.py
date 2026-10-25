import unittest
from datetime import date, datetime, timezone

from agent.data.macro import AlfredAdapter, point_in_time_values


UTC = timezone.utc


def availability(_series, _observation_date, vintage):
    return datetime(vintage.year, vintage.month, vintage.day, 14, 0, tzinfo=UTC)


def observation(
    value="3.5", *, date_value="2026-01-01",
    realtime_start="2026-01-10", realtime_end="2026-03-01",
):
    return {
        "date": date_value,
        "realtime_start": realtime_start,
        "realtime_end": realtime_end,
        "value": value,
    }


def fetch(adapter, **changes):
    arguments = {
        "series_id": "UNRATE",
        "observation_start": "2026-01-01",
        "observation_end": "2026-03-01",
        "realtime_start": "2026-01-01",
        "realtime_end": "2026-03-01",
        "as_of": "2026-02-01T15:00:00Z",
        "retrieved_at": "2026-03-02T00:00:00Z",
        "units": "Percent",
        "transformation": "lin",
    }
    arguments.update(changes)
    return adapter.fetch(**arguments)


class AlfredAdapterTests(unittest.TestCase):
    def test_preserves_vintages_missing_values_and_age(self):
        rows = [
            observation("3.5", realtime_start="2026-01-10"),
            observation("3.6", realtime_start="2026-01-20", realtime_end="2026-02-20"),
            observation(".", date_value="2026-02-01", realtime_start="2026-02-01"),
        ]
        adapter = AlfredAdapter(
            api_key="secret", availability_resolver=availability,
            requester=lambda _url, _params: {"observations": rows},
        )

        result = fetch(adapter)

        self.assertEqual(len(result), 3)
        self.assertNotEqual(result[0].id, result[1].id)
        self.assertEqual(result[0].values["units"], "Percent")
        self.assertEqual(result[0].values["transformation"], "lin")
        self.assertEqual(result[1].vintage_id, "2026-01-20")
        self.assertNotIn("realtime_end", result[1].values)
        self.assertNotIn("2026-02-20", result[1].id)
        self.assertTrue(result[2].values["missing"])
        self.assertIsNone(result[2].values["value"])
        joined = point_in_time_values(
            result, as_of="2026-02-01T15:00:00Z",
            series_specs={"UNRATE": ("Percent", "lin")},
        )
        self.assertGreaterEqual(joined["UNRATE"]["age_seconds"], 0)

    def test_filters_future_observations_vintages_and_availability(self):
        rows = [
            observation(date_value="2026-02-02", realtime_start="2026-01-10"),
            observation(date_value="2026-01-01", realtime_start="2026-02-02"),
            observation(date_value="2026-01-01", realtime_start="2026-02-01"),
        ]
        adapter = AlfredAdapter(
            api_key="secret", availability_resolver=availability,
            requester=lambda _url, _params: {"observations": rows},
        )

        result = fetch(adapter, as_of="2026-02-01T13:59:59Z")

        self.assertEqual(result, ())

    def test_requires_defensible_date_only_availability(self):
        adapter = AlfredAdapter(
            api_key="secret", availability_resolver=lambda *_args: None,
            requester=lambda _url, _params: {"observations": [observation()]},
        )
        with self.assertRaisesRegex(ValueError, "no defensible availability"):
            fetch(adapter)

    def test_request_parameters_are_explicit(self):
        calls = []

        def requester(endpoint, params):
            calls.append((endpoint, dict(params)))
            return {"observations": []}

        adapter = AlfredAdapter(
            api_key="secret", availability_resolver=availability, requester=requester,
        )
        fetch(adapter)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1]["series_id"], "UNRATE")
        self.assertEqual(calls[0][1]["units"], "lin")
        self.assertEqual(calls[0][1]["file_type"], "json")

    def test_errors_do_not_echo_api_key_or_provider_payload(self):
        secret = "DO-NOT-LEAK"

        def failed(_endpoint, params):
            raise RuntimeError(f"bad key {params['api_key']}")

        adapter = AlfredAdapter(
            api_key=secret, availability_resolver=availability, requester=failed,
        )
        with self.assertRaises(RuntimeError) as caught:
            fetch(adapter)
        self.assertNotIn(secret, str(caught.exception))

        provider_error = AlfredAdapter(
            api_key=secret, availability_resolver=availability,
            requester=lambda *_args: {
                "observations": [], "error_code": 400,
                "error_message": f"invalid {secret}",
            },
        )
        with self.assertRaises(RuntimeError) as caught:
            fetch(provider_error)
        self.assertNotIn(secret, str(caught.exception))

    def test_rejects_malformed_and_out_of_range_payloads(self):
        for payload in (
            {},
            {"observations": "bad"},
            {"observations": [object()]},
            {"observations": [observation(date_value="2025-12-31")]},
            {"observations": [observation(value="nan")]},
        ):
            with self.subTest(payload=payload), self.assertRaises((RuntimeError, ValueError)):
                fetch(AlfredAdapter(
                    api_key="secret", availability_resolver=availability,
                    requester=lambda *_args, _payload=payload: _payload,
                ))

    def test_results_are_deterministic_and_deduplicate_identical_rows(self):
        rows = [
            observation(date_value="2026-02-01", realtime_start="2026-02-01"),
            observation(), observation(),
        ]
        adapter = AlfredAdapter(
            api_key="secret", availability_resolver=availability,
            requester=lambda *_args: {"observations": rows},
        )

        first = fetch(adapter)
        second = fetch(adapter)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        self.assertEqual(
            [item.values["observation_date"] for item in first],
            ["2026-01-01", "2026-02-01"],
        )

    def test_evidence_identity_does_not_depend_on_query_as_of(self):
        adapter = AlfredAdapter(
            api_key="secret", availability_resolver=availability,
            requester=lambda *_args: {"observations": [observation()]},
        )
        first = fetch(adapter, as_of="2026-02-01T15:00:00Z")
        second = fetch(adapter, as_of="2026-02-02T15:00:00Z")
        self.assertEqual(first, second)


class PointInTimeJoinTests(unittest.TestCase):
    def test_forward_fill_selects_latest_eligible_compatible_revision(self):
        rows = [
            observation("3.5", realtime_start="2026-01-10"),
            observation("3.6", realtime_start="2026-01-20"),
            observation(".", date_value="2026-02-01", realtime_start="2026-02-01"),
        ]
        adapter = AlfredAdapter(
            api_key="secret", availability_resolver=availability,
            requester=lambda *_args: {"observations": rows},
        )
        evidence = fetch(adapter)

        values = point_in_time_values(
            evidence, as_of="2026-02-01T15:00:00Z",
            series_specs={"UNRATE": ("Percent", "lin")},
        )

        self.assertEqual(values["UNRATE"]["value"], 3.6)
        self.assertEqual(values["UNRATE"]["observation_date"], "2026-01-01")
        incompatible = point_in_time_values(
            evidence, as_of="2026-02-01T15:00:00Z",
            series_specs={"UNRATE": ("Index", "lin")},
        )
        self.assertTrue(incompatible["UNRATE"]["missing"])


if __name__ == "__main__":
    unittest.main()
