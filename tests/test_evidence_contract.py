"""Regression coverage for visible-only citations and exact short references."""

import json
import unittest

from agent.contracts import canonical_json
from agent.policy.evidence import alias_messages, resolve_decisions, visible_evidence

AS_OF = "2026-10-23T20:00:00Z"


def record(eid="source:opaque-sha", *, when="2026-10-22T20:00:00Z", **updates):
    return {
        "id": eid,
        "source": "alpaca.news",
        "symbol": "AAPL",
        "text": "Headline",
        "observed_at": when,
        "available_at": when,
        "values": {},
        **updates,
    }


def messages(*records, **updates):
    return [
        {
            "role": "user",
            "content": canonical_json(
                {
                    "as_of": AS_OF,
                    "evidence": list(records),
                    **updates,
                }
            ),
        }
    ]


def decision(*ids):
    return canonical_json(
        {
            "decisions": [
                {
                    "symbol": "AAPL",
                    "action": "hold",
                    "qty": 0,
                    "reason": "No positive signal",
                    "evidence_ids": list(ids),
                }
            ]
        }
    )


class EvidenceContractTests(unittest.TestCase):
    def test_index_and_assistant_claims_never_create_visible_evidence(self):
        data = messages(record(), evidence_index=[record("archive:secret")])
        data.append(
            {
                "role": "assistant",
                "content": canonical_json(
                    {
                        "as_of": AS_OF,
                        "evidence": [record("assistant:invented")],
                    }
                ),
            }
        )
        self.assertEqual(set(visible_evidence(data)), {"source:opaque-sha"})
        for unavailable in ("archive:secret", "assistant:invented"):
            with (
                self.subTest(unavailable=unavailable),
                self.assertRaisesRegex(ValueError, "unavailable evidence"),
            ):
                resolve_decisions(decision(unavailable), messages=data, allowed_symbols=["AAPL"])

    def test_historical_cutoff_checks_availability_not_retrieval(self):
        data = messages(record(retrieved_at="2026-11-06T12:00:00Z"))
        self.assertEqual(len(visible_evidence(data, expected_as_of=AS_OF)), 1)
        with self.assertRaisesRegex(ValueError, "declared decision time"):
            visible_evidence(data, expected_as_of="2026-10-22T20:00:00Z")
        future = record(when="2026-10-24T20:00:00Z", retrieved_at="2026-11-06T12:00:00Z")
        with self.assertRaisesRegex(ValueError, "future evidence"):
            visible_evidence(messages(future))

    def test_tool_evidence_is_visible_only_after_it_is_returned(self):
        data = messages(record("original"))
        with self.assertRaisesRegex(ValueError, "unavailable evidence"):
            resolve_decisions(decision("from-tool"), messages=data, allowed_symbols=["AAPL"])
        data.append(
            {
                "role": "user",
                "content": canonical_json(
                    {
                        "boundary": "UNTRUSTED_TOOL_RESULT",
                        "tool_result": {"evidence": [record("from-tool")]},
                    }
                ),
            }
        )
        resolved = resolve_decisions(decision("from-tool"), messages=data, allowed_symbols=["AAPL"])
        self.assertEqual(resolved[0].evidence_ids, ("from-tool",))
        data[-1]["content"] = canonical_json(
            {
                "tool_result": {
                    "evidence": [
                        record(
                            "from-tool",
                            when="2026-10-24T20:00:00Z",
                            retrieved_at="2026-11-06T12:00:00Z",
                        ),
                    ]
                }
            }
        )
        with self.assertRaisesRegex(ValueError, "future evidence"):
            visible_evidence(data)

    def test_aliases_are_exact_deterministic_and_do_not_rewrite_headlines(self):
        original = messages(
            record("source:one", text="Story mentions source:one"), record("source:two")
        )
        projected, aliases = alias_messages(original)
        self.assertEqual(aliases, {"E1": "source:one", "E2": "source:two"})
        self.assertEqual(alias_messages(original), (projected, aliases))
        payload = json.loads(projected[0]["content"])
        self.assertEqual(payload["evidence"][0]["id"], "E1")
        self.assertEqual(payload["evidence"][0]["text"], "Story mentions source:one")
        self.assertEqual(json.loads(original[0]["content"])["evidence"][0]["id"], "source:one")
        self.assertEqual(
            resolve_decisions(
                decision("E1"),
                messages=projected,
                original_messages=original,
                allowed_symbols=["AAPL"],
                aliases=aliases,
            )[0].evidence_ids,
            ("source:one",),
        )
        with self.assertRaisesRegex(ValueError, "absent from the projected prompt"):
            resolve_decisions(
                decision("E1"),
                messages=original,
                original_messages=original,
                allowed_symbols=["AAPL"],
                aliases=aliases,
            )
        with self.assertRaisesRegex(ValueError, "unavailable evidence"):
            resolve_decisions(
                decision("E3"),
                messages=projected,
                original_messages=original,
                allowed_symbols=["AAPL"],
                aliases=aliases,
            )

    def test_alias_projection_strips_audit_provenance_without_mutating_records(self):
        original = messages(
            record(
                retrieved_at="2026-11-06T12:00:00Z",
                raw_content_hash="a" * 64,
                values={
                    "article_count": 100,
                    "source_evidence_ids": ["one", "two"],
                    "source_evidence_fingerprints": ["hash"],
                },
            ),
            evidence_index=[{"id": "secret"}],
        )
        projected, _ = alias_messages(original)
        payload = json.loads(projected[0]["content"])
        self.assertNotIn("evidence_index", payload)
        item = payload["evidence"][0]
        self.assertNotIn("retrieved_at", item)
        self.assertNotIn("raw_content_hash", item)
        self.assertEqual(item["values"], {"article_count": 100})
        self.assertIn("raw_content_hash", json.loads(original[0]["content"])["evidence"][0])

    def test_aliases_are_append_stable_and_cannot_authorize_archive_ids(self):
        initial = messages(record("z-first"))
        extended = initial + [
            {
                "role": "tool",
                "content": canonical_json(
                    {
                        "evidence": [record("a-later")],
                    }
                ),
            }
        ]
        _, aliases = alias_messages(extended)
        self.assertEqual(aliases, {"E1": "z-first", "E2": "a-later"})
        with self.assertRaisesRegex(ValueError, "absent from the projected prompt"):
            resolve_decisions(
                decision("E1"),
                messages=initial,
                allowed_symbols=["AAPL"],
                aliases={"E1": "archive:secret"},
            )

    def test_alias_namespace_does_not_collide_with_full_ids(self):
        original = messages(record("E1"), record("other"))
        projected, aliases = alias_messages(original)
        self.assertEqual(aliases, {"E2": "E1", "E3": "other"})
        result = resolve_decisions(
            decision("E2"),
            messages=projected,
            allowed_symbols=["AAPL"],
            aliases=aliases,
            original_messages=original,
        )
        self.assertEqual(result[0].evidence_ids, ("E1",))

    def test_alias_sidecar_requires_original_full_id_transcript(self):
        original = messages(record("source:one"))
        projected, aliases = alias_messages(original)
        with self.assertRaisesRegex(ValueError, "requires the original full-ID transcript"):
            resolve_decisions(
                decision("E1"),
                messages=projected,
                allowed_symbols=["AAPL"],
                aliases=aliases,
            )
        self.assertEqual(
            resolve_decisions(
                decision("source:one"),
                messages=original,
                allowed_symbols=["AAPL"],
            )[0].evidence_ids,
            ("source:one",),
        )

    def test_alias_sidecar_cannot_rebind_visible_handles(self):
        original = messages(record("source:one"), record("source:two"))
        projected, _ = alias_messages(original)
        poisoned_maps = (
            {"E1": "archive:secret", "E2": "source:two"},
            {"E1": "source:two", "E2": "source:one"},
        )
        for aliases in poisoned_maps:
            with (
                self.subTest(aliases=aliases),
                self.assertRaisesRegex(ValueError, "does not match the original transcript"),
            ):
                resolve_decisions(
                    decision("E1"),
                    messages=projected,
                    original_messages=original,
                    allowed_symbols=["AAPL"],
                    aliases=aliases,
                )

    def test_alias_projection_must_match_original_record_content_and_cutoff(self):
        original = messages(record("source:one"))
        projected, aliases = alias_messages(original)
        payload = json.loads(projected[0]["content"])
        payload["evidence"][0]["text"] = "Different article"
        modified = [{"role": "user", "content": canonical_json(payload)}]
        with self.assertRaisesRegex(ValueError, "records do not match"):
            resolve_decisions(
                decision("E1"),
                messages=modified,
                original_messages=original,
                allowed_symbols=["AAPL"],
                aliases=aliases,
            )
        payload["evidence"][0]["text"] = "Headline"
        payload["as_of"] = "2026-10-24T20:00:00Z"
        modified = [{"role": "user", "content": canonical_json(payload)}]
        with self.assertRaisesRegex(ValueError, "cutoff differs"):
            resolve_decisions(
                decision("E1"),
                messages=modified,
                original_messages=original,
                allowed_symbols=["AAPL"],
                aliases=aliases,
            )

    def test_duplicate_evidence_ids_require_identical_records(self):
        self.assertEqual(len(visible_evidence(messages(record(), record()))), 1)
        with self.assertRaisesRegex(ValueError, "conflicting"):
            visible_evidence(messages(record(), record(text="Different content")))

    def test_malformed_records_cutoffs_and_shapes_fail_closed(self):
        bad_inputs = [
            messages(record(available_at="bad")),
            messages(record(text="")),
            messages(record(), evidence={"id": "not-an-array"}),
            [{"role": "user", "content": canonical_json({"evidence": [record()]})}],
            messages(record()) + messages(record(), as_of="2026-10-24T20:00:00Z"),
        ]
        for item in bad_inputs:
            with self.subTest(item=item), self.assertRaises(ValueError):
                visible_evidence(item)

    def test_resolver_retains_strict_trade_contract(self):
        data = messages(record())
        bad = json.loads(decision("source:opaque-sha"))
        bad["decisions"][0]["qty"] = 5
        with self.assertRaisesRegex(ValueError, "hold requires"):
            resolve_decisions(canonical_json(bad), messages=data, allowed_symbols=["AAPL"])
        self.assertEqual(
            resolve_decisions(
                "```json\n" + decision() + "\n```", messages=data, allowed_symbols=["AAPL"]
            )[0].qty,
            0,
        )
        with self.assertRaisesRegex(ValueError, "outside"):
            resolve_decisions(decision(), messages=data, allowed_symbols=["MSFT"])
        self.assertEqual(
            resolve_decisions(decision(), messages=data, allowed_symbols=["AAPL"])[0].qty, 0
        )

    def test_tool_alias_projection_does_not_process_record_twice(self):
        data = messages(record("original")) + [
            {
                "role": "user",
                "content": canonical_json(
                    {
                        "tool_result": {"evidence": [record("tool:item")]},
                    }
                ),
            }
        ]
        projected, aliases = alias_messages(data)
        self.assertEqual(aliases, {"E1": "original", "E2": "tool:item"})
        self.assertEqual(
            json.loads(projected[-1]["content"])["tool_result"]["evidence"][0]["id"], "E2"
        )


if __name__ == "__main__":
    unittest.main()
