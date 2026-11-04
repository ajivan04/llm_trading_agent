import unittest
import json

from agent.compaction import CompactionConfig, compact_evidence, compaction_manifest
from agent.contracts import Evidence
from agent.compact_cli import compact_messages


def ev(eid, source, when, text, values=None, accession=None):
    retrieved = max(when, "2026-11-02T00:00:00Z")
    return Evidence(
        id=eid, source=source, symbol="AAPL", observed_at=when,
        available_at=when, retrieved_at=retrieved, text=text,
        values=values or {}, accession_id=accession,
    )


class CompactionTests(unittest.TestCase):
    def test_selects_recent_relevant_news_and_filing_text(self):
        rows = [
            ev("old", "alpaca.news", "2026-09-01T12:00:00Z", "old headline"),
            ev("new", "alpaca.news", "2026-10-30T12:00:00Z", "AI earnings guidance", {"relevance_score": 1}),
            ev("filing", "sec.filing_text", "2026-10-29T12:00:00Z", "MD&A revenue risk", accession="a1"),
        ]
        result = compact_evidence(rows, as_of="2026-10-31T00:00:00Z",
                                  config=CompactionConfig(max_news=1, max_filings=1))
        self.assertEqual([row["id"] for row in result], ["filing", "new"])
        self.assertEqual(result[1]["values"]["event_types"], ("earnings", "ai_product"))

    def test_rejects_future_evidence_and_preserves_ids(self):
        future = ev("future", "alpaca.news", "2026-11-01T00:00:00Z", "future")
        with self.assertRaisesRegex(ValueError, "future evidence"):
            compact_evidence([future], as_of="2026-10-31T00:00:00Z")

    def test_manifest_is_versioned_and_hashed(self):
        rows = [{"id": "x", "source": "alpaca.news"}]
        manifest = compaction_manifest(rows, as_of="2026-10-31T00:00:00Z")
        self.assertEqual(manifest["compaction_version"], "evidence-compaction-v2")
        self.assertEqual(manifest["record_count"], 1)
        self.assertEqual(len(manifest["compaction_hash"]), 64)

    def test_offline_message_compaction_selects_bounded_evidence(self):
        rows = [
            ev(f"n{i}", "alpaca.news", f"2026-10-{20+i:02d}T12:00:00Z",
               f"AI earnings guidance {i} " + ("x" * 500), {"relevance_score": 1.0})
            for i in range(3)
        ]
        user = {"as_of": "2026-10-31T00:00:00Z", "evidence": [item.to_dict() for item in rows]}
        messages = compact_messages(
            [{"role": "system", "content": "system"},
             {"role": "user", "content": json.dumps(user)}],
            config=CompactionConfig(max_news=2, max_text_chars=40),
        )
        payload = json.loads(messages[1]["content"])
        self.assertEqual(len(payload["evidence"]), 2)
        self.assertEqual({item["id"] for item in payload["evidence_index"]}, {"n0", "n1", "n2"})
        self.assertTrue(all(len(item["text"]) <= 40 for item in payload["evidence"]))
        self.assertEqual({item["id"] for item in payload["evidence"]}, {"n1", "n2"})

    def test_offline_projection_without_retrieval_metadata_is_supported(self):
        item = ev("n", "alpaca.news", "2026-10-30T12:00:00Z", "headline")
        projected = item.to_dict()
        projected.pop("retrieved_at")
        payload = {"as_of": "2026-10-31T00:00:00Z", "evidence": [projected]}
        messages = compact_messages([{"role": "user", "content": json.dumps(payload)}])
        self.assertEqual(json.loads(messages[0]["content"])["evidence"][0]["id"], "n")


if __name__ == "__main__":
    unittest.main()
