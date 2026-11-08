"""Notebook-style extraction accepts formatting without weakening trade validation."""

import json
import unittest

from agent.policy import QwenToolPolicy, ToolRegistry, ToolSpec, parse_decisions
from agent.policy.core import parse_policy_json
from tests.test_policy import FakeBackend, packed


class NotebookJSONTests(unittest.TestCase):
    def final(self):
        return {
            "decisions": [
                {
                    "symbol": "AAPL",
                    "action": "buy",
                    "qty": 1,
                    "reason": "Visible data",
                    "evidence_ids": ["e1"],
                }
            ]
        }

    def test_fences_prose_and_arrays_preserve_exact_decision_semantics(self):
        value = self.final()
        for raw in (
            json.dumps(value),
            "```json\n" + json.dumps(value) + "\n```",
            "Decision:\n" + json.dumps(value["decisions"]) + "\nDone.",
        ):
            result = parse_decisions(raw, allowed_symbols=["AAPL"], allowed_evidence_ids=["e1"])
            self.assertEqual(result[0].qty, 1)
            self.assertEqual(result[0].evidence_ids, ("e1",))
        with self.assertRaisesRegex(ValueError, "unambiguous"):
            parse_policy_json(json.dumps(value) + "\n" + json.dumps(value))
        value["decisions"][0]["evidence_ids"] = ["future"]
        with self.assertRaisesRegex(ValueError, "unavailable"):
            parse_decisions(
                "```json\n" + json.dumps(value) + "\n```",
                allowed_symbols=["AAPL"],
                allowed_evidence_ids=["e1"],
            )

    def test_fenced_tool_is_actually_dispatched_without_manufacturing_a_trade(self):
        registry = ToolRegistry("2026-01-01T00:00:00Z")
        registry.register(ToolSpec("lookup", "Lookup"), lambda *_: {"status": "ok"})
        backend = FakeBackend(
            ['```json\n{"tool":"lookup","arguments":{}}\n```', json.dumps(self.final())]
        )
        result = QwenToolPolicy(backend, registry, min_tool_calls=1).run(
            packed(), allowed_symbols=["AAPL"]
        )
        self.assertEqual(result.status, "success")
        self.assertEqual(len(result.transcript["tool_calls"]), 1)
        self.assertEqual(result.decisions[0].action, "buy")

    def test_malformed_outer_json_or_truncated_second_candidate_is_never_rescued(self):
        valid = json.dumps(self.final())
        rows = json.dumps(self.final()["decisions"])
        for raw in (
            '{"decisions":' + rows + ",}",
            '{"forbidden":"outer","decisions":' + rows + ",}",
            valid + '\n{"decisions":[',
            '{"decisions":' + rows + ',"extra":true}',
        ):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_decisions(raw, allowed_symbols=["AAPL"], allowed_evidence_ids=["e1"])
