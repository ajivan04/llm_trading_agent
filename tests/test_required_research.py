"""Assignment policies must research before returning a trading decision."""

import json
import unittest

from agent.policy import QwenToolPolicy, ToolRegistry, ToolSpec
from tests.test_policy import FakeBackend, packed

FINAL = json.dumps(
    {
        "decisions": [
            {
                "symbol": "AAPL",
                "action": "hold",
                "qty": 0,
                "reason": "uncertain",
                "evidence_ids": ["e1"],
            }
        ]
    }
)


class RequiredResearchTests(unittest.TestCase):
    def test_unresearched_final_is_rejected_and_only_completed_registered_tools_count(self):
        registry = ToolRegistry("2026-01-01T00:00:00Z")
        result = QwenToolPolicy(FakeBackend([FINAL]), registry, min_tool_calls=1).run(
            packed(), allowed_symbols=["AAPL"]
        )
        self.assertEqual(result.status, "missing_tool_research")
        self.assertFalse(result.decisions)
        for status in ("ok", "missing", "provider_error"):
            with self.subTest(status=status):
                registry = ToolRegistry("2026-01-01T00:00:00Z")
                registry.register(
                    ToolSpec("lookup", "Lookup"), lambda *_, value=status: {"status": value}
                )
                backend = FakeBackend([json.dumps({"tool": "lookup", "arguments": {}}), FINAL])
                result = QwenToolPolicy(backend, registry, min_tool_calls=1).run(
                    packed(), allowed_symbols=["AAPL"]
                )
                self.assertEqual(
                    result.status,
                    "success" if status != "provider_error" else "missing_tool_research",
                )
