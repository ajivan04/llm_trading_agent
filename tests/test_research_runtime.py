"""Offline protocol checks spanning real research context, tools and citations."""

import json
import unittest
from datetime import UTC, datetime, timedelta

from agent.context import ContextConfig, build_context
from agent.contracts import Evidence, canonical_json
from agent.policy import QwenToolPolicy, ToolRegistry, ToolSpec
from agent.policy.evidence import resolve_decisions, visible_evidence
from agent.research import ResearchPolicy, ResearchSettings

CUTOFF = "2026-11-06T22:00:00Z"


def record(eid="initial", symbol="AAPL", when=CUTOFF, text="known fact"):
    return Evidence(eid, "test", symbol, when, when, when, text)


def final(eid, symbol="AAPL", action="buy"):
    return canonical_json(
        {
            "decisions": [
                {
                    "symbol": symbol,
                    "action": action,
                    "qty": 0 if action == "hold" else 1,
                    "reason": "Visible evidence supports this decision.",
                    "evidence_ids": [eid] if eid else [],
                }
            ]
        }
    )


class Backend:
    policy_identity = {"model": "Qwen/Qwen2.5-1.5B-Instruct"}

    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.inputs = []

    @staticmethod
    def token_count(messages):
        # Exact, deterministic synthetic units; no downloaded tokenizer/model.
        return len(canonical_json(messages))

    def generate(self, messages, *, max_new_tokens):
        snapshot = json.loads(canonical_json(messages))
        self.inputs.append((snapshot, max_new_tokens))
        output = next(self.outputs)
        return output(snapshot) if callable(output) else output


def context(rows=None):
    return build_context(
        state={"as_of": CUTOFF},
        evidence=rows or [record()],
        token_counter=Backend.token_count,
        max_context_tokens=20000,
        reserved_final_tokens=100,
        config=ContextConfig(),
    )


class ResearchRuntimeTests(unittest.TestCase):
    def test_real_historical_tools_add_dynamic_citable_evidence(self):
        rows = []
        for index in range(45):
            day = datetime(2026, 9, 1, 20, tzinfo=UTC) + timedelta(days=index)
            rows.append(
                Evidence(
                    f"bar-{index}",
                    "yfinance",
                    "AAPL",
                    day,
                    day,
                    day,
                    "Completed daily bar",
                    {
                        "session": day.date().isoformat(),
                        "raw": {
                            "open": 100 + index,
                            "high": 101 + index,
                            "low": 99 + index,
                            "close": 100 + index,
                        },
                        "volume": 1000,
                        "dividends": 0,
                        "stock_splits": 0,
                        "adjustment_policy": {"auto_adjust": False},
                    },
                )
            )

        def cite_quote(messages):
            visible = visible_evidence(messages, expected_as_of=CUTOFF)
            quote = next(
                eid
                for eid, value in visible.items()
                if value["source"] == "derived.historical_quote"
            )
            return final(quote)

        backend = Backend(
            [
                '<tool_call>{"name":"get_global_quote","arguments":{"symbol":"AAPL"}}</tool_call>',
                cite_quote,
            ]
        )
        settings = ResearchSettings(
            max_context_tokens=20000, max_new_tokens=100, tool_context_reserve=2000
        )
        result = ResearchPolicy(backend, settings=settings)(
            state={"as_of": CUTOFF, "cash": 100000, "positions": {}},
            as_of=CUTOFF,
            evidence=rows,
            universe=["AAPL"],
        )
        self.assertEqual(result["status"], "success")
        self.assertTrue(
            result["decisions"][0].evidence_ids[0].startswith("derived.historical_quote:AAPL:")
        )
        self.assertEqual(result["tool_calls"][0]["tool"], "get_global_quote")
        self.assertEqual(len(backend.inputs), 2)
        for messages, output_budget in backend.inputs:
            self.assertLessEqual(backend.token_count(messages) + output_budget, 20000)
        original_visible = visible_evidence(result["original_messages"])
        self.assertIn(result["decisions"][0].evidence_ids[0], original_visible)
        self.assertNotIn("future", canonical_json(backend.inputs))

    def test_future_tool_evidence_is_blocked_before_next_generation(self):
        registry = ToolRegistry(CUTOFF)
        registry.register(
            ToolSpec("lookup", "Lookup"),
            lambda *_: {
                "evidence": [record("future-secret", when="2026-11-07T00:00:00Z").to_dict()]
            },
        )
        backend = Backend(['{"tool":"lookup","arguments":{}}', final("E1", action="hold")])
        prompt = context()
        result = QwenToolPolicy(backend, registry, continue_tool_errors=True).run(
            prompt.packed, allowed_symbols=["AAPL"], original_messages=prompt.original_messages
        )
        self.assertEqual(result.status, "success")
        self.assertNotIn("future-secret", canonical_json(backend.inputs))
        self.assertEqual(result.transcript["tool_calls"][0]["result"]["status"], "unavailable")

    def test_unknown_tool_can_recover_and_xml_tool_request_is_supported(self):
        registry = ToolRegistry(CUTOFF)
        registry.register(ToolSpec("lookup", "Lookup"), lambda *_: {"evidence": []})
        backend = Backend(
            [
                '{"tool":"unknown","arguments":{}}',
                '<tool_call>{"name":"lookup","arguments":{}}</tool_call>',
                final("E1", action="hold"),
            ]
        )
        prompt = context()
        result = QwenToolPolicy(backend, registry, continue_tool_errors=True).run(
            prompt.packed, allowed_symbols=["AAPL"], original_messages=prompt.original_messages
        )
        self.assertEqual(result.status, "success")
        self.assertEqual(len(result.transcript["tool_calls"]), 2)

    def test_tool_evidence_is_omitted_instead_of_exceeding_context(self):
        prompt = context()
        limit = Backend.token_count(prompt.packed.messages) + 700
        registry = ToolRegistry(CUTOFF)
        registry.register(
            ToolSpec("lookup", "Lookup"),
            lambda *_: {"evidence": [record("large", text="x" * 10000).to_dict()]},
        )
        backend = Backend(['{"tool":"lookup","arguments":{}}', final("E1", action="hold")])
        result = QwenToolPolicy(
            backend,
            registry,
            token_counter=backend.token_count,
            max_context_tokens=limit,
            max_new_tokens=100,
        ).run(prompt.packed, allowed_symbols=["AAPL"], original_messages=prompt.original_messages)
        self.assertEqual(result.status, "success")
        self.assertEqual(result.transcript["omitted_tool_evidence_ids"], ("large",))
        self.assertNotIn("large", visible_evidence(backend.inputs[-1][0]))
        for messages, reserve in backend.inputs:
            self.assertLessEqual(backend.token_count(messages) + reserve, limit)

    def test_turn_limit_stops_unending_tool_requests(self):
        prompt = context()
        registry = ToolRegistry(CUTOFF)
        registry.register(ToolSpec("lookup", "Lookup"), lambda *_: {"evidence": []})
        backend = Backend(['{"tool":"lookup","arguments":{}}'] * 2)
        result = QwenToolPolicy(backend, registry, max_turns=2).run(
            prompt.packed, allowed_symbols=["AAPL"], original_messages=prompt.original_messages
        )
        self.assertEqual(result.failure, "tool_budget_exhausted")
        self.assertEqual(len(backend.inputs), 2)

    def test_citations_require_visible_same_symbol_evidence_for_trades(self):
        prompt = context([record("msft", symbol="MSFT")])
        kwargs = dict(
            messages=prompt.packed.messages,
            allowed_symbols=["AAPL", "MSFT"],
            aliases=prompt.aliases,
            original_messages=prompt.original_messages,
        )
        with self.assertRaisesRegex(ValueError, "another symbol"):
            resolve_decisions(final("E1", symbol="AAPL"), **kwargs)
        with self.assertRaisesRegex(ValueError, "require visible"):
            resolve_decisions(final(None), **kwargs)
        with self.assertRaisesRegex(ValueError, "unavailable"):
            resolve_decisions(final("not-visible"), **kwargs)
        self.assertEqual(resolve_decisions(final("E1", symbol="MSFT"), **kwargs)[0].symbol, "MSFT")

    def test_research_rejects_future_input_and_state_cutoff_mismatch(self):
        policy = ResearchPolicy(Backend([]))
        with self.assertRaisesRegex(ValueError, "future evidence"):
            policy(
                state={"as_of": CUTOFF},
                as_of=CUTOFF,
                universe=["AAPL"],
                evidence=[record(when="2026-11-07T00:00:00Z")],
            )
        with self.assertRaisesRegex(ValueError, "must agree"):
            policy(
                state={"as_of": "2026-11-07T00:00:00Z"},
                as_of=CUTOFF,
                universe=["AAPL"],
                evidence=[],
            )


if __name__ == "__main__":
    unittest.main()
