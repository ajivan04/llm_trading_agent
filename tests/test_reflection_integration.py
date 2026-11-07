"""Confirmed lessons pass only eligible content into the bounded policy context."""

import json
import unittest

from agent.memory import TradingMemory
from agent.policy.evidence import visible_evidence
from agent.research import ResearchPolicy, ResearchSettings
from agent.training.core import EpisodeExample, build_trajectory_sft_example
from tests.test_reflection import AT, NOW, SITUATION, Backend, outcome
from tests.test_research_runtime import Backend as ResearchBackend
from tests.test_research_runtime import final
from tests.test_trajectories import bars
from tests.test_trajectory_masks import NativeTokenizer


class ReflectionIntegrationTests(unittest.TestCase):
    def test_retrieved_eligible_lesson_reaches_policy_but_never_sft_assistant_labels(self):
        from agent.reflection import generate_reflection

        closed = outcome(source="historical.replay")
        generated = generate_reflection(
            Backend(), outcome=closed, situation=SITUATION, as_of=AT, clock=lambda: NOW
        )
        with TradingMemory(":memory:", fold="validation", clock=lambda: NOW) as memory:
            memory.add_memory(SITUATION, generated.lesson, outcome=closed, result=generated)
            eligible = memory.context_memories("AAPL", as_of=AT)
            self.assertEqual(memory.context_memories("AAPL", as_of="2026-11-06T21:59:59Z"), [])

            def close_decision(messages):
                references = visible_evidence(messages)
                quote = next(
                    key
                    for key, item in references.items()
                    if item["source"] == "derived.historical_quote"
                )
                return final(quote, action="hold")

            backend = ResearchBackend(
                [
                    '{"tool":"get_global_quote","arguments":{"symbol":"AAPL"}}',
                    close_decision,
                ]
            )
            policy = ResearchPolicy(
                backend,
                settings=ResearchSettings(
                    max_context_tokens=30000, max_new_tokens=100, tool_context_reserve=4000
                ),
            )
            result = policy(
                state={"as_of": AT, "cash": 100000, "equity": 100000, "positions": {}},
                as_of=AT,
                evidence=[bar.to_evidence() for bar in bars()["AAPL"]],
                universe=("AAPL",),
                memories=eligible,
            )
            self.assertEqual(result["status"], "success")
            for messages, _ in backend.inputs:
                user = json.loads(messages[1]["content"])
                self.assertEqual(user["memories"][0]["reflection"], generated.lesson)
            example = EpisodeExample(
                "reflection-trajectory",
                "validation",
                AT,
                "2026-11-09T21:00:00Z",
                "snapshot",
                "account",
                "evidence",
                tuple(result["messages"]),
                result["completion"],
            )
            built = build_trajectory_sft_example(
                example, NativeTokenizer(), max_length=50000, allowed_splits=("validation",)
            )
            learned = "".join(chr(token) for token in built.labels if token != -100)
            self.assertNotIn(generated.lesson, learned)
            self.assertNotIn(SITUATION, learned)
            self.assertNotIn('"tool_result"', learned)
            self.assertIn('"get_global_quote"', learned)
            self.assertIn('"decisions"', learned)

    def test_future_memory_is_rejected_before_any_generation(self):
        backend = ResearchBackend([])
        policy = ResearchPolicy(backend, settings=ResearchSettings(max_context_tokens=30000))
        with self.assertRaisesRegex(ValueError, "historically available"):
            policy(
                state={"as_of": AT, "cash": 100000, "equity": 100000, "positions": {}},
                as_of=AT,
                evidence=[bar.to_evidence() for bar in bars()["AAPL"]],
                universe=("AAPL",),
                memories=[{"available_at": "2026-11-07T00:00:00Z", "reflection": "future outcome"}],
            )
        self.assertEqual(backend.inputs, [])
