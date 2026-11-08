"""Full paper orchestration with scripted research and injected broker faults."""

import contextlib
import io
import unittest
from datetime import timedelta

from agent.context import build_context
from agent.contracts import Evidence, canonical_json
from agent.runtime import PaperTradingLoop
from tests import test_paper_execution as fixtures

NOW = fixtures.NOW


class Policy:
    def __init__(self):
        self.calls = 0
        self.fail = False

    def __call__(self, *, state, as_of, evidence, universe, memories=()):
        self.calls += 1
        if self.fail:
            return {"status": "parse_failure", "failure": "invalid_json"}
        context = build_context(
            state=state,
            evidence=evidence,
            memories=memories,
            token_counter=lambda _: 1,
            max_context_tokens=4096,
            reserved_final_tokens=512,
        )
        completion = canonical_json(
            {
                "decisions": [
                    {
                        "symbol": "AAPL",
                        "action": "buy",
                        "qty": 5,
                        "reason": "Visible source",
                        "evidence_ids": [next(iter(context.aliases))],
                    }
                ]
            }
        )
        return {
            "status": "success",
            "messages": context.packed.messages,
            "original_messages": context.original_messages,
            "completion": completion,
            "evidence_aliases": context.aliases,
            "tool_calls": [],
        }


class PaperRuntimeTests(unittest.TestCase):
    setUp = fixtures.PaperExecutionTests.setUp
    tearDown = fixtures.PaperExecutionTests.tearDown

    def loop(self, policy=None, source=None, clock=None):
        self.policy = policy or Policy()

        def provider(cutoff, universe):
            return (
                Evidence(
                    "article", "alpaca.news", "AAPL", NOW, NOW, NOW, "Actual archived source text."
                ),
            )

        return PaperTradingLoop(
            self.policy, self.runner, source or provider, clock=clock or (lambda: NOW)
        )

    def run_cycle(self, loop, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return loop.run_once(cycle_id="stable-cycle", **kwargs)

    def test_complete_cycle_and_restart_reuses_plan_without_generation_or_duplicate(self):
        loop = self.loop()
        first = self.run_cycle(loop, submit=True)
        second = self.run_cycle(loop, submit=True)
        self.assertEqual(first, second)
        self.assertEqual(first["status"], "completed")
        self.assertEqual(self.policy.calls, 1)
        self.assertEqual(len(self.broker.posts), 1)

    def test_ambiguous_submit_recovery_uses_durable_proposal(self):
        from agent.execution import ExecutionPaused

        loop = self.loop()
        self.broker.lose_response = True
        with self.assertRaises(ExecutionPaused):
            self.run_cycle(loop, submit=True)
        result = self.run_cycle(loop, submit=True)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(self.policy.calls, 1)
        self.assertEqual(len(self.broker.posts), 1)

    def test_invalid_generation_is_retained_without_fabricated_trade(self):
        policy = Policy()
        policy.fail = True
        loop = self.loop(policy)
        self.assertEqual(self.run_cycle(loop, submit=True)["status"], "invalid_policy")
        self.run_cycle(loop, submit=True)
        self.assertEqual(policy.calls, 1)
        self.assertEqual(self.broker.posts, [])

    def test_expired_generation_signal_is_not_sent(self):
        moments = iter([NOW, NOW + timedelta(minutes=6)])
        loop = self.loop(clock=lambda: next(moments))
        self.assertEqual(self.run_cycle(loop, submit=True)["status"], "expired_signal")
        self.assertEqual(self.broker.posts, [])

    def test_dry_run_requires_a_distinct_submitting_cycle_id(self):
        loop = self.loop()
        result = self.run_cycle(loop, submit=False)
        self.assertEqual(result["execution"][0]["status"], "dry_run")
        with self.assertRaisesRegex(ValueError, "different cycle IDs"):
            self.run_cycle(loop, submit=True)
        self.assertEqual(self.broker.posts, [])

    def test_future_provider_evidence_never_reaches_policy(self):
        future = NOW + timedelta(days=1)
        loop = self.loop(
            source=lambda *_: (
                Evidence("future", "test", "AAPL", future, future, future, "secret future"),
            )
        )
        with self.assertRaisesRegex(ValueError, "future evidence"):
            self.run_cycle(loop, submit=True)
        self.assertEqual(self.policy.calls, 0)
        self.assertEqual(self.broker.posts, [])


    def test_nonmapping_policy_result_is_recorded_invalid_without_orders(self):
        loop = self.loop(policy=lambda **_: None)
        result = self.run_cycle(loop, submit=True)
        self.assertEqual(result['status'], 'invalid_policy')
        self.assertEqual(self.broker.posts, [])
