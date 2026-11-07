import json
import unittest
from dataclasses import replace
from datetime import UTC, datetime

from agent.reflection import ClosedTradeOutcome, generate_reflection, reflect_on_trade
from agent.training.core import MODEL_NAME, MODEL_REVISION

NOW = datetime(2026, 11, 6, 22, tzinfo=UTC)
AT = NOW.isoformat()
IDENTITY = {
    "model_name": MODEL_NAME,
    "model_revision": MODEL_REVISION,
    "adapter_hash": "base-model",
}
SITUATION = "AAPL guidance uncertainty and changing capital spending."


def outcome(**changes):
    values = dict(
        trade_id="closed-aapl-1",
        symbol="AAPL",
        quantity=2,
        opened_at="2026-11-02T15:00:00Z",
        closed_at="2026-11-06T20:00:00Z",
        confirmed_at="2026-11-06T20:00:01Z",
        entry_fill_ids=("entry-1",),
        exit_fill_ids=("exit-1",),
        entry_notional=200,
        exit_notional=210,
        fees=1,
    )
    return ClosedTradeOutcome(**{**values, **changes})


class Backend:
    policy_identity = IDENTITY

    def __init__(self, output=None, tokens=100):
        self.output = output
        self.tokens = tokens
        self.calls = []

    def token_count(self, messages):
        self.messages = messages
        return self.tokens

    def generate(self, messages, *, max_new_tokens):
        self.calls.append(max_new_tokens)
        if isinstance(self.output, Exception):
            raise self.output
        if self.output is not None:
            return self.output
        observed = json.loads(messages[1]["content"])
        return json.dumps(
            {
                **{
                    key: observed[key] for key in ("symbol", "action", "result", "net_realized_pnl")
                },
                "lesson": f"AAPL's closing sale realized a {observed['result']} after fees.",
                "uncertainty": (
                    "This does not establish that the initial thesis caused the outcome."
                ),
            }
        )


def reflection(item=None, backend=None, **kwargs):
    return generate_reflection(
        backend or Backend(),
        outcome=item or outcome(),
        situation=SITUATION,
        as_of=AT,
        clock=lambda: NOW,
        **kwargs,
    )


class ReflectionTests(unittest.TestCase):
    def test_confirmed_closed_lot_uses_realized_pnl_not_portfolio_equity(self):
        item = outcome()
        self.assertEqual(item.net_realized_pnl, 9)
        backend = Backend()
        result = reflection(item, backend)
        self.assertEqual(result.status, "success")
        payload = json.loads(backend.messages[1]["content"])
        self.assertEqual(payload["net_realized_pnl"], 9)
        self.assertEqual(payload["symbol"], "AAPL")
        self.assertEqual(payload["action"], "sell")
        self.assertEqual(payload["result"], "gain")
        self.assertEqual(backend.calls, [128])
        self.assertEqual(result.outcome_hash, item.outcome_hash)
        self.assertEqual(len(result.prompt_hash), 64)

    def test_unmatured_missing_fills_and_invalid_amounts_rejected(self):
        with self.assertRaisesRegex(ValueError, "future"):
            reflection(outcome(confirmed_at="2026-11-07T20:00:00Z"))
        for changes in (
            {"entry_fill_ids": ()},
            {"exit_fill_ids": ("entry-1",)},
            {"quantity": 0},
            {"fees": float("nan")},
            {"source": "live"},
            {"side": "buy"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                outcome(**changes)

    def test_invalid_model_and_context_fail_before_generation(self):
        backend = Backend(tokens=10000)
        result = reflection(backend=backend)
        self.assertEqual(result.failure, "reflection_context_budget")
        self.assertEqual(backend.calls, [])
        backend.policy_identity = {**IDENTITY, "model_name": "another-model"}
        with self.assertRaisesRegex(ValueError, "pinned Qwen"):
            reflection(backend=backend)

    def test_bad_output_or_exception_produces_no_lesson_or_secret_echo(self):
        for output in (
            "```json\n{}\n```",
            '{"lesson":"x"}',
            json.dumps({"lesson": "x" * 300, "uncertainty": "y"}),
            RuntimeError("provider echoed SECRET"),
        ):
            result = reflection(backend=Backend(output))
            self.assertEqual(result.status, "failed")
            self.assertEqual(result.lesson, "")
            self.assertNotIn("SECRET", str(result))
        backend = Backend(
            json.dumps(
                {
                    "symbol": "AAPL",
                    "action": "sell",
                    "result": "gain",
                    "net_realized_pnl": 9,
                    "lesson": "SECRET",
                    "uncertainty": "unknown",
                }
            )
        )
        self.assertEqual(
            reflection(backend=backend, secret_values=("SECRET",)).failure,
            "unsafe_reflection_output",
        )

    def test_model_must_echo_authoritative_stock_action_and_realized_outcome(self):
        good = {
            "symbol": "AAPL",
            "action": "sell",
            "result": "gain",
            "net_realized_pnl": 9,
            "lesson": "Observed gain.",
            "uncertainty": "Cause is uncertain.",
        }
        for changed in (
            {"symbol": "MSFT"},
            {"action": "buy"},
            {"result": "loss"},
            {"net_realized_pnl": 10000},
            {"net_realized_pnl": True},
        ):
            with self.subTest(changed=changed):
                self.assertEqual(
                    reflection(backend=Backend(json.dumps({**good, **changed}))).status, "failed"
                )

    def test_paper_generation_cannot_be_backdated_and_replay_is_explicit(self):
        earlier = "2026-11-06T21:00:00Z"
        result = generate_reflection(
            Backend(), outcome=outcome(), situation=SITUATION, as_of=earlier, clock=lambda: NOW
        )
        self.assertEqual(result.available_at, AT)
        result = generate_reflection(
            Backend(),
            outcome=outcome(source="historical.replay"),
            situation=SITUATION,
            as_of=earlier,
            clock=lambda: NOW,
        )
        self.assertEqual(result.available_at, earlier.replace("Z", "+00:00"))
        with self.assertRaisesRegex(ValueError, "future cutoff"):
            generate_reflection(
                Backend(),
                outcome=outcome(),
                situation=SITUATION,
                as_of="2026-11-07T00:00:00Z",
                clock=lambda: NOW,
            )

    def test_notebook_wrapper_requires_confirmed_outcome_and_matching_decision(self):
        with self.assertRaisesRegex(ValueError, "confirmed closing lot"):
            reflect_on_trade(None, Backend(), SITUATION, {}, 100000, 110000)
        with self.assertRaisesRegex(ValueError, "differs"):
            reflect_on_trade(
                None,
                Backend(),
                SITUATION,
                {"symbol": "MSFT", "action": "sell"},
                100000,
                110000,
                outcome=outcome(),
                as_of=AT,
            )
        # Historical logical time avoids using present wall-clock time in replay.
        text = reflect_on_trade(
            None,
            Backend(),
            SITUATION,
            {"symbol": "AAPL", "action": "sell"},
            100000,
            110000,
            outcome=outcome(source="historical.replay"),
            as_of=AT,
        )
        self.assertIn("AAPL", text)
        self.assertNotIn("10000", text)
        with self.assertRaises(ValueError):
            replace(reflection(), lesson="x" * 301)
