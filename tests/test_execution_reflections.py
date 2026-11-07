import json
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

from agent.contracts import canonical_json
from agent.execution.contracts import BrokerOrder
from agent.execution.outbox import Outbox
from agent.execution.reflections import process_confirmed_reflections
from agent.memory import TradingMemory
from tests.test_reflection import AT, IDENTITY, NOW, Backend


class Ledger:
    def __init__(self, root, initial=None):
        self.outbox = Outbox(Path(root) / "outbox.sqlite")
        self.outbox.bind(
            account_id="paper-account", run_id="run", config_hash="config", policy_identity=IDENTITY
        )
        with self.outbox.db:
            self.outbox.db.execute(
                "INSERT INTO metadata VALUES('initial_positions',?)",
                (canonical_json(initial or {}),),
            )
        self.memory = TradingMemory(Path(root) / "memory.sqlite", run_id="run", clock=lambda: NOW)
        self.count = 0

    def close(self):
        self.memory.close()
        self.outbox.close()

    def intent(self, side, qty):
        self.count += 1
        return self.outbox.put(
            {
                "run_id": "run",
                "cycle_id": str(self.count),
                "symbol": "AAPL",
                "side": side,
                "qty": qty,
                "limit_price": 500 if side == "buy" else 1,
                "position_before": 0,
                "evidence_ids": ("eligible-source",),
            }
        )

    def fill(self, client_id, qty, average, *, full=True, confirmed=True, seconds=0):
        payload = json.loads(self.outbox.get(client_id)["payload"])
        self.outbox.confirm(
            BrokerOrder(
                "broker-" + client_id,
                client_id,
                "AAPL",
                payload["side"],
                payload["qty"],
                qty,
                average,
                "filled" if full else "partially_filled",
                payload["limit_price"],
            ),
            confirmed_at=(NOW - timedelta(minutes=10) + timedelta(seconds=seconds)).isoformat(),
        )
        if confirmed:
            self.outbox.confirm_position(client_id)

    def run(self, backend=None, memory=None, **kwargs):
        return process_confirmed_reflections(
            self.outbox,
            memory or self.memory,
            backend or Backend(),
            as_of=AT,
            clock=lambda: NOW,
            **kwargs,
        )


class ExecutionReflectionTests(unittest.TestCase):
    def test_open_positions_or_marks_never_produce_trade_reflections(self):
        with tempfile.TemporaryDirectory() as root:
            ledger = Ledger(root)
            try:
                buy = ledger.intent("buy", 2)
                ledger.fill(buy, 2, 100)
                backend = Backend()
                result = ledger.run(backend)
                self.assertEqual(result["generated"], 0)
                self.assertEqual(backend.calls, [])
                self.assertEqual(ledger.memory.memories, [])
            finally:
                ledger.close()

    def test_confirmed_closing_lot_stores_exact_realized_gain_once(self):
        with tempfile.TemporaryDirectory() as root:
            ledger = Ledger(root)
            try:
                buy, sell = ledger.intent("buy", 2), ledger.intent("sell", 2)
                ledger.fill(buy, 2, 100)
                ledger.fill(sell, 2, 110, seconds=1)
                backend = Backend()
                result = ledger.run(backend)
                self.assertEqual(result["stored"], 1)
                entry = ledger.memory.memories[0]
                self.assertEqual(entry["net_realized_pnl"], 20)
                self.assertTrue(
                    entry["outcome"]["entry_fill_ids"][0].startswith("broker-cumulative:")
                )
                self.assertEqual(entry["outcome"]["fees"], 0)
                self.assertIn("omits regulatory fees", result["fee_assumption"])
                repeated = ledger.run(backend)
                self.assertEqual(repeated["generated"], 0)
                self.assertEqual(len(backend.calls), 1)
                self.assertEqual(len(ledger.memory.memories), 1)
            finally:
                ledger.close()

    def test_partial_fill_deltas_match_fifo_without_cumulative_double_counting(self):
        with tempfile.TemporaryDirectory() as root:
            ledger = Ledger(root)
            try:
                buy = ledger.intent("buy", 4)
                ledger.fill(buy, 2, 100, full=False)
                ledger.fill(buy, 4, 105, seconds=1)
                sell = ledger.intent("sell", 3)
                ledger.fill(sell, 1, 120, full=False, seconds=2)
                ledger.fill(sell, 3, (120 + 250) / 3, seconds=3)
                result = ledger.run(max_reflections=10)
                self.assertEqual(result["stored"], 3)
                self.assertEqual(
                    sum(row["outcome"]["quantity"] for row in ledger.memory.memories), 3
                )
                self.assertAlmostEqual(
                    sum(row["net_realized_pnl"] for row in ledger.memory.memories), 60
                )
                self.assertEqual(ledger.run()["stored"], 0)
            finally:
                ledger.close()

    def test_unknown_initial_holdings_are_consumed_without_claiming_their_cost_basis(self):
        with tempfile.TemporaryDirectory() as root:
            ledger = Ledger(root, initial={"AAPL": 2})
            try:
                buy, sell = ledger.intent("buy", 2), ledger.intent("sell", 3)
                ledger.fill(buy, 2, 100)
                ledger.fill(sell, 3, 110, seconds=1)
                result = ledger.run()
                self.assertEqual(result["stored"], 1)
                row = ledger.memory.memories[0]
                self.assertEqual(row["outcome"]["quantity"], 1)
                self.assertEqual(row["net_realized_pnl"], 10)
            finally:
                ledger.close()

    def test_unreconciled_positions_or_ambiguous_submissions_block_generation(self):
        with tempfile.TemporaryDirectory() as root:
            ledger = Ledger(root)
            try:
                buy = ledger.intent("buy", 1)
                ledger.fill(buy, 1, 100, confirmed=False)
                backend = Backend()
                self.assertEqual(ledger.run(backend)["status"], "awaiting_reconciliation")
                ledger.outbox.confirm_position(buy)
                ambiguous = ledger.intent("sell", 1)
                ledger.outbox.transition(ambiguous, "unknown")
                self.assertEqual(ledger.run(backend)["status"], "awaiting_reconciliation")
                self.assertEqual(backend.calls, [])
            finally:
                ledger.close()

    def test_failed_or_crashed_generation_is_not_retried_automatically(self):
        for output in ("invalid JSON", SystemExit("simulated abrupt generation exit")):
            with self.subTest(output=type(output).__name__), tempfile.TemporaryDirectory() as root:
                ledger = Ledger(root)
                try:
                    buy, sell = ledger.intent("buy", 1), ledger.intent("sell", 1)
                    ledger.fill(buy, 1, 100)
                    ledger.fill(sell, 1, 90, seconds=1)
                    if isinstance(output, SystemExit):

                        class Crash(Backend):
                            def generate(self, *args, failure=output, **kwargs):
                                raise failure

                        with self.assertRaises(SystemExit):
                            ledger.run(Crash())
                    else:
                        self.assertEqual(ledger.run(Backend(output))["failed"], 1)
                    backend = Backend()
                    result = ledger.run(backend)
                    self.assertEqual(backend.calls, [])
                    self.assertEqual(ledger.memory.memories, [])
                    if isinstance(output, SystemExit):
                        self.assertEqual(result["uncertain"], 1)
                finally:
                    ledger.close()

    def test_persisted_success_can_retry_only_idempotent_memory_write(self):
        with tempfile.TemporaryDirectory() as root:
            ledger = Ledger(root)
            try:
                buy, sell = ledger.intent("buy", 1), ledger.intent("sell", 1)
                ledger.fill(buy, 1, 100)
                ledger.fill(sell, 1, 110, seconds=1)

                class UnavailableMemory:
                    fold = "paper"
                    run_id = "run"

                    def add_memory(self, *args, **kwargs):
                        raise RuntimeError("provider SECRET must not be persisted")

                backend = Backend()
                self.assertEqual(
                    ledger.run(backend, memory=UnavailableMemory())["memory_pending"], 1
                )
                self.assertEqual(ledger.run(backend)["stored"], 1)
                self.assertEqual(len(backend.calls), 1)
                audit = canonical_json(
                    [
                        dict(row)
                        for row in ledger.outbox.db.execute("SELECT * FROM reflection_attempts")
                    ]
                )
                self.assertNotIn("SECRET", audit)
            finally:
                ledger.close()


class ReflectionLedgerArchiveTests(unittest.TestCase):
    def test_original_entry_reason_and_language_evidence_are_used(self):
        from agent.context import build_context
        from agent.contracts import Evidence, content_hash

        with tempfile.TemporaryDirectory() as root:
            ledger = Ledger(root)
            try:
                buy, sell = ledger.intent("buy", 1), ledger.intent("sell", 1)
                at = (NOW - timedelta(minutes=10)).isoformat()
                source = Evidence(
                    "news-source",
                    "alpaca.news",
                    "AAPL",
                    at,
                    at,
                    at,
                    "AI demand raised the product revenue outlook.",
                )
                context = build_context(
                    state={"as_of": at},
                    evidence=(source,),
                    token_counter=lambda _: 1,
                    max_context_tokens=4096,
                    reserved_final_tokens=512,
                )
                archive = {
                    "decision_at": at,
                    "decisions": [
                        {
                            "symbol": "AAPL",
                            "action": "buy",
                            "reason": "AI product demand strengthened the margin outlook.",
                        }
                    ],
                    "response": {"messages": context.packed.messages},
                }
                with ledger.outbox.db:
                    ledger.outbox.db.execute(
                        "CREATE TABLE research_cycles("
                        "cycle_id TEXT PRIMARY KEY,payload TEXT,payload_hash TEXT)"
                    )
                    ledger.outbox.db.execute(
                        "INSERT INTO research_cycles VALUES(?,?,?)",
                        ("1", canonical_json(archive), content_hash(archive)),
                    )
                ledger.fill(buy, 1, 100)
                ledger.fill(sell, 1, 110, seconds=1)
                ledger.run()
                situation = ledger.memory.memories[0]["situation"]
                self.assertIn("AI product demand", situation)
                self.assertIn("AI demand raised", situation)
            finally:
                ledger.close()

    def test_saved_generated_result_and_ledger_scope_are_checked_before_recovery(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as root:
            ledger = Ledger(root)
            try:
                buy, sell = ledger.intent("buy", 1), ledger.intent("sell", 1)
                ledger.fill(buy, 1, 100)
                ledger.fill(sell, 1, 110, seconds=1)
                with patch.object(ledger.memory, "add_memory", side_effect=RuntimeError("offline")):
                    ledger.run()
                row = ledger.outbox.db.execute("SELECT result FROM reflection_attempts").fetchone()
                tampered = json.loads(row["result"])
                tampered["lesson"] = "Changed after generation."
                with ledger.outbox.db:
                    ledger.outbox.db.execute(
                        "UPDATE reflection_attempts SET result=?", (canonical_json(tampered),)
                    )
                backend = Backend()
                with self.assertRaisesRegex(ValueError, "checksum"):
                    ledger.run(backend)
                self.assertEqual(backend.calls, [])
                ledger.memory.run_id = "another-run"
                with self.assertRaisesRegex(ValueError, "scope differ"):
                    ledger.run(backend)
            finally:
                ledger.close()
