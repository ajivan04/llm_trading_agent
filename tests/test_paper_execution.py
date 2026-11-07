"""Fault-injected paper execution tests; never contact Alpaca."""

import json
import tempfile
import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from agent.config import Config
from agent.contracts import Decision
from agent.execution import (
    BrokerOrder,
    BrokerSnapshot,
    ExecutionPaused,
    Outbox,
    PaperExecutor,
    Quote,
)

NOW = datetime(2026, 11, 6, 16, tzinfo=UTC)


class FakeBroker:
    def __init__(self):
        self.orders = {}
        self.cash = 100000
        self.positions = {}
        self.posts = []
        self.lose_response = False
        self.account_id = "paper-account"
        self.stale_quote = False
        self.external = ()
        self.hide_lookup = False
        self.market_open = True

    def snapshot(self):
        return BrokerSnapshot(
            self.account_id,
            self.cash,
            100000,
            self.cash,
            dict(self.positions),
            tuple(
                o
                for o in self.orders.values()
                if o.status not in {"filled", "canceled", "expired", "rejected"}
            )
            + self.external,
            self.market_open,
            NOW.isoformat(),
            next_close=(NOW + timedelta(hours=5)).isoformat(),
        )

    def lookup(self, client_id):
        return None if self.hide_lookup else self.orders.get(client_id)

    def quote(self, symbol):
        at = NOW - timedelta(hours=1) if self.stale_quote else NOW
        return Quote(symbol, 99.9, 100, at.isoformat())

    def tradable(self, symbol):
        return True

    def submit(self, **kwargs):
        self.posts.append(kwargs)
        order = BrokerOrder(
            "broker-" + kwargs["client_id"],
            kwargs["client_id"],
            kwargs["symbol"],
            kwargs["side"],
            kwargs["qty"],
            0,
            None,
            "new",
            kwargs["limit_price"],
        )
        self.orders[order.client_id] = order
        if self.lose_response:
            raise TimeoutError("DO NOT LOG PROVIDER SECRETS")
        return order


def buy(symbol="AAPL", qty=5):
    return Decision(symbol, "buy", qty, "supported", ("e1",))


class PaperExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.box = Outbox(Path(self.temp.name) / "outbox.sqlite")
        self.broker = FakeBroker()
        self.cfg = Config(
            universe=("AAPL", "MSFT"), sector_map=(("AAPL", "technology"), ("MSFT", "technology"))
        )
        self.runner = PaperExecutor(
            self.broker,
            self.box,
            self.cfg,
            run_id="run",
            policy_identity={"adapter": "hash"},
            clock=lambda: NOW,
        )

    def tearDown(self):
        self.box.close()
        self.temp.cleanup()

    def execute(self, decisions=None, **kwargs):
        return self.runner.execute(
            decisions or [buy()],
            cycle_id=kwargs.pop("cycle_id", "cycle"),
            allowed_evidence_ids={"e1"},
            submit=True,
            **kwargs,
        )

    def test_accept_lost_response_reconciles_without_duplicate(self):
        self.broker.lose_response = True
        with self.assertRaisesRegex(ExecutionPaused, "uncertain"):
            self.execute()
        self.assertEqual(len(self.broker.posts), 1)
        result = self.execute()
        self.assertEqual(result[0]["status"], "already_recorded")
        self.assertEqual(len(self.broker.posts), 1)
        self.assertNotIn(
            "PROVIDER SECRETS",
            json.dumps([dict(row) for row in self.box.db.execute("SELECT * FROM events")]),
        )

    def test_404_after_timeout_keeps_global_pause(self):
        self.broker.lose_response = True
        with self.assertRaises(ExecutionPaused):
            self.execute()
        self.broker.hide_lookup = True
        with self.assertRaisesRegex(ExecutionPaused, "unresolved"):
            self.execute([buy("MSFT")], cycle_id="next")
        self.assertEqual(len(self.broker.posts), 1)

    def test_restart_after_marking_submitting_never_reposts(self):
        payload = dict(
            run_id="run",
            cycle_id="cycle",
            symbol="AAPL",
            side="buy",
            qty=5,
            limit_price=100.05,
            evidence_ids=("e1",),
            position_before=0,
        )
        client = self.box.put(payload)
        self.box.transition(client, "submitting")
        with self.assertRaises(ExecutionPaused):
            self.execute()
        self.assertEqual(self.broker.posts, [])

    def test_partial_fill_deltas_are_cumulative_not_new_average(self):
        quote = self.broker.quote
        self.broker.quote = lambda symbol: Quote(symbol, 499, 500, NOW.isoformat())
        self.execute()
        self.broker.quote = quote
        client = self.broker.posts[0]["client_id"]
        original = self.broker.orders[client]
        self.broker.orders[client] = replace(
            original, filled_qty=2, filled_avg_price=100, status="partially_filled"
        )
        self.broker.positions = {"AAPL": 2}
        self.broker.cash = 99800
        self.runner.reconcile()
        self.broker.orders[client] = replace(
            original, filled_qty=5, filled_avg_price=102, status="filled"
        )
        self.broker.positions = {"AAPL": 5}
        self.broker.cash = 99490
        self.runner.reconcile()
        self.runner.reconcile()
        deltas = self.box.db.execute(
            "SELECT qty,notional FROM fill_deltas ORDER BY rowid"
        ).fetchall()
        self.assertEqual([tuple(row) for row in deltas], [(2, 200), (3, 310)])
        self.assertEqual(self.box.active(), [])
        # Price changes/new output cannot create a second order in the same cycle.
        self.execute()
        self.assertEqual(len(self.broker.posts), 1)

    def test_position_lag_blocks_even_after_terminal_confirmation(self):
        self.execute()
        client = self.broker.posts[0]["client_id"]
        self.broker.orders[client] = replace(
            self.broker.orders[client], status="filled", filled_qty=5, filled_avg_price=100
        )
        for _ in range(2):
            with self.assertRaisesRegex(ExecutionPaused, "position"):
                self.runner.reconcile()
        self.broker.positions = {"AAPL": 5}
        self.broker.cash = 99500
        self.runner.reconcile()
        self.assertEqual(self.box.active(), [])

    def test_stale_quotes_external_orders_and_account_change_block(self):
        self.broker.stale_quote = True
        with self.assertRaisesRegex(ExecutionPaused, "quote"):
            self.execute()
        self.broker.stale_quote = False
        self.broker.external = (
            BrokerOrder("x", "external", "AAPL", "buy", 1, 0, None, "new", 100),
        )
        with self.assertRaisesRegex(ExecutionPaused, "external"):
            self.execute()
        self.broker.external = ()
        self.broker.account_id = "other-account"
        with self.assertRaisesRegex(ValueError, "binding"):
            self.execute()
        self.assertEqual(self.broker.posts, [])

    def test_pending_sale_reserves_shares_and_no_proceeds(self):
        self.broker.positions = {"AAPL": 5}
        sell = Decision("AAPL", "sell", 4, "supported", ("e1",))
        self.execute([sell])
        result = self.execute([sell], cycle_id="next")
        self.assertEqual(result[0]["status"], "pending_order")
        self.assertEqual(len(self.broker.posts), 1)

    def test_unsent_intent_can_be_discarded_but_unknown_cannot(self):
        payload = dict(
            run_id="run",
            cycle_id="old",
            symbol="AAPL",
            side="buy",
            qty=5,
            limit_price=100.05,
            evidence_ids=("e1",),
            position_before=0,
        )
        client = self.box.put(payload)
        self.box.discard_unsent(client)
        self.assertEqual(self.box.active(), [])
        with self.assertRaises(ValueError):
            self.box.discard_unsent(client)

    def test_dry_run_batch_reserves_combined_sector_exposure(self):
        cfg = replace(
            self.cfg,
            max_shares=500,
            max_position_weight=0.3,
            max_sector_weight=0.3,
            max_gross_weight=0.8,
        )
        self.runner.config = cfg
        result = self.runner.execute(
            [buy(qty=200), buy("MSFT", qty=200)],
            cycle_id="dry",
            allowed_evidence_ids={"e1"},
            submit=False,
        )
        self.assertEqual(result[0]["status"], "dry_run")
        self.assertEqual(result[1]["status"], "risk_rejected")
        self.assertEqual(self.broker.posts, [])
        self.assertEqual(self.box.active(), [])

    def test_two_runners_cannot_share_active_execution_lock(self):
        other = Outbox(self.box.path)
        try:
            with self.box.lock():
                with self.assertRaisesRegex(RuntimeError, "already active"):
                    with other.lock():
                        pass
        finally:
            other.close()


class RegressionTests(unittest.TestCase):
    setUp = PaperExecutionTests.setUp
    tearDown = PaperExecutionTests.tearDown
    execute = PaperExecutionTests.execute

    def test_pending_limit_to_mark_gain_cannot_inflate_equity(self):
        self.runner.config = replace(self.cfg, max_shares=500)
        self.runner.reconcile()
        payload = dict(
            run_id="run",
            cycle_id="old",
            symbol="AAPL",
            side="buy",
            qty=100,
            limit_price=10,
            evidence_ids=("e1",),
            position_before=0,
        )
        client = self.box.put(payload)
        self.box.transition(client, "submitting")
        self.broker.orders[client] = BrokerOrder(
            "broker-old", client, "AAPL", "buy", 100, 0, None, "new", 10
        )
        result = self.execute([buy("MSFT", qty=105)], cycle_id="next")
        self.assertEqual(result[0]["status"], "risk_rejected")
        self.assertEqual(len(self.broker.posts), 0)

    def test_clock_advancing_during_quotes_blocks_submission(self):
        current = [NOW]
        self.runner.clock = lambda: current[0]
        original = self.broker.quote

        def slow(symbol):
            quote = original(symbol)
            current[0] += timedelta(seconds=61)
            return quote

        self.broker.quote = slow
        with self.assertRaises(ExecutionPaused):
            self.execute()
        self.assertEqual(self.broker.posts, [])

    def test_unsupported_status_preserves_reservation_and_pauses(self):
        self.execute()
        client = self.broker.posts[0]["client_id"]
        for status in ("pending_cancel", "replaced", "suspended", "made_up"):
            self.broker.orders[client] = replace(self.broker.orders[client], status=status)
            with self.assertRaisesRegex(ExecutionPaused, "broker order state"):
                self.execute([buy("MSFT")], cycle_id="next")
        self.assertEqual(len(self.broker.posts), 1)

    def test_close_buffer_blocks_day_order_queue(self):
        original = self.broker.snapshot
        self.broker.snapshot = lambda: replace(
            original(), next_close=(NOW + timedelta(seconds=4)).isoformat()
        )
        with self.assertRaisesRegex(ExecutionPaused, "close boundary"):
            self.execute()
        self.assertEqual(self.broker.posts, [])

    def test_confirmed_fill_cash_lag_uses_local_confirmed_notional_budget(self):
        self.execute()
        client = self.broker.posts[0]["client_id"]
        self.broker.orders[client] = replace(
            self.broker.orders[client], status="filled", filled_qty=5, filled_avg_price=100
        )
        self.broker.positions = {"AAPL": 5}
        self.broker.cash = 100000  # Broker cash component temporarily lags the fill.
        with self.assertRaisesRegex(ExecutionPaused, "broker cash"):
            self.runner.reconcile()
        self.assertEqual(self.box.cash_budget(self.broker.cash), 99500)
        self.broker.cash = 99500
        self.runner.reconcile()

    def test_submitting_intent_cannot_be_discarded(self):
        payload = dict(
            run_id="run",
            cycle_id="old",
            symbol="AAPL",
            side="buy",
            qty=5,
            limit_price=100.05,
            evidence_ids=("e1",),
            position_before=0,
        )
        client = self.box.put(payload)
        self.box.transition(client, "submitting")
        with self.assertRaises(ValueError):
            self.box.discard_unsent(client)

    def test_corrupted_intent_or_delta_ledger_cannot_resume(self):
        self.execute()
        client = self.broker.posts[0]["client_id"]
        self.box.db.execute("UPDATE intents SET payload='{}' WHERE client_id=?", (client,))
        self.box.db.commit()
        with self.assertRaises((ValueError, KeyError)):
            self.runner.reconcile()
        self.assertEqual(len(self.broker.posts), 1)

    def test_future_broker_fill_timestamp_blocks_reconciliation(self):
        self.execute()
        client = self.broker.posts[0]["client_id"]
        self.broker.orders[client] = replace(
            self.broker.orders[client],
            status="filled",
            filled_qty=5,
            filled_avg_price=100,
            filled_at=(NOW + timedelta(days=1)).isoformat(),
        )
        with self.assertRaisesRegex(ValueError, "future"):
            self.runner.reconcile()
