"""Only injected transports; endpoint/redirect and broker-read guards."""

import unittest
from datetime import timedelta
from unittest.mock import Mock, patch

from agent.execution.alpaca_paper import AlpacaPaperBroker, BrokerUnavailable, _NoRedirect
from tests.test_paper_execution import NOW


class AlpacaPaperTests(unittest.TestCase):
    def test_single_submit_fixed_paper_endpoint_and_secret_safe_failure(self):
        calls = []

        def transport(method, url, payload):
            calls.append((method, url, payload))
            raise TimeoutError("MY-PRIVATE-TOKEN")

        broker = AlpacaPaperBroker("key", "MY-PRIVATE-TOKEN", transport=transport)
        with self.assertRaisesRegex(BrokerUnavailable, "^paper submission outcome") as failure:
            broker.submit(client_id="stable", symbol="AAPL", side="buy", qty=1, limit_price=100)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], "https://paper-api.alpaca.markets/v2/orders")
        self.assertFalse(calls[0][2]["extended_hours"])
        self.assertEqual(calls[0][2]["time_in_force"], "ioc")
        self.assertNotIn("MY-PRIVATE-TOKEN", str(failure.exception))
        with self.assertRaises(ValueError):
            broker.quote("../v2/account")

    def test_redirect_handler_refuses_every_destination_without_forwarding(self):
        self.assertIsNone(
            _NoRedirect().redirect_request(
                None, None, 302, "", {}, "https://api.alpaca.markets/v2/orders"
            )
        )
        opener = Mock()
        opener.open.side_effect = OSError("secret")
        broker = AlpacaPaperBroker("key", "secret")
        with patch("agent.execution.alpaca_paper.build_opener", return_value=opener) as create:
            with self.assertRaises(BrokerUnavailable):
                broker.submit(client_id="stable", symbol="AAPL", side="buy", qty=1, limit_price=100)
            self.assertIsInstance(create.call_args.args[0], _NoRedirect)
            opener.open.assert_called_once()
            self.assertEqual(opener.open.call_args.kwargs["timeout"], 30)

    def test_oldest_component_time_and_truncated_orders(self):
        times = iter([NOW, NOW + timedelta(seconds=90)])

        def transport(_method, url, _payload):
            if "/account" in url:
                return {
                    "id": "account",
                    "cash": "100000",
                    "equity": "100000",
                    "buying_power": "400000",
                    "status": "ACTIVE",
                }
            if "/positions" in url:
                return []
            if "/clock" in url:
                return {"is_open": True, "next_close": (NOW + timedelta(hours=5)).isoformat()}
            return []

        broker = AlpacaPaperBroker("key", "secret", transport=transport, clock=lambda: next(times))
        snapshot = broker.snapshot()
        self.assertEqual(snapshot.observed_at, NOW.isoformat())

        def capped(method, url, payload):
            return [{}] * 500 if "/orders?" in url else transport(method, url, payload)

        broker = AlpacaPaperBroker("key", "secret", transport=capped, clock=lambda: NOW)
        with self.assertRaisesRegex(BrokerUnavailable, "incomplete"):
            broker.snapshot()

    def test_asset_uses_official_class_alias_not_position_asset_class(self):
        broker = AlpacaPaperBroker(
            "key",
            "secret",
            transport=lambda *_: {
                "symbol": "AAPL",
                "class": "us_equity",
                "status": "active",
                "tradable": True,
            },
        )
        self.assertTrue(broker.tradable("AAPL"))

    def test_documented_rest_shapes_execute_and_reconcile_one_order(self):
        import tempfile
        from pathlib import Path

        from agent.config import Config
        from agent.contracts import Decision
        from agent.execution import Outbox, PaperExecutor

        orders, posts = {}, []

        def transport(method, url, payload):
            if method == "POST":
                posts.append(dict(payload))
                order = {
                    **payload,
                    "id": "broker-order",
                    "filled_qty": "0",
                    "filled_avg_price": None,
                    "filled_at": None,
                    "status": "new",
                }
                orders[order["client_order_id"]] = order
                return order
            if "/orders:by_client_order_id?" in url:
                from urllib.parse import parse_qs, urlparse

                return orders.get(parse_qs(urlparse(url).query)["client_order_id"][0])
            if "/orders?" in url:
                return list(orders.values())
            if "/account" in url:
                return {
                    "id": "paper-account",
                    "cash": "100000",
                    "equity": "100000",
                    "buying_power": "400000",
                    "status": "ACTIVE",
                    "currency": "USD",
                }
            if "/positions" in url:
                return []
            if "/clock" in url:
                return {"is_open": True, "next_close": (NOW + timedelta(hours=5)).isoformat()}
            if "/assets/" in url:
                return {
                    "symbol": "AAPL",
                    "class": "us_equity",
                    "status": "active",
                    "tradable": True,
                }
            if "/stocks/quotes/latest?" in url:
                self.assertIn("feed=iex", url)
                return {"quotes": {"AAPL": {"bp": 99.9, "ap": 100, "t": NOW.isoformat()}}}
            raise AssertionError("unexpected broker route")

        with tempfile.TemporaryDirectory() as root:
            box = Outbox(Path(root) / "execution.sqlite")
            try:
                broker = AlpacaPaperBroker("key", "secret", transport=transport, clock=lambda: NOW)
                runner = PaperExecutor(
                    broker,
                    box,
                    Config(),
                    run_id="run",
                    policy_identity={"adapter": "pinned"},
                    clock=lambda: NOW,
                )
                decision = Decision("AAPL", "buy", 5, "Known evidence", ("source",))
                for _ in range(2):
                    runner.execute(
                        [decision], cycle_id="fixed", allowed_evidence_ids={"source"}, submit=True
                    )
                self.assertEqual(len(posts), 1)
                self.assertEqual(posts[0]["time_in_force"], "ioc")
                self.assertEqual(posts[0]["limit_price"], "100.05")
            finally:
                box.close()
