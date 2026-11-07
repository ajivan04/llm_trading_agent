"""Durable intents and cumulative broker confirmations, protected by a runner lock."""

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from agent.contracts import canonical_json, content_hash, parse_timestamp

from .contracts import TERMINAL


class Outbox:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=5)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
          CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS intents(
            client_id TEXT PRIMARY KEY, payload TEXT NOT NULL, payload_hash TEXT NOT NULL,
            status TEXT NOT NULL,
            broker_id TEXT, filled_qty INTEGER NOT NULL DEFAULT 0,
            filled_notional REAL NOT NULL DEFAULT 0,
            position_confirmed INTEGER NOT NULL DEFAULT 0);
          CREATE TABLE IF NOT EXISTS fill_deltas(
            id TEXT PRIMARY KEY, client_id TEXT NOT NULL, qty INTEGER NOT NULL,
            notional REAL NOT NULL, confirmed_at TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS events(
            sequence INTEGER PRIMARY KEY, kind TEXT NOT NULL, payload TEXT NOT NULL);
        """)
        self.db.commit()
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(intents)")}
        if "payload_hash" not in columns:
            self.db.close()
            raise ValueError("unversioned outbox cannot be resumed; preserve it for review")

    def close(self):
        self.db.close()

    @contextmanager
    def lock(self):
        import fcntl

        with self.path.with_suffix(self.path.suffix + ".lock").open("a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("paper execution runner is already active") from None
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def bind(self, *, account_id, run_id, config_hash, policy_identity):
        identity = canonical_json(
            {
                "endpoint": "https://paper-api.alpaca.markets",
                "account_id": account_id,
                "run_id": run_id,
                "config_hash": config_hash,
                "policy_identity": policy_identity,
            }
        )
        row = self.db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
        if row and row["value"] != identity:
            raise ValueError("outbox account/run/config/policy binding differs")
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO metadata VALUES('identity',?)", (identity,))

    def cash_budget(self, broker_cash):
        row = self.db.execute("SELECT value FROM metadata WHERE key='cash_anchor'").fetchone()
        if row is None:
            with self.db:
                self.db.execute("INSERT INTO metadata VALUES('cash_anchor',?)", (str(broker_cash),))
            anchor = broker_cash
        else:
            anchor = float(row["value"])
        expected = anchor
        for row in self.all():
            payload = json.loads(row["payload"])
            expected += row["filled_notional"] * (1 if payload["side"] == "sell" else -1)
        return min(broker_cash, expected)

    def cycle_order(self, run_id, cycle_id, symbol):
        key = (
            "qwen-" + content_hash({"run_id": run_id, "cycle_id": cycle_id, "symbol": symbol})[:40]
        )
        return self.get(key)

    def peak(self, equity):
        row = self.db.execute("SELECT value FROM metadata WHERE key='peak_equity'").fetchone()
        result = max(float(row["value"]) if row else equity, equity)
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO metadata VALUES('peak_equity',?)", (str(result),)
            )
        return result

    def record(self, kind, payload):
        with self.db:
            self.db.execute(
                "INSERT INTO events(kind,payload) VALUES(?,?)", (kind, canonical_json(payload))
            )

    def put(self, payload):
        client_id = (
            "qwen-"
            + content_hash({key: payload[key] for key in ("run_id", "cycle_id", "symbol")})[:40]
        )
        encoded = canonical_json(payload)
        old = self.get(client_id)
        if old and old["payload"] != encoded:
            raise ValueError("client order identity collision")
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO intents(client_id,payload,payload_hash,status) "
                "VALUES(?,?,?,?)",
                (client_id, encoded, content_hash(payload), "intent"),
            )
        return client_id

    def _verified(self, row):
        if row is None:
            return None
        try:
            payload = json.loads(row["payload"])
            identity = (
                "qwen-"
                + content_hash({key: payload[key] for key in ("run_id", "cycle_id", "symbol")})[:40]
            )
        except (ValueError, KeyError, TypeError):
            raise ValueError("durable intent payload is invalid") from None
        if content_hash(payload) != row["payload_hash"] or identity != row["client_id"]:
            raise ValueError("durable intent checksum or client identity failed")
        total = self.db.execute(
            "SELECT COALESCE(SUM(qty),0),COALESCE(SUM(notional),0) "
            "FROM fill_deltas WHERE client_id=?",
            (row["client_id"],),
        ).fetchone()
        if total[0] != row["filled_qty"] or abs(total[1] - row["filled_notional"]) > 1e-6:
            raise ValueError("cumulative broker fills differ from durable delta ledger")
        return row

    def all(self):
        return [
            self._verified(row) for row in self.db.execute("SELECT * FROM intents ORDER BY rowid")
        ]

    def get(self, client_id):
        return self._verified(
            self.db.execute("SELECT * FROM intents WHERE client_id=?", (client_id,)).fetchone()
        )

    def active(self):
        return [
            row
            for row in self.all()
            if row["status"] not in TERMINAL | {"risk_rejected"}
            or (row["filled_qty"] > 0 and not row["position_confirmed"])
        ]

    def confirm_position(self, client_id):
        with self.db:
            self.db.execute(
                "UPDATE intents SET position_confirmed=1 WHERE client_id=?", (client_id,)
            )

    def discard_unsent(self, client_id):
        row = self.get(client_id)
        if row is None or row["status"] != "intent" or row["broker_id"] is not None:
            raise ValueError("only a never-submitted intent can be discarded")
        with self.db:
            self.db.execute(
                "UPDATE intents SET status='risk_rejected' WHERE client_id=?", (client_id,)
            )

    def transition(self, client_id, status):
        row = self.get(client_id)
        if row is None or row["status"] in TERMINAL | {"risk_rejected"}:
            raise ValueError("terminal or missing intent cannot change local state")
        if status not in {"submitting", "unknown", "risk_rejected"} or (
            status in {"submitting", "risk_rejected"} and row["status"] != "intent"
        ):
            raise ValueError("ambiguous or submitted intent cannot return to submission")
        with self.db:
            self.db.execute("UPDATE intents SET status=? WHERE client_id=?", (status, client_id))

    def confirm(self, order, *, confirmed_at):
        if order.filled_at and parse_timestamp(order.filled_at) > parse_timestamp(confirmed_at):
            raise ValueError("broker fill timestamp is in the future")
        row = self.get(order.client_id)
        if row is None:
            raise ValueError("broker confirmation has no durable intent")
        intent = json.loads(row["payload"])
        if (order.symbol, order.side, order.qty, order.limit_price) != (
            intent["symbol"],
            intent["side"],
            intent["qty"],
            intent["limit_price"],
        ):
            raise ValueError("broker order differs from durable intent")
        if row["broker_id"] and row["broker_id"] != order.broker_id:
            raise ValueError("client ID identifies multiple broker orders")
        if order.filled_qty and (
            (order.side == "buy" and order.filled_avg_price > intent["limit_price"] + 1e-6)
            or (order.side == "sell" and order.filled_avg_price < intent["limit_price"] - 1e-6)
        ):
            raise ValueError("broker cumulative execution violates the order limit")
        qty = order.filled_qty
        notional = qty * (order.filled_avg_price or 0)
        if qty < row["filled_qty"] or notional < row["filled_notional"] - 1e-6:
            raise ValueError("broker cumulative fill regressed")
        if qty == row["filled_qty"] and abs(notional - row["filled_notional"]) > 1e-6:
            raise ValueError("broker fill price correction requires review")
        if row["status"] in TERMINAL and order.status != row["status"]:
            raise ValueError("broker terminal state changed")
        delta_qty, delta_notional = qty - row["filled_qty"], notional - row["filled_notional"]
        if delta_qty and delta_notional <= 0:
            raise ValueError("invalid cumulative fill delta")
        with self.db:
            if delta_qty:
                key = content_hash({"broker_order_id": order.broker_id, "filled_qty": qty})
                self.db.execute(
                    "INSERT INTO fill_deltas VALUES(?,?,?,?,?)",
                    (key, order.client_id, delta_qty, delta_notional, confirmed_at),
                )
            self.db.execute(
                """UPDATE intents SET status=?,broker_id=?,filled_qty=?,
                              filled_notional=?,position_confirmed=? WHERE client_id=?""",
                (
                    order.status,
                    order.broker_id,
                    qty,
                    notional,
                    0 if delta_qty else row["position_confirmed"],
                    order.client_id,
                ),
            )
            self.db.execute(
                "INSERT INTO events(kind,payload) VALUES(?,?)",
                ("broker_confirmation", canonical_json(asdict(order))),
            )
