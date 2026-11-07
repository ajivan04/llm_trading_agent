"""Independent risk, reservation and reconciliation before any paper submission."""

import json
from dataclasses import replace
from datetime import UTC, datetime
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal

from agent.contracts import Decision, parse_timestamp

from .contracts import ACTIVE, TERMINAL


class ExecutionPaused(RuntimeError):
    pass


class PaperExecutor:
    def __init__(
        self,
        broker,
        outbox,
        config,
        *,
        run_id,
        policy_identity,
        clock=None,
        max_quote_age=60,
        max_snapshot_age=30,
    ):
        if not run_id or not config.paper_trading_only:
            raise ValueError("a paper-only run identity is required")
        self.broker, self.outbox, self.config = broker, outbox, config
        self.run_id, self.policy_identity = run_id, policy_identity
        self.clock = clock or (lambda: datetime.now(UTC))
        if max_quote_age <= 0 or max_snapshot_age <= 0:
            raise ValueError("freshness bounds must be positive")
        self.max_quote_age, self.max_snapshot_age = max_quote_age, max_snapshot_age

    def _reconcile(self):
        snapshot = self.broker.snapshot()
        self.outbox.bind(
            account_id=snapshot.account_id,
            run_id=self.run_id,
            config_hash=self.config.config_hash,
            policy_identity=self.policy_identity,
        )
        self.outbox.cash_budget(snapshot.cash)
        with self.outbox.db:
            self.outbox.db.execute(
                "INSERT OR IGNORE INTO metadata VALUES('initial_positions',?)",
                (json.dumps(dict(snapshot.positions)),),
            )
        reconciled = []
        for row in self.outbox.active():
            if row["status"] == "intent":
                continue  # Persisted, never POSTed: still independently recheck risk.
            order = self.broker.lookup(row["client_id"])
            if order is None:
                # 404 after an ambiguous POST does NOT authorize another POST.
                if row["status"] in TERMINAL:
                    raise ExecutionPaused("confirmed order lookup unavailable during settlement")
                self.outbox.transition(row["client_id"], "unknown")
            else:
                if order.status not in ACTIVE | TERMINAL:
                    raise ExecutionPaused("unsupported or unresolved broker order state")
                self.outbox.confirm(order, confirmed_at=self.clock().isoformat())
                reconciled.append((json.loads(row["payload"]), order))
        # Account/position reads follow order confirmations, never predate them.
        snapshot = self.broker.snapshot()
        self.outbox.bind(
            account_id=snapshot.account_id,
            run_id=self.run_id,
            config_hash=self.config.config_hash,
            policy_identity=self.policy_identity,
        )
        for payload, order in reconciled:
            expected = payload["position_before"] + order.filled_qty * (
                1 if order.side == "buy" else -1
            )
            if snapshot.positions.get(order.symbol, 0) != expected:
                raise ExecutionPaused("broker position has not reconciled with confirmed fills")
            self.outbox.confirm_position(order.client_id)
        if any(row["status"] in {"submitting", "unknown"} for row in self.outbox.active()):
            raise ExecutionPaused("unresolved submission; reconcile by client order ID")
        known = {
            row["client_id"] for row in self.outbox.db.execute("SELECT client_id FROM intents")
        }
        if any(order.client_id not in known for order in snapshot.open_orders):
            raise ExecutionPaused("external open orders require reconciliation")
        if set(snapshot.positions) - set(self.config.universe):
            raise ExecutionPaused("broker holdings outside configured universe")
        for order in snapshot.open_orders:
            if order.status not in ACTIVE:
                raise ExecutionPaused("unsupported or unresolved broker order state")
            row = self.outbox.get(order.client_id)
            if row is None or order.filled_qty != row["filled_qty"] or order.status in TERMINAL:
                raise ExecutionPaused("broker orders changed during snapshot; reconcile again")
            self.outbox.confirm(order, confirmed_at=self.clock().isoformat())
            if row is None or order.status in TERMINAL:
                raise ExecutionPaused("inconsistent open order snapshot")
        # A known active order absent from open-order list needs a definitive lookup.
        open_ids = {order.client_id for order in snapshot.open_orders}
        if any(
            row["status"] != "intent" and row["client_id"] not in open_ids
            for row in self.outbox.active()
        ):
            raise ExecutionPaused("broker order/account snapshots are not coherent yet")
        # All position/cash changes must reconcile with this run's confirmed fills.
        # Splits, external fills and account resets need explicit operator review;
        # they must not silently corrupt FIFO cost basis or inflate risk equity.
        initial = json.loads(
            self.outbox.db.execute(
                "SELECT value FROM metadata WHERE key='initial_positions'"
            ).fetchone()["value"]
        )
        expected_positions = dict(initial)
        expected_cash = float(
            self.outbox.db.execute("SELECT value FROM metadata WHERE key='cash_anchor'").fetchone()[
                "value"
            ]
        )
        ledger = self.outbox.all()
        for row in ledger:
            intent = json.loads(row["payload"])
            sign = 1 if intent["side"] == "buy" else -1
            expected_positions[intent["symbol"]] = (
                expected_positions.get(intent["symbol"], 0) + sign * row["filled_qty"]
            )
            expected_cash -= sign * row["filled_notional"]
        expected_positions = {s: q for s, q in expected_positions.items() if q}
        actual_positions = {s: q for s, q in snapshot.positions.items() if q}
        if expected_positions != actual_positions:
            raise ExecutionPaused(
                "broker positions differ from confirmed ledger; review external actions"
            )
        if abs(expected_cash - snapshot.cash) > max(0.02, len(ledger) * 0.01):
            raise ExecutionPaused("broker cash differs from confirmed ledger; await reconciliation")
        age = (self.clock() - parse_timestamp(snapshot.observed_at)).total_seconds()
        if not 0 <= age <= self.max_snapshot_age or snapshot.blocked:
            raise ExecutionPaused("stale or blocked broker account")
        self.outbox.peak(snapshot.equity)
        return snapshot

    def reconcile(self):
        with self.outbox.lock():
            return self._reconcile()

    def _quote(self, symbol, quote=None):
        quote = self.broker.quote(symbol) if quote is None else quote
        age = (self.clock() - parse_timestamp(quote.observed_at)).total_seconds()
        if quote.symbol != symbol or quote.feed != "iex" or not 0 <= age <= self.max_quote_age:
            raise ExecutionPaused("stale, future or unsupported-feed execution quote")
        return quote

    def _risk(self, decision, snapshot, allowed_evidence, dry_pending=()):
        from trading_agent import Portfolio, RiskEngine

        active = self.outbox.active()
        symbols = set(snapshot.positions) | {decision.symbol}
        payloads = [(row, json.loads(row["payload"])) for row in active]
        payloads.extend(({"filled_qty": 0}, payload) for payload in dry_pending)
        symbols.update(p["symbol"] for _, p in payloads)
        if hasattr(self.broker, "quotes"):
            batch = self.broker.quotes(tuple(sorted(symbols)))
            if set(batch) != symbols:
                raise ExecutionPaused("execution quote batch is incomplete")
            quotes = {symbol: self._quote(symbol, batch[symbol]) for symbol in symbols}
        else:
            quotes = {symbol: self._quote(symbol) for symbol in symbols}
        prices = {symbol: quote.ask for symbol, quote in quotes.items()}
        positions, cash = dict(snapshot.positions), self.outbox.cash_budget(snapshot.cash)
        reserved_sells = {}
        for row, payload in payloads:
            remaining = payload["qty"] - row["filled_qty"]
            if payload["side"] == "buy":
                cash -= remaining * payload["limit_price"]
                positions[payload["symbol"]] = positions.get(payload["symbol"], 0) + remaining
            else:
                reserved_sells[payload["symbol"]] = (
                    reserved_sells.get(payload["symbol"], 0) + remaining
                )
        if decision.action == "sell" and decision.qty > (
            snapshot.positions.get(decision.symbol, 0) - reserved_sells.get(decision.symbol, 0)
        ):
            return "reserved_or_insufficient_holdings", None
        quote = quotes[decision.symbol]
        raw = quote.ask if decision.action == "buy" else quote.bid
        factor = Decimal(1) + (
            Decimal(str(self.config.slippage_bps))
            / Decimal(10000)
            * (1 if decision.action == "buy" else -1)
        )
        limit = float(
            (Decimal(str(raw)) * factor).quantize(
                Decimal(".01"), rounding=ROUND_CEILING if decision.action == "buy" else ROUND_FLOOR
            )
        )
        if limit <= 0:
            return "invalid_limit", None
        prices[decision.symbol] = limit

        # Pending sales never fund new purchases or release risk exposure early.
        class BrokerEquityPortfolio(Portfolio):
            def equity(self, _prices):
                return snapshot.equity  # Never recognize gains on hypothetical pending fills.

        portfolio = BrokerEquityPortfolio(
            min(cash, snapshot.buying_power), positions, self.outbox.peak(snapshot.equity)
        )
        # The executable limit already includes the configured price buffer.
        reason = RiskEngine(replace(self.config, slippage_bps=0)).check(
            decision, portfolio, prices, set(allowed_evidence)
        )
        self._checked_quotes = tuple(quotes.values())
        self._fresh(snapshot)
        return reason, limit

    def _fresh(self, snapshot):
        now = self.clock()
        if (
            not snapshot.next_close
            or (parse_timestamp(snapshot.next_close) - now).total_seconds() <= 5
        ):
            raise ExecutionPaused("market close boundary; do not queue an overnight signal")
        if (
            not 0
            <= (now - parse_timestamp(snapshot.observed_at)).total_seconds()
            <= self.max_snapshot_age
        ):
            raise ExecutionPaused("broker snapshot expired during execution checks")
        for quote in self._checked_quotes:
            if (
                not 0
                <= (now - parse_timestamp(quote.observed_at)).total_seconds()
                <= self.max_quote_age
            ):
                raise ExecutionPaused("execution quote expired during execution checks")

    def execute(self, decisions, *, cycle_id, allowed_evidence_ids, submit=False):
        if not cycle_id or type(submit) is not bool:
            raise ValueError("stable cycle ID and explicit submit flag required")
        decisions = tuple(decisions)
        if not all(isinstance(d, Decision) for d in decisions):
            raise TypeError("execution requires independently validated decisions")
        if len({d.symbol for d in decisions}) != len(decisions):
            raise ValueError("duplicate decision symbols")
        results, dry_pending = [], []
        with self.outbox.lock():
            snapshot = self._reconcile()
            if not snapshot.market_open:
                raise ExecutionPaused("market is closed")
            for decision in decisions:
                if decision.symbol not in self.config.universe:
                    results.append(
                        {
                            "symbol": decision.symbol,
                            "status": "risk_rejected",
                            "reason": "outside_universe",
                        }
                    )
                    continue
                if decision.action == "hold":
                    results.append({"symbol": decision.symbol, "status": "hold"})
                    continue
                existing = self.outbox.cycle_order(self.run_id, cycle_id, decision.symbol)
                if existing is not None:
                    results.append(
                        {
                            "symbol": decision.symbol,
                            "status": "already_recorded",
                            "client_id": existing["client_id"],
                        }
                    )
                    continue
                # Refresh per proposal; previously submitted orders reserve exposure.
                snapshot = self._reconcile()
                if not snapshot.market_open or not self.broker.tradable(decision.symbol):
                    raise ExecutionPaused("market closed or asset not tradable")
                if any(
                    json.loads(row["payload"])["symbol"] == decision.symbol
                    for row in self.outbox.active()
                ):
                    results.append({"symbol": decision.symbol, "status": "pending_order"})
                    continue
                reason, limit = self._risk(decision, snapshot, allowed_evidence_ids, dry_pending)
                if reason:
                    results.append(
                        {"symbol": decision.symbol, "status": "risk_rejected", "reason": reason}
                    )
                    continue
                payload = {
                    "run_id": self.run_id,
                    "cycle_id": cycle_id,
                    "symbol": decision.symbol,
                    "side": decision.action,
                    "qty": decision.qty,
                    "limit_price": limit,
                    "evidence_ids": decision.evidence_ids,
                    "position_before": snapshot.positions.get(decision.symbol, 0),
                }
                if not submit:
                    results.append(
                        {"symbol": decision.symbol, "status": "dry_run", "intent": payload}
                    )
                    dry_pending.append(payload)
                    continue
                client_id = self.outbox.put(payload)
                row = self.outbox.get(client_id)
                if row["status"] != "intent":
                    results.append(
                        {
                            "symbol": decision.symbol,
                            "status": "already_recorded",
                            "client_id": client_id,
                        }
                    )
                    continue
                self._fresh(snapshot)
                self.outbox.transition(client_id, "submitting")  # Durable before the one POST.
                try:
                    order = self.broker.submit(
                        client_id=client_id,
                        symbol=decision.symbol,
                        side=decision.action,
                        qty=decision.qty,
                        limit_price=limit,
                    )
                    if order.status not in ACTIVE | TERMINAL:
                        raise ExecutionPaused("unsupported submission response state")
                    self.outbox.confirm(order, confirmed_at=self.clock().isoformat())
                except Exception:
                    self.outbox.transition(client_id, "unknown")
                    self.outbox.record("submission_uncertain", {"client_id": client_id})
                    raise ExecutionPaused("submission uncertain; no blind retry") from None
                results.append(
                    {"symbol": decision.symbol, "status": order.status, "client_id": client_id}
                )
            self.outbox.record(
                "execution_cycle", {"cycle_id": cycle_id, "submitted": submit, "results": results}
            )
        return results
