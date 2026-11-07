"""Fixed paper-only REST transport; POST is attempted exactly once."""

import json
import os
import re
from datetime import UTC, datetime
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener

from agent.contracts import normalize_timestamp

from .contracts import BrokerOrder, BrokerSnapshot, Quote, amount, shares

PAPER_URL = "https://paper-api.alpaca.markets"
DATA_URL = "https://data.alpaca.markets"


class BrokerUnavailable(RuntimeError):
    pass


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Never forward paper credentials to any redirected endpoint.


class AlpacaPaperBroker:
    def __init__(self, api_key, secret_key, *, transport=None, clock=None):
        if not api_key or not secret_key:
            raise ValueError("paper Alpaca credentials are required")
        self._headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret_key}
        self._transport = transport or self._request
        self.clock = clock or (lambda: datetime.now(UTC))

    @classmethod
    def from_environment(cls):
        return cls(os.environ.get("ALPACA_API_KEY", ""), os.environ.get("ALPACA_SECRET_KEY", ""))

    def _request(self, method, url, payload=None):
        data = json.dumps(payload).encode() if payload is not None else None
        request = Request(
            url,
            data=data,
            method=method,
            headers={**self._headers, "Content-Type": "application/json"},
        )
        try:
            # No SDK retry layer: even a 429/504 POST is ambiguous until reconciled.
            with build_opener(_NoRedirect()).open(request, timeout=30) as response:
                return json.loads(response.read())
        except HTTPError as exc:
            if method == "GET" and exc.code == 404:
                return None
            raise BrokerUnavailable("paper broker request failed") from None
        except Exception:
            raise BrokerUnavailable("paper broker request failed") from None

    def _get(self, path, params=None, *, data=False):
        url = (DATA_URL if data else PAPER_URL) + path
        if params:
            url += "?" + urlencode(params)
        try:
            return self._transport("GET", url, None)
        except Exception:
            raise BrokerUnavailable("paper broker read unavailable") from None

    @staticmethod
    def _order(row):
        if not isinstance(row, dict):
            raise BrokerUnavailable("invalid broker order response")
        return BrokerOrder(
            row["id"],
            row["client_order_id"],
            row["symbol"],
            row["side"],
            shares(row["qty"]),
            shares(row["filled_qty"]),
            amount(row["filled_avg_price"], positive=True) if row.get("filled_avg_price") else None,
            row["status"],
            amount(row["limit_price"], positive=True) if row.get("limit_price") else None,
            row.get("filled_at"),
            row["time_in_force"],
            row["type"],
            row["extended_hours"],
        )

    def lookup(self, client_id):
        row = self._get("/v2/orders:by_client_order_id", {"client_order_id": client_id})
        order = None if row is None else self._order(row)
        if order is not None and order.client_id != client_id:
            raise BrokerUnavailable("broker lookup returned another client order")
        return order

    def snapshot(self):
        started = self.clock()
        account = self._get("/v2/account")
        positions = self._get("/v2/positions")
        clock = self._get("/v2/clock")
        orders = self._get("/v2/orders", {"status": "open", "limit": 500, "direction": "asc"})
        if not isinstance(orders, list) or len(orders) >= 500:
            raise BrokerUnavailable("open order list may be incomplete")
        if not isinstance(positions, list) or not isinstance(account, dict):
            raise BrokerUnavailable("invalid broker snapshot")
        holdings = {}
        for row in positions:
            if row.get("asset_class") != "us_equity" or row.get("side") != "long":
                raise BrokerUnavailable("unsupported broker position")
            if row["symbol"] in holdings:
                raise BrokerUnavailable("duplicate broker position")
            holdings[row["symbol"]] = shares(row["qty"])
        return BrokerSnapshot(
            account["id"],
            amount(account["cash"]),
            amount(account["equity"], positive=True),
            amount(account["buying_power"]),
            holdings,
            tuple(self._order(row) for row in orders),
            clock["is_open"] is True,
            normalize_timestamp(started),
            bool(
                account.get("trading_blocked")
                or account.get("account_blocked")
                or account.get("trade_suspended_by_user")
                or account.get("currency", "USD") != "USD"
                or account.get("status") != "ACTIVE"
            ),
            clock.get("next_close"),
        )

    @staticmethod
    def _symbol(symbol):
        if not isinstance(symbol, str) or not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,14}", symbol):
            raise ValueError("unsupported asset symbol")
        return symbol

    def quote(self, symbol):
        symbol = self._symbol(symbol)
        row = self._get(f"/v2/stocks/{symbol}/quotes/latest", {"feed": "iex"}, data=True)
        q = row["quote"]
        if row.get("symbol") != symbol:
            raise BrokerUnavailable("quote symbol mismatch")
        return Quote(symbol, amount(q["bp"], positive=True), amount(q["ap"], positive=True), q["t"])

    def quotes(self, symbols):
        symbols = tuple(self._symbol(s) for s in symbols)
        if not symbols or len(symbols) > 100 or len(set(symbols)) != len(symbols):
            raise ValueError("execution quote batch must contain 1..100 unique symbols")
        row = self._get(
            "/v2/stocks/quotes/latest", {"symbols": ",".join(symbols), "feed": "iex"}, data=True
        )
        records = row.get("quotes", {})
        if set(records) != set(symbols):
            raise BrokerUnavailable("execution quote batch is incomplete")
        return {
            symbol: Quote(
                symbol, amount(q["bp"], positive=True), amount(q["ap"], positive=True), q["t"]
            )
            for symbol, q in records.items()
        }

    def tradable(self, symbol):
        symbol = self._symbol(symbol)
        row = self._get("/v2/assets/" + symbol)
        return bool(
            row
            and row.get("symbol") == symbol
            and row.get("class", row.get("asset_class")) == "us_equity"
            and row.get("status") == "active"
            and row.get("tradable") is True
        )

    def submit(self, *, client_id, symbol, side, qty, limit_price):
        symbol = self._symbol(symbol)
        if side not in {"buy", "sell"}:
            raise ValueError("invalid order side")
        amount(limit_price, positive=True)
        payload = {
            "client_order_id": client_id,
            "symbol": symbol,
            "side": side,
            "qty": str(shares(qty, zero=False)),
            "type": "limit",
            "limit_price": format(limit_price, ".2f"),
            "time_in_force": "ioc",
            "extended_hours": False,
        }
        try:
            return self._order(self._transport("POST", PAPER_URL + "/v2/orders", payload))
        except Exception:
            raise BrokerUnavailable("paper submission outcome requires reconciliation") from None
