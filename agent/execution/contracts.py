"""Broker-authoritative, long-only paper execution snapshots."""

import math
from collections.abc import Mapping
from dataclasses import dataclass

from agent.contracts import normalize_timestamp

TERMINAL = frozenset({"filled", "canceled", "expired", "rejected"})
ACTIVE = frozenset({"new", "accepted", "partially_filled", "pending_new", "accepted_for_bidding"})


def amount(value, *, positive=False):
    if isinstance(value, bool):
        raise ValueError("invalid broker amount")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result <= 0):
        raise ValueError("invalid broker amount")
    return result


def shares(value, *, zero=True):
    result = amount(value, positive=not zero)
    if not result.is_integer():
        raise ValueError("fractional/short positions and orders are unsupported")
    return int(result)


@dataclass(frozen=True)
class BrokerOrder:
    broker_id: str
    client_id: str
    symbol: str
    side: str
    qty: int
    filled_qty: int
    filled_avg_price: float | None
    status: str
    limit_price: float | None = None
    filled_at: str | None = None
    time_in_force: str = "ioc"
    order_type: str = "limit"
    extended_hours: bool = False

    def __post_init__(self):
        if (
            self.time_in_force != "ioc"
            or self.order_type != "limit"
            or self.extended_hours is not False
        ):
            raise ValueError("only regular-session IOC limit orders are supported")
        if not self.broker_id or not self.client_id or not self.symbol:
            raise ValueError("broker order identity missing")
        if self.side not in {"buy", "sell"} or shares(self.qty, zero=False) != self.qty:
            raise ValueError("invalid broker order")
        if shares(self.filled_qty) > self.qty:
            raise ValueError("broker filled quantity exceeds intent")
        if self.filled_qty and self.filled_avg_price is None:
            raise ValueError("filled order requires cumulative average price")
        if self.filled_avg_price is not None:
            amount(self.filled_avg_price, positive=True)
        if self.limit_price is not None:
            amount(self.limit_price, positive=True)
        if self.status == "filled" and self.filled_qty != self.qty:
            raise ValueError("filled status contradicts quantity")
        if self.filled_at is not None:
            normalize_timestamp(self.filled_at)


@dataclass(frozen=True)
class BrokerSnapshot:
    account_id: str
    cash: float
    equity: float
    buying_power: float
    positions: Mapping[str, int]
    open_orders: tuple[BrokerOrder, ...]
    market_open: bool
    observed_at: str
    blocked: bool = False
    next_close: str | None = None

    def __post_init__(self):
        if not self.account_id:
            raise ValueError("account identity missing")
        for value in (self.cash, self.buying_power):
            amount(value)
        amount(self.equity, positive=True)
        for value in self.positions.values():
            shares(value)
        normalize_timestamp(self.observed_at)


@dataclass(frozen=True)
class Quote:
    symbol: str
    bid: float
    ask: float
    observed_at: str
    feed: str = "iex"

    def __post_init__(self):
        amount(self.bid, positive=True)
        amount(self.ask, positive=True)
        if self.ask < self.bid:
            raise ValueError("crossed quote")
        normalize_timestamp(self.observed_at)
