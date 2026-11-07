"""One-shot delayed lessons from reconciled paper-ledger FIFO closing lots."""

from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

from agent.contracts import canonical_json, content_hash, normalize_timestamp, parse_timestamp
from agent.policy.evidence import visible_evidence
from agent.reflection import ClosedTradeOutcome, ReflectionResult, generate_reflection

PAPER_FEE_ASSUMPTION = (
    "Alpaca paper simulator omits regulatory fees and dividends; matched closing lots use fees=0. "
    "This is not a claim about real-money trading costs."
)


def _entry_context(outbox, payload) -> dict[str, Any]:
    """Use only the immutable buy-time research, never newly fetched market text."""
    exists = outbox.db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='research_cycles'"
    ).fetchone()
    if not exists:
        return {"source_status": "entry_research_archive_unavailable"}
    row = outbox.db.execute(
        "SELECT payload,payload_hash FROM research_cycles WHERE cycle_id=?",
        (payload.get("cycle_id"),),
    ).fetchone()
    if row is None:
        return {"source_status": "entry_research_archive_unavailable"}
    saved = json.loads(row["payload"])
    if content_hash(saved) != row["payload_hash"]:
        raise ValueError("entry research archive checksum failed")
    reason = next(
        (
            decision.get("reason", "")
            for decision in saved.get("decisions", ())
            if decision.get("symbol") == payload["symbol"] and decision.get("action") == "buy"
        ),
        "",
    )
    response = saved.get("response", {})
    messages = response.get("messages", ())
    context = []
    if messages:
        visible = visible_evidence(messages, expected_as_of=saved["decision_at"])
        for item in visible.values():
            if item.get("symbol") in (None, payload["symbol"]) and item.get("source") in {
                "alpaca.news",
                "alpha_vantage.news",
                "sec.filing_text",
            }:
                context.append({"source": item["source"], "text": item["text"][:180]})
                if len(context) == 2:
                    break
    return {
        "entry_reason": str(reason)[:160],
        "entry_language_context": context,
        "source_status": "recorded_entry_research",
    }


def _closed_lots(outbox, *, as_of: str) -> list[tuple[ClosedTradeOutcome, str]]:
    initial = outbox.db.execute(
        "SELECT value FROM metadata WHERE key='initial_positions'"
    ).fetchone()
    identity = outbox.db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
    if initial is None or identity is None:
        raise ValueError("reflection requires a bound reconciled ledger and initial holdings")
    positions = json.loads(initial["value"])
    if not isinstance(positions, dict) or any(
        not isinstance(symbol, str) or type(qty) is not int or qty < 0
        for symbol, qty in positions.items()
    ):
        raise ValueError("initial ledger holdings are invalid")
    lots = defaultdict(deque)
    for symbol, qty in positions.items():
        if qty:
            lots[symbol].append({"qty": qty, "id": None})
    outcomes = []
    rows = outbox.db.execute("""
        SELECT d.id,d.qty,d.notional,d.confirmed_at,i.payload,i.position_confirmed
        FROM fill_deltas d JOIN intents i ON i.client_id=d.client_id ORDER BY d.rowid
    """)
    for row in rows:
        at = normalize_timestamp(row["confirmed_at"])
        if parse_timestamp(at) > parse_timestamp(as_of):
            continue
        if not row["position_confirmed"]:
            raise ValueError("reflection ledger contains fills awaiting position reconciliation")
        payload = json.loads(row["payload"])
        symbol, side = payload["symbol"], payload["side"]
        qty, amount = row["qty"], row["notional"]
        if (
            type(qty) is not int
            or qty <= 0
            or isinstance(amount, bool)
            or not isinstance(amount, (int, float))
            or not math.isfinite(amount)
            or amount <= 0
        ):
            raise ValueError("invalid confirmed ledger delta")
        unit = amount / qty
        if side == "buy":
            lots[symbol].append(
                {
                    "qty": qty,
                    "original_qty": qty,
                    "id": row["id"],
                    "unit": unit,
                    "at": at,
                    "payload": payload,
                }
            )
            continue
        if side != "sell":
            raise ValueError("reflection ledger is long-only")
        remaining = qty
        while remaining:
            if not lots[symbol]:
                raise ValueError("sell delta exceeds attributable and initial ledger holdings")
            entry = lots[symbol][0]
            matched = min(entry["qty"], remaining)
            if entry["id"] is not None:
                trade_id = (
                    "paper-closed:"
                    + content_hash(
                        {
                            "ledger_identity": json.loads(identity["value"]),
                            "entry_delta": entry["id"],
                            "exit_delta": row["id"],
                            "entry_offset": entry["original_qty"] - entry["qty"],
                            "exit_offset": qty - remaining,
                            "quantity": matched,
                        }
                    )[:40]
                )
                # These identify cumulative broker-confirmation deltas, not
                # exchange execution IDs. Receipt timestamps are conservative.
                outcome = ClosedTradeOutcome(
                    trade_id=trade_id,
                    symbol=symbol,
                    quantity=matched,
                    opened_at=entry["at"],
                    closed_at=at,
                    confirmed_at=at,
                    entry_fill_ids=("broker-cumulative:" + entry["id"],),
                    exit_fill_ids=("broker-cumulative:" + row["id"],),
                    entry_notional=entry["unit"] * matched,
                    exit_notional=unit * matched,
                    fees=0,
                    source="alpaca.paper",
                )
                situation = canonical_json(
                    {
                        "symbol": symbol,
                        "action": "sell",
                        "matched_quantity": matched,
                        "entry_cycle": str(entry["payload"].get("cycle_id", ""))[:100],
                        "exit_cycle": str(payload.get("cycle_id", ""))[:100],
                        **_entry_context(outbox, entry["payload"]),
                        "fill_identity_kind": (
                            "cumulative broker confirmation delta, not exchange fill ID"
                        ),
                        "fee_assumption": "paper simulator fees=0; no dividend attribution",
                    }
                )
                # Evidence IDs and cycle IDs are bounded trusted intent fields.
                # They identify the original research without revealing provider
                # content or inventing a situation from future market movement.
                if len(situation) > 1200:
                    situation = canonical_json(
                        {
                            "symbol": symbol,
                            "action": "sell",
                            "matched_quantity": matched,
                            "entry_reason": _entry_context(outbox, entry["payload"]).get(
                                "entry_reason", "unavailable"
                            ),
                            "fee_assumption": "paper simulator fees=0",
                        }
                    )
                outcomes.append((outcome, situation))
            entry["qty"] -= matched
            remaining -= matched
            if not entry["qty"]:
                lots[symbol].popleft()
    return outcomes


def process_confirmed_reflections(
    outbox,
    memory,
    backend,
    *,
    as_of: str,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    max_reflections: int = 2,
) -> dict[str, Any]:
    """Queue matured matched lots and generate each lesson at most once.

    Attempt state commits before Qwen is called. A crashed/uncertain generation
    is never retried automatically. A saved successful response can safely retry
    only its idempotent memory write, without changing the lesson or any trade.
    This function never submits or modifies a broker order.
    """
    cutoff = normalize_timestamp(as_of)
    if memory.fold != "paper":
        raise ValueError("broker-confirmed lessons require the paper memory namespace")
    if type(max_reflections) is not int or not 1 <= max_reflections <= 20:
        raise ValueError("reflection cycle budget must be 1..20")
    if parse_timestamp(cutoff) > parse_timestamp(clock()):
        raise ValueError("paper reflection cutoff cannot be in the future")
    summary = {
        "queued": 0,
        "generated": 0,
        "stored": 0,
        "failed": 0,
        "uncertain": 0,
        "memory_pending": 0,
        "fee_assumption": PAPER_FEE_ASSUMPTION,
    }
    with outbox.lock():
        outbox.all()  # Verify intent and cumulative delta integrity before producing a lesson.
        if outbox.db.execute(
            "SELECT 1 FROM intents WHERE status IN ('submitting','unknown') "
            "OR (filled_qty>0 AND position_confirmed=0) LIMIT 1"
        ).fetchone():
            return {**summary, "status": "awaiting_reconciliation"}
        outcomes = _closed_lots(outbox, as_of=cutoff)
        ledger_identity = json.loads(
            outbox.db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()["value"]
        )
        if memory.run_id != ledger_identity["run_id"]:
            raise ValueError("paper reflection memory and ledger run scope differ")
        expected_payloads = {
            item.trade_id: canonical_json({"outcome": asdict(item), "situation": situation})
            for item, situation in outcomes
        }
        with outbox.db:
            outbox.db.execute("""CREATE TABLE IF NOT EXISTS reflection_attempts(
                outcome_id TEXT PRIMARY KEY, outcome_hash TEXT NOT NULL, payload TEXT NOT NULL,
                status TEXT NOT NULL, attempted_at TEXT, result TEXT, result_hash TEXT,
                memory_id TEXT, error TEXT)
            """)
            for outcome, situation in outcomes:
                old = outbox.db.execute(
                    "SELECT outcome_hash FROM reflection_attempts WHERE outcome_id=?",
                    (outcome.trade_id,),
                ).fetchone()
                if old is not None and old["outcome_hash"] != outcome.outcome_hash:
                    raise ValueError("confirmed outcome changed after reflection was queued")
                outbox.db.execute(
                    "INSERT OR IGNORE INTO reflection_attempts"
                    "(outcome_id,outcome_hash,payload,status) "
                    "VALUES(?,?,?,'queued')",
                    (
                        outcome.trade_id,
                        outcome.outcome_hash,
                        canonical_json({"outcome": asdict(outcome), "situation": situation}),
                    ),
                )
            uncertain = outbox.db.execute(
                "UPDATE reflection_attempts SET status='uncertain',"
                "error='previous_generation_uncertain' "
                "WHERE status='attempting'"
            ).rowcount
            summary["uncertain"] = uncertain
        rows = list(
            outbox.db.execute(
                "SELECT * FROM reflection_attempts WHERE status IN ('queued','generated') "
                "ORDER BY rowid"
            )
        )
        summary["queued"] = sum(row["status"] == "queued" for row in rows)
        attempts = 0
        for row in rows:
            payload = json.loads(row["payload"])
            outcome = ClosedTradeOutcome(**payload["outcome"])
            if parse_timestamp(outcome.confirmed_at) > parse_timestamp(cutoff):
                continue
            if (
                outcome.outcome_hash != row["outcome_hash"]
                or expected_payloads.get(outcome.trade_id) != row["payload"]
            ):
                raise ValueError("queued reflection no longer matches the confirmed ledger")
            if row["status"] == "queued":
                if attempts >= max_reflections:
                    continue
                attempts += 1
                with outbox.db:
                    outbox.db.execute(
                        "UPDATE reflection_attempts SET status='attempting',attempted_at=? "
                        "WHERE outcome_id=?",
                        (normalize_timestamp(clock()), outcome.trade_id),
                    )
                try:
                    result = generate_reflection(
                        backend,
                        outcome=outcome,
                        situation=payload["situation"],
                        as_of=cutoff,
                        clock=clock,
                    )
                except Exception:
                    result = None
                if result is None or result.status != "success":
                    with outbox.db:
                        outbox.db.execute(
                            "UPDATE reflection_attempts SET status='failed',"
                            "error='generation_failed' "
                            "WHERE outcome_id=?",
                            (outcome.trade_id,),
                        )
                    summary["failed"] += 1
                    continue
                with outbox.db:
                    outbox.db.execute(
                        "UPDATE reflection_attempts SET status='generated',result=?,result_hash=? "
                        "WHERE outcome_id=?",
                        (
                            canonical_json(asdict(result)),
                            content_hash(asdict(result)),
                            outcome.trade_id,
                        ),
                    )
                summary["generated"] += 1
            else:
                recorded_result = json.loads(row["result"])
                if content_hash(recorded_result) != row["result_hash"]:
                    raise ValueError("saved reflection result checksum failed")
                result = ReflectionResult(**recorded_result)
            try:
                memory_id = memory.add_memory(
                    payload["situation"], result.lesson, outcome=outcome, result=result
                )
            except Exception:
                with outbox.db:
                    outbox.db.execute(
                        "UPDATE reflection_attempts SET error='memory_persistence_failed' "
                        "WHERE outcome_id=?",
                        (outcome.trade_id,),
                    )
                summary["memory_pending"] += 1
                continue
            with outbox.db:
                outbox.db.execute(
                    "UPDATE reflection_attempts SET status='stored',memory_id=?,error=NULL "
                    "WHERE outcome_id=?",
                    (memory_id, outcome.trade_id),
                )
            summary["stored"] += 1
    return {**summary, "status": "complete"}
