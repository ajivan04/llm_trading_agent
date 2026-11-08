"""Actual base-policy trajectories, outcome labels, and GOOD-only SFT exports.

Replay execution data stays private until the policy returns. These are simulated
portfolio outcomes, never Alpaca fills or oracle-selected decisions.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agent.config import Config
from agent.contracts import Decision, Evidence, Fill, canonical_json, content_hash, parse_timestamp
from agent.data.prices import DailyPriceBar
from agent.datasets import DatasetArtifact, DatasetManifest, _audit_payload, load_dataset_artifact
from agent.policy.evidence import resolve_decisions, visible_evidence
from agent.training.core import (
    MODEL_NAME,
    MODEL_REVISION,
    EpisodeExample,
    validate_chronological_splits,
)

PROTOCOL_VERSION = "base-good-trajectories-v1"


@dataclass(frozen=True)
class TrajectorySettings:
    universe: tuple[str, ...]
    train_end: str
    validation_end: str
    test_end: str
    initial_cash: float = 100_000.0
    cost_bps: float = 5.0
    max_qty: int = 50
    min_tool_calls: int = 1
    history_sessions: int = 130
    label_threshold: float = 0.0
    label_rule: str = "net_equity_change"
    sector_map: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        universe = tuple(item.strip().upper() for item in self.universe)
        if not universe or len(set(universe)) != len(universe) or any(not s for s in universe):
            raise ValueError("trajectory universe must be nonempty and unique")
        object.__setattr__(self, "universe", universe)
        if not (
            parse_timestamp(self.train_end)
            < parse_timestamp(self.validation_end)
            < parse_timestamp(self.test_end)
        ):
            raise ValueError("trajectory split boundaries must be ordered")
        for name in ("initial_cash", "cost_bps", "label_threshold"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.initial_cash <= 0 or self.cost_bps >= 10_000:
            raise ValueError("invalid trajectory cash or costs")
        for name in ("max_qty", "history_sessions"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if type(self.min_tool_calls) is not int or self.min_tool_calls < 0:
            raise ValueError("min_tool_calls must be nonnegative")
        if self.label_rule != "net_equity_change":
            raise ValueError("assignment trajectory labels use net_equity_change")
        if self.sector_map and not set(universe) <= set(dict(self.sector_map)):
            raise ValueError("sector_map must cover the complete trajectory universe")


@dataclass(frozen=True)
class TrajectoryCollection:
    settings: TrajectorySettings
    split: str
    policy_identity: Mapping[str, Any]
    source_hash: str
    records: tuple[Mapping[str, Any], ...]
    planned_sessions: tuple[str, ...]
    skipped_windows: tuple[Mapping[str, str], ...] = ()

    @property
    def counts(self) -> dict[str, Any]:
        labels = Counter(str(row["label"]) for row in self.records)
        exclusions = Counter(
            str(row["exclusion_reason"]) for row in self.records if row.get("exclusion_reason")
        )
        raw_labels = Counter(str(row["portfolio_change_label"]) for row in self.records)
        return {
            "sessions": len(self.records),
            "labels": dict(labels),
            "raw_portfolio_labels": dict(raw_labels),
            "good_export_candidates": labels.get("good", 0),
            "exclusions": dict(exclusions),
            "fills": sum(len(row["fills"]) for row in self.records),
            "tool_calls": sum(row["tool_call_count"] for row in self.records),
        }


def _identity(value: Any, *, evaluation_only: bool = False) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("base collection requires actual backend policy_identity")
    identity = json.loads(canonical_json(value))
    if (
        identity.get("model_name") != MODEL_NAME
        or identity.get("model_revision") != MODEL_REVISION
        or (not evaluation_only and identity.get("adapter_hash") != "base-model")
        or (not evaluation_only and identity.get("collection_purpose") == "evaluation")
        or not isinstance(identity.get("adapter_hash"), str)
        or not identity.get("adapter_hash")
    ):
        raise ValueError(
            "assignment trajectories must come from the pinned base policy, without adapter"
        )
    if not isinstance(identity.get("protocol_hash"), str) or not identity["protocol_hash"]:
        raise ValueError("policy_identity requires a prompt/tool/decoding protocol_hash")
    if evaluation_only:
        identity["collection_purpose"] = "evaluation"
    _audit_payload(identity)
    return identity


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(canonical_json(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _read_journal_records(root: Path) -> list[dict[str, Any]]:
    records = []
    hashes = [None]
    previous_hash = None
    for expected, path in enumerate(sorted(root.glob("episode-*.json"))):
        if path.name != f"episode-{expected:06d}.json":
            raise ValueError("trajectory journal episodes are not a complete prefix")
        envelope = json.loads(path.read_text())
        if set(envelope) != {"record", "record_hash", "previous_record_hash"}:
            raise ValueError("trajectory journal has an invalid record envelope")
        if (
            envelope["previous_record_hash"] != previous_hash
            or content_hash(envelope["record"]) != envelope["record_hash"]
        ):
            raise ValueError("trajectory journal record hash mismatch")
        records.append(envelope["record"])
        previous_hash = envelope["record_hash"]
        hashes.append(previous_hash)
    commit_path = root / "collection-state.json"
    expected_marker = {
        "record_count": len(records),
        "record_hash": previous_hash,
    }
    marker = json.loads(commit_path.read_text()) if commit_path.is_file() else None
    if marker != expected_marker:
        # A crash may occur after the complete envelope rename, before marker rename.
        # Only the immediately preceding, hash-verified commit can be advanced.
        recoverable = len(records) > 0 and marker == {
            "record_count": len(records) - 1,
            "record_hash": hashes[-2],
        }
        if recoverable:
            _write_json(commit_path, expected_marker)
        else:
            raise ValueError(
                "trajectory journal commit marker is incomplete or differs from its records"
            )
    return records


@contextmanager
def _journal(root: Path | None, contract: Mapping[str, Any], resume: bool):
    if root is None:
        if resume:
            raise ValueError("resume requires a trajectory journal directory")
        yield []
        return
    import fcntl

    root.mkdir(parents=True, exist_ok=True)
    with (root / ".collector.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("trajectory journal is already being collected") from exc
        manifest = root / "collection-manifest.json"
        if manifest.exists():
            if not resume:
                raise ValueError(
                    "trajectory journal exists; resume explicitly or use a new directory"
                )
            if json.loads(manifest.read_text()) != json.loads(canonical_json(contract)):
                raise ValueError("trajectory resume contract changed")
        else:
            if resume:
                raise ValueError("trajectory journal has no collection manifest")
            if any(path.name != ".collector.lock" for path in root.iterdir()):
                raise ValueError("trajectory journal is not empty")
            _write_json(manifest, contract)
            _write_json(root / "collection-state.json", {"record_count": 0, "record_hash": None})
        records = _read_journal_records(root)
        yield records


def load_collection(journal: Path) -> TrajectoryCollection:
    """Load a checksum-verified completed prefix without loading a model."""
    contract = json.loads((journal / "collection-manifest.json").read_text())
    if contract.get("version") != PROTOCOL_VERSION:
        raise ValueError("unsupported trajectory collection protocol")
    payload = dict(contract["settings"])
    payload["universe"] = tuple(payload["universe"])
    payload["sector_map"] = tuple(tuple(row) for row in payload["sector_map"])
    settings = TrajectorySettings(**payload)
    identity = _identity(
        contract["policy_identity"], evaluation_only=contract.get("evaluation_only", False)
    )
    with _journal(journal, contract, True) as records:
        planned = tuple(contract["planned_sessions"])
        if len(records) > len(planned) or any(
            row["session"] != planned[index] or row["split"] != contract["split"]
            for index, row in enumerate(records)
        ):
            raise ValueError("trajectory journal records differ from the collection plan")
        return TrajectoryCollection(
            settings,
            contract["split"],
            identity,
            contract["source_hash"],
            tuple(records),
            planned,
            tuple(contract["skipped_windows"]),
        )


def _prompt_check(messages: Sequence[Mapping[str, Any]]) -> None:
    for message in messages:
        _audit_payload(message)
        content = message.get("content")
        if isinstance(content, str):
            try:
                decoded = json.loads(content)
            except json.JSONDecodeError:
                continue
            _audit_payload(decoded)


def _capture(result: Any, *, as_of: str) -> dict[str, Any]:
    """Retain invalid model text for diagnosis, but never unsafe prompt data."""
    if not isinstance(result, Mapping):
        raise ValueError("decision callback must return a recorded policy result")
    messages = result.get("messages")
    raw = result.get("completion")
    if not isinstance(messages, (list, tuple)) or not messages or not isinstance(raw, str):
        raise ValueError(
            "policy result requires actual final-generation messages and raw completion"
        )
    _prompt_check(messages)
    visible_evidence(messages, expected_as_of=as_of)
    calls = result.get("tool_calls", ())
    _audit_payload(calls)
    if isinstance(calls, int) and not isinstance(calls, bool) and calls >= 0:
        count = calls
    elif isinstance(calls, (list, tuple)):
        count = len(calls)
    else:
        raise ValueError("tool_calls must be a count or a recorded call list")
    observed = 0
    for index, message in enumerate(messages[:-1]):
        if message.get("role") != "assistant" or not isinstance(message.get("content"), str):
            continue
        text = message["content"].strip()
        if text.startswith("<tool_call>") and text.endswith("</tool_call>"):
            text = text[len("<tool_call>") : -len("</tool_call>")].strip()
        try:
            request = json.loads(text)
        except json.JSONDecodeError:
            continue
        if (
            not isinstance(request, Mapping)
            or not isinstance(request.get("arguments"), Mapping)
            or not isinstance(request.get("tool", request.get("name")), str)
        ):
            continue
        following = messages[index + 1]
        try:
            returned = json.loads(following.get("content", ""))
        except (TypeError, json.JSONDecodeError):
            continue
        if (
            following.get("role") in {"user", "tool"}
            and isinstance(returned, Mapping)
            and ("tool_result" in returned or "evidence" in returned)
        ):
            observed += 1
    if observed != count:
        raise ValueError("reported tool count differs from actual tool turns in the model prompt")
    return {
        "messages": json.loads(canonical_json(messages)),
        "completion": raw,
        "tool_calls": calls,
        "tool_call_count": count,
    }


def _result(
    result: Any, *, as_of: str, universe: tuple[str, ...], state: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(result, Mapping) or result.get("status") != "success":
        raise ValueError("failed policy output cannot become an executable trajectory")
    captured = _capture(result, as_of=as_of)
    messages, raw = captured["messages"], captured["completion"]
    if messages[-1].get("role") == "assistant":
        raise ValueError("messages must be the prefix before the final assistant completion")
    accounts = []
    for message in messages:
        if message.get("role") != "user":
            continue
        try:
            payload = json.loads(message["content"])
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(payload, Mapping):
            account = payload.get("state", payload.get("account_state"))
            if isinstance(account, Mapping):
                accounts.append(account)
    if not accounts or any(
        any(account.get(key) != state[key] for key in ("cash", "positions", "equity"))
        for account in accounts
    ):
        raise ValueError("policy prompt does not contain the actual replay account state")
    aliases = result.get("evidence_aliases")
    original = result.get("original_messages")
    _audit_payload(aliases or {})
    if original:
        _prompt_check(original)
    parsed = resolve_decisions(
        raw,
        messages=messages,
        allowed_symbols=universe,
        aliases=aliases,
        original_messages=original,
    )
    reported = result.get("decisions")
    if reported is not None:
        reported = tuple(
            item if isinstance(item, Decision) else Decision(**item) for item in reported
        )
        if reported != parsed:
            raise ValueError("recorded decisions differ from the actual model completion")
    allowed = visible_evidence(original if aliases else messages, expected_as_of=as_of)
    return {
        **captured,
        "decisions": parsed,
        "allowed": set(allowed),
        "evidence_aliases": aliases or {},
        "original_messages": original or [],
    }


def _corporate_actions(portfolio: Any, bars: Mapping[str, DailyPriceBar]) -> None:
    for symbol, bar in bars.items():
        held = portfolio.positions.get(symbol, 0)
        if held and bar.stock_splits not in {0, 1}:
            split_qty = held * bar.stock_splits
            if not split_qty.is_integer():
                raise ValueError("fractional split holdings require an explicit replay policy")
            portfolio.positions[symbol] = int(split_qty)
        if held and bar.dividends:
            portfolio.cash += portfolio.positions[symbol] * bar.dividends


def collect_trajectories(
    bars_by_symbol: Mapping[str, Sequence[DailyPriceBar]],
    *,
    decide: Callable[..., Any],
    settings: TrajectorySettings,
    evidence_provider: Callable[[str, str], Sequence[Evidence]] | None = None,
    evidence_source_hash: str | None = None,
    split: str = "train",
    decision_sessions: Sequence[str] | None = None,
    policy_identity: Mapping[str, Any] | None = None,
    next_session_resolver: Callable[[str], str] | None = None,
    journal: Path | None = None,
    resume: bool = False,
    max_sessions: int | None = None,
    evaluation_only: bool = False,
    frozen_release=None,
    frozen_test_permit=None,
) -> TrajectoryCollection:
    """Collect actual decisions; each split starts an independent cash portfolio.

    Default train collection never invokes the policy on validation or test dates.
    Validation collection is explicit; test collection is deliberately unavailable.
    ``max_sessions`` is a total prefix target, enabling a small smoke then resume.
    """
    if split == "test":
        from agent.release import FrozenRelease, PendingTestPermit

        if (
            not evaluation_only
            or not isinstance(frozen_release, FrozenRelease)
            or not isinstance(frozen_test_permit, PendingTestPermit)
        ):
            raise ValueError("held-out replay requires a verified frozen release")
        frozen_release.authorize_test(
            settings=settings,
            decision_sessions=decision_sessions,
            next_session_resolver=next_session_resolver,
        )
        frozen_test_permit.verify(
            frozen_release, journal=journal, evidence_source_hash=evidence_source_hash
        )
    elif split not in {"train", "validation"}:
        raise ValueError("only train or explicit validation trajectory collection is allowed")
    if type(evaluation_only) is not bool or (evaluation_only and split == "train"):
        raise ValueError("evaluation replay cannot use the training split")
    identity = _identity(
        policy_identity or getattr(decide, "policy_identity", None), evaluation_only=evaluation_only
    )
    if split == "test" and identity["adapter_hash"] != frozen_release.manifest["adapter_hash"]:
        raise ValueError("held-out policy differs from the frozen selected adapter")
    if evidence_provider is not None and (
        not isinstance(evidence_source_hash, str)
        or len(evidence_source_hash) != 64
        or any(char not in "0123456789abcdef" for char in evidence_source_hash)
    ):
        raise ValueError("evidence provider requires its immutable evidence_source_hash")
    if max_sessions is not None and (type(max_sessions) is not int or max_sessions <= 0):
        raise ValueError("max_sessions must be positive")
    if set(settings.universe) - set(bars_by_symbol):
        raise ValueError("bars must cover the complete configured universe")
    by_symbol = {}
    for symbol in settings.universe:
        rows = sorted(bars_by_symbol[symbol], key=lambda item: item.session)
        if any(row.symbol != symbol for row in rows) or len({r.session for r in rows}) != len(rows):
            raise ValueError("bar symbols must agree and sessions must be unique")
        by_symbol[symbol] = {row.session: row for row in rows}
    sessions = sorted(set.union(*(set(rows) for rows in by_symbol.values())))
    wanted = set(decision_sessions) if decision_sessions is not None else None
    if wanted is not None and (len(wanted) != len(decision_sessions) or wanted - set(sessions)):
        raise ValueError("decision sessions must be unique source sessions")
    windows, skipped = [], []
    train_end, validation_end = (
        parse_timestamp(settings.train_end),
        parse_timestamp(settings.validation_end),
    )
    for current, following in zip(sessions, sessions[1:], strict=False):
        if wanted is not None and current not in wanted:
            continue
        if any(current not in rows or following not in rows for rows in by_symbol.values()):
            skipped.append({"session": current, "reason": "incomplete_universe_marks"})
            continue
        current_bars = {s: rows[current] for s, rows in by_symbol.items()}
        next_bars = {s: rows[following] for s, rows in by_symbol.items()}
        at = max(row.close_at for row in current_bars.values())
        outcome_at = max(row.close_at for row in next_bars.values())
        if split == "test":
            eligible = following in frozen_release.manifest["test_execution_sessions"]
        else:
            eligible = (
                parse_timestamp(outcome_at) <= train_end
                if split == "train"
                else train_end < parse_timestamp(at)
                and parse_timestamp(outcome_at) <= validation_end
            )
        if not eligible:
            continue
        if next_session_resolver is not None and next_session_resolver(current) != following:
            skipped.append({"session": current, "reason": "nonconsecutive_exchange_session"})
            continue
        windows.append((current, following, at, outcome_at))
    if not windows:
        raise ValueError("no complete eligible trajectory windows for the requested split")
    planned = tuple(row[0] for row in windows)
    if split == "test" and planned != tuple(frozen_release.manifest["test_decision_sessions"]):
        raise ValueError("all frozen held-out windows must be complete before any policy calls")
    source_hash = content_hash(
        {
            "price_bars": {s: tuple(by_symbol[s].values()) for s in settings.universe},
            "evidence_source_hash": evidence_source_hash,
        }
    )
    if split == "test":
        frozen_test_permit.verify(
            frozen_release,
            journal=journal,
            price_content_hash=content_hash(
                {s: tuple(by_symbol[s].values()) for s in settings.universe}
            ),
        )
    contract = {
        "version": PROTOCOL_VERSION,
        "evaluation_only": evaluation_only,
        "release_id": frozen_release.release_id if split == "test" else None,
        "test_attempt_id": frozen_test_permit.attempt_id if split == "test" else None,
        "settings": asdict(settings),
        "split": split,
        "policy_identity": identity,
        "source_hash": source_hash,
        "evidence_source_hash": evidence_source_hash,
        "planned_sessions": planned,
        "skipped_windows": skipped,
        "session_policy": "exchange-calendar" if next_session_resolver else "source-adjacent",
    }
    from trading_agent import Portfolio, RiskEngine

    cfg = Config(
        initial_cash=settings.initial_cash,
        universe=settings.universe,
        sector_map=settings.sector_map or tuple((s, "unknown") for s in settings.universe),
        max_shares=settings.max_qty,
        slippage_bps=settings.cost_bps,
    )
    risk = RiskEngine(cfg)
    portfolio = Portfolio(settings.initial_cash, peak_equity=settings.initial_cash)
    with _journal(journal, contract, resume) as records:
        if len(records) > len(windows):
            raise ValueError("journal extends beyond its planned sessions")
        for index, row in enumerate(records):
            if row["session"] != planned[index] or row["split"] != split:
                raise ValueError("journal episodes differ from planned collection prefix")
        if records:
            checkpoint = records[-1]["portfolio_after"]
            portfolio = Portfolio(
                checkpoint["cash"], dict(checkpoint["positions"]), checkpoint["peak_equity"]
            )
        last_outcome_session = str(records[-1]["outcome_at"])[:10] if records else None
        target = len(windows) if max_sessions is None else min(max_sessions, len(windows))
        if len(records) > target:
            raise ValueError("journal is beyond the requested total session target")
        for index in range(len(records), target):
            current, following, at, outcome_at = windows[index]
            if last_outcome_session is not None:
                for idle in sessions:
                    if last_outcome_session < idle <= current:
                        if any(idle not in rows for rows in by_symbol.values()):
                            raise ValueError(
                                "missing universe marks between selected decision sessions"
                            )
                        idle_bars = {s: rows[idle] for s, rows in by_symbol.items()}
                        _corporate_actions(portfolio, idle_bars)
                        portfolio.peak_equity = max(
                            portfolio.peak_equity,
                            portfolio.equity({s: bar.close for s, bar in idle_bars.items()}),
                        )
            current_bars = {s: rows[current] for s, rows in by_symbol.items()}
            prices = {s: row.close for s, row in current_bars.items()}
            before = portfolio.equity(prices)
            portfolio.peak_equity = max(portfolio.peak_equity, before)
            state = {
                "cash": portfolio.cash,
                "positions": dict(portfolio.positions),
                "equity": before,
                "as_of": at,
                "paper_only": True,
            }
            evidence_by_id = {}
            for symbol, rows in by_symbol.items():
                past = [bar for session, bar in rows.items() if session <= current]
                for bar in past[-settings.history_sessions :]:
                    item = bar.to_evidence()
                    evidence_by_id[item.id] = item
                extras = evidence_provider(symbol, at) if evidence_provider is not None else ()
                for item in extras:
                    if (
                        not isinstance(item, Evidence)
                        or not item.eligible(at)
                        or item.symbol not in {None, symbol}
                    ):
                        raise ValueError(
                            "trajectory evidence provider returned ineligible evidence"
                        )
                    if item.id in evidence_by_id and evidence_by_id[item.id] != item:
                        raise ValueError("conflicting trajectory evidence IDs")
                    evidence_by_id[item.id] = item
            # The callback receives only decision-time inputs. Following-session
            # bars are accessed below only after its response has been captured.
            failure, recorded, captured = None, None, None
            error_location = None
            response = {}
            started = time.monotonic()
            print(
                canonical_json(
                    {
                        "event": "policy_decision_start",
                        "split": split,
                        "session": current,
                        "index": index + 1,
                        "total": target,
                    }
                ),
                flush=True,
            )
            try:
                response = decide(
                    state=json.loads(canonical_json(state)),
                    as_of=at,
                    evidence=tuple(evidence_by_id.values()),
                    universe=settings.universe,
                )
                captured = _capture(response, as_of=at)
                if response.get("status") != "success":
                    raise ValueError("policy did not return a successful decision")
                recorded = _result(response, as_of=at, universe=settings.universe, state=state)
            except Exception as exc:
                failure = type(exc).__name__
                # Record code location, never exception values or frame locals:
                # provider errors may contain credentials or source payloads.
                frame = exc.__traceback__
                while frame.tb_next is not None:
                    frame = frame.tb_next
                error_location = {
                    "module": frame.tb_frame.f_globals.get("__name__"),
                    "function": frame.tb_frame.f_code.co_name,
                    "line": frame.tb_lineno,
                }
            initial_rejections = {}
            if recorded is not None:
                initial_rejections = {
                    d.symbol: risk.check(d, portfolio, prices, recorded["allowed"])
                    for d in recorded["decisions"]
                }
            next_bars = {s: rows[following] for s, rows in by_symbol.items()}
            _corporate_actions(portfolio, next_bars)
            opens = {s: bar.open for s, bar in next_bars.items()}
            closes = {s: bar.close for s, bar in next_bars.items()}
            no_action = portfolio.equity(closes)
            fills, rejections = [], []
            if recorded is not None:
                for number, decision in enumerate(recorded["decisions"]):
                    rejection = initial_rejections[decision.symbol]
                    if rejection is None and decision.action != "hold":
                        rejection = risk.check(decision, portfolio, opens, recorded["allowed"])
                    if rejection is not None:
                        rejections.append({"symbol": decision.symbol, "reason": rejection})
                        continue
                    if decision.action == "hold":
                        continue
                    rate = settings.cost_bps / 10_000
                    fill = Fill(
                        content_hash((split, at, number, decision))[:32],
                        decision.symbol,
                        decision.action,
                        decision.qty,
                        opens[decision.symbol]
                        * (1 + rate if decision.action == "buy" else 1 - rate),
                        next_bars[decision.symbol].open_at,
                    )
                    portfolio.apply(fill)
                    fills.append(fill.to_dict())
            after = portfolio.equity(closes)
            portfolio.peak_equity = max(portfolio.peak_equity, after)
            change = after - before
            economic_label = (
                "good"
                if change > settings.label_threshold
                else "bad"
                if change < -settings.label_threshold
                else "flat"
            )
            exclusion = None
            if recorded is None:
                label, exclusion = "invalid", "invalid_policy_output"
            elif rejections:
                label, exclusion = "invalid", "execution_rejection"
            elif recorded["tool_call_count"] < settings.min_tool_calls:
                label, exclusion = "unlabeled", "missing_required_tool_research"
            elif not fills:
                label, exclusion = "unlabeled", "no_executed_trade"
            elif change > settings.label_threshold:
                label = "good"
            elif change < -settings.label_threshold:
                label, exclusion = "bad", "negative_net_equity_change"
            else:
                label, exclusion = "unlabeled", "flat_net_equity_change"
            trace = recorded or captured or {}
            safe_response = response if isinstance(response, Mapping) else {}
            row = {
                "example_id": f"trajectory:{split}:{current}",
                "split": split,
                "session": current,
                "decision_at": at,
                "outcome_at": outcome_at,
                "state": state,
                "status": "success" if recorded is not None else "invalid",
                "messages": trace.get("messages", []),
                "completion": trace.get("completion", ""),
                "decisions": [d.to_dict() for d in recorded["decisions"]] if recorded else [],
                "tool_calls": trace.get("tool_calls", []),
                "tool_call_count": trace.get("tool_call_count", 0),
                "evidence_aliases": recorded["evidence_aliases"] if recorded else {},
                "original_messages": recorded["original_messages"] if recorded else [],
                "error_class": failure,
                "error_location": error_location,
                "context_diagnostics": {
                    key: value
                    for key, value in (
                        safe_response.get("context_diagnostics", {})
                        if isinstance(safe_response.get("context_diagnostics"), Mapping)
                        else {}
                    ).items()
                    if key in {"minimum_input_tokens", "input_token_budget"}
                    and type(value) is int
                    and value >= 0
                },
                "policy_status": safe_response.get("status", "policy_setup_failure"),
                "policy_failure": safe_response.get("failure"),
                "inference_seconds": time.monotonic() - started,
                "initial_input_tokens": safe_response.get("initial_input_tokens"),
                "fills": fills,
                "execution_rejections": rejections,
                "equity_before": before,
                "equity_after": after,
                "net_equity_change": change,
                "no_action_equity": no_action,
                "portfolio_change_label": economic_label,
                "incremental_pnl": after - no_action,
                "label": label,
                "exclusion_reason": exclusion,
                "portfolio_after": {
                    "cash": portfolio.cash,
                    "positions": dict(portfolio.positions),
                    "peak_equity": portfolio.peak_equity,
                },
            }
            records.append(json.loads(canonical_json(row)))
            last_outcome_session = following
            if journal is not None:
                digest = content_hash(records[-1])
                _write_json(
                    journal / f"episode-{index:06d}.json",
                    {
                        "record": records[-1],
                        "record_hash": digest,
                        "previous_record_hash": content_hash(records[-2]) if index else None,
                    },
                )
                _write_json(
                    journal / "collection-state.json",
                    {
                        "record_count": len(records),
                        "record_hash": digest,
                    },
                )
            print(
                canonical_json(
                    {
                        "event": "trajectory_collected",
                        "split": split,
                        "completed": index + 1,
                        "target": target,
                        "session": current,
                        "label": label,
                        "tool_calls": row["tool_call_count"],
                        "policy_status": row["policy_status"],
                        "policy_failure": row["policy_failure"],
                        "error_class": row["error_class"],
                        "error_location": row["error_location"],
                        "context_diagnostics": row["context_diagnostics"],
                    }
                ),
                flush=True,
            )
    return TrajectoryCollection(
        settings, split, identity, source_hash, tuple(records), planned, tuple(skipped)
    )


def merge_train_collections(collections):
    """Pool independent chronological training folds without fabricating continuity."""
    collections = tuple(collections)
    if not collections:
        raise ValueError("at least one independent training fold is required")
    first = collections[0]
    records, sessions, seen = [], [], set()
    for number, collection in enumerate(collections):
        _identity(collection.policy_identity)
        if (
            collection.split != "train"
            or collection.settings != first.settings
            or collection.policy_identity != first.policy_identity
            or len(collection.records) != len(collection.planned_sessions)
        ):
            raise ValueError("training folds must be complete with matching policy/source settings")
        if seen.intersection(collection.planned_sessions):
            raise ValueError("training folds overlap; do not duplicate exposure")
        seen.update(collection.planned_sessions)
        sessions.extend(collection.planned_sessions)
        records.extend(
            {**row, "source_fold": number, "independent_fold": True} for row in collection.records
        )
    return TrajectoryCollection(
        first.settings,
        "train",
        first.policy_identity,
        content_hash({"independent_fold_sources": [c.source_hash for c in collections]}),
        tuple(sorted(records, key=lambda r: r["session"])),
        tuple(sorted(sessions)),
        tuple(row for c in collections for row in c.skipped_windows),
    )


def export_good_dataset(
    train: TrajectoryCollection, validation: TrajectoryCollection, output: Path
) -> DatasetArtifact:
    """Export genuine successful model completions, never oracle replacements."""
    if train.split != "train" or validation.split != "validation":
        raise ValueError("GOOD export requires independent train and validation collections")
    if train.settings != validation.settings or train.policy_identity != validation.policy_identity:
        raise ValueError("collections must use identical settings and base policy protocol")
    _identity(train.policy_identity)
    if output.exists():
        raise ValueError("trajectory output already exists; choose a new directory")
    settings = train.settings
    examples = []
    for collection in (train, validation):
        if len(collection.records) != len(collection.planned_sessions):
            raise ValueError("finish the planned collection before exporting the GOOD dataset")
        good = [row for row in collection.records if row["label"] == "good"]
        if not good:
            raise ValueError(
                f"no valid GOOD {collection.split} trajectories; do not fabricate targets"
            )
        for row in good:
            if (
                row["status"] != "success"
                or not row["fills"]
                or row["execution_rejections"]
                or row["net_equity_change"] <= settings.label_threshold
                or row["tool_call_count"] < settings.min_tool_calls
            ):
                raise ValueError("GOOD record does not satisfy the declared eligibility filter")
            _result(row, as_of=row["decision_at"], universe=settings.universe, state=row["state"])
            messages = tuple(row["messages"])
            examples.append(
                EpisodeExample(
                    row["example_id"],
                    collection.split,
                    row["decision_at"],
                    row["outcome_at"],
                    content_hash({"state": row["state"], "messages": messages}),
                    content_hash(row["state"]),
                    content_hash(tuple(visible_evidence(messages).values())),
                    messages,
                    row["completion"],
                )
            )
    validate_chronological_splits(
        examples,
        train_end=settings.train_end,
        validation_end=settings.validation_end,
        test_end=settings.test_end,
    )
    splits = {
        split: tuple(row.example_id for row in examples if row.split == split)
        for split in ("train", "validation", "test")
    }
    source_hashes = tuple(sorted((content_hash(train), content_hash(validation))))
    regime = "base-policy-good-trajectories-v1"
    filter_hash = content_hash((train.planned_sessions, validation.planned_sessions))
    identity = {
        "examples": examples,
        "splits": splits,
        "purged": (),
        "source_hashes": source_hashes,
        "boundaries": (settings.train_end, settings.validation_end, settings.test_end),
        "regime_id": regime,
        "decision_session_filter_hash": filter_hash,
    }
    manifest = DatasetManifest(
        content_hash(identity),
        splits,
        (),
        source_hashes,
        settings.train_end,
        settings.validation_end,
        settings.test_end,
        regime,
        filter_hash,
    )
    artifact = DatasetArtifact(tuple(examples), manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        artifact.write(staging / "dataset")
        for collection in (train, validation):
            (staging / f"trajectories-{collection.split}.jsonl").write_text(
                "".join(canonical_json(row) + "\n" for row in collection.records), encoding="utf-8"
            )
        _write_json(
            staging / "trajectory-manifest.json",
            {
                "protocol_version": PROTOCOL_VERSION,
                "dataset_hash": manifest.dataset_hash,
                "policy_identity": train.policy_identity,
                "settings": asdict(settings),
                "label_rule": (
                    "GOOD iff valid researched executed trade and net equity change > threshold"
                ),
                "economic_label_rule": "good/bad/flat from net portfolio equity change after costs",
                "filter_explanation": (
                    "HOLD-only, failed/rejected and unresearched outputs are not SFT targets"
                ),
                "supervision_mode": "trajectory-final",
                "train_counts": train.counts,
                "validation_counts": validation.counts,
                "test_policy_calls": 0,
                "corporate_action_policy": "integer splits; dividends credited on source ex-date",
            },
        )
        load_dataset_artifact(staging / "dataset")
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return artifact
