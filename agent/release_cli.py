"""Immutable selection packaging and one-shot, resumable October test."""

import argparse
import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

from agent.artifacts import atomic_json, check_files, file_hash
from agent.contracts import Evidence, canonical_json, content_hash, parse_timestamp
from agent.data.prices import DailyPriceBar
from agent.evaluation.registry import ExperimentRegistry
from agent.evaluation.research import summarize_research_replay, write_research_result
from agent.release import (
    DECISION_WEEK,
    EXECUTION_WEEK,
    FrozenRelease,
    PendingTestPermit,
    freeze_release,
)
from agent.trajectories import TrajectorySettings, collect_trajectories
from agent.trajectory_cli import _calendar, _rows


def heldout_plan(release, prepared):
    release.verify_runtime()
    prepared = Path(prepared)
    hydration = json.loads((prepared / "hydration-manifest.json").read_text())
    if (
        not hydration.get("heldout_news_requested")
        or hydration.get("test_release_id") != release.release_id
    ):
        raise ValueError("held-out source must be hydrated for this already-frozen release")
    parent_hashes = hydration["source_file_hashes"]
    if any(
        parent_hashes.get(name) != value
        for name, value in release.manifest["source_file_hashes"].items()
    ):
        raise ValueError("held-out source is not an extension of the frozen training source")
    check_files(prepared, hydration["output_file_hashes"])
    if (
        file_hash(prepared / "price-bars.jsonl")
        != release.manifest["source_file_hashes"]["price-bars.jsonl"]
    ):
        raise ValueError("held-out prices changed after selection")
    cfg = json.loads((release.path / "paper-config.json").read_text())
    original = json.loads((release.path / "validation.json").read_text())["report"]["protocol"]
    settings_row = dict(original["settings"])
    settings_row["universe"] = tuple(settings_row["universe"])
    settings_row["sector_map"] = tuple(map(tuple, settings_row["sector_map"]))
    settings = TrajectorySettings(**settings_row)
    bars = {symbol: [] for symbol in cfg["universe"]}
    for row in _rows(prepared / "price-bars.jsonl"):
        if row["symbol"] in bars:
            bars[row["symbol"]].append(DailyPriceBar(**row))
    calendar = _calendar()
    indexed = {s: {b.session: b for b in rows} for s, rows in bars.items()}
    for decision, execution in zip(DECISION_WEEK, EXECUTION_WEEK, strict=True):
        if calendar.next(decision).session != execution:
            raise ValueError("frozen execution calendar mapping differs")
        for rows in indexed.values():
            for session in (decision, execution):
                if session not in rows:
                    raise ValueError("held-out week has incomplete universe marks")
                bounds = calendar.bounds(session)
                if (
                    rows[session].open_at != bounds.open_at
                    or rows[session].close_at != bounds.close_at
                ):
                    raise ValueError("source timestamps differ from verified execution calendar")
    evidence = tuple(Evidence(**row) for row in _rows(prepared / "evidence.jsonl"))
    coverage = []
    for day in DECISION_WEEK:
        cutoff = calendar.bounds(day).close_at
        eligible = [
            r
            for r in evidence
            if r.symbol in (*cfg["universe"], None)
            and r.eligible(cutoff)
            and parse_timestamp(cutoff)
            - parse_timestamp(
                r.observed_at
                if r.source in {"alpaca.news", "alpha_vantage.news"}
                else r.available_at
            )
            <= timedelta(days=60)
        ]
        coverage.append(
            {
                "decision_session": day,
                "news": sum(r.source in {"alpaca.news", "alpha_vantage.news"} for r in eligible),
                "sec_text": sum(r.source == "sec.filing_text" for r in eligible),
            }
        )
    if not sum(r["news"] for r in coverage) or not sum(r["sec_text"] for r in coverage):
        raise ValueError("held-out NLP source coverage is missing")
    hashes = {
        "price-bars.jsonl": file_hash(prepared / "price-bars.jsonl"),
        "evidence.jsonl": file_hash(prepared / "evidence.jsonl"),
        "hydration-manifest.json": file_hash(prepared / "hydration-manifest.json"),
    }
    return (
        bars,
        evidence,
        settings,
        calendar,
        {
            "release_id": release.release_id,
            "price_content_hash": content_hash(
                {s: tuple(sorted(rows, key=lambda b: b.session)) for s, rows in bars.items()}
            ),
            "source_file_hashes": hashes,
            "decisions": DECISION_WEEK,
            "executions": EXECUTION_WEEK,
            "starting_cash": 100000,
            "memory": "disabled",
            "coverage": coverage,
        },
    )


def run_heldout(release, prepared, output, registry_path, *, resume=False, decide=None):
    bars, evidence, settings, calendar, contract = heldout_plan(release, prepared)
    output, registry_path = Path(output).resolve(), Path(registry_path).resolve()
    if registry_path.is_relative_to(output):
        raise ValueError("assessment registry must live outside the result directory")
    contract["journal_path"] = str(output / "journal")
    output.mkdir(parents=True, exist_ok=True)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    import fcntl

    with registry_path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        registry = ExperimentRegistry(registry_path, release_id=release.manifest["assessment_id"])
        try:
            if resume:
                attempt = registry.resume_pending(contract)
            else:
                if any(output.iterdir()):
                    raise ValueError("held-out output is not empty; resume its original attempt")
                attempt = registry.begin(contract, split="test")
            permit = PendingTestPermit(registry_path, attempt.attempt_id, contract)
            sidecar = output / "attempt.json"
            if sidecar.exists() and json.loads(sidecar.read_text()) != {
                "attempt_id": attempt.attempt_id,
                "contract_hash": content_hash(contract),
            }:
                raise ValueError("held-out attempt sidecar differs from registry reservation")
            atomic_json(
                output / "attempt.json",
                {"attempt_id": attempt.attempt_id, "contract_hash": content_hash(contract)},
            )
            if decide is None:
                from agent.context import ContextConfig
                from agent.policy.loader import HuggingFaceQwenBackend
                from agent.research import ResearchPolicy, ResearchSettings

                values = dict(release.manifest["research_settings"])
                values["context"] = ContextConfig(**values["context"])
                decide = ResearchPolicy(
                    HuggingFaceQwenBackend(
                        adapter=release.path / "adapter",
                        max_context_tokens=values["max_context_tokens"],
                    ),
                    settings=ResearchSettings(**values),
                    mode="historical",
                    cache_path=output / "journal/tools.sqlite",
                )
            identity = dict(decide.policy_identity)
            validated = json.loads((release.path / "validation.json").read_text())["report"][
                "protocol"
            ]["research_policy_identity"]
            if identity != validated:
                raise ValueError("held-out research loop differs from the frozen validated policy")
            identity["protocol_hash"] = content_hash(
                {"policy": identity["protocol_hash"], "frozen_test": contract}
            )

            def provider(symbol, cutoff):
                return tuple(
                    r
                    for r in evidence
                    if r.symbol in (symbol, None)
                    and r.eligible(cutoff)
                    and parse_timestamp(cutoff)
                    - parse_timestamp(
                        r.observed_at
                        if r.source in {"alpaca.news", "alpha_vantage.news"}
                        else r.available_at
                    )
                    <= timedelta(days=60)
                )

            journal = output / "journal"
            collection = collect_trajectories(
                bars,
                decide=decide,
                settings=settings,
                evidence_provider=provider,
                evidence_source_hash=contract["source_file_hashes"]["evidence.jsonl"],
                split="test",
                decision_sessions=DECISION_WEEK,
                policy_identity=identity,
                next_session_resolver=lambda d: calendar.next(d).session,
                journal=journal,
                resume=resume and (journal / "collection-manifest.json").exists(),
                evaluation_only=True,
                frozen_release=release,
                frozen_test_permit=permit,
            )
            result = summarize_research_replay(collection, bars)
            result = replace(
                result,
                protocol={
                    **result.protocol,
                    "frozen_release_id": release.release_id,
                    "attempt_id": attempt.attempt_id,
                    "contract": contract,
                    "five_day_sharpe_interpretation": "descriptive only",
                },
            )
            check_files(prepared, contract["source_file_hashes"])
            release.verify_runtime()
            write_research_result(result, output)
            registry.finish(attempt.attempt_id, metrics=result.metrics)
            return result
        finally:
            # Operational interruption remains pending: only identical attempt+prefix may resume.
            registry.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    freeze = commands.add_parser("freeze")
    for name in ("adapter", "prepared", "validation", "config", "output"):
        freeze.add_argument("--" + name, type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--release", type=Path, required=True)
    test = commands.add_parser("test")
    for name in ("release", "prepared", "output", "registry"):
        test.add_argument("--" + name, type=Path, required=True)
    test.add_argument("--resume", action="store_true")
    test.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    if args.action == "freeze":
        release = freeze_release(
            adapter=args.adapter,
            prepared=args.prepared,
            validation=args.validation,
            config=args.config,
            output=args.output,
        )
        print(canonical_json({"release_id": release.release_id, "output": str(args.output)}))
    elif args.action == "verify":
        release = FrozenRelease(args.release)
        release.verify_runtime()
        print(canonical_json({"release_id": release.release_id, "verified": True}))
    else:
        release = FrozenRelease(args.release)
        if args.plan_only:
            print(canonical_json(heldout_plan(release, args.prepared)[-1]))
        else:
            result = run_heldout(
                release, args.prepared, args.output, args.registry, resume=args.resume
            )
            print(canonical_json(result.metrics), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
