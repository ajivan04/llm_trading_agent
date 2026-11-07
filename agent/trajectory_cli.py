"""Collect genuine base-model research trajectories and export GOOD-only SFT data."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from agent.contracts import Evidence, canonical_json, content_hash, parse_timestamp
from agent.data.prices import DailyPriceBar
from agent.datasets import load_dataset_artifact
from agent.trajectories import (
    TrajectorySettings,
    collect_trajectories,
    export_good_dataset,
    load_collection,
)
from agent.universe import BASELINE_UNIVERSE


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("collect", "plan"):
        command = commands.add_parser(name, help="collect or inspect an immutable session plan")
        command.add_argument("--prepared", type=Path, required=True)
        command.add_argument("--split", choices=("train", "validation"), default="train")
        command.add_argument(
            "--sessions",
            type=int,
            help="most recent eligible sessions; default train 60 / validation 20",
        )
        command.add_argument("--start", help="inclusive decision date, no earlier than 2016-01-01")
        command.add_argument("--end", help="exclusive decision date")
        command.add_argument(
            "--symbols", nargs="+", help="explicit research fixture universe override"
        )
        command.add_argument("--max-context-tokens", type=int, default=4096)
        command.add_argument("--max-new-tokens", type=int, default=512)
        command.add_argument("--max-turns", type=int, default=6)
        command.add_argument("--news-lookback-days", type=int, default=60)
        command.add_argument("--sec-lookback-days", type=int, default=60)
        command.add_argument(
            "--allow-missing-nlp",
            action="store_true",
            help="explicit diagnostic-only allowance for incomplete news/SEC archive",
        )
        if name == "collect":
            command.add_argument("--journal", type=Path, required=True)
            command.add_argument(
                "--max-sessions",
                type=int,
                help="total collected prefix target for a resumable smoke",
            )
            command.add_argument("--resume", action="store_true")
            command.add_argument("--require-cuda", action="store_true")
    export = commands.add_parser(
        "export", help="export verified completed journals without loading Qwen"
    )
    export.add_argument("--train-journal", type=Path, required=True)
    export.add_argument("--validation-journal", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    return parser


def _rows(path: Path) -> list[dict[str, Any]]:
    result = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid or incomplete {path.name} row {number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path.name} row {number} must be an object")
            result.append(row)
    return result


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _calendar():
    import exchange_calendars

    from agent.data.calendar import XNYSCalendar

    return XNYSCalendar(exchange_calendars.get_calendar("XNYS", start="2016-01-01"))


def _policy(args):
    # Import/load nothing model-sized until the source/session plan passes.
    from agent.policy.loader import HuggingFaceQwenBackend
    from agent.research import ResearchPolicy, ResearchSettings

    if args.require_cuda:
        import torch

        if not torch.cuda.is_available():
            raise ValueError("CUDA unavailable; refusing CPU collection in the GPU job")
    backend = HuggingFaceQwenBackend(adapter=None, max_context_tokens=args.max_context_tokens)
    return ResearchPolicy(
        backend,
        settings=ResearchSettings(
            max_context_tokens=args.max_context_tokens,
            max_new_tokens=args.max_new_tokens,
            max_turns=args.max_turns,
        ),
        mode="historical",
        cache_path=args.journal / "tools.sqlite",
    )


def _plan(args):
    source = args.prepared.resolve()
    required = (
        "dataset/manifest.json",
        "preparation-manifest.json",
        "price-bars.jsonl",
        "evidence.jsonl",
    )
    hashes = {name: _file_hash(source / name) for name in required}
    manifest = json.loads((source / "dataset/manifest.json").read_text())
    prepared_manifest = json.loads((source / "preparation-manifest.json").read_text())
    universe = tuple(args.symbols or BASELINE_UNIVERSE)
    settings = TrajectorySettings(
        universe, manifest["train_end"], manifest["validation_end"], manifest["test_end"]
    )
    for name in (
        "max_context_tokens",
        "max_new_tokens",
        "max_turns",
        "news_lookback_days",
        "sec_lookback_days",
    ):
        if type(getattr(args, name)) is not int or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.max_new_tokens >= args.max_context_tokens:
        raise ValueError("context must reserve usable input and completion space")
    count = args.sessions if args.sessions is not None else 60 if args.split == "train" else 20
    if type(count) is not int or count <= 0:
        raise ValueError("sessions must be positive")
    lower = date.fromisoformat(args.start or "2016-01-01")
    upper = date.fromisoformat(args.end) if args.end else None
    if lower < date(2016, 1, 1) or (upper is not None and upper <= lower):
        raise ValueError("decision dates must start in 2016 or later with an ordered end")
    bars: dict[str, list[DailyPriceBar]] = defaultdict(list)
    source_rows = _rows(source / "price-bars.jsonl")
    if prepared_manifest.get("source_hash") != content_hash(tuple(source_rows)):
        raise ValueError("prepared price archive differs from its source hash")
    compaction_provenance = None
    if prepared_manifest.get("dataset_hash") != manifest.get("dataset_hash"):
        compaction_path = source / "compaction-manifest.json"
        if not compaction_path.is_file():
            raise ValueError(
                "prepared dataset manifest identity differs without compaction lineage"
            )
        compaction = json.loads(compaction_path.read_text())
        if (
            compaction.get("dataset_hash") != manifest.get("dataset_hash")
            or compaction.get("compaction_version") != "evidence-compaction-v2"
        ):
            raise ValueError("prepared dataset manifest differs from compaction lineage")
        load_dataset_artifact(source / "dataset")
        hashes["compaction-manifest.json"] = _file_hash(compaction_path)
        compaction_provenance = {
            "version": compaction["compaction_version"],
            "source_target_link_verified": (
                compaction.get("source_dataset_hash") == prepared_manifest.get("dataset_hash")
            ),
            "preserved_price_hash_verified": True,
            "preserved_evidence_count_verified": True,
            "legacy_targets_consumed": False,
            "legacy_compaction_source_link_unverified": (
                compaction.get("source_dataset_hash") != prepared_manifest.get("dataset_hash")
            ),
        }
    for row in source_rows:
        item = DailyPriceBar(**row)
        if item.symbol in settings.universe:
            bars[item.symbol].append(item)
    if set(settings.universe) - set(bars):
        raise ValueError(
            "source bars do not cover the requested universe; no silent symbol removal"
        )
    by_symbol = {symbol: {row.session: row for row in rows} for symbol, rows in bars.items()}
    if any(len(rows) != len(by_symbol[symbol]) for symbol, rows in bars.items()):
        raise ValueError("duplicate source symbol/session bars")
    calendar = _calendar()
    candidates = []
    sessions = sorted(set.union(*(set(rows) for rows in by_symbol.values())))
    for current in sessions:
        if date.fromisoformat(current) < lower or (upper and date.fromisoformat(current) >= upper):
            continue
        if any(current not in rows for rows in by_symbol.values()):
            continue
        at = max(rows[current].close_at for rows in by_symbol.values())
        if args.split == "train" and parse_timestamp(at) >= parse_timestamp(settings.train_end):
            continue
        if args.split == "validation" and not (
            parse_timestamp(settings.train_end)
            < parse_timestamp(at)
            < parse_timestamp(settings.validation_end)
        ):
            continue
        following = calendar.next(current).session
        if any(following not in rows for rows in by_symbol.values()):
            continue
        outcome = max(rows[following].close_at for rows in by_symbol.values())
        boundary = settings.train_end if args.split == "train" else settings.validation_end
        if parse_timestamp(outcome) <= parse_timestamp(boundary):
            candidates.append(current)
    if len(candidates) < count:
        raise ValueError(
            f"only {len(candidates)} complete eligible {args.split} sessions; requested {count}"
        )
    selected = tuple(candidates[-count:])
    evidence = tuple(Evidence(**row) for row in _rows(source / "evidence.jsonl"))
    if prepared_manifest.get("counts", {}).get("evidence") != len(evidence):
        raise ValueError("prepared evidence archive count differs from manifest")
    coverage = []
    for day in selected:
        cutoff = max(rows[day].close_at for rows in by_symbol.values())
        eligible = [
            item
            for item in evidence
            if item.symbol in (*settings.universe, None)
            and item.eligible(cutoff)
            and parse_timestamp(cutoff) - parse_timestamp(item.available_at)
            <= timedelta(
                days=(
                    args.sec_lookback_days
                    if item.source.startswith("sec.")
                    else args.news_lookback_days
                )
            )
        ]
        coverage.append(
            {
                "session": day,
                "raw_news_records": sum(
                    item.source in {"alpaca.news", "alpha_vantage.news"} for item in eligible
                ),
                "sec_text_records": sum(item.source == "sec.filing_text" for item in eligible),
            }
        )
    raw_news = sum(row["raw_news_records"] for row in coverage)
    filing_text = sum(row["sec_text_records"] for row in coverage)
    if not args.allow_missing_nlp and (not raw_news or not filing_text):
        raise ValueError(
            "news-sec collection requires archived raw articles and SEC text; "
            "aggregate counts alone are not NLP evidence"
        )
    if hashes != {name: _file_hash(source / name) for name in hashes}:
        raise ValueError("prepared source changed during planning; finish the artifact transfer")
    summary = {
        "split": args.split,
        "sessions": selected,
        "source_file_hashes": hashes,
        "universe": settings.universe,
        "universe_size": len(settings.universe),
        "raw_news_records": raw_news,
        "sec_text_records": filing_text,
        "decision_window_coverage": coverage,
        "diagnostic_missing_nlp_allowed": args.allow_missing_nlp,
        "source_dataset_hash": manifest["dataset_hash"],
        "compaction_provenance": compaction_provenance,
        "settings": asdict(settings),
    }
    return bars, evidence, settings, selected, calendar, summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "export":
        train = load_collection(args.train_journal)
        validation = load_collection(args.validation_journal)
        artifact = export_good_dataset(train, validation, args.output)
        print(
            canonical_json(
                {
                    "output": str(args.output),
                    "dataset_hash": artifact.manifest.dataset_hash,
                    "train_counts": train.counts,
                    "validation_counts": validation.counts,
                }
            ),
            flush=True,
        )
        return 0
    bars, evidence, settings, sessions, calendar, summary = _plan(args)
    print(canonical_json({"event": "trajectory_plan", **summary}), flush=True)
    if args.command == "plan":
        return 0
    policy = _policy(args)
    identity = dict(policy.policy_identity)
    identity["protocol_hash"] = content_hash(
        {
            "research_protocol": identity["protocol_hash"],
            "source_file_hashes": summary["source_file_hashes"],
            "collector_code": _file_hash(Path(__file__).with_name("trajectories.py")),
            "cli_code": _file_hash(Path(__file__)),
            "news_lookback_days": args.news_lookback_days,
            "sec_lookback_days": args.sec_lookback_days,
        }
    )
    extras: dict[str | None, list[Evidence]] = defaultdict(list)
    for item in evidence:
        if item.source not in {"yfinance", "derived.price_features"}:
            extras[item.symbol].append(item)

    def provider(symbol: str, as_of: str):
        cutoff = parse_timestamp(as_of)
        result = []
        for item in (*extras[symbol], *extras[None]):
            days = (
                args.sec_lookback_days
                if item.source.startswith("sec.")
                else args.news_lookback_days
            )
            if item.eligible(as_of) and cutoff - parse_timestamp(item.available_at) <= timedelta(
                days=days
            ):
                result.append(item)
        return tuple(result)

    collection = collect_trajectories(
        bars,
        decide=policy,
        settings=settings,
        evidence_provider=provider,
        evidence_source_hash=summary["source_file_hashes"]["evidence.jsonl"],
        split=args.split,
        decision_sessions=sessions,
        policy_identity=identity,
        next_session_resolver=lambda day: calendar.next(day).session,
        journal=args.journal,
        resume=args.resume,
        max_sessions=args.max_sessions,
    )
    print(
        canonical_json(
            {
                "event": "trajectory_collection_complete",
                "journal": str(args.journal),
                "counts": collection.counts,
                "completed_plan": len(collection.records) == len(collection.planned_sessions),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
