"""Resume-safe raw-news augmentation; preserves prices and SEC, never old targets."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from agent.artifacts import atomic_json, check_files
from agent.contracts import Evidence, canonical_json, content_hash, parse_timestamp
from agent.data.news import AlpacaNewsAdapter
from agent.source_bundle import load_source, write_source
from agent.trajectory_cli import _file_hash, _plan, _rows, build_parser


def hydration_plan(
    source: Path, *, train_sessions=60, validation_sessions=20, symbols=None, historical_sessions=0
):
    summaries = []
    requests = [("train", train_sessions, "recent"), ("validation", validation_sessions, "recent")]
    if historical_sessions:
        requests.append(("train", historical_sessions, "midterm"))
    for split, count, cohort in requests:
        args = build_parser().parse_args(
            [
                "plan",
                "--prepared",
                str(source),
                "--split",
                split,
                "--sessions",
                str(count),
                "--cohort",
                cohort,
                "--allow-missing-nlp",
                *(["--symbols", *symbols] if symbols else []),
            ]
        )
        bars, _, _, sessions, _, summary = _plan(args)
        cutoffs = [
            max(b.close_at for rows in bars.values() for b in rows if b.session == day)
            for day in sessions
        ]
        summaries.append({**summary, "cutoffs": cutoffs})
    intervals = sorted(
        (
            min(parse_timestamp(t) for t in p["cutoffs"]) - timedelta(days=60),
            max(parse_timestamp(t) for t in p["cutoffs"]),
        )
        for p in summaries
    )
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    end = max(b for _, b in merged)
    if end >= parse_timestamp("2026-10-23T20:00:00Z"):
        raise ValueError("training/validation hydration cannot enter the held-out decision week")
    windows, window_as_of = [], []
    for start, stop in merged:
        while start < stop:
            following = min(start + timedelta(days=7), stop)
            windows.append((start.isoformat(), following.isoformat()))
            window_as_of.append(stop.isoformat())
            start = following
    return {
        "version": "raw-news-hydration-v1",
        "source": str(source.resolve()),
        "source_file_hashes": summaries[0]["source_file_hashes"],
        "dataset_hash": summaries[0]["source_dataset_hash"],
        "universe": summaries[0]["universe"],
        "plans": summaries,
        "windows": windows,
        "window_as_of": window_as_of,
        "as_of": end.isoformat(),
        "page_limit": 50,
        "latest_only": False,
        "code_hash": _file_hash(Path(__file__)),
    }


def heldout_hydration_plan(source, frozen_release):
    from agent.release import DECISION_WEEK
    from agent.trajectory_cli import _calendar

    frozen_release.verify_runtime()
    check_files(source, frozen_release.manifest["source_file_hashes"])
    universe = json.loads((frozen_release.path / "paper-config.json").read_text())["universe"]
    source = Path(source).resolve()
    hashes = dict(frozen_release.manifest["source_file_hashes"])
    metadata = load_source(source)
    plan = {
        "version": "raw-news-hydration-v1",
        "source": str(source),
        "source_file_hashes": hashes,
        "universe": universe,
        "dataset_hash": metadata["dataset_hash"],
        "page_limit": 50,
        "latest_only": False,
        "code_hash": _file_hash(Path(__file__)),
    }
    calendar = _calendar()
    cutoffs = [calendar.bounds(day).close_at for day in DECISION_WEEK]
    start = parse_timestamp(cutoffs[0]) - timedelta(days=60)
    end = parse_timestamp(cutoffs[-1])
    windows = []
    while start < end:
        following = min(start + timedelta(days=7), end)
        windows.append((start.isoformat(), following.isoformat()))
        start = following
    plan.update(
        {
            "windows": windows,
            "window_as_of": [end.isoformat()] * len(windows),
            "as_of": end.isoformat(),
            "plans": [{"split": "frozen-test-source", "cutoffs": cutoffs}],
            "test_release_id": frozen_release.release_id,
        }
    )
    return plan


def _validate_records(records, plan, start, end):
    index = [tuple(w) for w in plan["windows"]].index((start, end))
    cutoff = plan.get("window_as_of", [plan["as_of"]] * len(plan["windows"]))[index]
    for item in records:
        if (
            item.source != "alpaca.news"
            or item.symbol not in plan["universe"]
            or not item.eligible(cutoff)
            or not parse_timestamp(start)
            <= parse_timestamp(item.observed_at)
            <= parse_timestamp(end)
        ):
            raise ValueError("news provider returned records outside hydration contract")


def hydrate(
    source: Path, output: Path, *, adapter, plan, resume=False, clock=None, frozen_release=None
):
    """No policy inference or label rewriting. Completed chunks never refetch."""
    clock = clock or (lambda: datetime.now(UTC))
    source, output = source.resolve(), output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("hydration output must be separate and nonnested")
    if plan.get("source") != str(source) or plan.get("version") != "raw-news-hydration-v1":
        raise ValueError("hydration plan belongs to a different source or version")
    if plan.get("test_release_id"):
        from agent.release import FrozenRelease

        if not isinstance(frozen_release, FrozenRelease):
            raise ValueError("test-source hydration requires its verified frozen release")
        expected = heldout_hydration_plan(source, frozen_release)
        if canonical_json(plan) != canonical_json(expected):
            raise ValueError("held-out news request differs from frozen source plan")
    elif parse_timestamp(plan["as_of"]) >= parse_timestamp("2026-10-23T20:00:00Z"):
        raise ValueError("hydration cannot request held-out decision news")
    if plan.get("code_hash") != _file_hash(Path(__file__)):
        raise ValueError("hydration implementation changed after planning")
    check_files(source, plan["source_file_hashes"])
    if output.exists():
        raise ValueError("hydration output already exists; original data is never overwritten")
    work = output.with_name(output.name + ".hydrating")
    work.mkdir(parents=True, exist_ok=True)
    import fcntl

    with (work / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        contract = work / "contract.json"
        normalized = json.loads(canonical_json(plan))
        if contract.exists():
            if not resume or json.loads(contract.read_text()) != normalized:
                raise ValueError("hydration resume contract differs or resume not explicit")
        elif resume:
            raise ValueError("no hydration contract to resume")
        else:
            if any(p.name != ".lock" for p in work.iterdir()):
                raise ValueError("nonempty hydration work has no verified contract")
            atomic_json(contract, plan)
        for index, (start, end) in enumerate(plan["windows"]):
            path = work / f"chunk-{index:04d}.json"
            cutoff = plan.get("window_as_of", [plan["as_of"]] * len(plan["windows"]))[index]
            request_identity = {
                "plan_hash": content_hash(plan),
                "index": index,
                "start": start,
                "end": end,
                "as_of": cutoff,
            }
            if path.exists():
                saved = json.loads(path.read_text())
                if (
                    content_hash(saved["records"]) != saved["records_hash"]
                    or saved.get("request") != request_identity
                ):
                    raise ValueError("hydration chunk checksum or request binding failed")
                records = tuple(Evidence(**row) for row in saved["records"])
                _validate_records(records, plan, start, end)
                continue
            print(
                canonical_json(
                    {
                        "event": "news_chunk_start",
                        "index": index + 1,
                        "total": len(plan["windows"]),
                        "start": start,
                        "end": end,
                    }
                ),
                flush=True,
            )
            records = tuple(
                adapter.fetch(
                    plan["universe"],
                    time_from=start,
                    time_to=end,
                    as_of=cutoff,
                    retrieved_at=clock(),
                    limit=50,
                    latest_only=False,
                )
            )
            rows = [item.to_dict() for item in records]
            _validate_records(records, plan, start, end)
            saved = {
                "request": request_identity,
                "records": rows,
                "records_hash": content_hash(rows),
            }
            atomic_json(path, saved)
            print(
                canonical_json(
                    {"event": "news_chunk_done", "index": index + 1, "records": len(rows)}
                ),
                flush=True,
            )
        if {name: _file_hash(source / name) for name in plan["source_file_hashes"]} != plan[
            "source_file_hashes"
        ]:
            raise ValueError("source archive changed during hydration")
        staging = work / "bundle"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir()
        shutil.copy2(source / "price-bars.jsonl", staging / "price-bars.jsonl")
        if (source / "price-basis-manifest.json").is_file():
            shutil.copy2(
                source / "price-basis-manifest.json", staging / "price-basis-manifest.json"
            )
        # Preserve old SEC/raw records, but never publish the old oracle/PPO targets
        # as newly generated training examples.
        evidence = {}
        for row in _rows(source / "evidence.jsonl"):
            old = evidence.get(row["id"])
            if old is not None and content_hash(old) != content_hash(row):
                raise ValueError("source evidence has conflicting duplicate IDs")
            evidence[row["id"]] = row
        old_ids = set(evidence)
        for index in range(len(plan["windows"])):
            saved = json.loads((work / f"chunk-{index:04d}.json").read_text())
            start, end = plan["windows"][index]
            cutoff = plan.get("window_as_of", [plan["as_of"]] * len(plan["windows"]))[index]
            expected_request = {
                "plan_hash": content_hash(plan),
                "index": index,
                "start": start,
                "end": end,
                "as_of": cutoff,
            }
            if (
                saved.get("request") != expected_request
                or content_hash(saved["records"]) != saved["records_hash"]
            ):
                raise ValueError("hydration chunk changed before publication")
            _validate_records(tuple(Evidence(**r) for r in saved["records"]), plan, start, end)
            for row in saved["records"]:
                Evidence(**row)
                old = evidence.get(row["id"])
                # Retrieval time can differ for the same article on a chunk boundary.
                if old is not None:
                    a, b = dict(old), dict(row)
                    a.pop("retrieved_at", None)
                    b.pop("retrieved_at", None)
                    if content_hash(a) != content_hash(b):
                        raise ValueError("conflicting article versions share an ID")
                else:
                    evidence[row["id"]] = row
        with (staging / "evidence.jsonl").open("w") as handle:
            for eid in sorted(evidence):
                handle.write(canonical_json(evidence[eid]) + "\n")
        metadata = load_source(source)
        write_source(
            staging,
            metadata,
            universe=metadata["universe"] or tuple(plan["universe"]),
            scope="heldout" if plan.get("test_release_id") else metadata["scope"],
            parent_source_id=metadata["source_id"],
            test_release_id=plan.get("test_release_id"),
        )
        output_hashes = {
            name: _file_hash(staging / name)
            for name in ("price-bars.jsonl", "evidence.jsonl", "source.json")
        }
        coverage = []
        raw_times = [
            (
                parse_timestamp(row["available_at"]),
                parse_timestamp(row["observed_at"]),
                row.get("symbol"),
            )
            for row in evidence.values()
            if row["source"] in {"alpaca.news", "alpha_vantage.news"}
        ]
        universe = set(plan["universe"])
        for split_plan in plan["plans"]:
            for cutoff in split_plan["cutoffs"]:
                at = parse_timestamp(cutoff)
                count = sum(
                    symbol in universe
                    and available <= at
                    and timedelta(0) <= at - published <= timedelta(days=60)
                    for available, published, symbol in raw_times
                )
                coverage.append({"split": split_plan["split"], "as_of": cutoff, "raw_news": count})
        manifest = {
            "plan": plan,
            "source_file_hashes": plan["source_file_hashes"],
            "output_file_hashes": output_hashes,
            "raw_news_evidence_records_added": len(set(evidence) - old_ids),
            "coverage": coverage,
            "old_targets_rebuilt": False,
            "policy_calls": 0,
            "heldout_news_requested": bool(plan.get("test_release_id")),
            "test_release_id": plan.get("test_release_id"),
            "upstream_compaction_provenance": plan["plans"][0].get("compaction_provenance"),
        }
        (staging / "hydration-manifest.json").write_text(canonical_json(manifest) + "\n")
        check_files(source, plan["source_file_hashes"])
        preserved = {
            name: value
            for name, value in plan["source_file_hashes"].items()
            if name == "price-bars.jsonl"
        }
        check_files(staging, preserved)
        if output.exists():
            raise ValueError("hydration output appeared during collection; refusing to replace it")
        staging.rename(output)
        return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--train-sessions", type=int, default=60)
    parser.add_argument("--validation-sessions", type=int, default=20)
    parser.add_argument("--historical-sessions", type=int, default=0)
    parser.add_argument("--symbols", nargs="+")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--release",
        type=Path,
        help="only after selection: augment a separate source for frozen test",
    )
    args = parser.parse_args(argv)
    frozen = None
    if args.release:
        from agent.release import FrozenRelease

        frozen = FrozenRelease(args.release)
        plan = heldout_hydration_plan(args.source, frozen)
    else:
        plan = hydration_plan(
            args.source,
            train_sessions=args.train_sessions,
            validation_sessions=args.validation_sessions,
            symbols=args.symbols,
            historical_sessions=args.historical_sessions,
        )
    print(canonical_json({"event": "hydration_plan", "plan": plan}), flush=True)
    if args.plan_only:
        return 0
    if args.env_file:
        from agent.env import load_credentials

        load_credentials(args.env_file)
    import os

    last = [0.0]

    def request(params):
        delay = 0.35 - (time.monotonic() - last[0])
        if delay > 0:
            time.sleep(delay)
        last[0] = time.monotonic()
        return AlpacaNewsAdapter._default_request(params)

    adapter = AlpacaNewsAdapter(
        os.environ.get("ALPACA_API_KEY", ""), os.environ.get("ALPACA_SECRET_KEY", ""), request
    )
    manifest = hydrate(
        args.source,
        args.output,
        adapter=adapter,
        plan=plan,
        resume=args.resume,
        frozen_release=frozen,
    )
    print(
        canonical_json(
            {"output": str(args.output), "added": manifest["raw_news_evidence_records_added"]}
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
