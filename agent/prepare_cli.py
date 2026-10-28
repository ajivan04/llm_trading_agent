"""Download daily prices and prepare SFT, DPO, and PPO artifacts."""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from agent.data import XNYSCalendar, YFinancePriceAdapter
from agent.evaluation import build_midterm_regime_plan
from agent.preparation import PreparationSettings, prepare_training_artifacts
from agent.universe import BASELINE_UNIVERSE


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="first session, YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="exclusive price end, YYYY-MM-DD")
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--validation-end", required=True)
    parser.add_argument("--test-end", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--symbols", nargs="+", default=list(BASELINE_UNIVERSE))
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--trade-qty", type=int, default=5)
    parser.add_argument(
        "--regime", choices=("none", "midterm-election"), default="none",
        help="restrict decision sessions to the declared event-regime windows",
    )
    parser.add_argument("--target-election-year", type=int, default=2026)
    parser.add_argument("--historical-election-years", type=int, nargs="+",
                        default=[2022, 2018, 2014, 2010, 2006])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    symbols = tuple(symbol.strip().upper() for symbol in args.symbols)
    if not symbols or any(not item for item in symbols) or len(set(symbols)) != len(symbols):
        raise ValueError("symbols must be nonempty and unique")
    calendar = XNYSCalendar()

    def bounds(_symbol, session):
        item = calendar.bounds(session)
        return item.open_at, item.close_at

    adapter = YFinancePriceAdapter(bounds)
    retrieved_at = datetime.now(UTC)
    bars = {
        symbol: adapter.fetch(
            symbol, start=args.start, end=args.end, retrieved_at=retrieved_at,
        )
        for symbol in symbols
    }
    regime_plan = None
    allowed_sessions = ()
    if args.regime == "midterm-election":
        regime_plan = build_midterm_regime_plan(
            target_year=args.target_election_year,
            historical_years=tuple(args.historical_election_years),
        )
        allowed_sessions = regime_plan.decision_sessions()
    prepared = prepare_training_artifacts(
        bars,
        settings=PreparationSettings(
            train_end=args.train_end, validation_end=args.validation_end,
            test_end=args.test_end, cost_bps=args.cost_bps, trade_qty=args.trade_qty,
            allowed_decision_sessions=allowed_sessions,
            regime_id=regime_plan.regime if regime_plan is not None else "chronological-v1",
        ),
        next_session_resolver=lambda session: calendar.next(session).session,
    )
    prepared.write(args.output)
    if regime_plan is not None:
        (args.output / "regime-plan.json").write_text(
            json.dumps(regime_plan.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({
        "artifact_hash": prepared.artifact_hash,
        "dataset_hash": prepared.dataset.manifest.dataset_hash,
        "output": str(args.output), "symbols": len(symbols),
        "training_examples": len(prepared.dataset.manifest.split_example_ids["train"]),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

