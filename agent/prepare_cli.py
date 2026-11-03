"""Download daily prices and prepare SFT, DPO, and PPO artifacts."""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

from agent.contracts import parse_timestamp
from agent.data import (
    AlphaVantageNewsAdapter,
    SecDataAdapter,
    XNYSCalendar,
    YFinancePriceAdapter,
    aggregate_news,
)
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
    parser.add_argument(
        "--evidence-mode", choices=("price-only", "news", "sec", "news-sec"),
        default="price-only",
        help="point-in-time evidence sources to include in policy prompts",
    )
    parser.add_argument("--news-lookback-days", type=int, default=60)
    parser.add_argument("--sec-lookback-days", type=int, default=60)
    parser.add_argument("--news-limit", type=int, default=50)
    return parser


def _sec_request(url: str, *, headers: dict[str, str]):
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=30) as response:  # noqa: S310 - fixed SEC HTTPS endpoints
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # pragma: no cover - network dependent
        raise RuntimeError("SEC request failed") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("SEC returned an invalid response")
    return payload


def _sec_text_request(url: str, *, headers: dict[str, str]) -> str:
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=30) as response:  # noqa: S310 - fixed SEC HTTPS endpoints
            return response.read().decode("utf-8", errors="replace")
    except Exception as exc:  # pragma: no cover - network dependent
        raise RuntimeError("SEC filing text request failed") from exc


def _eligible_news(items, *, as_of: str, lookback_days: int):
    cutoff_at = parse_timestamp(as_of)
    return tuple(
        item for item in items
        if item.eligible(as_of)
        and (cutoff_at - parse_timestamp(item.available_at)).days <= lookback_days
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (args.news_lookback_days <= 0 or args.sec_lookback_days <= 0
            or not 1 <= args.news_limit <= 1000):
        raise ValueError("lookback windows must be positive and news limit must be 1..1000")
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

    evidence_provider = None
    if args.evidence_mode != "price-only":
        cutoff = datetime.fromisoformat(args.end).replace(tzinfo=UTC)
        news_by_symbol: dict[str, tuple] = defaultdict(tuple)
        sec_by_symbol: dict[str, tuple] = defaultdict(tuple)
        sec_text_cache: dict[str, object] = {}
        if args.evidence_mode in {"news", "news-sec"}:
            api_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
            if not api_key:
                raise ValueError("ALPHA_VANTAGE_API_KEY is required for news evidence")
            news_adapter = AlphaVantageNewsAdapter(api_key)
            # Fetch the complete historical range once; the per-snapshot provider
            # below applies the lookback relative to each decision cutoff. A final-
            # window fetch would silently omit 2006/2010/2014/2018 event weeks.
            news_start = datetime.fromisoformat(args.start).replace(tzinfo=UTC)
            for symbol in symbols:
                news_by_symbol[symbol] = news_adapter.fetch(
                    (symbol,), time_from=news_start, time_to=cutoff,
                    as_of=cutoff, retrieved_at=retrieved_at, limit=args.news_limit,
                )
        if args.evidence_mode in {"sec", "news-sec"}:
            user_agent = os.environ.get("SEC_USER_AGENT", "")
            if not user_agent:
                raise ValueError("SEC_USER_AGENT is required for SEC evidence")
            sec_adapter = SecDataAdapter(
                user_agent, lambda url, headers: _sec_request(url, headers=headers),
                text_request=lambda url, headers: _sec_text_request(url, headers=headers),
            )
            ticker_map = sec_adapter.fetch_ticker_map()
            for symbol in symbols:
                cik = ticker_map.get(symbol)
                if cik is None:
                    raise ValueError(f"SEC ticker map has no CIK for {symbol}")
                filings = sec_adapter.fetch_submissions(
                    cik, symbol=symbol, as_of=cutoff, retrieved_at=retrieved_at,
                )
                sec_by_symbol[symbol] = filings

        def evidence_provider(symbol: str, as_of: str):
            extras = []
            if args.evidence_mode in {"news", "news-sec"}:
                eligible_news = _eligible_news(
                    news_by_symbol[symbol], as_of=as_of,
                    lookback_days=args.news_lookback_days,
                )
                extras.append(aggregate_news(
                    eligible_news, symbol=symbol, as_of=as_of, retrieved_at=retrieved_at,
                ))
            if args.evidence_mode in {"sec", "news-sec"}:
                cutoff_at = parse_timestamp(as_of)
                eligible_sec = tuple(
                    item for item in sec_by_symbol[symbol]
                    if item.eligible(as_of)
                    and (cutoff_at - parse_timestamp(item.available_at)).days
                    <= args.sec_lookback_days
                )
                extras.extend(eligible_sec[-5:])
                for filing in eligible_sec[-5:]:
                    if filing.id not in sec_text_cache:
                        sec_text_cache[filing.id] = sec_adapter.fetch_filing_text(
                            filing, as_of=as_of, retrieved_at=retrieved_at,
                        )
                    extras.append(sec_text_cache[filing.id])
            return tuple(extras)
    prepared = prepare_training_artifacts(
        bars,
        settings=PreparationSettings(
            train_end=args.train_end, validation_end=args.validation_end,
            test_end=args.test_end, cost_bps=args.cost_bps, trade_qty=args.trade_qty,
            allowed_decision_sessions=allowed_sessions,
            regime_id=regime_plan.regime if regime_plan is not None else "chronological-v1",
            evidence_mode=args.evidence_mode,
        ),
        next_session_resolver=lambda session: calendar.next(session).session,
        evidence_provider=evidence_provider,
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
        "evidence_mode": args.evidence_mode,
        "evidence_records": len(prepared.evidence_rows),
        "sec_evidence_scope": (
            "submissions-metadata-and-bounded-text"
            if args.evidence_mode in {"sec", "news-sec"} else None
        ),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

