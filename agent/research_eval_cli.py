"""Run resumable account-aware research comparisons on validation only."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

from agent.contracts import canonical_json, content_hash, parse_timestamp
from agent.evaluation.research import evaluate_research_policy, write_research_result
from agent.policy.loader import HuggingFaceQwenBackend
from agent.research import ResearchPolicy, ResearchSettings
from agent.trajectory_cli import _file_hash, _plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--adapter", type=Path, help="fixed trained adapter; omitted means frozen base"
    )
    parser.add_argument(
        "--sessions", type=int, default=20, help="fixed most recent validation plan"
    )
    parser.add_argument(
        "--max-sessions", type=int, help="total prefix target; smoke then resume same plan"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--symbols", nargs="+", help="explicit fixture universe override")
    parser.add_argument("--benchmark-symbol", help="optional explicit single-symbol benchmark")
    parser.add_argument("--max-context-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--news-lookback-days", type=int, default=60)
    parser.add_argument("--sec-lookback-days", type=int, default=60)
    parser.add_argument(
        "--allow-missing-nlp",
        action="store_true",
        help="diagnostic only; never represents an NLP-complete comparison",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.split = "validation"
    bars, evidence, settings, sessions, calendar, summary = _plan(args)
    print(canonical_json({"event": "research_validation_plan", **summary}), flush=True)
    if args.plan_only:
        return 0
    if args.adapter is not None:
        if not (args.adapter / "adapter_config.json").is_file() or not (
            (args.adapter / "adapter_model.safetensors").is_file()
            or (args.adapter / "adapter_model.bin").is_file()
        ):
            raise ValueError("adapter must contain a complete saved PEFT policy")
    if args.require_cuda:
        import torch

        if not torch.cuda.is_available():
            raise ValueError("CUDA unavailable; refusing CPU inference in the GPU job")
    backend = HuggingFaceQwenBackend(
        adapter=args.adapter, max_context_tokens=args.max_context_tokens
    )
    research = ResearchPolicy(
        backend,
        settings=ResearchSettings(
            max_context_tokens=args.max_context_tokens,
            max_new_tokens=args.max_new_tokens,
            max_turns=args.max_turns,
        ),
        mode="historical",
        cache_path=args.output / "journal/tools.sqlite",
    )
    base_identity = dict(research.policy_identity)
    identity = {
        **base_identity,
        "protocol_hash": content_hash(
            {
                "research_protocol": base_identity["protocol_hash"],
                "source_file_hashes": summary["source_file_hashes"],
                "collector_code": _file_hash(Path(__file__).with_name("trajectories.py")),
                "evaluator_code": _file_hash(Path(__file__).parent / "evaluation/research.py"),
                "cli_code": _file_hash(Path(__file__)),
                "mode": "validation-only-account-aware",
                "news_lookback_days": args.news_lookback_days,
                "sec_lookback_days": args.sec_lookback_days,
                "benchmark_symbol": args.benchmark_symbol,
                "memory": "disabled",
            }
        ),
    }
    extras = defaultdict(list)
    for item in evidence:
        if item.source not in {"yfinance", "derived.price_features"}:
            extras[item.symbol].append(item)

    def provider(symbol, as_of):
        cutoff = parse_timestamp(as_of)
        return tuple(
            item
            for item in (*extras[symbol], *extras[None])
            if item.eligible(as_of)
            and cutoff - parse_timestamp(item.available_at)
            <= timedelta(
                days=(
                    args.sec_lookback_days
                    if item.source.startswith("sec.")
                    else args.news_lookback_days
                )
            )
        )

    result = evaluate_research_policy(
        bars,
        decide=research,
        settings=settings,
        decision_sessions=sessions,
        next_session_resolver=lambda day: calendar.next(day).session,
        policy_identity=identity,
        evidence_provider=provider,
        evidence_source_hash=summary["source_file_hashes"]["evidence.jsonl"],
        journal=args.output / "journal",
        resume=args.resume,
        max_sessions=args.max_sessions,
        benchmark_symbol=args.benchmark_symbol,
    )
    if base_identity != research.policy_identity:
        raise ValueError("policy files changed during evaluation; preserve journal and investigate")
    result = replace(
        result,
        protocol={
            **result.protocol,
            "source_coverage": summary["decision_window_coverage"],
            "diagnostic_missing_nlp_allowed": args.allow_missing_nlp,
            "memory": {"mode": "disabled", "initial_digest": content_hash(())},
        },
    )
    write_research_result(result, args.output)
    print(
        canonical_json(
            {
                "event": "research_validation_complete",
                "output": str(args.output),
                "metrics": result.metrics,
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
