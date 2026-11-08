"""Compile bounded train-only NLP decision episodes; never run policy inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from collections import defaultdict
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

from agent.context import ContextConfig, build_context
from agent.contracts import canonical_json, content_hash, parse_timestamp
from agent.datasets import load_dataset_artifact
from agent.features import PriceFeatureBuilder
from agent.policy.evidence import DECISION_INSTRUCTIONS, alias_messages, visible_evidence
from agent.policy.loader import HuggingFaceQwenBackend
from agent.trajectory_cli import _plan

from .environment import ReplayEpisode, critical_ppo_code_hashes

INSTRUCTION = (
    "This is a final-decision-only PPO episode. All available research is supplied. "
    "Do not request tools or change the bound decision cutoff. Respect the actual "
    "cash and held shares; short selling and margin are disabled. "
    "Unavailable news or filings must not be invented. " + DECISION_INSTRUCTIONS
)


def compile_ppo(args, *, backend=None):
    if args.output.exists():
        raise ValueError("PPO output exists; choose a new compiled artifact directory")
    # Reuse audited archive identity, complete exchange windows, and raw-NLP gate.
    args.split, args.start, args.end, args.symbols = (
        "train",
        None,
        None,
        getattr(args, "symbols", None),
    )
    args.news_lookback_days = args.sec_lookback_days = 60
    args.allow_missing_nlp = False
    args.max_context_tokens, args.max_turns = args.max_length, 6
    bars, evidence, settings, sessions, calendar, source = _plan(args)
    sft_artifact = load_dataset_artifact(args.sft_dataset)
    if sft_artifact.manifest.train_end != settings.train_end:
        raise ValueError("PPO source and assignment SFT have different training boundaries")
    if sft_artifact.manifest.regime_id != "base-policy-good-trajectories-v1":
        raise ValueError("PPO warm-start dataset must be actual base-policy GOOD trajectories")
    backend = backend or HuggingFaceQwenBackend(max_context_tokens=args.max_length)
    indexed = {symbol: {row.session: row for row in rows} for symbol, rows in bars.items()}
    extras = defaultdict(list)
    for item in evidence:
        if item.source in {
            "alpaca.news",
            "alpha_vantage.news",
            "sec.filing_text",
            "sec.submissions",
        }:
            extras[item.symbol].append(item)
    context_config = ContextConfig(max_news=4, max_filings=2, max_text_chars=800)
    feature_builder = PriceFeatureBuilder()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="." + args.output.name + "-", dir=args.output.parent))
    count, maximum, skipped = 0, 0, []
    try:
        with (staging / "ppo-train.jsonl").open("w") as handle:
            for session in sessions:
                following = calendar.next(session).session
                for symbol in sorted(settings.universe):
                    current, future = indexed[symbol][session], indexed[symbol][following]
                    if parse_timestamp(future.close_at) > parse_timestamp(settings.train_end):
                        raise ValueError("PPO horizon crosses training boundary")
                    history = [b.to_evidence() for b in bars[symbol] if b.session <= session][-130:]
                    feature = feature_builder.build(history, symbol=symbol, as_of=current.close_at)
                    language = [
                        item
                        for item in (*extras[symbol], *extras[None])
                        if item.eligible(current.close_at)
                        and parse_timestamp(current.close_at)
                        - parse_timestamp(
                            item.observed_at
                            if item.source in {"alpaca.news", "alpha_vantage.news"}
                            else item.available_at
                        )
                        <= timedelta(days=60)
                    ]
                    pool = [history[-1], feature, *language]
                    for quantity in (0, 5):
                        cash = settings.initial_cash - quantity * current.close
                        if cash <= 0:
                            skipped.append(
                                {"symbol": symbol, "session": session, "held_qty": quantity}
                            )
                            continue
                        state = {
                            "as_of": current.close_at,
                            "cash": cash,
                            "positions": {symbol: quantity} if quantity else {},
                            "equity": cash + quantity * current.close,
                            "universe": [symbol],
                            "protocol": "final-decision-nlp-v1",
                            "memory": "disabled",
                        }
                        required = [history[-1].id]
                        for sources in ({"alpaca.news", "alpha_vantage.news"}, {"sec.filing_text"}):
                            available = [item for item in language if item.source in sources]
                            if available:
                                required.append(
                                    max(available, key=lambda item: (item.available_at, item.id)).id
                                )
                        context = build_context(
                            state=state,
                            evidence=pool,
                            tools=(),
                            token_counter=backend.token_count,
                            max_context_tokens=args.max_length - 220,
                            reserved_final_tokens=args.max_new_tokens + 8,
                            config=context_config,
                            required_evidence_ids=tuple(required),
                        )
                        original = json.loads(canonical_json(context.original_messages))
                        original[0]["content"] = INSTRUCTION
                        messages, aliases = alias_messages(original)
                        tokens = backend.token_count(messages)
                        if tokens + args.max_new_tokens + 8 > args.max_length:
                            raise ValueError(
                                "PPO final-decision prompt exceeds native context budget"
                            )
                        maximum = max(maximum, tokens)
                        episode = ReplayEpisode(
                            episode_id=f"ppo-nlp:{session}:{symbol}:held-{quantity}",
                            split="train",
                            decision_at=current.close_at,
                            outcome_at=future.close_at,
                            messages=tuple(messages),
                            evidence_ids=tuple(visible_evidence(messages)),
                            symbol=symbol,
                            next_open=future.open,
                            outcome_close=future.close,
                            starting_cash=cash,
                            starting_qty=quantity,
                            max_qty=settings.max_qty,
                            cost_bps=settings.cost_bps,
                            decision_close=current.close,
                            enforce_risk=True,
                            evidence_aliases=aliases,
                            original_messages=tuple(original),
                        )
                        handle.write(canonical_json(asdict(episode)) + "\n")
                        count += 1
                print(
                    canonical_json(
                        {
                            "event": "ppo_compile",
                            "session": session,
                            "episodes": count,
                            "max_input_tokens": maximum,
                        }
                    ),
                    flush=True,
                )
        if count == 0:
            raise ValueError("no complete train-only PPO slots compiled")
        digest = hashlib.sha256((staging / "ppo-train.jsonl").read_bytes()).hexdigest()
        protocol = {
            "version": "final-decision-nlp-v1",
            "critical_code_hashes": critical_ppo_code_hashes(),
            "reward_horizon": (
                "next execution open to same-session close; held overnight gap excluded"
            ),
            "episodes": count,
            "episodes_sha256": digest,
            "train_end": settings.train_end,
            "source_file_hashes": source["source_file_hashes"],
            "sft_dataset_hash": sft_artifact.manifest.dataset_hash,
            "source_coverage": source["decision_window_coverage"],
            "max_length": args.max_length,
            "reserved_output_tokens": args.max_new_tokens,
            "max_input_tokens": maximum,
            "scenarios": ["cash-only", "held-five-shares"],
            "skipped_scenarios": skipped,
            "memory": "disabled",
            "objective": (
                "final decisions with fixed supplied evidence; tool calls are not PPO optimized"
            ),
            "risk": "Config defaults; unknown shared sector; decision-close and next-open checks",
            "settings": asdict(settings),
            "test_policy_calls": 0,
            "context_config": asdict(context_config),
        }
        if any(
            hashlib.sha256((args.prepared / name).read_bytes()).hexdigest() != digest
            for name, digest in source["source_file_hashes"].items()
        ):
            raise ValueError("PPO source changed during compilation")
        protocol["protocol_hash"] = content_hash(protocol)
        (staging / "ppo-protocol.json").write_text(canonical_json(protocol) + "\n")
        (staging / "ppo-environment.json").write_text(
            canonical_json(
                {
                    "episodes_path": str((args.output / "ppo-train.jsonl").resolve()),
                    "protocol_path": str((args.output / "ppo-protocol.json").resolve()),
                }
            )
            + "\n"
        )
        if args.output.exists():
            raise ValueError("PPO output appeared during compilation; preserve existing artifacts")
        staging.rename(args.output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return protocol


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--sft-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sessions", type=int, default=60)
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="explicit fixture universe; preserve full universe for assignment",
    )
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args(argv)
    result = compile_ppo(args)
    print(
        canonical_json(
            {
                "output": str(args.output),
                "episodes": result["episodes"],
                "protocol_hash": result["protocol_hash"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
