"""Run one bounded paper research cycle; defaults to dry-run order proposals."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from agent.config import Config
from agent.contracts import canonical_json, content_hash
from agent.execution import AlpacaPaperBroker, Outbox, PaperExecutor
from agent.live_context import LiveResearchSeed
from agent.memory import TradingMemory
from agent.policy.loader import HuggingFaceQwenBackend
from agent.research import ResearchPolicy
from agent.runtime import PaperTradingLoop


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "cycle", "discard-unsent"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--release", type=Path)
    parser.add_argument("--outbox", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cycle-id")
    parser.add_argument("--client-id")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--submit-paper", action="store_true")
    parser.add_argument("--memory", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.env_file is not None:
        from agent.env import load_credentials

        load_credentials(args.env_file)
    if args.submit_paper and args.action != "cycle":
        raise ValueError("paper submission is available only for a research cycle")
    if args.action == "cycle" and not args.cycle_id:
        raise ValueError("cycle requires a stable --cycle-id")
    frozen = None
    settings = None
    if args.release:
        from agent.artifacts import check_files
        from agent.context import ContextConfig
        from agent.release import FrozenRelease
        from agent.research import ResearchSettings

        frozen = FrozenRelease(args.release)
        frozen.verify_runtime()
        if args.adapter is not None or args.config is not None:
            raise ValueError("frozen paper runs cannot override adapter or risk config")
        check_files(args.prepared, frozen.manifest["source_file_hashes"])
        args.adapter, args.config = frozen.path / "adapter", frozen.path / "paper-config.json"
        values = dict(frozen.manifest["research_settings"])
        values["context"] = ContextConfig(**values["context"])
        settings = ResearchSettings(**values)
    elif args.adapter is None or args.config is None:
        raise ValueError("provide --release, or both --adapter and --config for an experiment")
    config = Config.from_dict(json.loads(args.config.read_text()))
    backend = HuggingFaceQwenBackend(
        adapter=args.adapter, max_context_tokens=settings.max_context_tokens if settings else 4096
    )
    source = LiveResearchSeed(args.prepared, cache_path=args.outbox.with_suffix(".tools.sqlite"))
    if frozen is not None:
        check_files(args.prepared, frozen.manifest["source_file_hashes"])
    policy = ResearchPolicy(backend, mode="live", cache_path=source.cache_path, settings=settings)
    root = Path(__file__).parent
    code_paths = [
        *sorted((root / "execution").glob("*.py")),
        root / "runtime.py",
        root / "live_context.py",
        root / "memory.py",
        root / "reflection.py",
        root.parent / "trading_agent.py",
        root / "config.py",
    ]
    code_hashes = {
        str(p.relative_to(root.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in code_paths
    }
    identity = {
        **policy.policy_identity,
        "frozen_release_id": frozen.release_id if frozen else None,
        "paper_memory": frozen.manifest["paper_memory"] if frozen else "delayed-confirmed",
        "runtime_hash": content_hash(
            {
                "code": code_hashes,
                "source": source.source_hash,
                "config": asdict(config),
                "memory_namespace": (args.run_id, "paper"),
                "memory_path": str(args.memory.resolve()),
                "signal_ttl_seconds": 300,
                "order_type": "ioc-limit-no-extended-hours",
                "live_prefetch_symbols": source.max_symbols,
            }
        ),
    }
    box = Outbox(args.outbox)
    try:
        executor = PaperExecutor(
            AlpacaPaperBroker.from_environment(),
            box,
            config,
            run_id=args.run_id,
            policy_identity=identity,
        )
        if args.action == "status":
            snapshot = executor.reconcile()
            print(
                canonical_json(
                    {
                        "cash": snapshot.cash,
                        "equity": snapshot.equity,
                        "positions": snapshot.positions,
                        "market_open": snapshot.market_open,
                        "pending_intents": len(box.active()),
                    }
                ),
                flush=True,
            )
        elif args.action == "discard-unsent":
            if not args.client_id:
                raise ValueError("discard-unsent requires --client-id")
            with box.lock():
                box.discard_unsent(args.client_id)
            print(canonical_json({"status": "discarded_never_submitted_intent"}))
        else:
            with TradingMemory(
                args.memory, experiment_id=args.run_id, run_id=args.run_id, fold="paper"
            ) as memory:
                result = PaperTradingLoop(
                    policy, executor, source, memory=memory, frozen_release=frozen
                ).run_once(cycle_id=args.cycle_id, submit=args.submit_paper)
            print(
                canonical_json({"status": result["status"], "cycle_id": args.cycle_id}), flush=True
            )
    finally:
        box.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
