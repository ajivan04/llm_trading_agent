"""Run one bounded paper research cycle; defaults to dry-run order proposals."""

import argparse
import contextlib
import hashlib
import json
import math
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from agent.config import Config
from agent.contracts import canonical_json, content_hash
from agent.execution import AlpacaPaperBroker, Outbox, PaperExecutor
from agent.live_context import LiveResearchSeed
from agent.memory import TradingMemory
from agent.policy.loader import HuggingFaceQwenBackend
from agent.research import ResearchPolicy
from agent.runtime import PaperTradingLoop


def paper_cycle_ids(prefix, iterations):
    return [f"{prefix}-{index:06d}" for index in range(iterations)]


class _CycleLog:
    def on_cycle_result(self, result):
        response = result.get("response", {})
        state = result.get("state", {})
        print(
            canonical_json(
                {
                    "event": "paper_iteration",
                    "cycle_id": result.get("cycle_id"),
                    "status": result["status"],
                    "equity": state.get("equity"),
                    "positions": state.get("positions"),
                    "memory_count": state.get("memory_count"),
                    "memory_retrieval": result.get("memories", ()),
                    "source_refresh": state.get("source_refresh"),
                    "tool_calls": response.get("tool_calls", ()),
                    "completion": response.get("completion", ""),
                    "decisions": result.get("decisions", ()),
                    "execution": result.get("execution", ()),
                }
            ),
            flush=True,
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "cycle", "discard-unsent"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--release", type=Path)
    parser.add_argument("--max-context-tokens", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--outbox", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cycle-id")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--interval", type=float, default=60)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--client-id")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--submit-paper", action="store_true")
    parser.add_argument("--memory", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.env_file is not None:
        from agent.env import load_credentials

        load_credentials(args.env_file)
    if (
        not 1 <= args.iterations <= 100000
        or not math.isfinite(args.interval)
        or not 0 <= args.interval <= 3600
    ):
        raise ValueError("use a finite positive iteration count and interval 0..3600 seconds")
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
        if (
            args.adapter is not None
            or args.config is not None
            or args.max_context_tokens is not None
            or args.max_new_tokens is not None
        ):
            raise ValueError("frozen paper runs cannot override adapter or risk config")
        check_files(args.prepared, frozen.manifest["source_file_hashes"])
        args.adapter, args.config = frozen.path / "adapter", frozen.path / "paper-config.json"
        values = dict(frozen.manifest["research_settings"])
        values["context"] = ContextConfig(**values["context"])
        settings = ResearchSettings(**values)
    elif args.adapter is None or args.config is None:
        raise ValueError("provide --release, or both --adapter and --config for an experiment")
    if settings is None:
        from agent.research import ResearchSettings

        settings = ResearchSettings(
            max_context_tokens=args.max_context_tokens or 4096,
            max_new_tokens=args.max_new_tokens or 512,
        )
    if args.submit_paper and frozen is None:
        from agent.release import verify_trained_adapter

        verify_trained_adapter(args.adapter)
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
                from agent.notebook_api import NotebookSession, run_trading_loop

                loop = PaperTradingLoop(
                    policy, executor, source, memory=memory, frozen_release=frozen
                )
                session = NotebookSession(policy, config.universe, source, paper_loop=loop)
                if box.db.execute(
                    "SELECT 1 FROM research_cycles WHERE cycle_id=?", (args.cycle_id,)
                ).fetchone():
                    raise ValueError(
                        "legacy unsuffixed cycle ID already exists; "
                        "reconcile its saved cycle before starting a new session"
                    )
                ids = paper_cycle_ids(args.cycle_id, args.iterations)
                log_dir = args.log_dir or args.outbox.parent / "trading_logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                day = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
                with (log_dir / f"{day}.log").open("a") as log:
                    original_stdout = sys.stdout

                    class Tee:
                        def write(self, text):
                            original_stdout.write(text)
                            log.write(text)

                        def flush(self):
                            original_stdout.flush()
                            log.flush()

                    with contextlib.redirect_stdout(Tee()):
                        results = run_trading_loop(
                            loop,
                            None,
                            session,
                            memory=memory,
                            n_iterations=args.iterations,
                            cycle_ids=ids,
                            submit=args.submit_paper,
                            interval_seconds=args.interval,
                            collector=_CycleLog(),
                        )
                        if not results or not any(
                            item["status"] == "completed" for item in results
                        ):
                            return 2
    finally:
        box.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
