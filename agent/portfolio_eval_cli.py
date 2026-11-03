"""Run a sequential $100,000 portfolio replay from model predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent.evaluation.portfolio import replay_predictions, write_result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-cash", type=float, default=100_000.0)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--max-qty", type=int, default=50)
    args = parser.parse_args(argv)
    result = replay_predictions(
        args.prepared, args.predictions, split=args.split,
        initial_cash=args.initial_cash, cost_bps=args.cost_bps, max_qty=args.max_qty,
    )
    write_result(result, args.output)
    print(json.dumps({"output": str(args.output), "metrics": result.metrics}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
