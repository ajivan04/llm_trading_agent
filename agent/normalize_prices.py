"""Create a nominal-share source bundle without overwriting archived Yahoo data."""

import argparse
import shutil
import tempfile
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from agent.artifacts import atomic_json, check_files, file_hash
from agent.contracts import canonical_json, content_hash
from agent.data.prices import DailyPriceBar
from agent.price_basis import normalize_bundle
from agent.source_bundle import load_source, write_source
from agent.trajectory_cli import _rows


def normalize_source(source, output, *, tail_fetcher=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents or output.exists():
        raise ValueError("normalization requires a new nonnested output directory")
    metadata = load_source(source)
    inputs = metadata["file_hashes"]
    raw = _rows(source / "price-bars.jsonl")
    if metadata["price_hash"] is not None and metadata["price_hash"] != content_hash(tuple(raw)):
        raise ValueError("original price hash failed")
    grouped = defaultdict(list)
    for row in raw:
        row = {**row, "price_basis": row.get("price_basis", "yahoo-split-adjusted")}
        grouped[row["symbol"]].append(DailyPriceBar(**row))
    print(canonical_json({"event": "price_basis_tail_check", "symbols": len(grouped)}), flush=True)
    normalized, tails = normalize_bundle(grouped, fetcher=tail_fetcher)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="." + output.name + "-", dir=output.parent))
    try:
        rows = tuple(asdict(b) for symbol in sorted(normalized) for b in normalized[symbol])
        with (stage / "price-bars.jsonl").open("w") as handle:
            for row in rows:
                handle.write(canonical_json(row) + "\n")
        retained = [
            r
            for r in _rows(source / "evidence.jsonl")
            if r["source"] not in {"yfinance", "derived.price_features"}
        ]
        with (stage / "evidence.jsonl").open("w") as handle:
            for row in retained:
                handle.write(canonical_json(row) + "\n")
        summary = {
            "version": "nominal-share-reconstruction-v1",
            "input_file_hashes": inputs,
            "output_price_hash": file_hash(stage / "price-bars.jsonl"),
            "tail_actions": tails,
            "policy_sees_future_adjustment_factors": False,
            "provider_basis": "Yahoo split-adjusted OHLC/volume/dividends; auto_adjust=False",
            "volume_rounding": "nearest original-share integer",
            "old_targets_rebuilt": False,
            "source_originally_labeled_raw": any("price_basis" not in r for r in raw),
        }
        atomic_json(stage / "price-basis-manifest.json", summary)
        write_source(
            stage,
            metadata,
            universe=metadata["universe"] or tuple(sorted(grouped)),
            parent_source_id=metadata["source_id"],
        )
        check_files(source, inputs)
        if output.exists():
            raise ValueError("normalization output appeared during build")
        stage.rename(output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = normalize_source(args.source, args.output)
    print(canonical_json({"output": str(args.output), "version": result["version"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
