"""Create a nominal-share source bundle without overwriting archived Yahoo data."""

import argparse
import json
import shutil
import tempfile
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from agent.artifacts import atomic_json, check_files, file_hash
from agent.contracts import canonical_json, content_hash
from agent.data.prices import DailyPriceBar
from agent.datasets import load_dataset_artifact
from agent.price_basis import normalize_bundle
from agent.trajectory_cli import _rows


def normalize_source(source, output, *, tail_fetcher=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents or output.exists():
        raise ValueError("normalization requires a new nonnested output directory")
    names = [
        "price-bars.jsonl",
        "evidence.jsonl",
        "preparation-manifest.json",
        *(
            "dataset/" + n
            for n in (
                "manifest.json",
                "examples.jsonl",
                "train.jsonl",
                "validation.jsonl",
                "test.jsonl",
            )
        ),
    ]
    inputs = {n: file_hash(source / n) for n in names}
    artifact = load_dataset_artifact(source / "dataset")
    prep = json.loads((source / "preparation-manifest.json").read_text())
    raw = _rows(source / "price-bars.jsonl")
    if prep["source_hash"] != content_hash(tuple(raw)):
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
        (stage / "dataset").mkdir()
        for name in names:
            if name.startswith("dataset/"):
                shutil.copy2(source / name, stage / name)
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
        updated = {
            **prep,
            "dataset_hash": artifact.manifest.dataset_hash,
            "source_preparation_dataset_hash": prep["dataset_hash"],
            "source_artifact_hash": prep.get("artifact_hash"),
            "upstream_counts": prep.get("counts"),
            "source_hash": content_hash(rows),
            "archive_kind": "nominal-research-source-v1",
            "price_basis_verified": True,
            "old_training_targets_usable": False,
            "counts": {"evidence": len(retained), "ppo_train": 0, "dpo_train": 0},
            "artifact_hash": content_hash(summary),
        }
        atomic_json(stage / "preparation-manifest.json", updated)
        check_files(source, inputs)
        load_dataset_artifact(stage / "dataset")
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
