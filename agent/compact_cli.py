"""Offline compaction of prepared policy prompts without provider calls."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from agent.compaction import COMPACTION_VERSION, CompactionConfig, compact_evidence
from agent.contracts import Evidence, canonical_json, content_hash


def compact_messages(
    messages: Any, *, config: CompactionConfig = CompactionConfig(),
    include_evidence_index: bool = True,
) -> Any:
    if not isinstance(messages, list):
        return messages
    output = []
    for message in messages:
        item = dict(message)
        content = item.get("content")
        if item.get("role") == "user" and isinstance(content, str):
            try:
                payload = json.loads(content)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict) and isinstance(payload.get("evidence"), list):
                as_of = payload.get("as_of")
                if not isinstance(as_of, str):
                    raise ValueError("policy payload is missing an as_of timestamp")
                evidence = []
                for entry in payload["evidence"]:
                    if not isinstance(entry, dict):
                        continue
                    # Policy-visible historical projections intentionally omit
                    # retrieval/audit fields. Reconstruct the minimum contract
                    # without changing eligibility or inventing a later time.
                    normalized = dict(entry)
                    normalized.setdefault("retrieved_at", as_of)
                    normalized.setdefault("transformation_version", "policy-projection-v1")
                    evidence.append(Evidence(**normalized))
                compacted = list(compact_evidence(
                    evidence, as_of=as_of, config=config,
                ))
                payload["evidence"] = compacted
                # The complete index is audit metadata, not model evidence. The
                # CLI writes it to a sidecar so it cannot consume the context
                # window. Keep the option for callers/tests that need inline
                # provenance.
                if include_evidence_index:
                    payload["evidence_index"] = [
                        {
                            "id": item.id, "source": item.source, "symbol": item.symbol,
                            "available_at": item.available_at,
                        }
                        for item in evidence
                    ]
                item["content"] = canonical_json(payload)
        output.append(item)
    return output


def compact_prompt(
    prompt: Any, *, config: CompactionConfig = CompactionConfig(),
    include_evidence_index: bool = True,
) -> Any:
    if not isinstance(prompt, str):
        return prompt
    try:
        messages = json.loads(prompt)
    except json.JSONDecodeError:
        return prompt
    if isinstance(messages, list):
        return canonical_json(compact_messages(
            messages, config=config, include_evidence_index=include_evidence_index,
        ))
    return prompt


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")


def _compact_dataset(root: Path, *, config: CompactionConfig) -> str:
    dataset = root / "dataset"
    examples = _read_jsonl(dataset / "examples.jsonl")
    audit_index: list[dict[str, Any]] = []
    for row in examples:
        before = row.get("messages", [])
        for message_index, message in enumerate(before if isinstance(before, list) else []):
            if message.get("role") != "user" or not isinstance(message.get("content"), str):
                continue
            try:
                payload = json.loads(message["content"])
            except json.JSONDecodeError:
                continue
            entries = payload.get("evidence") if isinstance(payload, dict) else None
            if isinstance(entries, list):
                audit_index.append({
                    "example_id": row.get("example_id"),
                    "message_index": message_index,
                    "evidence": [
                        {"id": item.get("id"), "source": item.get("source"),
                         "symbol": item.get("symbol"), "available_at": item.get("available_at")}
                        for item in entries if isinstance(item, dict)
                    ],
                })
        row["messages"] = compact_messages(
            before, config=config, include_evidence_index=False,
        )
    _write_jsonl(dataset / "examples.jsonl", examples)
    for split in ("train", "validation", "test"):
        _write_jsonl(dataset / f"{split}.jsonl", [
            row for row in examples if row.get("split") == split
        ])
    manifest = json.loads((dataset / "manifest.json").read_text())
    identity = {
        "examples": examples, "splits": manifest["split_example_ids"],
        "purged": manifest.get("purged_episode_ids", []),
        "source_hashes": manifest.get("source_hashes", []),
        "boundaries": (manifest.get("train_end"), manifest.get("validation_end"),
                       manifest.get("test_end")),
    }
    for key in ("regime_id", "decision_session_filter_hash"):
        if key in manifest:
            identity[key] = manifest[key]
    manifest["dataset_hash"] = content_hash(identity)
    (dataset / "manifest.json").write_text(canonical_json(manifest), encoding="utf-8")
    (root / "compaction-evidence-index.jsonl").write_text(
        "".join(canonical_json(item) + "\n" for item in audit_index), encoding="utf-8"
    )
    return manifest["dataset_hash"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-news", type=int, default=8)
    parser.add_argument("--max-filings", type=int, default=3)
    parser.add_argument("--max-text-chars", type=int, default=1200)
    args = parser.parse_args(argv)
    if args.input.resolve() == args.output.resolve():
        raise ValueError("input and output must be different directories")
    shutil.copytree(args.input, args.output, dirs_exist_ok=True)
    config = CompactionConfig(
        max_news=args.max_news, max_filings=args.max_filings,
        max_text_chars=args.max_text_chars,
    )
    dataset_hash = _compact_dataset(args.output, config=config)
    for name in ("dpo-train.jsonl", "dpo-validation.jsonl"):
        rows = _read_jsonl(args.output / name)
        for row in rows:
            pair = row.get("pair")
            if isinstance(pair, dict) and "prompt" in pair:
                pair["prompt"] = compact_prompt(
                    pair["prompt"], config=config, include_evidence_index=False,
                )
        _write_jsonl(args.output / name, rows)
    ppo = _read_jsonl(args.output / "ppo-train.jsonl")
    for row in ppo:
        row["messages"] = compact_messages(
            row.get("messages", []), config=config, include_evidence_index=False,
        )
    _write_jsonl(args.output / "ppo-train.jsonl", ppo)
    environment_path = args.output / "ppo-environment.json"
    if environment_path.exists():
        environment = json.loads(environment_path.read_text())
        if isinstance(environment, dict):
            environment["episodes_path"] = str((args.output / "ppo-train.jsonl").resolve())
            environment_path.write_text(canonical_json(environment) + "\n", encoding="utf-8")
    source_hash = content_hash({"input": str(args.input.resolve()),
                                "dataset_hash": dataset_hash})
    (args.output / "compaction-manifest.json").write_text(canonical_json({
        "source": str(args.input.resolve()), "dataset_hash": dataset_hash,
        "compaction_version": COMPACTION_VERSION, "source_hash": source_hash,
    }) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "dataset_hash": dataset_hash}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
