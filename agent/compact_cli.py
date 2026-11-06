"""Offline compaction of prepared policy prompts without provider calls."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from agent.compaction import COMPACTION_VERSION, CompactionConfig, compact_evidence
from agent.contracts import Evidence, canonical_json, content_hash
from agent.datasets import load_dataset_artifact
from agent.policy import parse_decisions
from agent.policy.evidence import DECISION_INSTRUCTIONS, alias_messages, visible_evidence
from agent.preferences import CandidateOutcome, build_audited_dpo_pair
from agent.training import DPOCandidate

_REQUIRED_JSONL = (
    "dataset/examples.jsonl", "dataset/train.jsonl", "dataset/validation.jsonl",
    "dataset/test.jsonl", "dpo-train.jsonl", "dpo-validation.jsonl",
    "ppo-train.jsonl", "outcome-audit.jsonl", "price-bars.jsonl", "evidence.jsonl",
)
_REQUIRED_JSON = ("dataset/manifest.json", "preparation-manifest.json", "ppo-environment.json")


def compact_messages(
    messages: Any, *, config: CompactionConfig | None = None,
    include_evidence_index: bool = True,
) -> Any:
    if not isinstance(messages, list):
        return messages
    config = config or CompactionConfig()
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
                        raise ValueError("policy evidence records must be objects")
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
                else:
                    payload.pop("evidence_index", None)
                item["content"] = canonical_json(payload)
        output.append(item)
    return output


def compact_prompt(
    prompt: Any, *, config: CompactionConfig | None = None,
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
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"incomplete or invalid JSON at {path.name}:{number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected an object at {path.name}:{number}")
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _instructions(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = [dict(item) for item in messages]
    if result and result[0].get("role") == "system":
        if DECISION_INSTRUCTIONS not in result[0].get("content", ""):
            result[0]["content"] += "\n" + DECISION_INSTRUCTIONS
    else:
        result.insert(0, {"role": "system", "content": DECISION_INSTRUCTIONS})
    return result


def _repair_response(raw: str, *, source_ids: set[str], visible_ids: set[str],
                     symbol: str, full_to_alias: dict[str, str],
                     visible_sources: tuple[str, ...] = (),
                     repair_log: dict[str, Any] | None = None) -> str:
    # Validate BEFORE filtering: an unknown/hallucinated ID must not disappear
    # into a seemingly valid repaired target.
    decisions = parse_decisions(raw, allowed_symbols=(symbol,), allowed_evidence_ids=source_ids)
    rows = []
    removed = 0
    rewritten = False
    for decision in decisions:
        row = decision.to_dict()
        kept = [eid for eid in decision.evidence_ids if eid in visible_ids]
        if decision.evidence_ids and not kept:
            raise ValueError("target has no supporting evidence remaining after compaction")
        removed_count = len(decision.evidence_ids) - len(kept)
        removed += removed_count
        if removed_count:
            # A rationale may contain details supported only by an omitted
            # filing/article. Do not preserve those claims after hiding the
            # evidence or invent a new economic rationale for the same label.
            row["reason"] = "Historical supervised action label; selected evidence sources: " + (
                ", ".join(sorted(set(visible_sources))) or "supplied records"
            ) + "."
            rewritten = True
        row["evidence_ids"] = [full_to_alias.get(eid, eid) for eid in kept]
        rows.append(row)
    emitted_ids = {full_to_alias.get(eid, eid) for eid in visible_ids}
    response = canonical_json({"decisions": rows})
    parse_decisions(response, allowed_symbols=(symbol,), allowed_evidence_ids=emitted_ids)
    if repair_log is not None:
        repair_log.update({
            "source_completion_hash": content_hash(raw),
            "citations_removed": removed, "reason_rewritten": rewritten,
        })
    return response


def _outcome(row: dict[str, Any]) -> CandidateOutcome:
    payload = dict(row)
    payload["candidate"] = DPOCandidate(**payload["candidate"])
    return CandidateOutcome(**payload)


def _repair_dpo(row: dict[str, Any], *, examples: dict[str, dict[str, Any]],
                original_examples: dict[str, dict[str, Any]],
                audits: dict[str, dict[str, Any]],
                visible_before_alias: dict[str, set[str]],
                mappings: dict[str, dict[str, str]], expected_split: str,
                repair_notes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    chosen, rejected = _outcome(row["chosen_outcome"]), _outcome(row["rejected_outcome"])
    original_pair = build_audited_dpo_pair(chosen, rejected)
    audit_identity = {
        "state_hash": original_pair.pair.state_hash,
        "simulator_hash": chosen.simulator_hash, "horizon_end": chosen.horizon_end,
    }
    valid_hashes = {
        content_hash({**audit_identity, "outcomes": (chosen, rejected)}),
        content_hash({**audit_identity, "outcomes": (rejected, chosen)}),
    }
    if row.get("audit_hash") not in valid_hashes:
        raise ValueError("DPO audit_hash does not match the original audited candidates")
    # Earlier compactors changed pair.prompt without updating candidates or
    # hashes. Accept that known transformation only when it exactly matches the
    # verified dataset row. All audited responses/state remain independently checked.
    ids = {chosen.candidate.candidate_id.rsplit(":", 1)[0],
           rejected.candidate.candidate_id.rsplit(":", 1)[0]}
    if len(ids) != 1:
        raise ValueError("DPO alternatives do not identify the same episode")
    example_id = ids.pop()
    example, original = examples[example_id], original_examples[example_id]
    if example["split"] != expected_split:
        raise ValueError("DPO example has the wrong split")
    pair = row["pair"]
    if pair["prompt"] not in {
        original_pair.pair.prompt, canonical_json(original["messages"]),
    }:
        raise ValueError("DPO prompt does not match its source state")
    for key in ("chosen", "rejected", "state_hash", "chosen_id", "rejected_id",
                "preference_margin", "split"):
        if pair.get(key) != getattr(original_pair.pair, key):
            raise ValueError(f"DPO {key} does not match its audited candidates")
    source_ids = set(audits[example_id]["evidence_ids"])
    mapping = mappings[example_id]
    reverse = {value: key for key, value in mapping.items()}
    prompt = canonical_json(example["messages"])
    rebuilt = []
    for candidate_outcome in (chosen, rejected):
        candidate = candidate_outcome.candidate
        if (candidate.snapshot_id != original["snapshot_id"]
                or candidate.account_state_hash != original["account_state_hash"]):
            raise ValueError("DPO candidate state does not match dataset")
        note = {}
        response = _repair_response(
            candidate.response, source_ids=source_ids,
            visible_ids=visible_before_alias[example_id], symbol=audits[example_id]["symbol"],
            full_to_alias=reverse,
            visible_sources=tuple(item["source"] for item in visible_evidence(
                example["messages"], expected_as_of=example["decision_at"],
            ).values()), repair_log=note,
        )
        repair_notes[example_id]["dpo"][candidate.candidate_id] = note
        rebuilt.append(replace(candidate_outcome, candidate=replace(
            candidate, prompt=prompt, response=response,
            evidence_hash=example["evidence_hash"],
        )))
    return asdict(build_audited_dpo_pair(*rebuilt))


def compact_artifact(source: Path, output: Path, *, config: CompactionConfig,
                     short_evidence_ids: bool = False) -> dict[str, Any]:
    """Create a new, independently verified artifact, without copying trained outputs."""
    source, output = source.resolve(), output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("input and output must be separate, non-nested directories")
    if output.exists():
        raise ValueError("output already exists; choose a new artifact directory")
    previous_compaction = source / "compaction-manifest.json"
    if previous_compaction.exists():
        previous = json.loads(previous_compaction.read_text(encoding="utf-8"))
        if previous.get("short_evidence_ids"):
            raise ValueError("compact from the full-ID source, not an already aliased artifact")
    # A complete source is required. Do not publish partial copies downloaded
    # while this command is running, or silently drop missing optional splits.
    source_files = (*_REQUIRED_JSONL, *_REQUIRED_JSON)
    hashes = {name: _digest(source / name) for name in source_files}
    source_dataset = load_dataset_artifact(source / "dataset")
    tables = {name: _read_jsonl(source / name) for name in _REQUIRED_JSONL}
    manifests = {name: json.loads((source / name).read_text(encoding="utf-8"))
                 for name in _REQUIRED_JSON}
    if not all(isinstance(value, dict) for value in manifests.values()):
        raise ValueError("prepared manifests must be JSON objects")
    audits = {row["example_id"]: row for row in tables["outcome-audit.jsonl"]}
    examples = tables["dataset/examples.jsonl"]
    if len(audits) != len(tables["outcome-audit.jsonl"]) or set(audits) != {
        row["example_id"] for row in examples
    }:
        raise ValueError("outcome audit must uniquely cover every dataset example")
    originals = {row["example_id"]: json.loads(canonical_json(row)) for row in examples}
    mappings: dict[str, dict[str, str]] = {}
    visible_before_alias: dict[str, set[str]] = {}
    repair_notes: dict[str, dict[str, Any]] = {}
    provenance = []
    for row in examples:
        eid = row["example_id"]
        audit = audits[eid]
        if audit["split"] != row["split"]:
            raise ValueError("outcome split differs from dataset example")
        original_visible = visible_evidence(
            row["messages"], expected_as_of=row["decision_at"],
        )
        if not set(original_visible) <= set(audit["evidence_ids"]):
            raise ValueError("source prompt evidence is outside source outcome provenance")
        messages = _instructions(compact_messages(
            row["messages"], config=config, include_evidence_index=False,
        ))
        selected = visible_evidence(messages, expected_as_of=row["decision_at"])
        source_ids = set(audit["evidence_ids"])
        if not set(selected) <= source_ids:
            raise ValueError("visible evidence is outside source outcome provenance")
        visible_before_alias[eid] = set(selected)
        mapping = {}
        if short_evidence_ids:
            messages, mapping = alias_messages(messages)
        mappings[eid] = mapping
        reverse = {value: key for key, value in mapping.items()}
        repair_notes[eid] = {"sft": {}, "dpo": {}}
        row["completion"] = _repair_response(
            row["completion"], source_ids=source_ids, visible_ids=set(selected),
            symbol=audit["symbol"], full_to_alias=reverse,
            visible_sources=tuple(item["source"] for item in selected.values()),
            repair_log=repair_notes[eid]["sft"],
        )
        row["messages"] = messages
        row["evidence_hash"] = content_hash(tuple(visible_evidence(messages).values()))
        provenance.append({
            "example_id": eid, "source_evidence_ids": audit["evidence_ids"],
            "visible_evidence_ids": list(visible_evidence(messages)),
            "alias_to_full": mapping,
            "source_prompt_hash": content_hash(originals[eid]["messages"]),
            "selected_full_evidence_hash": content_hash(tuple(selected.values())),
            "model_prompt_hash": content_hash(messages),
            "target_repairs": repair_notes[eid],
        })
    by_id = {row["example_id"]: row for row in examples}
    for name, split in (("dpo-train.jsonl", "train"), ("dpo-validation.jsonl", "validation")):
        tables[name] = [_repair_dpo(
            row, examples=by_id, original_examples=originals, audits=audits,
            visible_before_alias=visible_before_alias, mappings=mappings, expected_split=split,
            repair_notes=repair_notes,
        ) for row in tables[name]]
    seen_ppo = set()
    for row in tables["ppo-train.jsonl"]:
        eid = row["episode_id"]
        example = by_id[eid]
        if eid in seen_ppo or row["split"] != "train" or example["split"] != "train":
            raise ValueError("PPO rows must be unique train episodes")
        seen_ppo.add(eid)
        if row["messages"] != originals[eid]["messages"]:
            raise ValueError("PPO messages do not match verified source dataset")
        if set(row["evidence_ids"]) != set(audits[eid]["evidence_ids"]):
            raise ValueError("PPO evidence allowlist differs from source audit")
        row["messages"] = example["messages"]
        row["evidence_ids"] = list(visible_evidence(row["messages"]))
    if seen_ppo != {row["example_id"] for row in examples if row["split"] == "train"}:
        raise ValueError("PPO rows must cover the complete training split")
    for row in tables["outcome-audit.jsonl"]:
        row["evidence_ids"] = list(visible_evidence(by_id[row["example_id"]]["messages"]))
    manifest = manifests["dataset/manifest.json"]
    identity = {
        "examples": examples, "splits": manifest["split_example_ids"],
        "purged": manifest.get("purged_episode_ids", []),
        "source_hashes": manifest.get("source_hashes", []),
        "boundaries": (manifest["train_end"], manifest["validation_end"], manifest["test_end"]),
        "regime_id": manifest.get("regime_id", "chronological-v1"),
        "decision_session_filter_hash": manifest.get("decision_session_filter_hash", ""),
    }
    manifest["regime_id"] = identity["regime_id"]
    manifest["decision_session_filter_hash"] = identity["decision_session_filter_hash"]
    manifest["dataset_hash"] = content_hash(identity)
    # Ensure a growing/in-progress source transfer cannot publish a mixed artifact.
    if hashes != {name: _digest(source / name) for name in source_files}:
        raise ValueError("source artifact changed during compaction; finish transfer first")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        (staging / "dataset").mkdir()
        for name, rows in tables.items():
            if name.startswith("dataset/") and name != "dataset/examples.jsonl":
                rows = [row for row in examples if row["split"] == Path(name).stem]
            _write_jsonl(staging / name, rows)
        _write_jsonl(staging / "compaction-evidence-index.jsonl", provenance)
        (staging / "dataset/manifest.json").write_text(canonical_json(manifest) + "\n")
        environment = {"episodes_path": str(output / "ppo-train.jsonl")}
        (staging / "ppo-environment.json").write_text(canonical_json(environment) + "\n")
        if (source / "regime-plan.json").is_file():
            # Only this explicitly allowed planning file is copied. No adapter,
            # checkpoint, cache, evaluation or credentials enter the new artifact.
            shutil.copyfile(source / "regime-plan.json", staging / "regime-plan.json")
        load_dataset_artifact(staging / "dataset")
        generated_hashes = {str(path.relative_to(staging)): _digest(path)
                            for path in sorted(staging.rglob("*")) if path.is_file()}
        artifact_hash = content_hash({"dataset_hash": manifest["dataset_hash"],
                                      "files": generated_hashes})
        preparation = dict(manifests["preparation-manifest.json"])
        preparation.update({
            "artifact_hash": artifact_hash, "dataset_hash": manifest["dataset_hash"],
            "source_artifact_hash": preparation.get("artifact_hash"),
            "transformation": "compaction-contract-v3",
        })
        (staging / "preparation-manifest.json").write_text(canonical_json(preparation) + "\n")
        result = {
            "source_dataset_hash": source_dataset.manifest.dataset_hash,
            "source_hash": content_hash(hashes), "source_file_hashes": hashes,
            "dataset_hash": manifest["dataset_hash"], "artifact_hash": artifact_hash,
            "compaction_version": COMPACTION_VERSION,
            "contract_version": "compaction-contract-v3", "config": asdict(config),
            "short_evidence_ids": short_evidence_ids,
        }
        (staging / "compaction-manifest.json").write_text(canonical_json(result) + "\n")
        if output.exists():
            raise ValueError("output appeared during compaction; refusing overwrite")
        staging.rename(output)
        return result
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-news", type=int, default=8)
    parser.add_argument("--max-filings", type=int, default=3)
    parser.add_argument("--max-text-chars", type=int, default=1200)
    parser.add_argument("--short-evidence-ids", action="store_true",
                        help="Use short local evidence handles with an archived provenance map")
    args = parser.parse_args(argv)
    config = CompactionConfig(
        max_news=args.max_news, max_filings=args.max_filings,
        max_text_chars=args.max_text_chars,
    )
    result = compact_artifact(args.input, args.output, config=config,
                              short_evidence_ids=args.short_evidence_ids)
    print(json.dumps({"output": str(args.output), "dataset_hash": result["dataset_hash"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
