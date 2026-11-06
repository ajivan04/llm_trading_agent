import json
import tempfile
import unittest
from pathlib import Path

from agent.compact_cli import compact_artifact, compact_messages, compact_prompt
from agent.compaction import CompactionConfig
from agent.contracts import Evidence, content_hash
from agent.datasets import load_dataset_artifact
from agent.policy import parse_decisions
from agent.policy.evidence import resolve_decisions, visible_evidence
from agent.preparation import PreparationSettings, prepare_training_artifacts
from agent.training.cli import _audited_dpo_pair
from tests.test_preparation import bar


def prepared_source(root):
    sessions = [
        ("2026-01-02", 100, 101), ("2026-01-05", 102, 103),
        ("2026-01-06", 104, 102), ("2026-01-07", 101, 100),
        ("2026-01-08", 99, 103), ("2026-01-09", 104, 105),
    ]
    bars = tuple(bar(day, f"{day}T14:30:00Z", f"{day}T21:00:00Z", opened, closed)
                 for day, opened, closed in sessions)
    filings = tuple(Evidence(
        id=f"sec-{index}", source="sec.filing_text", symbol="AAPL",
        observed_at=f"2026-01-01T{index + 10}:00:00Z",
        available_at=f"2026-01-01T{index + 10}:00:00Z",
        retrieved_at="2026-02-01T00:00:00Z", text=f"filing {index}",
        accession_id=f"accession-{index}", values={"form": "8-K"},
    ) for index in range(3))
    artifact = prepare_training_artifacts(
        {"AAPL": bars},
        settings=PreparationSettings(
            "2026-01-05T23:59:59Z", "2026-01-07T23:59:59Z",
            "2026-01-09T23:59:59Z", evidence_mode="sec",
        ),
        next_session_resolver={first.session: second.session
                               for first, second in zip(bars, bars[1:], strict=False)}.__getitem__,
        evidence_provider=lambda _symbol, _as_of: filings,
    )
    artifact.write(root)
    return artifact


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


class CompactArtifactTests(unittest.TestCase):
    def test_repairs_every_training_path_without_copying_models_or_results(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "source", Path(directory) / "output"
            original = prepared_source(source)
            (source / "sft").mkdir()
            (source / "sft/adapter_model.safetensors").write_bytes(b"old model")
            (source / "eval-sft").mkdir()
            (source / "eval-sft/evaluation.json").write_text('{"stale":true}')
            before = (source / "dataset/examples.jsonl").read_bytes()
            result = compact_artifact(source, output, config=CompactionConfig(max_filings=1))
            self.assertEqual(before, (source / "dataset/examples.jsonl").read_bytes())
            self.assertFalse((output / "sft").exists())
            self.assertFalse((output / "eval-sft").exists())
            artifact = load_dataset_artifact(output / "dataset")
            self.assertNotEqual(result["dataset_hash"], original.dataset.manifest.dataset_hash)
            for example in artifact.examples:
                ids = set(visible_evidence(example.messages))
                self.assertIn("sec-2", ids)
                self.assertNotIn("sec-0", ids)
                decision = parse_decisions(
                    example.completion, allowed_symbols=("AAPL",), allowed_evidence_ids=ids,
                )[0]
                self.assertEqual(set(decision.evidence_ids), ids)
            for name, split in (("dpo-train.jsonl", "train"),
                                ("dpo-validation.jsonl", "validation")):
                for row in jsonl(output / name):
                    pair = _audited_dpo_pair(row, split)
                    self.assertEqual(row["chosen_outcome"]["candidate"]["prompt"], pair.prompt)
                    self.assertEqual(row["chosen_outcome"]["candidate"]["response"], pair.chosen)
                    ids = set(visible_evidence(json.loads(pair.prompt)))
                    for response in (pair.chosen, pair.rejected):
                        parse_decisions(
                            response, allowed_symbols=("AAPL",), allowed_evidence_ids=ids,
                        )
                    self.assertNotEqual(row["audit_hash"], original.dpo[split][0].audit_hash)
            for row in jsonl(output / "ppo-train.jsonl"):
                self.assertEqual(set(row["evidence_ids"]), set(visible_evidence(row["messages"])))
            new_audits = jsonl(output / "outcome-audit.jsonl")
            self.assertEqual(len(new_audits), len(original.outcome_audit))
            for old, new in zip(original.outcome_audit, new_audits, strict=True):
                self.assertEqual(old["objectives"], new["objectives"])
                self.assertEqual(old["chosen"], new["chosen"])
                self.assertNotIn("sec-0", new["evidence_ids"])

    def test_short_handles_have_exact_provenance_and_valid_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "source", Path(directory) / "output"
            prepared_source(source)
            full_output = Path(directory) / "full"
            compact_artifact(source, full_output, config=CompactionConfig(max_filings=1))
            full_examples = {example.example_id: example
                             for example in load_dataset_artifact(full_output / "dataset").examples}
            compact_artifact(source, output, config=CompactionConfig(max_filings=1),
                             short_evidence_ids=True)
            provenance = jsonl(output / "compaction-evidence-index.jsonl")
            artifact = load_dataset_artifact(output / "dataset")
            for example, index in zip(artifact.examples, provenance, strict=True):
                mapping = index["alias_to_full"]
                self.assertIn("sec-2", mapping.values())
                self.assertNotIn("sec-0", mapping.values())
                self.assertEqual(set(mapping), set(visible_evidence(example.messages)))
                decisions = parse_decisions(example.completion, allowed_symbols=("AAPL",),
                                            allowed_evidence_ids=mapping)
                self.assertEqual(set(decisions[0].evidence_ids), set(mapping))
                resolved = resolve_decisions(
                    example.completion, messages=example.messages,
                    allowed_symbols=("AAPL",), aliases=mapping,
                    original_messages=full_examples[example.example_id].messages,
                )
                self.assertEqual(set(resolved[0].evidence_ids), set(mapping.values()))
            for row in jsonl(output / "dpo-train.jsonl"):
                pair = _audited_dpo_pair(row, "train")
                ids = visible_evidence(json.loads(pair.prompt))
                parse_decisions(pair.chosen, allowed_symbols=("AAPL",), allowed_evidence_ids=ids)

    def test_recovers_legacy_prompt_only_compaction_with_verified_original_audits(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "source", Path(directory) / "output"
            prepared_source(source)
            config = CompactionConfig(max_filings=1)
            examples = jsonl(source / "dataset/examples.jsonl")
            for row in examples:
                row["messages"] = compact_messages(
                    row["messages"], config=config, include_evidence_index=False,
                )
            for name, rows in (("examples", examples), *(
                (split, [row for row in examples if row["split"] == split])
                for split in ("train", "validation", "test")
            )):
                (source / f"dataset/{name}.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows))
            manifest = json.loads((source / "dataset/manifest.json").read_text())
            manifest["dataset_hash"] = content_hash({
                "examples": examples, "splits": manifest["split_example_ids"],
                "purged": manifest["purged_episode_ids"],
                "source_hashes": manifest["source_hashes"],
                "boundaries": (
                    manifest["train_end"], manifest["validation_end"], manifest["test_end"],
                ),
                "regime_id": manifest["regime_id"],
                "decision_session_filter_hash": manifest["decision_session_filter_hash"],
            })
            (source / "dataset/manifest.json").write_text(json.dumps(manifest))
            for name in ("dpo-train.jsonl", "dpo-validation.jsonl"):
                rows = jsonl(source / name)
                for row in rows:
                    row["pair"]["prompt"] = compact_prompt(
                        row["pair"]["prompt"], config=config, include_evidence_index=False,
                    )
                (source / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
            ppo = jsonl(source / "ppo-train.jsonl")
            for row in ppo:
                row["messages"] = compact_messages(
                    row["messages"], config=config, include_evidence_index=False,
                )
            (source / "ppo-train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in ppo))
            compact_artifact(source, output, config=config)
            for row in jsonl(output / "dpo-train.jsonl"):
                _audited_dpo_pair(row, "train")

    def test_rejects_tampered_audit_and_leaves_no_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "source", Path(directory) / "output"
            prepared_source(source)
            rows = jsonl(source / "dpo-train.jsonl")
            response = json.loads(rows[0]["chosen_outcome"]["candidate"]["response"])
            response["decisions"][0]["reason"] = "tampered rationale"
            rows[0]["chosen_outcome"]["candidate"]["response"] = json.dumps(response)
            (source / "dpo-train.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "audit_hash"):
                compact_artifact(source, output, config=CompactionConfig())
            self.assertFalse(output.exists())

    def test_existing_or_nested_output_is_rejected_without_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            prepared_source(source)
            for output in (source, source / "nested", source.parent):
                with self.assertRaises(ValueError):
                    compact_artifact(source, output, config=CompactionConfig())
            output = Path(directory) / "existing"
            output.mkdir()
            (output / "keep.txt").write_text("preserve")
            with self.assertRaisesRegex(ValueError, "already exists"):
                compact_artifact(source, output, config=CompactionConfig())
            self.assertEqual((output / "keep.txt").read_text(), "preserve")

    def test_truncated_source_does_not_publish_an_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "source", Path(directory) / "output"
            prepared_source(source)
            with (source / "dpo-train.jsonl").open("a") as handle:
                handle.write('{"unfinished":')
            with self.assertRaisesRegex(ValueError, "incomplete or invalid JSON"):
                compact_artifact(source, output, config=CompactionConfig())
            self.assertFalse(output.exists())
            self.assertEqual(list(output.parent.glob(".output-*")), [])

    def test_rejects_fabricated_citation_before_dropping_it(self):
        from agent.compact_cli import _repair_response

        raw = json.dumps({"decisions": [{
            "symbol": "AAPL", "action": "buy", "qty": 5, "reason": "unchanged",
            "evidence_ids": ["real", "invented"],
        }]})
        with self.assertRaisesRegex(ValueError, "unavailable evidence"):
            _repair_response(raw, source_ids={"real"}, visible_ids={"real"},
                             symbol="AAPL", full_to_alias={})

    def test_removing_support_also_removes_hidden_rationale_claims(self):
        from agent.compact_cli import _repair_response

        raw = json.dumps({"decisions": [{
            "symbol": "AAPL", "action": "buy", "qty": 5,
            "reason": "Hidden filing says revenue doubled to an unprecedented figure.",
            "evidence_ids": ["visible", "hidden"],
        }]})
        note = {}
        repaired = _repair_response(
            raw, source_ids={"visible", "hidden"}, visible_ids={"visible"}, symbol="AAPL",
            full_to_alias={}, visible_sources=("yfinance",), repair_log=note,
        )
        decision = json.loads(repaired)["decisions"][0]
        self.assertEqual((decision["action"], decision["qty"]), ("buy", 5))
        self.assertEqual(decision["evidence_ids"], ["visible"])
        self.assertNotIn("revenue doubled", repaired)
        self.assertIn("Historical supervised action label", decision["reason"])
        self.assertEqual(note["citations_removed"], 1)
        self.assertTrue(note["reason_rewritten"])
        self.assertEqual(note["source_completion_hash"], content_hash(raw))

    def test_removes_preexisting_inline_index(self):
        item = Evidence(
            id="visible", source="alpaca.news", symbol="AAPL", text="headline",
            observed_at="2026-01-01T10:00:00Z", available_at="2026-01-01T10:00:00Z",
            retrieved_at="2026-01-01T10:00:00Z",
        )
        messages = [{"role": "user", "content": json.dumps({
            "as_of": "2026-01-02T10:00:00Z", "evidence": [item.to_dict()],
            "evidence_index": [{"id": "not-policy-evidence"}],
        })}]
        compacted = compact_messages(messages, include_evidence_index=False)
        self.assertNotIn("evidence_index", json.loads(compacted[0]["content"]))


if __name__ == "__main__":
    unittest.main()
