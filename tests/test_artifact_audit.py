import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent.audit_cli import audit_prepared
from agent.compact_cli import compact_artifact
from agent.compaction import CompactionConfig
from agent.contracts import canonical_json
from agent.evaluation import EvaluationOutcome
from agent.training import EpisodeExample
from agent.training.cli import main as training_main
from agent.training.validation import validate_sft_targets
from tests.test_compact_artifact import prepared_source


def example(citation="price"):
    return EpisodeExample(
        "AAPL:2026-01-05", "train", "2026-01-05T21:00:00Z", "2026-01-06T21:00:00Z",
        "snapshot", "account", "evidence",
        ({"role": "user", "content": canonical_json({
            "as_of": "2026-01-05T21:00:00Z",
            "evidence": [{"id": "price", "source": "test", "symbol": "AAPL",
                          "observed_at": "2026-01-05T20:00:00Z",
                          "available_at": "2026-01-05T20:00:00Z", "text": "price"}],
        })},),
        canonical_json({"decisions": [{"symbol": "AAPL", "action": "buy", "qty": 1,
                                       "reason": "price", "evidence_ids": [citation]}]}),
    )


class ArtifactAuditTests(unittest.TestCase):
    def test_repaired_short_reference_artifact_passes_offline_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            source, repaired = Path(directory) / "source", Path(directory) / "repaired"
            prepared_source(source)
            compact_artifact(source, repaired, config=CompactionConfig(max_filings=1),
                             short_evidence_ids=True)
            report = audit_prepared(repaired)
        self.assertTrue(report["target_contract_passed"])
        for split in ("train", "validation", "test"):
            self.assertGreater(report["splits"][split]["examples"], 0)
            self.assertEqual(report["splits"][split]["invalid_targets"], 0)

    def test_target_contract_checks_actual_visible_ids(self):
        validate_sft_targets([example()])
        with self.assertRaisesRegex(ValueError, "repair prompt/target"):
            validate_sft_targets([example("archive-only")])

    def test_target_rejects_prompt_cutoff_mismatch(self):
        with self.assertRaisesRegex(ValueError, "cutoff differs"):
            validate_sft_targets([replace(example(), decision_at="2026-01-05T20:30:00Z")])

    def test_audit_reports_invalid_counts_without_exposing_generations(self):
        dataset = SimpleNamespace(
            manifest=SimpleNamespace(dataset_hash="hash"),
            examples=(example("archive-only"),),
        )
        with patch("agent.audit_cli.load_dataset_artifact", return_value=dataset):
            result = audit_prepared(Path("unused"))
        self.assertFalse(result["target_contract_passed"])
        self.assertEqual(result["splits"]["train"]["invalid_targets"], 1)
        self.assertEqual(result["splits"]["train"]["failure_counts"],
                         {"unavailable_evidence": 1})
        self.assertNotIn("archive-only", json.dumps(result))

    def test_invalid_target_stops_cli_before_model_loading(self):
        row = asdict(example("archive-only"))
        with (patch("agent.training.cli._jsonl", return_value=[row]),
              patch("agent.training.cli._actual_data_hashes", return_value=("hash",)),
              patch("agent.training.cli._verify_sft_manifest", return_value="verified"),
              patch("agent.training.cli.train_sft") as train):
            with self.assertRaisesRegex(ValueError, "invalid supervised target"):
                training_main([
                    "sft", "--train", "train.jsonl", "--validation", "validation.jsonl",
                    "--output", "unused", "--prompt-version", "test",
                ])
            train.assert_not_called()

    def test_saved_prediction_audit_binds_dataset_and_records_file_identity(self):
        row = replace(example(), split="validation")
        dataset = SimpleNamespace(manifest=SimpleNamespace(dataset_hash="hash"), examples=(row,))
        outcome = EvaluationOutcome(row.example_id, "validation", "AAPL", ("price",),
                                    {"buy": 0.02, "hold": 0, "sell": -0.02}, "buy")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "predictions.jsonl"
            path.write_text(canonical_json({"example_id": row.example_id,
                                            "generation": row.completion}) + "\n")
            metadata = root / "evaluation.json"
            metadata.write_text(canonical_json({
                "split": "validation", "config": {"dataset_hash": "hash", "adapter_hash": "a"},
            }))
            with (patch("agent.audit_cli.load_dataset_artifact", return_value=dataset),
                  patch("agent.audit_cli.load_outcomes", return_value=(outcome,))):
                report = audit_prepared(root, predictions=path)
                details = report["validation_predictions"]
                self.assertEqual(len(details["file_sha256"]), 64)
                self.assertEqual(details["provenance"]["config"]["adapter_hash"], "a")
                metadata.write_text(canonical_json({
                    "split": "validation", "config": {"dataset_hash": "stale"},
                }))
                with self.assertRaisesRegex(ValueError, "different dataset identity"):
                    audit_prepared(root, predictions=path)

    def test_ppo_environment_preflight_precedes_gpu_model_loading(self):
        with (patch("agent.training.cli._actual_data_hashes", return_value=("hash",)),
              patch("agent.training.cli._environment", side_effect=ValueError("bad replay")),
              patch("agent.training.cli.QwenPPOOptimizer.from_pretrained") as load):
            with self.assertRaisesRegex(ValueError, "bad replay"):
                training_main([
                    "ppo", "--adapter-checkpoint", "existing-sft", "--output", "unused",
                    "--prompt-version", "test", "--updates", "10",
                    "--environment-factory",
                    "agent.training.environment:create_training_environment",
                ])
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
