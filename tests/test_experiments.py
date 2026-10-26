import tempfile
import unittest
from pathlib import Path

from agent.evaluation import (
    ArtifactManifest, ComparisonPlan, ExperimentRegistry, ExperimentSpec,
    artifact_hash, evaluate_variant, run_variant, verify_artifact,
)


def spec(name="sft", method="sft", **changes):
    values = dict(
        name=name, method=method, dataset_hash="data", split_manifest_hash="splits",
        simulator_hash="sim", prompt_hash="prompt", tool_schema_hash="tools",
        feature_version="features", retrieval_version="bm25",
        risk_config_hash="risk", execution_config_hash="execution",
        seed_budget=(7,), evidence_token_budget=4096, sample_budget=100,
        compute_budget_steps=10,
    )
    values.update(changes)
    return ExperimentSpec(**values)


class ExperimentContractTests(unittest.TestCase):
    def test_comparison_rejects_unfair_budget_or_data_changes(self):
        ComparisonPlan("fair", (spec("base", "base"), spec()))
        with self.assertRaisesRegex(ValueError, "sample_budget"):
            ComparisonPlan("unfair", (spec("base", "base"), spec(sample_budget=101)))
        with self.assertRaisesRegex(ValueError, "dataset_hash"):
            ComparisonPlan("unfair", (spec("base", "base"), spec(dataset_hash="other")))

    def test_artifact_hash_detects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint"
            path.mkdir()
            (path / "adapter.safetensors").write_bytes(b"adapter")
            digest = artifact_hash(path)
            manifest = ArtifactManifest(
                spec().spec_hash, "code", "lock", "model", "tokenizer",
                digest, digest, None, 1.0, 100, {"loss": 1.0},
            )
            verify_artifact(path, manifest)
            (path / "adapter.safetensors").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_artifact(path, manifest)

    def test_variant_records_evaluation_with_artifact_lineage(self):
        item = spec()
        manifest = ArtifactManifest(
            item.spec_hash, "code", "lock", "model", "tokenizer",
            "adapter", "checkpoint", None, 1.0, None, {"loss": 0.5},
        )
        attempt = run_variant(
            item, split="validation", trainer=lambda _spec: manifest,
            evaluator=lambda _spec, _artifact, split: {"split": split, "return": 0.1},
            registry=ExperimentRegistry(),
        )
        self.assertEqual(attempt.status, "completed")
        self.assertEqual(attempt.metrics["split"], "validation")
        with self.assertRaisesRegex(ValueError, "frozen artifact"):
            run_variant(
                item, split="test", trainer=lambda _spec: manifest,
                evaluator=lambda *_args: {}, registry=ExperimentRegistry(),
            )
        frozen = evaluate_variant(
            item, artifact=manifest, split="test",
            evaluator=lambda _spec, _artifact, split: {"split": split},
            registry=ExperimentRegistry(),
        )
        self.assertEqual(frozen.metrics["split"], "test")


if __name__ == "__main__":
    unittest.main()
