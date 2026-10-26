"""Deterministic replay evaluation, benchmarks, and experiment controls."""

from .benchmarks import MarketSession, benchmark_curves
from .metrics import block_bootstrap_return_ci, exposure_match_returns, financial_metrics
from .model import EvaluationOutcome, load_outcomes, score_generations
from .registry import ExperimentRegistry, ExperimentRunner
from .experiments import (
    ArtifactManifest, ComparisonPlan, ExperimentSpec, artifact_hash,
    evaluate_variant, run_variant, verify_artifact,
)
from .splits import EpisodeWindow, Fold, purged_split, rolling_folds

__all__ = [
    "EpisodeWindow", "ExperimentRegistry", "ExperimentRunner", "Fold", "MarketSession",
    "benchmark_curves", "block_bootstrap_return_ci", "exposure_match_returns",
    "financial_metrics", "purged_split", "rolling_folds",
    "EvaluationOutcome", "load_outcomes", "score_generations",
    "ArtifactManifest", "ComparisonPlan", "ExperimentSpec", "artifact_hash",
    "evaluate_variant", "run_variant", "verify_artifact",
]

