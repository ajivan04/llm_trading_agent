"""Deterministic replay evaluation, benchmarks, and experiment controls."""

from .benchmarks import MarketSession, benchmark_curves
from .metrics import block_bootstrap_return_ci, exposure_match_returns, financial_metrics
from .registry import ExperimentRegistry, ExperimentRunner
from .splits import EpisodeWindow, Fold, purged_split, rolling_folds

__all__ = [
    "EpisodeWindow", "ExperimentRegistry", "ExperimentRunner", "Fold", "MarketSession",
    "benchmark_curves", "block_bootstrap_return_ci", "exposure_match_returns",
    "financial_metrics", "purged_split", "rolling_folds",
]

