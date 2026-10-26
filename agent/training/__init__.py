"""Training data, validation, and optimization primitives."""

from .core import (
    MODEL_NAME,
    DPOCandidate,
    DPOPair,
    EpisodeExample,
    PPOTransition,
    PPORollout,
    RewardComponents,
    SFTExample,
    TrainingManifest,
    build_dpo_pair,
    build_sft_example,
    compute_gae,
    ppo_objective,
    validate_chronological_splits,
    validate_fresh_rollout,
)

__all__ = [
    "MODEL_NAME", "DPOCandidate", "DPOPair", "EpisodeExample",
    "PPOTransition", "PPORollout", "RewardComponents", "SFTExample",
    "TrainingManifest", "build_dpo_pair", "build_sft_example",
    "compute_gae", "ppo_objective", "validate_chronological_splits",
    "validate_fresh_rollout",
]

