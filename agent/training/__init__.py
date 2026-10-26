"""Training data, validation, and optimization primitives."""

from .core import (
    MODEL_NAME,
    MODEL_REVISION,
    DPOCandidate,
    DPOPair,
    EpisodeExample,
    PPORollout,
    PPOTransition,
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
from .environment import HistoricalPPOEnvironment, ReplayEpisode, create_training_environment
from .execution import (
    LoRASettings,
    PPOEnvironment,
    PPOOptimizer,
    TrainerSettings,
    TrainingResult,
    smoke_test_adapter,
    train_dpo,
    train_ppo,
    train_sft,
)
from .qwen_ppo import QwenPPOOptimizer, QwenPPOSettings

__all__ = [
    "MODEL_NAME", "MODEL_REVISION", "DPOCandidate", "DPOPair", "EpisodeExample",
    "PPOTransition", "PPORollout", "RewardComponents", "SFTExample",
    "TrainingManifest", "build_dpo_pair", "build_sft_example",
    "compute_gae", "ppo_objective", "validate_chronological_splits",
    "validate_fresh_rollout", "LoRASettings", "PPOEnvironment", "PPOOptimizer",
    "TrainerSettings", "TrainingResult", "train_dpo", "train_ppo", "train_sft",
    "smoke_test_adapter",
    "QwenPPOOptimizer", "QwenPPOSettings",
    "HistoricalPPOEnvironment", "ReplayEpisode", "create_training_environment",
]

