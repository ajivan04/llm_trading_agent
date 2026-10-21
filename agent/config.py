"""Validated immutable configuration for research and paper-trading runs."""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Mapping

from .contracts import FrozenDict, JsonContract, content_hash

POLICY_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


@dataclass(frozen=True)
class Config(JsonContract):
    model_name: str = POLICY_MODEL
    initial_cash: float = 100_000.0
    universe: tuple[str, ...] = ("AAPL", "MSFT", "SPY")
    sector_map: tuple[tuple[str, str], ...] = (
        ("AAPL", "technology"), ("MSFT", "technology"), ("SPY", "broad"))
    max_shares: int = 50
    max_position_weight: float = 0.10
    max_sector_weight: float = 0.30
    max_gross_weight: float = 0.80
    cash_reserve_weight: float = 0.20
    max_drawdown: float = 0.10
    slippage_bps: float = 5.0
    news_ttl_hours: int = 24
    memory_top_k: int = 2
    max_tool_turns: int = 6
    max_evidence_age_hours: int = 96
    seed: int = 7
    prompt_version: str = "v1"
    paper_trading_only: bool = True
    cache_transformation_version: str = "v1"

    def __post_init__(self) -> None:
        if self.model_name != POLICY_MODEL:
            raise ValueError("The assignment fixes the policy model to Qwen 2.5 1.5B Instruct")
        if self.paper_trading_only is not True:
            raise ValueError("Real-money trading is prohibited")
        universe = tuple(symbol.strip().upper() for symbol in self.universe)
        if not universe or any(not symbol for symbol in universe) or len(universe) != len(set(universe)):
            raise ValueError("universe must be nonempty and unique")
        object.__setattr__(self, "universe", universe)
        sectors = tuple((symbol.strip().upper(), sector.strip().lower()) for symbol, sector in self.sector_map)
        if any(not symbol or not sector for symbol, sector in sectors) or len(dict(sectors)) != len(sectors):
            raise ValueError("sector_map entries must be nonempty with unique symbols")
        if set(universe) - set(dict(sectors)):
            raise ValueError("Every universe symbol requires a sector mapping")
        object.__setattr__(self, "sector_map", sectors)
        if not isinstance(self.initial_cash, (int, float)) or isinstance(self.initial_cash, bool) or not math.isfinite(self.initial_cash) or self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive and finite")
        for name in ("max_position_weight", "max_sector_weight", "max_gross_weight",
                     "cash_reserve_weight", "max_drawdown"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be between zero and one")
        if self.max_position_weight > self.max_gross_weight or self.max_sector_weight > self.max_gross_weight:
            raise ValueError("Position and sector limits cannot exceed gross exposure")
        if not isinstance(self.slippage_bps, (int, float)) or isinstance(self.slippage_bps, bool) or not math.isfinite(self.slippage_bps) or not 0 <= self.slippage_bps < 10_000:
            raise ValueError("slippage_bps must be finite and below 10000")
        for name in ("max_shares", "news_ttl_hours", "memory_top_k", "max_tool_turns",
                     "max_evidence_age_hours"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        if not self.prompt_version.strip() or not self.cache_transformation_version.strip():
            raise ValueError("Version identifiers must be nonempty")

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> Config:
        if not isinstance(values, Mapping):
            raise TypeError("Config input must be a mapping")
        valid = {item.name for item in fields(cls)}
        unknown = set(values) - valid
        if unknown:
            raise ValueError(f"Unknown config fields: {sorted(unknown)}")
        payload = dict(values)
        if "universe" in payload:
            payload["universe"] = tuple(payload["universe"])
        if "sector_map" in payload:
            sector_map = payload["sector_map"]
            if isinstance(sector_map, Mapping):
                sector_map = sector_map.items()
            payload["sector_map"] = tuple(tuple(item) for item in sector_map)
        return cls(**payload)

    @property
    def config_hash(self) -> str:
        return content_hash(self)

    @property
    def sectors(self) -> FrozenDict:
        return FrozenDict(dict(self.sector_map))
