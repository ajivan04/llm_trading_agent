"""Audited same-state DPO preferences from independent simulator resets."""
from __future__ import annotations

import math
from dataclasses import dataclass

from agent.contracts import content_hash, normalize_timestamp
from agent.training import DPOCandidate, DPOPair, build_dpo_pair


@dataclass(frozen=True)
class CandidateOutcome:
    candidate: DPOCandidate
    reset_id: str
    simulator_hash: str
    horizon_end: str
    net_return: float
    execution_costs: float
    risk_penalty: float = 0.0

    def __post_init__(self) -> None:
        if not self.reset_id.strip() or not self.simulator_hash.strip():
            raise ValueError("reset_id and simulator_hash must be nonempty")
        object.__setattr__(self, "horizon_end", normalize_timestamp(self.horizon_end))
        for name in ("net_return", "execution_costs", "risk_penalty"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.execution_costs < 0 or self.risk_penalty < 0:
            raise ValueError("costs and risk penalties must be nonnegative")
        expected = self.net_return - self.risk_penalty
        if not math.isclose(self.candidate.objective, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("candidate objective must equal net return minus risk penalty")


@dataclass(frozen=True)
class AuditedDPOPair:
    pair: DPOPair
    audit_hash: str
    chosen_outcome: CandidateOutcome
    rejected_outcome: CandidateOutcome


def build_audited_dpo_pair(
    first: CandidateOutcome, second: CandidateOutcome, *, minimum_margin: float = 0.0,
) -> AuditedDPOPair:
    if first.reset_id == second.reset_id:
        raise ValueError("DPO alternatives require independent simulator reset IDs")
    if first.simulator_hash != second.simulator_hash:
        raise ValueError("DPO alternatives require the same simulator")
    if first.horizon_end != second.horizon_end:
        raise ValueError("DPO alternatives require the same outcome horizon")
    pair = build_dpo_pair(first.candidate, second.candidate, minimum_margin=minimum_margin)
    chosen, rejected = (
        (first, second)
        if pair.chosen_id == first.candidate.candidate_id else (second, first)
    )
    audit_hash = content_hash({
        "state_hash": pair.state_hash, "simulator_hash": first.simulator_hash,
        "horizon_end": first.horizon_end, "outcomes": (first, second),
    })
    return AuditedDPOPair(pair, audit_hash, chosen, rejected)
