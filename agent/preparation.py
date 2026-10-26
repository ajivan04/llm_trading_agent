"""Prepare verified SFT, DPO, and PPO inputs from historical daily bars."""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from agent.contracts import Decision, canonical_json, content_hash
from agent.data.prices import DailyPriceBar
from agent.datasets import DatasetArtifact, HistoricalDatasetBuilder, HistoricalEpisode
from agent.features import FeatureParameters, PriceFeatureBuilder
from agent.preferences import AuditedDPOPair, CandidateOutcome, build_audited_dpo_pair
from agent.snapshots import SnapshotBuilder
from agent.training import DPOCandidate


@dataclass(frozen=True)
class PreparationSettings:
    train_end: str
    validation_end: str
    test_end: str
    initial_cash: float = 100_000.0
    starting_qty: int = 5
    trade_qty: int = 5
    max_qty: int = 50
    cost_bps: float = 5.0
    minimum_preference_margin: float = 1e-7
    universe_version: str = "assignment-baseline-v1"

    def __post_init__(self) -> None:
        for name in ("initial_cash", "cost_bps", "minimum_preference_margin"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric")
            if not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.initial_cash <= 0 or self.cost_bps >= 10_000:
            raise ValueError("initial_cash/cost_bps are invalid")
        for name in ("starting_qty", "trade_qty", "max_qty"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.trade_qty > min(self.starting_qty, self.max_qty):
            raise ValueError("trade_qty cannot exceed starting_qty or max_qty")


@dataclass(frozen=True)
class PreparedArtifacts:
    dataset: DatasetArtifact
    dpo: Mapping[str, tuple[AuditedDPOPair, ...]]
    outcome_audit: tuple[Mapping[str, object], ...]
    ppo_train: tuple[Mapping[str, object], ...]
    source_rows: tuple[Mapping[str, object], ...]
    source_hash: str
    settings_hash: str

    @property
    def artifact_hash(self) -> str:
        return content_hash({
            "dataset_hash": self.dataset.manifest.dataset_hash,
            "dpo": self.dpo,
            "outcome_audit": self.outcome_audit,
            "ppo_train": self.ppo_train,
            "source_hash": self.source_hash,
            "settings_hash": self.settings_hash,
        })

    def write(self, output: Path) -> None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        self.dataset.write(output / "dataset")
        _write_jsonl(output / "price-bars.jsonl", self.source_rows)
        _write_jsonl(output / "dpo-train.jsonl", self.dpo.get("train", ()))
        _write_jsonl(output / "dpo-validation.jsonl", self.dpo.get("validation", ()))
        _write_jsonl(output / "outcome-audit.jsonl", self.outcome_audit)
        _write_jsonl(output / "ppo-train.jsonl", self.ppo_train)
        _atomic_write(output / "ppo-environment.json", canonical_json({
            "episodes_path": str((output / "ppo-train.jsonl").resolve()),
        }) + "\n")
        _atomic_write(output / "preparation-manifest.json", canonical_json({
            "artifact_hash": self.artifact_hash,
            "dataset_hash": self.dataset.manifest.dataset_hash,
            "source_hash": self.source_hash,
            "settings_hash": self.settings_hash,
            "counts": {
                "sft": {
                    split: len(self.dataset.manifest.split_example_ids[split])
                    for split in ("train", "validation", "test")
                },
                "dpo_train": len(self.dpo.get("train", ())),
                "dpo_validation": len(self.dpo.get("validation", ())),
                "ppo_train": len(self.ppo_train),
            },
        }) + "\n")


def _atomic_write(path: Path, payload: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, values: Sequence[object]) -> None:
    _atomic_write(path, "".join(canonical_json(value) + "\n" for value in values))


def _completion(
    symbol: str, action: str, qty: int, evidence_ids: tuple[str, ...],
) -> str:
    return canonical_json({"decisions": [Decision(
        symbol=symbol, action=action, qty=qty,
        reason=(
            "Counterfactual expert label selected after modeled costs; the policy input "
            "contains only point-in-time price and feature evidence."
        ),
        evidence_ids=evidence_ids,
    ).to_dict()]})


def _objectives(
    current_close: float, next_open: float, outcome_close: float, *,
    settings: PreparationSettings,
) -> dict[str, float]:
    equity = settings.initial_cash + settings.starting_qty * current_close
    rate = settings.cost_bps / 10_000
    quantity = settings.trade_qty
    return {
        "buy": quantity * (outcome_close - next_open * (1 + rate)) / equity,
        "sell": quantity * (next_open * (1 - rate) - outcome_close) / equity,
        "hold": 0.0,
    }


def _candidate_outcome(
    example, episode: HistoricalEpisode, prompt: str, objectives: Mapping[str, float],
    next_open: float, action: str, simulator_hash: str, settings: PreparationSettings,
) -> CandidateOutcome:
    response = _completion(
        episode.decisions[0].symbol, action,
        0 if action == "hold" else settings.trade_qty,
        episode.snapshot.evidence_ids,
    )
    candidate = DPOCandidate(
        candidate_id=f"{example.example_id}:{action}", split=example.split,
        prompt=prompt, response=response, snapshot_id=example.snapshot_id,
        account_state_hash=example.account_state_hash, evidence_hash=example.evidence_hash,
        objective=objectives[action],
    )
    return CandidateOutcome(
        candidate=candidate, reset_id=f"{example.example_id}:reset:{action}",
        simulator_hash=simulator_hash, horizon_end=example.outcome_at,
        net_return=objectives[action],
        execution_costs=(
            0.0 if action == "hold"
            else settings.trade_qty * next_open * settings.cost_bps / 10_000
        ),
    )


def prepare_training_artifacts(
    bars_by_symbol: Mapping[str, Sequence[DailyPriceBar]], *,
    settings: PreparationSettings,
    next_session_resolver: Callable[[str], str],
    feature_parameters: FeatureParameters | None = None,
) -> PreparedArtifacts:
    """Build method inputs while keeping future prices outside observations."""
    if not bars_by_symbol:
        raise ValueError("at least one symbol is required")
    feature_builder = PriceFeatureBuilder(feature_parameters)
    snapshot_builder = SnapshotBuilder()
    episodes: list[HistoricalEpisode] = []
    outcomes: dict[str, tuple[dict[str, float], float, float]] = {}
    source_rows: list[Mapping[str, object]] = []

    for raw_symbol, raw_bars in sorted(bars_by_symbol.items()):
        symbol = raw_symbol.strip().upper()
        if not symbol or symbol != raw_symbol:
            raise ValueError("bar keys must be normalized uppercase symbols")
        bars = tuple(sorted(raw_bars, key=lambda item: item.session))
        if len(bars) < 2 or any(item.symbol != symbol for item in bars):
            raise ValueError(f"{symbol} requires at least two matching bars")
        if len({item.session for item in bars}) != len(bars):
            raise ValueError(f"{symbol} contains duplicate sessions")
        source_rows.extend(asdict(item) for item in bars)
        history = []
        for current, following in zip(bars, bars[1:], strict=False):
            if next_session_resolver(current.session) != following.session:
                raise ValueError(
                    f"{symbol} has a missing or nonconsecutive session after {current.session}"
                )
            current_evidence = current.to_evidence()
            history.append(current_evidence)
            feature = feature_builder.build(
                history, symbol=symbol, as_of=current.close_at,
            )
            evidence = (current_evidence, feature)
            evidence_ids = tuple(item.id for item in evidence)
            account = {
                "cash": settings.initial_cash,
                "equity": settings.initial_cash + settings.starting_qty * current.close,
                "positions": {symbol: settings.starting_qty},
                "paper_only": True,
            }
            snapshot = snapshot_builder.build(
                as_of=current.close_at, universe=(symbol,),
                universe_version=settings.universe_version, evidence=evidence,
                compact_evidence=({
                    "evidence_id": feature.id, "symbol": symbol,
                    "features": feature.values["features"],
                },),
                feature_version=feature_builder.parameters.version,
                account_state=account, source_health={"yfinance": "historical_complete"},
                market_session_id=current.session,
            )
            objectives = _objectives(
                current.close, following.open, following.close, settings=settings,
            )
            action = max(("buy", "hold", "sell"), key=lambda item: (objectives[item], item))
            episode_id = f"{symbol}:{current.session}"
            episodes.append(HistoricalEpisode(
                episode_id=episode_id, snapshot=snapshot, evidence=evidence,
                decisions=(Decision(
                    symbol, action, 0 if action == "hold" else settings.trade_qty,
                    "Best training-period counterfactual after modeled costs.", evidence_ids,
                ),),
                outcome_at=following.close_at,
            ))
            outcomes[episode_id] = (objectives, following.open, following.close)

    dataset = HistoricalDatasetBuilder().build(
        episodes, train_end=settings.train_end,
        validation_end=settings.validation_end, test_end=settings.test_end,
    )
    episodes_by_id = {item.episode_id: item for item in episodes}
    dpo: dict[str, list[AuditedDPOPair]] = {"train": [], "validation": []}
    audits: list[Mapping[str, object]] = []
    ppo_rows: list[Mapping[str, object]] = []
    simulator_hash = content_hash({
        "name": "one-step-counterfactual-v1", "cost_bps": settings.cost_bps,
        "starting_qty": settings.starting_qty, "trade_qty": settings.trade_qty,
    })
    for example in dataset.examples:
        episode = episodes_by_id[example.example_id]
        objectives, next_open, outcome_close = outcomes[example.example_id]
        ordered = sorted(objectives, key=lambda action: (objectives[action], action))
        rejected_action, chosen_action = ordered[0], ordered[-1]
        margin = objectives[chosen_action] - objectives[rejected_action]
        prompt = canonical_json(example.messages)
        if margin >= settings.minimum_preference_margin and example.split in dpo:
            pair = build_audited_dpo_pair(
                _candidate_outcome(
                    example, episode, prompt, objectives, next_open,
                    chosen_action, simulator_hash, settings,
                ),
                _candidate_outcome(
                    example, episode, prompt, objectives, next_open,
                    rejected_action, simulator_hash, settings,
                ),
                minimum_margin=settings.minimum_preference_margin,
            )
            dpo[example.split].append(pair)
        audits.append({
            "example_id": example.example_id, "split": example.split,
            "symbol": episode.decisions[0].symbol,
            "evidence_ids": episode.snapshot.evidence_ids,
            "simulator_hash": simulator_hash, "horizon_end": example.outcome_at,
            "objectives": objectives, "chosen": chosen_action,
            "rejected": rejected_action, "preference_margin": margin,
        })
        if example.split == "train":
            ppo_rows.append({
                "episode_id": example.example_id, "split": "train",
                "decision_at": example.decision_at, "outcome_at": example.outcome_at,
                "messages": example.messages,
                "evidence_ids": episode.snapshot.evidence_ids,
                "symbol": episode.decisions[0].symbol,
                "next_open": next_open, "outcome_close": outcome_close,
                "starting_cash": settings.initial_cash,
                "starting_qty": settings.starting_qty, "max_qty": settings.max_qty,
                "cost_bps": settings.cost_bps, "benchmark_return": 0.0,
            })
    if not dataset.manifest.split_example_ids["train"]:
        raise ValueError("prepared dataset has no training examples")
    ordered_sources = tuple(sorted(
        source_rows, key=lambda row: (str(row["symbol"]), str(row["session"])),
    ))
    return PreparedArtifacts(
        dataset=dataset, dpo={name: tuple(rows) for name, rows in dpo.items()},
        outcome_audit=tuple(audits), ppo_train=tuple(ppo_rows),
        source_rows=ordered_sources, source_hash=content_hash(ordered_sources),
        settings_hash=content_hash(settings),
    )

