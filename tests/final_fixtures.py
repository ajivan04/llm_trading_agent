"""Synthetic release fixtures; scripted decisions are never empirical model results."""

from dataclasses import asdict, replace
from pathlib import Path

from agent.artifacts import atomic_json
from agent.config import Config
from agent.contracts import Evidence, content_hash
from agent.data.prices import DailyPriceBar
from agent.evaluation.research import evaluate_research_policy, write_research_result
from agent.policy.loader import (
    POLICY_MODEL,
    POLICY_REVISION,
    HuggingFaceQwenBackend,
    adapter_fingerprint,
)
from agent.preparation import PreparationSettings, prepare_training_artifacts
from agent.release import source_hashes
from agent.research import ResearchPolicy, ResearchSettings
from agent.trajectories import TrajectorySettings
from agent.trajectory_cli import _calendar
from tests.test_trajectories import Policy


def release_inputs(root):
    root = Path(root)
    calendar = _calendar()
    sessions = calendar.calendar.sessions_in_range("2026-05-01", "2026-10-30")
    bars = []
    for session in sessions:
        day = str(session.date())
        bounds = calendar.bounds(day)
        bars.append(
            DailyPriceBar(
                "AAPL",
                day,
                100,
                101,
                99,
                100,
                100,
                1000,
                0,
                0,
                bounds.open_at,
                bounds.close_at,
                "2026-11-08T00:00:00Z",
            )
        )

    def extras(symbol, cutoff):
        return tuple(
            Evidence(
                source + ":" + cutoff[:10],
                source,
                symbol,
                cutoff,
                cutoff,
                "2026-11-08T00:00:00Z",
                "Synthetic research text.",
            )
            for source in ("alpaca.news", "sec.filing_text")
        )

    prepared = root / "source"
    prepared_data = prepare_training_artifacts(
        {"AAPL": tuple(bars)},
        settings=PreparationSettings(
            "2026-09-01T00:00:00Z",
            "2026-10-24T00:00:00Z",
            "2026-11-03T23:59:59Z",
            evidence_mode="news-sec",
        ),
        evidence_provider=extras,
        next_session_resolver=lambda d: calendar.next(d).session,
    )
    prepared_data.write(prepared)
    adapter = root / "sft"
    adapter.mkdir()
    atomic_json(
        adapter / "adapter_config.json",
        {"base_model_name_or_path": POLICY_MODEL, "peft_type": "LORA", "r": 8},
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"synthetic-unit-test-weights")
    manifest = {
        "method": "sft",
        "model_name": POLICY_MODEL,
        "model_revision": POLICY_REVISION,
        "tokenizer_revision": POLICY_REVISION,
        "steps": 8,
        "adapter_config": {"supervision_mode": "trajectory-final"},
    }
    atomic_json(
        adapter / "training_manifest.json",
        {
            "manifest": manifest,
            "manifest_hash": content_hash(manifest),
            "metrics": {
                "status": "completed",
                "supervision_mode": "trajectory-final",
                "completed_steps": 8,
                "planned_steps": 8,
                "trainable_parameter_fraction": 0.0005,
                "inference_adapter_fingerprint": adapter_fingerprint(adapter),
            },
        },
    )
    research_settings = ResearchSettings()
    identity = ResearchPolicy(
        HuggingFaceQwenBackend(adapter=adapter), settings=research_settings
    ).policy_identity
    policy = Policy()
    policy.policy_identity = identity
    settings = TrajectorySettings(
        ("AAPL",), "2026-09-01T00:00:00Z", "2026-10-24T00:00:00Z", "2026-11-03T23:59:59Z"
    )
    validation = root / "validation"
    days = tuple(
        str(s.date()) for s in calendar.calendar.sessions_in_range("2026-09-25", "2026-10-22")
    )
    from agent.artifacts import file_hash

    result = evaluate_research_policy(
        {"AAPL": tuple(bars)},
        decide=policy,
        settings=settings,
        decision_sessions=days,
        next_session_resolver=lambda d: calendar.next(d).session,
        evidence_provider=extras,
        evidence_source_hash=file_hash(prepared / "evidence.jsonl"),
        journal=validation / "journal",
    )
    result = replace(
        result,
        protocol={
            **result.protocol,
            "source_file_hashes": source_hashes(prepared),
            "research_settings": asdict(research_settings),
            "research_policy_identity": identity,
            "diagnostic_missing_nlp_allowed": False,
            "memory": {"mode": "disabled"},
        },
    )
    write_research_result(result, validation)
    config = root / "config.json"
    atomic_json(config, asdict(Config(universe=("AAPL",), sector_map=(("AAPL", "unknown"),))))
    return {
        "adapter": adapter,
        "prepared": prepared,
        "validation": validation,
        "config": config,
    }, policy
