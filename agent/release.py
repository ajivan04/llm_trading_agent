"""Freeze checked inference assets and authorize a fixed historical test schedule."""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

from agent.artifacts import atomic_json, check_files, file_hash
from agent.config import Config
from agent.contracts import canonical_json, content_hash
from agent.policy.loader import ADAPTER_FILES, POLICY_MODEL, POLICY_REVISION, adapter_fingerprint

EXECUTION_WEEK = ("2026-10-26", "2026-10-27", "2026-10-28", "2026-10-29", "2026-10-30")
DECISION_WEEK = ("2026-10-23", "2026-10-26", "2026-10-27", "2026-10-28", "2026-10-29")
LIVE_PAPER_WEEK = ("2026-11-09", "2026-11-10", "2026-11-11", "2026-11-12", "2026-11-13")


def code_hashes(root=None):
    root = Path(root or Path(__file__).parent.parent)
    files = sorted((root / "agent").rglob("*.py"))
    files += [root / name for name in ("trading_agent.py", "pyproject.toml", "uv.lock")]
    if any(p.is_symlink() for p in files):
        raise ValueError("frozen code cannot contain symlinks")
    return {str(p.relative_to(root)): file_hash(p) for p in files}


def source_hashes(prepared):
    names = (
        "price-bars.jsonl",
        "evidence.jsonl",
        "preparation-manifest.json",
        "dataset/manifest.json",
    )
    result = {name: file_hash(Path(prepared) / name) for name in names}
    for name in (
        "hydration-manifest.json",
        "compaction-manifest.json",
        "price-basis-manifest.json",
    ):
        if (Path(prepared) / name).is_file():
            result[name] = file_hash(Path(prepared) / name)
    return result


def freeze_release(*, adapter, prepared, validation, config, output):
    adapter, prepared, validation, output = map(Path, (adapter, prepared, validation, output))
    if output.exists():
        raise ValueError("release output already exists; releases are immutable")
    inference_hash = adapter_fingerprint(adapter)
    report = json.loads((validation / "research_manifest.json").read_text())
    metrics = json.loads((validation / "research_metrics.json").read_text())
    protocol = report["protocol"]
    if report["metrics_hash"] != content_hash(metrics) or report["protocol_hash"] != content_hash(
        protocol
    ):
        raise ValueError("validation summary checksums failed")
    if (
        protocol["split"] != "validation"
        or not metrics.get("complete_plan")
        or metrics.get("sessions", 0) < 20
        or metrics.get("valid_schema_rate", 0) < 0.95
        or metrics.get("executed_trades", 0) < 1
        or metrics.get("test_policy_calls") != 0
        or protocol.get("diagnostic_missing_nlp_allowed", True)
    ):
        raise ValueError(
            "release requires complete NLP validation, schema>=95%, and executed trades"
        )
    if protocol["policy_identity"].get("adapter_hash") != inference_hash:
        raise ValueError("validation did not evaluate the selected inference assets")
    hashes = source_hashes(prepared)
    if protocol.get("source_file_hashes") != hashes:
        raise ValueError("validation source differs from the frozen source")
    cfg = Config.from_dict(json.loads(Path(config).read_text()))
    settings = protocol["settings"]
    if (
        tuple(settings["universe"]) != cfg.universe
        or settings["initial_cash"] != cfg.initial_cash
        or cfg.initial_cash != 100000
        or settings["max_qty"] != cfg.max_shares
        or settings["cost_bps"] != cfg.slippage_bps
    ):
        raise ValueError("paper and replay cash/universe/order limits differ")
    sectors = settings["sector_map"] or [(s, "unknown") for s in cfg.universe]
    defaults = Config()
    if tuple(map(tuple, sectors)) != cfg.sector_map or any(
        getattr(cfg, name) != getattr(defaults, name)
        for name in (
            "max_position_weight",
            "max_sector_weight",
            "max_gross_weight",
            "cash_reserve_weight",
            "max_drawdown",
        )
    ):
        raise ValueError("paper risk rules differ from the validated replay defaults")
    training_root = adapter if (adapter / "training_manifest.json").is_file() else adapter.parent
    training = json.loads((training_root / "training_manifest.json").read_text())
    manifest, tm = training["manifest"], training["metrics"]
    if training.get("manifest_hash") != content_hash(manifest):
        raise ValueError("training provenance checksum failed")
    if (
        manifest["model_name"] != POLICY_MODEL
        or manifest["model_revision"] != POLICY_REVISION
        or manifest["tokenizer_revision"] != POLICY_REVISION
    ):
        raise ValueError("release requires a completed pinned training run")
    if manifest["method"] == "sft":
        if (
            training_root != adapter
            or tm.get("inference_adapter_fingerprint") != adapter_fingerprint(adapter)
            or tm.get("status") != "completed"
            or not 0 < tm.get("trainable_parameter_fraction", 1) < 0.001
            or tm.get("supervision_mode") != "trajectory-final"
            or tm.get("completed_steps", 0) < tm.get("planned_steps", 1)
        ):
            raise ValueError("SFT release requires completed assignment final-only supervision")
    elif manifest["method"] == "ppo":
        status = json.loads((training_root / "training_status.json").read_text())
        if (
            status.get("status") != "completed"
            or status["completed_updates"] < status["target_updates"]
        ):
            raise ValueError("PPO release is a stopped or partial experiment")
        from agent.training.execution import _path_hash
        from agent.training.qwen_ppo import PPO_IMPLEMENTATION_VERSION

        state = json.loads((training_root / "ppo_state.json").read_text())
        contract = state["run_contract"]
        parent = contract.get("initial_sft") or {}
        if (
            state.get("implementation_version") != PPO_IMPLEMENTATION_VERSION
            or state.get("run_fingerprint") != content_hash(contract)
            or state.get("adapter_hash") != _path_hash(adapter)
            or parent.get("supervision_mode") != "trajectory-final"
            or not parent.get("inference_weights_verified")
            or parent.get("status") != "completed"
            or parent.get("completed_steps", 0) < parent.get("planned_steps", 1)
            or not 0 < parent.get("trainable_parameter_fraction", 1) < 0.001
            or contract.get("prompt_version") != "final-decision-nlp-v1"
        ):
            raise ValueError("PPO release lacks corrected final-only SFT parent provenance")
        training = {**training, "ppo_state_provenance": state, "training_status": status}
    else:
        raise ValueError("final release candidates are assignment SFT or corrected PPO")
    root = Path(__file__).parent.parent
    current_code = code_hashes(root)
    if protocol.get("research_policy_identity", {}).get("model_revision") != POLICY_REVISION:
        raise ValueError("validation must record its pinned research policy identity")
    # Validated native policy settings are copied, never silently replaced by defaults.
    research_settings = protocol["research_settings"]
    from agent.context import ContextConfig
    from agent.evaluation.research import _implementation_hashes
    from agent.policy.loader import HuggingFaceQwenBackend
    from agent.research import ResearchPolicy, ResearchSettings

    config_settings = dict(research_settings)
    config_settings["context"] = ContextConfig(**config_settings["context"])
    expected = ResearchPolicy(
        HuggingFaceQwenBackend(adapter=adapter), settings=ResearchSettings(**config_settings)
    ).policy_identity
    if expected != protocol["research_policy_identity"]:
        raise ValueError("policy/tokenizer/tool code differs from completed validation")
    if protocol["policy_identity"].get("evaluation_code_hashes") != _implementation_hashes():
        raise ValueError("risk/evaluation code changed after validation")
    from agent.trajectories import load_collection

    journal = load_collection(validation / "journal")
    if (
        content_hash(journal.records) != report["record_hash"]
        or len(journal.records) != report["record_count"]
        or len(journal.records) != len(journal.planned_sessions)
    ):
        raise ValueError("validation journal is incomplete or differs from summary")
    if journal.policy_identity != protocol["policy_identity"] or canonical_json(
        asdict(journal.settings)
    ) != canonical_json(protocol["settings"]):
        raise ValueError("validation summary identity/settings differ from its journal")
    from agent.data.prices import DailyPriceBar
    from agent.evaluation.research import summarize_research_replay
    from agent.trajectory_cli import _rows

    bars = {s: [] for s in cfg.universe}
    for row in _rows(prepared / "price-bars.jsonl"):
        if row["symbol"] in bars:
            bars[row["symbol"]].append(DailyPriceBar(**row))
    benchmark_symbols = protocol["benchmark"]["symbols"]
    benchmark_symbol = benchmark_symbols[0] if len(benchmark_symbols) == 1 else None
    recomputed = summarize_research_replay(journal, bars, benchmark_symbol=benchmark_symbol)
    if content_hash(recomputed.metrics) != content_hash(metrics):
        raise ValueError("validation metrics do not match its complete journal")
    if any(row["status"] == "success" and row["tool_call_count"] < 1 for row in journal.records):
        raise ValueError("valid release decisions must perform actual tool research")
    base = output.parent
    base.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="." + output.name + "-", dir=base))
    try:
        (stage / "adapter").mkdir()
        for name in ADAPTER_FILES:
            if (adapter / name).is_file():
                shutil.copy2(adapter / name, stage / "adapter" / name)
        atomic_json(stage / "training_provenance.json", training)
        atomic_json(stage / "paper-config.json", asdict(cfg))
        atomic_json(stage / "validation.json", {"report": report, "metrics": metrics})
        for name in current_code:
            destination = stage / "code" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / name, destination)
        files = {str(p.relative_to(stage)): file_hash(p) for p in stage.rglob("*") if p.is_file()}
        body = {
            "version": "frozen-trading-release-v1",
            "model_name": POLICY_MODEL,
            "model_revision": POLICY_REVISION,
            "tokenizer_revision": POLICY_REVISION,
            "adapter_hash": inference_hash,
            "files": files,
            "code_hashes": current_code,
            "source_file_hashes": hashes,
            "research_settings": research_settings,
            "historical_memory": "disabled",
            "paper_memory": "delayed-confirmed-empty-initial",
            "memory_enrichment_validated": False,
            "test_execution_sessions": EXECUTION_WEEK,
            "test_decision_sessions": DECISION_WEEK,
            "live_paper_sessions": LIVE_PAPER_WEEK,
            "assessment_id": "assignment-2026-october",
            "validation_metrics": metrics,
            "training_method": manifest["method"],
            "base_weights_bundled": False,
        }
        atomic_json(stage / "release.json", {"release_id": content_hash(body), "manifest": body})
        check_files(prepared, hashes)
        if adapter_fingerprint(adapter) != inference_hash or code_hashes(root) != current_code:
            raise ValueError("inference assets/code changed during packaging")
        if output.exists():
            raise ValueError("release output appeared during packaging")
        stage.rename(output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return FrozenRelease(output)


class FrozenRelease:
    def __init__(self, path):
        self.path = Path(path).resolve()
        payload = json.loads((self.path / "release.json").read_text())
        self.manifest, self.release_id = payload["manifest"], payload["release_id"]
        self.verify()

    def verify(self):
        if (
            self.manifest.get("version") != "frozen-trading-release-v1"
            or self.manifest.get("model_name") != POLICY_MODEL
            or self.manifest.get("model_revision") != POLICY_REVISION
            or self.manifest.get("tokenizer_revision") != POLICY_REVISION
            or self.manifest.get("assessment_id") != "assignment-2026-october"
            or tuple(self.manifest.get("live_paper_sessions", ())) != LIVE_PAPER_WEEK
        ):
            raise ValueError("release model, assessment namespace or paper week differs")
        if self.release_id != content_hash(self.manifest):
            raise ValueError("frozen release manifest checksum failed")
        check_files(self.path, self.manifest["files"])
        actual_code = code_hashes(self.path / "code")
        if actual_code != self.manifest["code_hashes"]:
            raise ValueError("frozen module set changed")
        if set(p.name for p in (self.path / "adapter").iterdir()) != {
            Path(name).name for name in self.manifest["files"] if name.startswith("adapter/")
        }:
            raise ValueError("frozen adapter contains unexpected files")
        if adapter_fingerprint(self.path / "adapter") != self.manifest["adapter_hash"]:
            raise ValueError("frozen adapter changed")
        if (
            tuple(self.manifest["test_execution_sessions"]) != EXECUTION_WEEK
            or tuple(self.manifest["test_decision_sessions"]) != DECISION_WEEK
        ):
            raise ValueError("held-out October execution schedule changed")

    def verify_runtime(self):
        self.verify()
        if code_hashes() != self.manifest["code_hashes"]:
            raise ValueError("active code differs from the exact frozen module set")

    def authorize_test(self, *, settings, decision_sessions, next_session_resolver):
        self.verify_runtime()
        cfg = Config.from_dict(json.loads((self.path / "paper-config.json").read_text()))
        original = json.loads((self.path / "validation.json").read_text())["report"]["protocol"][
            "settings"
        ]
        if canonical_json(asdict(settings)) != canonical_json(original):
            raise ValueError("held-out replay settings differ from validated release")
        if (
            settings.initial_cash != 100000
            or tuple(settings.universe) != cfg.universe
            or tuple(decision_sessions) != DECISION_WEEK
            or tuple(next_session_resolver(day) for day in decision_sessions) != EXECUTION_WEEK
        ):
            raise ValueError("held-out execution/capital/universe plan differs from frozen release")
        return self.release_id


class PendingTestPermit:
    """A live registry reservation bound to one release, source and journal root."""

    def __init__(self, registry_path, attempt_id, contract):
        self.registry_path = Path(registry_path).resolve()
        self.attempt_id, self.contract = attempt_id, contract

    def verify(self, release, *, journal, evidence_source_hash=None, price_content_hash=None):
        import sqlite3

        if not self.registry_path.is_file():
            raise ValueError("durable held-out reservation is missing")
        db = sqlite3.connect(self.registry_path)
        try:
            row = db.execute(
                "SELECT a.config_hash,a.status,a.release_id,l.attempt_id "
                "FROM experiment_attempts a JOIN final_test_locks l "
                "ON l.attempt_id=a.attempt_id WHERE a.attempt_id=?",
                (self.attempt_id,),
            ).fetchone()
        finally:
            db.close()
        if (
            row is None
            or row[0] != content_hash(self.contract)
            or row[1] != "pending"
            or row[2] != release.manifest["assessment_id"]
            or self.contract["release_id"] != release.release_id
            or str(Path(journal).resolve()) != self.contract["journal_path"]
        ):
            raise ValueError("held-out permit is not the identical pending attempt/journal")
        if (
            evidence_source_hash is not None
            and evidence_source_hash != self.contract["source_file_hashes"]["evidence.jsonl"]
        ):
            raise ValueError("held-out evidence differs from reserved source")
        if (
            price_content_hash is not None
            and price_content_hash != self.contract["price_content_hash"]
        ):
            raise ValueError("held-out prices differ from reserved source")
