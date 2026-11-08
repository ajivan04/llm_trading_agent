"""One durable paper research cycle; broker execution is an independent boundary."""

import json
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import UTC, datetime

from agent.contracts import canonical_json, content_hash, parse_timestamp
from agent.execution import ExecutionPaused
from agent.trajectories import _result


class PaperTradingLoop:
    def __init__(
        self,
        policy,
        executor,
        evidence_provider,
        *,
        memory=None,
        clock=None,
        max_signal_age=300,
        frozen_release=None,
    ):
        self.policy, self.executor, self.evidence_provider = policy, executor, evidence_provider
        self.memory = memory
        self.frozen_release = frozen_release
        self.clock = clock or (lambda: datetime.now(UTC))
        if max_signal_age <= 0:
            raise ValueError("signal age must be positive")
        self.max_signal_age = max_signal_age
        if memory is not None and (memory.fold != "paper" or memory.run_id != executor.run_id):
            raise ValueError("paper memory must match the paper run namespace")
        if frozen_release is not None and memory is not None:
            marker = {
                "release_id": frozen_release.release_id,
                "run_id": executor.run_id,
                "initial_digest": content_hash(()),
            }
            row = executor.outbox.db.execute(
                "SELECT value FROM metadata WHERE key='frozen_memory_start'"
            ).fetchone()
            if row is None:
                if memory.memories:
                    raise ValueError("frozen paper memory must start empty in its run namespace")
                with executor.outbox.db:
                    executor.outbox.db.execute(
                        "INSERT INTO metadata VALUES('frozen_memory_start',?)",
                        (canonical_json(marker),),
                    )
            elif json.loads(row["value"]) != marker:
                raise ValueError("frozen paper memory belongs to a different release/run")
        executor.outbox.db.execute("""CREATE TABLE IF NOT EXISTS research_cycles(
            cycle_id TEXT PRIMARY KEY, payload TEXT NOT NULL, payload_hash TEXT NOT NULL)""")
        executor.outbox.db.commit()

    @contextmanager
    def _lock(self):
        import fcntl

        path = self.executor.outbox.path.with_suffix(".cycle.lock")
        with path.open("a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ExecutionPaused("paper research loop already active") from None
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def _persist(self, cycle_id, payload):
        text = canonical_json(payload)
        with self.executor.outbox.db:
            self.executor.outbox.db.execute(
                "INSERT OR REPLACE INTO research_cycles VALUES(?,?,?)",
                (cycle_id, text, content_hash(payload)),
            )

    def run_once(self, *, cycle_id, submit=False):
        if not isinstance(cycle_id, str) or not cycle_id.strip() or type(submit) is not bool:
            raise ValueError("stable cycle ID and explicit submit boolean required")
        with self._lock():
            if self.frozen_release is not None:
                self.frozen_release.verify_runtime()
            snapshot = self.executor.reconcile()
            row = self.executor.outbox.db.execute(
                "SELECT * FROM research_cycles WHERE cycle_id=?", (cycle_id,)
            ).fetchone()
            if row is not None:
                saved = json.loads(row["payload"])
                if content_hash(saved) != row["payload_hash"]:
                    raise ValueError("durable research cycle checksum failed")
                if saved["submit"] != submit:
                    raise ValueError("dry-run and submitting cycles must use different cycle IDs")
                if saved["status"] != "planned":
                    return saved
                plan = saved
            else:
                if not snapshot.market_open or snapshot.blocked:
                    raise ExecutionPaused("market closed or paper account blocked")
                refresh = getattr(self.evidence_provider, "refresh", None)
                if refresh is not None:
                    refresh(self.executor.config.universe, snapshot)
                if self.memory is not None:
                    from agent.execution.reflections import process_confirmed_reflections

                    process_confirmed_reflections(
                        self.executor.outbox,
                        self.memory,
                        self.policy.backend,
                        as_of=self.clock().isoformat(),
                        clock=self.clock,
                    )
                snapshot = self.executor.reconcile()
                if not snapshot.market_open or snapshot.blocked:
                    raise ExecutionPaused("market closed during source prefetch")
                cutoff = self.clock().isoformat()
                state = {
                    "as_of": cutoff,
                    "cash": snapshot.cash,
                    "positions": dict(snapshot.positions),
                    "equity": snapshot.equity,
                    "paper_only": True,
                    "source_refresh": getattr(self.evidence_provider, "status", {}),
                }
                evidence = tuple(self.evidence_provider(cutoff, self.executor.config.universe))
                if any(not item.eligible(cutoff) for item in evidence):
                    raise ValueError("future evidence cannot enter the paper loop")
                memories = ()
                if self.memory is not None:
                    query = (
                        " ".join(self.executor.config.universe)
                        + " "
                        + " ".join(item.text[:500] for item in evidence[-4:])
                    )
                    memories = self.memory.context_memories(query, as_of=cutoff)
                print(
                    canonical_json(
                        {"event": "paper_policy_start", "cycle_id": cycle_id, "as_of": cutoff}
                    ),
                    flush=True,
                )
                response = {}
                try:
                    response = self.policy(
                        state=state,
                        as_of=cutoff,
                        evidence=evidence,
                        universe=self.executor.config.universe,
                        memories=memories,
                    )
                    checked = _result(
                        response, as_of=cutoff, universe=self.executor.config.universe, state=state
                    )
                except Exception:
                    response = response if isinstance(response, Mapping) else {}
                    transcript = response.get("transcript")
                    transcript = transcript if isinstance(transcript, Mapping) else {}
                    plan = {
                        "status": "invalid_policy",
                        "cycle_id": cycle_id,
                        "submit": submit,
                        "decision_at": cutoff,
                        "policy_status": response.get("status", "generation_failure"),
                        "policy_failure": response.get("failure"),
                        "raw_outputs": transcript.get("raw_outputs", []),
                    }
                    self._persist(cycle_id, plan)
                    return plan
                plan = {
                    "status": "planned",
                    "cycle_id": cycle_id,
                    "submit": submit,
                    "decision_at": cutoff,
                    "response": response,
                    "state": state,
                    "decisions": [d.to_dict() for d in checked["decisions"]],
                    "allowed_evidence": sorted(checked["allowed"]),
                }
                self._persist(cycle_id, plan)  # Actual proposal durable before any order intent.
            age = (self.clock() - parse_timestamp(plan["decision_at"])).total_seconds()
            if not 0 <= age <= self.max_signal_age:
                plan["status"] = "expired_signal"
                self._persist(cycle_id, plan)
                return plan
            from agent.contracts import Decision

            decisions = tuple(Decision(**row) for row in plan["decisions"])
            # Every execution call independently refreshes broker state and quotes.
            if self.frozen_release is not None:
                self.frozen_release.verify_runtime()
            results = self.executor.execute(
                decisions,
                cycle_id=cycle_id,
                allowed_evidence_ids=plan["allowed_evidence"],
                submit=submit,
            )
            plan["status"], plan["execution"] = "completed", results
            self._persist(cycle_id, plan)
            print(
                canonical_json(
                    {
                        "event": "paper_cycle_complete",
                        "cycle_id": cycle_id,
                        "submit": submit,
                        "execution": results,
                    }
                ),
                flush=True,
            )
            return json.loads(canonical_json(plan))


def run_trading_loop(loop: PaperTradingLoop, *, cycle_ids, submit=False):
    """Bounded iterations with explicit durable IDs; never an implicit infinite trader."""
    return [loop.run_once(cycle_id=cycle_id, submit=submit) for cycle_id in cycle_ids]
