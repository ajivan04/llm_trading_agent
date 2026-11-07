import json
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

from agent.memory import TradingMemory
from agent.reflection import generate_reflection
from tests.test_reflection import AT, NOW, SITUATION, Backend, outcome, reflection


def stored(memory, item=None, situation=SITUATION, **kwargs):
    item = item or outcome()
    result = generate_reflection(
        Backend(), outcome=item, situation=situation, as_of=AT, clock=lambda: NOW
    )
    return memory.add_memory(situation, result.lesson, outcome=item, result=result, **kwargs)


class MemoryTests(unittest.TestCase):
    def test_durable_bm25_retrieval_and_context_cutoff(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "memory.sqlite"
            with TradingMemory(path, clock=lambda: NOW) as memory:
                memory_id = stored(memory)
                self.assertEqual(memory.get_memories("AAPL"), [reflection().lesson])
                self.assertEqual(memory.get_memories("unrelated"), [])
                context = memory.context_memories("AAPL", as_of=AT)
                self.assertEqual(context[0]["id"], memory_id)
                self.assertGreater(context[0]["score"], 0)
                self.assertEqual(memory.context_memories("AAPL", as_of="2026-11-06T21:59:59Z"), [])
                digest = memory.memory_digest
            with TradingMemory(path, clock=lambda: NOW) as memory:
                self.assertEqual(memory.memory_digest, digest)
                self.assertEqual(len(memory.get_memories("AAPL")), 1)

    def test_scopes_and_future_records_do_not_change_eligible_corpus(self):
        later = NOW + timedelta(days=2)
        future_item = outcome(
            trade_id="future", confirmed_at="2026-11-07T20:00:01Z", closed_at="2026-11-07T20:00:00Z"
        )
        future_result = generate_reflection(
            Backend(),
            outcome=future_item,
            situation="AAPL AAPL future",
            as_of=later.isoformat(),
            clock=lambda: later,
        )
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "memory.sqlite"
            with TradingMemory(path, clock=lambda: later) as memory:
                stored(memory)
                score = memory.context_memories("AAPL", as_of=AT)[0]["score"]
                memory.add_memory(
                    "AAPL AAPL future",
                    future_result.lesson,
                    outcome=future_item,
                    result=future_result,
                )
                self.assertEqual(memory.context_memories("AAPL", as_of=AT)[0]["score"], score)
            for scope in ({"run_id": "other"}, {"experiment_id": "other"}, {"fold": "test"}):
                with TradingMemory(path, clock=lambda: later, **scope) as memory:
                    self.assertEqual(memory.context_memories("AAPL", as_of=later.isoformat()), [])

    def test_replay_and_paper_provenance_and_explicit_replay_cutoff(self):
        with TradingMemory(":memory:", fold="validation", clock=lambda: NOW) as memory:
            with self.assertRaisesRegex(ValueError, "isolated"):
                stored(memory)
            stored(memory, item=outcome(source="historical.replay"))
            with self.assertRaisesRegex(ValueError, "explicit as_of"):
                memory.get_memories("AAPL")
            self.assertEqual(len(memory.get_memories("AAPL", as_of=AT)), 1)
        with TradingMemory(":memory:", clock=lambda: NOW) as memory:
            with self.assertRaisesRegex(ValueError, "isolated"):
                stored(memory, item=outcome(source="historical.replay"))

    def test_idempotent_first_lesson_and_changed_outcome_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "memory.sqlite"
            with (
                TradingMemory(path, clock=lambda: NOW) as first,
                TradingMemory(path, clock=lambda: NOW) as second,
            ):
                memory_id = stored(first)
                result = replace(reflection(), lesson="Another cautious lesson.")
                self.assertEqual(
                    second.add_memory(SITUATION, result.lesson, outcome=outcome(), result=result),
                    memory_id,
                )
                self.assertEqual(first.get_memories("AAPL"), [reflection().lesson])
                with self.assertRaisesRegex(ValueError, "changed outcome"):
                    stored(second, item=outcome(exit_notional=220))
                self.assertEqual(len(first.memories), 1)

    def test_unverified_failed_future_or_mismatched_lessons_not_persisted(self):
        with TradingMemory(":memory:", clock=lambda: NOW) as memory:
            failed = reflection(backend=Backend("invalid"))
            for result, text, situation in (
                (failed, "invented", SITUATION),
                (reflection(), "invented", SITUATION),
                (reflection(), reflection().lesson, "AAPL facts changed"),
                (
                    replace(reflection(), available_at=(NOW + timedelta(seconds=1)).isoformat()),
                    reflection().lesson,
                    SITUATION,
                ),
            ):
                with self.subTest(result=result.status), self.assertRaises(ValueError):
                    memory.add_memory(situation, text, outcome=outcome(), result=result)
            self.assertEqual(memory.memories, [])

    def test_corrupt_or_tampered_store_fails_closed(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "memory.sqlite"
            with TradingMemory(path, clock=lambda: NOW) as memory:
                stored(memory)
                raw = json.loads(
                    memory.db.execute("SELECT payload FROM reflection_memories").fetchone()[0]
                )
                raw["reflection"] = "tampered"
                memory.db.execute("UPDATE reflection_memories SET payload=?", (json.dumps(raw),))
                memory.db.commit()
                with self.assertRaisesRegex(ValueError, "integrity"):
                    memory.get_memories("AAPL")
            invalid = Path(root) / "corrupt.sqlite"
            invalid.write_text("not a database")
            with self.assertRaisesRegex(ValueError, "invalid"):
                TradingMemory(invalid)

    def test_invalid_scope_and_timestamps_rejected(self):
        with self.assertRaises(ValueError):
            TradingMemory(":memory:", fold="all")
        with TradingMemory(":memory:", clock=lambda: NOW) as memory:
            with self.assertRaises(ValueError):
                memory.context_memories("AAPL", as_of=datetime(2026, 1, 1))
            with self.assertRaises(ValueError):
                memory.context_memories("AAPL", as_of=AT, n_matches=0)
            self.assertEqual(memory.context_memories("", as_of=AT), [])
