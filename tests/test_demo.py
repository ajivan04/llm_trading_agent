import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from agent.contracts import Fill
from trading_agent import Config, DemoPolicy, Memory, Portfolio, Store, replay, synthetic_bars


class DemoIntegrationTests(unittest.TestCase):
    def test_portfolio_applies_fill_fees_on_both_sides(self):
        portfolio = Portfolio(1_000.0)
        portfolio.apply(Fill(
            "buy-1", "AAPL", "buy", 1, 100.0, "2026-01-05T14:30:00Z", fee=5.0,
        ))
        self.assertEqual(portfolio.cash, 895.0)

        portfolio.apply(Fill(
            "sell-1", "AAPL", "sell", 1, 110.0, "2026-01-06T14:30:00Z", fee=3.0,
        ))
        self.assertEqual(portfolio.cash, 1_002.0)

    def test_seeded_replay_is_repeatable(self):
        cfg = Config()
        bars = synthetic_bars(cfg, sessions=8)

        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_output = Path(first) / "run"
            first_summary = replay(cfg, bars, DemoPolicy(), first_output)
            second_summary = replay(cfg, bars, DemoPolicy(), Path(second) / "run")
            self.assertEqual(json.loads((first_output / "summary.json").read_text()), first_summary)
            self.assertIn("config", json.loads((first_output / "manifest.json").read_text()))
            episodes = [json.loads(line) for line in
                        (first_output / "episodes.jsonl").read_text().splitlines()]
            self.assertEqual(len(episodes), first_summary["episodes"])

        self.assertEqual(first_summary, second_summary)
        self.assertTrue(first_summary["synthetic_demo_only"])
        self.assertGreater(first_summary["episodes"], 0)

    def test_store_migrates_legacy_memory_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.sqlite"
            with sqlite3.connect(path) as database:
                database.execute(
                    "CREATE TABLE memories (id TEXT PRIMARY KEY, run_id TEXT, "
                    "available_at TEXT, situation TEXT, reflection TEXT)"
                )
                database.execute(
                    "INSERT INTO memories VALUES (?,?,?,?,?)",
                    ("legacy-id", "legacy-run", "2026-01-05T00:00:00+00:00",
                     "AAPL momentum", "legacy reflection"),
                )

            store = Store(path)
            try:
                columns = {
                    row["name"] for row in store.db.execute("PRAGMA table_info(memories)")
                }
                self.assertTrue({"experiment_id", "fold"} <= columns)
                migrated = store.db.execute(
                    "SELECT * FROM memories WHERE id='legacy-id'"
                ).fetchone()
                self.assertEqual(migrated["experiment_id"], "legacy")
                self.assertEqual(migrated["fold"], "legacy")
                results = Memory(store).retrieve(
                    "legacy", "legacy-run", "legacy", "AAPL",
                    datetime(2026, 1, 6, tzinfo=timezone.utc), 1,
                )
                self.assertEqual(results[0]["reflection"], "legacy reflection")
            finally:
                store.close()

    def test_sqlite_memory_isolates_experiment_run_and_fold(self):
        at = datetime(2026, 1, 5, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "state.sqlite")
            try:
                memory = Memory(store)
                store.memory_add("experiment-a", "run-a", "train", at,
                                 "AAPL momentum", "eligible")
                store.memory_add("experiment-b", "run-a", "train", at,
                                 "AAPL momentum", "wrong experiment")
                store.memory_add("experiment-a", "run-a", "test", at,
                                 "AAPL momentum", "wrong fold")

                results = memory.retrieve(
                    "experiment-a", "run-a", "train", "AAPL", at, 5
                )
                self.assertEqual([item["reflection"] for item in results], ["eligible"])
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
