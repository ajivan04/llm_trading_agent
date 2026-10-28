import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from agent.data.prices import DailyPriceBar
from agent.datasets import load_dataset_artifact
from agent.preparation import PreparationSettings, prepare_training_artifacts


def bar(session, opened, closed, price, next_price):
    return DailyPriceBar(
        "AAPL", session, price, max(price, next_price) + 1,
        min(price, next_price) - 1, next_price, next_price, 1_000_000, 0, 0,
        opened, closed, datetime(2026, 2, 1, tzinfo=UTC),
    )


class PreparationTests(unittest.TestCase):
    def setUp(self):
        sessions = [
            ("2026-01-02", 100, 101), ("2026-01-05", 102, 103),
            ("2026-01-06", 104, 102), ("2026-01-07", 101, 100),
            ("2026-01-08", 99, 103), ("2026-01-09", 104, 105),
        ]
        self.bars = tuple(bar(
            day, f"{day}T14:30:00Z", f"{day}T21:00:00Z", opened, closed,
        ) for day, opened, closed in sessions)
        self.settings = PreparationSettings(
            train_end="2026-01-05T23:59:59Z",
            validation_end="2026-01-07T23:59:59Z",
            test_end="2026-01-09T23:59:59Z",
        )

    def prepare(self, bars=None):
        rows = tuple(bars or self.bars)
        following = {
            first.session: second.session
            for first, second in zip(rows, rows[1:], strict=False)
        }
        return prepare_training_artifacts(
            {"AAPL": rows}, settings=self.settings,
            next_session_resolver=following.__getitem__,
        )

    def test_builds_all_training_inputs_without_future_observations(self):
        prepared = self.prepare()
        self.assertTrue(prepared.dataset.manifest.split_example_ids["train"])
        self.assertTrue(prepared.dataset.manifest.split_example_ids["validation"])
        self.assertTrue(prepared.dataset.manifest.split_example_ids["test"])
        self.assertTrue(prepared.dpo["train"])
        self.assertTrue(prepared.ppo_train)
        self.assertEqual(len(prepared.outcome_audit), len(prepared.dataset.examples))
        messages = json.dumps(prepared.ppo_train[0]["messages"])
        self.assertNotIn("next_open", messages)
        self.assertNotIn("outcome_close", messages)
        self.assertNotIn("retrieved_at", messages)
        self.assertEqual(prepared.ppo_train[0]["split"], "train")

    def test_written_artifacts_are_reproducible_and_verified(self):
        first = self.prepare()
        second = self.prepare()
        self.assertEqual(first.artifact_hash, second.artifact_hash)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            first.write(output)
            self.assertEqual(
                load_dataset_artifact(output / "dataset").manifest.dataset_hash,
                first.dataset.manifest.dataset_hash,
            )
            manifest = json.loads((output / "preparation-manifest.json").read_text())
            self.assertEqual(manifest["artifact_hash"], first.artifact_hash)
            config = json.loads((output / "ppo-environment.json").read_text())
            self.assertTrue(config["episodes_path"].endswith("ppo-train.jsonl"))

    def test_rejects_mixed_symbols_and_invalid_trade_size(self):
        wrong = list(self.bars)
        wrong[0] = DailyPriceBar(
            "MSFT", **{key: value for key, value in wrong[0].__dict__.items()
                       if key != "symbol"},
        )
        with self.assertRaisesRegex(ValueError, "matching bars"):
            prepare_training_artifacts(
                {"AAPL": wrong}, settings=self.settings,
                next_session_resolver=lambda _session: "2026-01-05",
            )
        with self.assertRaisesRegex(ValueError, "trade_qty"):
            PreparationSettings(
                self.settings.train_end, self.settings.validation_end,
                self.settings.test_end, starting_qty=1, trade_qty=2,
            )

    def test_filters_decision_sessions_but_keeps_warmup_bars(self):
        prepared = prepare_training_artifacts(
            {"AAPL": self.bars},
            settings=PreparationSettings(
                self.settings.train_end, self.settings.validation_end,
                self.settings.test_end,
                allowed_decision_sessions=("2026-01-02", "2026-01-06"),
            ),
            next_session_resolver=lambda session: {
                first.session: second.session
                for first, second in zip(self.bars, self.bars[1:], strict=False)
            }[session],
        )
        self.assertEqual(
            tuple(item.example_id for item in prepared.dataset.examples),
            ("AAPL:2026-01-02", "AAPL:2026-01-06"),
        )


if __name__ == "__main__":
    unittest.main()

