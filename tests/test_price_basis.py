import unittest
from dataclasses import replace

from agent.data.prices import DailyPriceBar
from agent.price_basis import complete_tail_actions, restore_nominal, split_adjusted_closes


def bar(day, price, split=0):
    return DailyPriceBar(
        "AAPL",
        day,
        price,
        price + 1,
        price - 1,
        price,
        price,
        1000,
        0,
        split,
        day + "T14:30:00Z",
        day + "T21:00:00Z",
        "2026-01-06T22:00:00Z",
        price_basis="yahoo-split-adjusted",
    )


class PriceBasisTests(unittest.TestCase):
    def test_restore_nominal_once_then_indicators_adjust_only_known_past_split(self):
        source = (bar("2026-01-02", 50), bar("2026-01-05", 55, 2))
        with self.assertRaisesRegex(ValueError, "nominal"):
            source[0].to_evidence()
        result = restore_nominal(source)
        self.assertEqual([r.close for r in result], [100, 55])
        self.assertEqual([r.volume for r in result], [500, 1000])
        self.assertEqual(restore_nominal(result), result)
        evidence = [r.to_evidence() for r in result]
        self.assertEqual(split_adjusted_closes(evidence), [50, 55])
        self.assertNotIn("future", str(evidence[0].to_dict()))
        self.assertNotIn("adjusted_close", evidence[0].values)

    def test_tail_actions_are_checked_through_original_capture_not_assumed_absent(self):
        source = {"AAPL": (bar("2026-01-02", 50),)}

        def fetch(_symbols, _start, _end):
            return {
                "AAPL": (
                    {"session": "2026-01-05", "ratio": 2},
                    {"session": "2026-01-06", "ratio": 0},
                )
            }

        tails = complete_tail_actions(source, fetcher=fetch)
        result = restore_nominal(source["AAPL"], tails["AAPL"])
        self.assertEqual(result[0].close, 100)
        with self.assertRaisesRegex(ValueError, "missing split-action tail"):
            complete_tail_actions(source, fetcher=lambda *_: {"AAPL": []})

    def test_contradictory_basis_and_mixed_series_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "calibration"):
            restore_nominal((bar("2026-01-02", 100), bar("2026-01-05", 55, 2)))
        with self.assertRaisesRegex(ValueError, "consistent"):
            restore_nominal(
                (bar("2026-01-02", 50), replace(bar("2026-01-05", 55), price_basis="nominal"))
            )

    def test_duplicate_nan_or_missing_tail_cannot_be_certified(self):
        source = {"AAPL": (bar("2026-01-02", 50),)}
        for rows in (
            [{"session": "2026-01-05", "ratio": 0}] * 2,
            [{"session": "2026-01-05", "ratio": float("nan")}],
            [{"session": "2026-01-05", "ratio": -1}],
        ):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                complete_tail_actions(source, fetcher=lambda *_, records=rows: {"AAPL": records})

    def test_source_clone_preserves_anchors_omits_stale_targets_and_rejects_corruption(self):
        import json
        import tempfile
        from pathlib import Path

        from agent.normalize_prices import normalize_source
        from tests.test_compact_artifact import prepared_source

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root / "source", root / "nominal"
            prepared_source(source)
            (source / ".env").write_text("do-not-copy")
            before = (source / "price-bars.jsonl").read_bytes()
            target = (source / "dataset/examples.jsonl").read_bytes()
            normalize_source(
                source, output, tail_fetcher=lambda *_: self.fail("nominal tail fetched")
            )
            self.assertEqual((source / "price-bars.jsonl").read_bytes(), before)
            self.assertEqual((output / "dataset/examples.jsonl").read_bytes(), target)
            self.assertFalse((output / ".env").exists())
            self.assertFalse((output / "ppo-train.jsonl").exists())
            prep = json.loads((output / "preparation-manifest.json").read_text())
            self.assertFalse(prep["old_training_targets_usable"])
            self.assertTrue(
                all(
                    json.loads(r)["source"] != "yfinance"
                    for r in (output / "evidence.jsonl").read_text().splitlines()
                )
            )
            (source / "price-bars.jsonl").write_text("")
            with self.assertRaisesRegex(ValueError, "price hash"):
                normalize_source(source, root / "corrupt")
