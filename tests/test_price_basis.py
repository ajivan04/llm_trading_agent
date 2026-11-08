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
            self.assertEqual((source / "dataset/examples.jsonl").read_bytes(), target)
            self.assertFalse((output / "dataset").exists())
            self.assertFalse((output / ".env").exists())
            self.assertFalse((output / "ppo-train.jsonl").exists())
            from agent.source_bundle import load_source

            self.assertEqual(load_source(output)["scope"], "research")
            self.assertFalse((output / "preparation-manifest.json").exists())
            self.assertTrue(
                all(
                    json.loads(r)["source"] != "yfinance"
                    for r in (output / "evidence.jsonl").read_text().splitlines()
                )
            )
            (source / "price-bars.jsonl").write_text("")
            with self.assertRaisesRegex(ValueError, "price hash"):
                normalize_source(source, root / "corrupt")


class BatchedYahooActionTests(unittest.TestCase):
    def frame(self):
        import pandas as pd

        return pd.DataFrame(
            {
                ("Adj Close", "AAPL"): [50.0, 51.0],
                ("Adj Close", "MSFT"): [200.0, 201.0],
                ("Stock Splits", "AAPL"): [2.0, 0.0],
                ("Stock Splits", "MSFT"): [0.0, 3.0],
            },
            index=pd.to_datetime(["2026-01-05", "2026-01-06"]),
        ).rename_axis(columns=["Price", "Ticker"])

    def source(self):
        return {
            "AAPL": (bar("2026-01-02", 50),),
            "MSFT": (replace(bar("2026-01-02", 200), symbol="MSFT"),),
        }

    def test_default_transport_selects_each_ticker_in_either_multiindex_orientation(self):
        from unittest.mock import patch

        from agent.price_basis import normalize_bundle

        frame = self.frame()
        for response in (frame, frame.swaplevel("Price", "Ticker", axis=1)):
            with self.subTest(names=response.columns.names):
                with patch("yfinance.download", return_value=response) as download:
                    normalized, tails = normalize_bundle(self.source())
                self.assertEqual(download.call_count, 1)
                self.assertEqual(tails["AAPL"], ({"session": "2026-01-05", "ratio": 2.0},))
                self.assertEqual(tails["MSFT"], ({"session": "2026-01-06", "ratio": 3.0},))
                self.assertEqual(normalized["AAPL"][0].close, 100)
                self.assertEqual(normalized["MSFT"][0].close, 600)
                self.assertEqual(normalized["AAPL"][0].price_basis, "nominal")

    def test_single_flat_response_allowed_only_for_single_requested_symbol(self):
        from unittest.mock import patch

        frame = self.frame().xs("AAPL", axis=1, level="Ticker")
        with patch("yfinance.download", return_value=frame):
            self.assertEqual(
                complete_tail_actions({"AAPL": self.source()["AAPL"]})["AAPL"][0]["ratio"], 2
            )
            with self.assertRaisesRegex(ValueError, "ticker-labeled"):
                complete_tail_actions(self.source())

    def test_missing_ticker_action_and_nan_fail_without_cross_symbol_substitution(self):
        from unittest.mock import patch

        frame = self.frame()
        bad_frames = (
            frame.drop(columns="MSFT", level="Ticker"),
            frame.drop(columns=("Stock Splits", "MSFT")),
            frame.rename_axis(columns=["Unknown", "Level"]),
        )
        for response in bad_frames:
            with (
                self.subTest(columns=list(response.columns)),
                patch("yfinance.download", return_value=response),
            ):
                with self.assertRaises(ValueError):
                    complete_tail_actions(self.source())
        frame[("Stock Splits", "MSFT")] = float("nan")
        with patch("yfinance.download", return_value=frame):
            with self.assertRaisesRegex(ValueError, "incomplete"):
                complete_tail_actions(self.source())
