import math
import unittest

from agent.contracts import Evidence
from agent.data.prices import DailyPriceBar
from agent.features import FeatureParameters, PriceFeatureBuilder
from agent.snapshots import SnapshotBuilder


def price_bar(
    number: int,
    *,
    symbol: str = "AAPL",
    session: str | None = None,
    close: float | None = None,
) -> Evidence:
    session = session or f"2026-01-{number:02d}"
    value = close if close is not None else float(number)
    timestamp = f"{session}T21:00:00Z"
    return DailyPriceBar(
        symbol=symbol,
        session=session,
        open=value - 0.25,
        high=value + 0.5,
        low=max(value - 0.5, 0.01),
        close=value,
        adjusted_close=value,
        volume=number * 10,
        dividends=0.0,
        stock_splits=0.0,
        open_at=f"{session}T14:30:00Z",
        close_at=timestamp,
        retrieved_at="2026-02-01T00:00:00Z",
    ).to_evidence()


PARAMETERS = FeatureParameters(
    return_window=2,
    sma_window=2,
    rsi_window=2,
    macd_fast=2,
    macd_slow=3,
    volatility_window=2,
    volume_window=2,
    annualization_sessions=252,
    version="test-v1",
)


class PriceFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.builder = PriceFeatureBuilder(PARAMETERS)

    def test_future_and_other_symbol_bars_are_never_consumed(self) -> None:
        eligible = [price_bar(index) for index in range(1, 5)]
        future = price_bar(5)
        other = price_bar(4, symbol="MSFT")

        with_future = self.builder.build(
            [future, other, *reversed(eligible)],
            symbol="aapl",
            as_of="2026-01-04T21:00:00Z",
        )
        without_future = self.builder.build(
            eligible,
            symbol="AAPL",
            as_of="2026-01-04T21:00:00Z",
        )

        self.assertEqual(with_future.id, without_future.id)
        self.assertEqual(
            with_future.values["source_evidence_ids"],
            tuple(bar.id for bar in eligible),
        )
        self.assertNotIn(future.id, with_future.values["source_evidence_ids"])

    def test_insufficient_history_is_explicitly_missing(self) -> None:
        result = self.builder.build(
            [price_bar(1)], symbol="AAPL", as_of="2026-01-01T21:00:00Z")
        self.assertEqual(
            result.values["features"],
            {
                "trailing_return": None,
                "sma_ratio": None,
                "rsi": None,
                "macd": None,
                "realized_volatility": None,
                "relative_volume": None,
            },
        )

    def test_features_are_deterministic_and_use_documented_windows(self) -> None:
        bars = [price_bar(index) for index in range(1, 5)]
        first = self.builder.build(
            reversed(bars), symbol="AAPL", as_of="2026-01-04T21:00:00Z")
        second = self.builder.build(
            bars, symbol="AAPL", as_of="2026-01-04T21:00:00Z")
        features = first.values["features"]

        self.assertEqual(first.to_json(), second.to_json())
        self.assertAlmostEqual(features["trailing_return"], 1.0)
        self.assertAlmostEqual(features["sma_ratio"], 4.0 / 3.5)
        self.assertEqual(features["rsi"], 100.0)
        self.assertAlmostEqual(features["macd"], 0.5)
        expected_vol = math.sqrt((((0.5 - 5 / 12) ** 2 + (1 / 3 - 5 / 12) ** 2)) * 252)
        self.assertAlmostEqual(features["realized_volatility"], expected_vol)
        self.assertAlmostEqual(features["relative_volume"], 1.6)
        self.assertEqual(first.values["feature_version"], "test-v1")
        self.assertEqual(
            first.values["price_basis"],
            "point_in_time_total_return_from_raw_close_actions",
        )
        self.assertEqual(first.values["provider_adjusted_close_usage"], "audit_only")

    def test_duplicate_symbol_session_is_rejected(self) -> None:
        first = price_bar(1)
        duplicate = price_bar(2, session="2026-01-01")
        with self.assertRaisesRegex(ValueError, "duplicate price bar"):
            self.builder.build(
                [first, duplicate], symbol="AAPL", as_of="2026-01-02T21:00:00Z")

    def test_provider_adjustments_are_not_feature_inputs(self) -> None:
        original = [price_bar(index) for index in range(1, 5)]
        revised = []
        for index in range(1, 5):
            session = f"2026-01-{index:02d}"
            value = float(index)
            revised.append(DailyPriceBar(
                symbol="AAPL", session=session, open=value - 0.25,
                high=value + 0.5, low=max(value - 0.5, 0.01), close=value,
                adjusted_close=value / 10, volume=index * 10, dividends=0.0,
                stock_splits=0.0, open_at=f"{session}T14:30:00Z",
                close_at=f"{session}T21:00:00Z",
                retrieved_at="2026-02-01T00:00:00Z",
            ).to_evidence())

        original_features = self.builder.build(
            original, symbol="AAPL", as_of="2026-01-04T21:00:00Z")
        revised_features = self.builder.build(
            revised, symbol="AAPL", as_of="2026-01-04T21:00:00Z")

        self.assertEqual(original_features.id, revised_features.id)
        self.assertEqual(original_features.to_json(), revised_features.to_json())

    def test_future_corporate_action_is_excluded(self) -> None:
        eligible = [price_bar(index) for index in range(1, 5)]
        future = DailyPriceBar(
            symbol="AAPL", session="2026-01-05", open=1.9, high=2.1, low=1.8,
            close=2.0, adjusted_close=2.0, volume=100, dividends=0.5,
            stock_splits=2.0, open_at="2026-01-05T14:30:00Z",
            close_at="2026-01-05T21:00:00Z", retrieved_at="2026-02-01T00:00:00Z",
        ).to_evidence()
        baseline = self.builder.build(
            eligible, symbol="AAPL", as_of="2026-01-04T21:00:00Z")
        with_future = self.builder.build(
            [*eligible, future], symbol="AAPL", as_of="2026-01-04T21:00:00Z")
        self.assertEqual(baseline.id, with_future.id)

    def test_eligible_split_and_dividend_remove_artificial_price_drop(self) -> None:
        before = price_bar(1, close=100.0)
        after = DailyPriceBar(
            symbol="AAPL", session="2026-01-02", open=49.0, high=51.0, low=48.0,
            close=49.5, adjusted_close=49.5, volume=100, dividends=0.5,
            stock_splits=2.0, open_at="2026-01-02T14:30:00Z",
            close_at="2026-01-02T21:00:00Z", retrieved_at="2026-02-01T00:00:00Z",
        ).to_evidence()
        parameters = FeatureParameters(
            return_window=1, sma_window=2, rsi_window=1, macd_fast=1,
            macd_slow=2, volatility_window=2, volume_window=1,
        )
        feature = PriceFeatureBuilder(parameters).build(
            [before, after], symbol="AAPL", as_of="2026-01-02T21:00:00Z")
        self.assertAlmostEqual(feature.values["features"]["trailing_return"], 0.0)


class SnapshotBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.builder = SnapshotBuilder()
        self.evidence = [price_bar(1), price_bar(2)]
        self.compact = (
            {"evidence_id": self.evidence[1].id, "summary": "second"},
            {"evidence_id": self.evidence[0].id, "summary": "first"},
        )

    def build(self, **changes):
        arguments = {
            "as_of": "2026-01-02T21:00:00Z",
            "universe": ("AAPL", "MSFT"),
            "universe_version": "u1",
            "evidence": self.evidence,
            "compact_evidence": self.compact,
            "feature_version": "f1",
            "account_state": {"cash": 1000.0, "positions": {"AAPL": 1}},
            "source_health": {"yfinance": {"status": "ok"}},
            "market_session_id": "XNYS-2026-01-02",
        }
        arguments.update(changes)
        return self.builder.build(**arguments)

    def test_preserves_evidence_and_exact_compact_payload_order(self) -> None:
        snapshot = self.build(evidence=reversed(self.evidence))
        self.assertEqual(snapshot.evidence_ids, tuple(bar.id for bar in reversed(self.evidence)))
        self.assertEqual(snapshot.compact_evidence, self.compact)
        with self.assertRaises(TypeError):
            snapshot.account_state["cash"] = 0
        with self.assertRaises(TypeError):
            snapshot.source_health["yfinance"]["status"] = "down"

    def test_rejects_future_duplicate_and_out_of_universe_evidence(self) -> None:
        with self.assertRaisesRegex(ValueError, "not eligible"):
            self.build(evidence=[*self.evidence, price_bar(3)])
        with self.assertRaisesRegex(ValueError, "IDs must be unique"):
            self.build(evidence=[self.evidence[0], self.evidence[0]])
        with self.assertRaisesRegex(ValueError, "outside the universe"):
            self.build(evidence=[price_bar(1, symbol="TSLA")])
        with self.assertRaisesRegex(ValueError, "universe symbols must be unique"):
            self.build(universe=("AAPL", "aapl"))

    def test_snapshot_hash_is_deterministic_and_changes_with_inputs(self) -> None:
        original = self.build()
        self.assertEqual(original.snapshot_id, self.build().snapshot_id)
        changed_payload = self.build(compact_evidence=(
            {"evidence_id": self.evidence[0].id, "summary": "changed"},))
        changed_account = self.build(account_state={"cash": 999.0})
        self.assertNotEqual(original.snapshot_id, changed_payload.snapshot_id)
        self.assertNotEqual(original.snapshot_id, changed_account.snapshot_id)

        with self.assertRaisesRegex(ValueError, "unknown evidence ID"):
            self.build(compact_evidence=({"evidence_id": "different"},))

    def test_snapshot_identity_uses_content_and_checks_feature_version(self) -> None:
        original = self.evidence[0]
        revised = Evidence(
            id=original.id, source=original.source, symbol=original.symbol,
            observed_at=original.observed_at, available_at=original.available_at,
            retrieved_at=original.retrieved_at, text="corrected record",
            values=original.values,
        )
        first = self.build(evidence=[original], compact_evidence=())
        second = self.build(evidence=[revised], compact_evidence=())
        self.assertNotEqual(first.snapshot_id, second.snapshot_id)

        feature = PriceFeatureBuilder(PARAMETERS).build(
            self.evidence, symbol="AAPL", as_of="2026-01-02T21:00:00Z")
        with self.assertRaisesRegex(ValueError, "does not match feature_version"):
            self.build(evidence=[*self.evidence, feature], feature_version="different")


if __name__ == "__main__":
    unittest.main()
