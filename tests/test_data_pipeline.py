import unittest
from datetime import datetime, timezone

from agent.data.prices import YFinancePriceAdapter
from agent.features import FeatureParameters, PriceFeatureBuilder
from agent.snapshots import SnapshotBuilder


UTC = timezone.utc


def session_bounds(_symbol, session):
    return (
        datetime(session.year, session.month, session.day, 14, 30, tzinfo=UTC),
        datetime(session.year, session.month, session.day, 21, 0, tzinfo=UTC),
    )


class HistoricalDataPipelineTests(unittest.TestCase):
    def test_adapter_features_and_snapshot_share_one_schema(self):
        rows = []
        for day, close in enumerate((100.0, 101.0, 103.0, 106.0), start=5):
            session = f"2026-01-{day:02d}"
            rows.append({
                "session": session,
                "Open": close - 1.0,
                "High": close + 1.0,
                "Low": close - 2.0,
                "Close": close,
                "Adj Close": close / 2.0,
                "Volume": day * 1_000,
                "Dividends": 0.0,
                "Stock Splits": 0.0,
            })

        adapter = YFinancePriceAdapter(session_bounds, lambda *args, **kwargs: rows)
        bars = adapter.fetch(
            "AAPL", start="2026-01-05", end="2026-01-10",
            retrieved_at="2026-01-10T22:00:00Z",
        )
        price_evidence = tuple(bar.to_evidence() for bar in bars)
        parameters = FeatureParameters(
            return_window=2, sma_window=2, rsi_window=2, macd_fast=2,
            macd_slow=3, volatility_window=2, volume_window=2,
            version="pipeline-v1",
        )
        feature = PriceFeatureBuilder(parameters).build(
            price_evidence, symbol="AAPL", as_of="2026-01-08T21:00:00Z",
        )
        snapshot = SnapshotBuilder().build(
            as_of="2026-01-08T21:00:00Z",
            universe=("AAPL",),
            universe_version="assignment-v1",
            evidence=(*price_evidence, feature),
            compact_evidence=(
                {"evidence_id": feature.id, "features": feature.values["features"]},
            ),
            feature_version="pipeline-v1",
            account_state={"cash": 100_000.0, "positions": {}},
            source_health={"yfinance": {"status": "ok"}},
            market_session_id="XNYS-2026-01-08",
        )

        self.assertEqual(
            feature.values["price_basis"],
            "point_in_time_total_return_from_raw_close_actions",
        )
        self.assertEqual(feature.values["latest_session"], "2026-01-08")
        self.assertEqual(snapshot.evidence_ids[-1], feature.id)
        self.assertEqual(snapshot.compact_evidence[0]["evidence_id"], feature.id)


if __name__ == "__main__":
    unittest.main()
