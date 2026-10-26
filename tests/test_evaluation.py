import unittest

from agent.evaluation import (
    EpisodeWindow, ExperimentRegistry, ExperimentRunner, MarketSession, benchmark_curves,
    block_bootstrap_return_ci, exposure_match_returns, financial_metrics,
    purged_split, rolling_folds,
)


def episode(name, decision, outcome):
    return EpisodeWindow(name, f"2026-01-{decision:02d}T21:00:00Z",
                         f"2026-01-{outcome:02d}T21:00:00Z")


class SplitTests(unittest.TestCase):
    def test_purges_outcome_overlap(self):
        result = purged_split(
            [episode("train", 1, 2), episode("cross", 2, 4),
             episode("validation", 4, 5), episode("test", 7, 8)],
            train_end="2026-01-03T00:00:00Z",
            validation_end="2026-01-06T00:00:00Z",
            test_end="2026-01-09T00:00:00Z",
        )
        self.assertEqual([item.episode_id for item in result["train"]], ["train"])
        self.assertEqual([item.episode_id for item in result["validation"]], ["validation"])
        self.assertEqual([item.episode_id for item in result["test"]], ["test"])
        self.assertEqual([item.episode_id for item in result["purged"]], ["cross"])

    def test_rolling_folds_are_disjoint(self):
        folds = rolling_folds(
            [f"2026-01-{day:02d}" for day in range(1, 13)],
            train_size=5, validation_size=2, test_size=2, step=2,
        )
        self.assertEqual(len(folds), 2)
        for fold in folds:
            self.assertFalse(set(fold.train_sessions) & set(fold.validation_sessions))
            self.assertFalse(set(fold.validation_sessions) & set(fold.test_sessions))


def market_sessions():
    values = []
    for index in range(8):
        values.append(MarketSession(
            f"2026-01-{index + 1:02d}",
            {"SPY": 100 + index, "AAA": 100, "BBB": 100},
            {"SPY": 101 + index, "AAA": 100 + 2 * index, "BBB": 100 - index},
        ))
    return values


class BenchmarkTests(unittest.TestCase):
    def test_empty_sessions_are_rejected_cleanly(self):
        with self.assertRaisesRegex(ValueError, "nonempty"):
            benchmark_curves([], initial_cash=100_000, benchmark_symbol="SPY")

    def test_cash_and_buy_hold_share_starting_capital(self):
        curves = benchmark_curves(
            market_sessions(), initial_cash=100_000, benchmark_symbol="SPY",
            momentum_lookback=2,
        )
        self.assertTrue(all(curve.equities[0] == 100_000 for curve in curves.values()))
        self.assertTrue(all(value == 100_000 for value in curves["cash"].equities))
        self.assertGreater(curves["buy_and_hold"].equities[-1], 100_000)

    def test_momentum_has_no_position_before_lookback_and_costs_apply_once(self):
        zero_cost = benchmark_curves(
            market_sessions(), initial_cash=100_000, benchmark_symbol="SPY",
            momentum_lookback=2, cost_bps=0,
        )["momentum"]
        with_cost = benchmark_curves(
            market_sessions(), initial_cash=100_000, benchmark_symbol="SPY",
            momentum_lookback=2, cost_bps=10,
        )["momentum"]
        self.assertEqual(zero_cost.equities[:4], (100_000.0,) * 4)
        self.assertGreater(zero_cost.equities[-1], with_cost.equities[-1])
        self.assertGreater(with_cost.turnover, 0)


class MetricsTests(unittest.TestCase):
    def test_metrics_and_short_sample_behavior(self):
        result = financial_metrics(
            [100, 110, 99], benchmark_equities=[100, 105, 105], turnover=20)
        self.assertAlmostEqual(result["total_return"], -0.01)
        self.assertAlmostEqual(result["excess_return"], -0.06)
        self.assertAlmostEqual(result["max_drawdown"], 0.1)
        short = financial_metrics([100])
        self.assertIsNone(short["annualized_sharpe"])
        self.assertEqual(short["n_returns"], 0)

    def test_block_bootstrap_is_seeded_and_exposure_matching_is_explicit(self):
        first = block_bootstrap_return_ci(
            [.01, -.02, .03, .01], block_size=2, samples=100, seed=7)
        second = block_bootstrap_return_ci(
            [.01, -.02, .03, .01], block_size=2, samples=100, seed=7)
        self.assertEqual(first, second)
        self.assertLessEqual(first["lower"], first["upper"])
        self.assertEqual(
            exposure_match_returns((.02, .03, .01), (1.0, .5, 0), target_gross=.5),
            (.01, .03, 0.0),
        )


class RegistryTests(unittest.TestCase):
    def test_records_all_attempts_and_final_test_is_one_shot(self):
        registry = ExperimentRegistry()
        first = registry.record({"lr": 1e-4}, split="validation", metrics={"return": .1})
        second = registry.record({"lr": 2e-4}, split="validation", metrics={"return": .2})
        self.assertNotEqual(first.config_hash, second.config_hash)
        registry.record({"lr": 2e-4}, split="test", metrics={"return": .15})
        with self.assertRaisesRegex(ValueError, "already been consumed"):
            registry.record({"lr": 1e-4}, split="test", metrics={"return": .3})
        self.assertEqual(len(registry.attempts), 3)

    def test_runner_records_evaluator_output_immutably(self):
        runner = ExperimentRunner(
            lambda config, split: {"score": config["score"], "split": split})
        attempt = runner.run({"score": 1.0}, split="validation")
        self.assertEqual(attempt.metrics["score"], 1.0)
        with self.assertRaises(TypeError):
            attempt.metrics["score"] = 2.0


if __name__ == "__main__":
    unittest.main()
