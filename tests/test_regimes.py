import unittest
from datetime import date

from agent.evaluation import build_midterm_regime_plan, midterm_election_date


class ElectionRegimeTests(unittest.TestCase):
    def test_midterm_dates_and_target_week(self):
        self.assertEqual(midterm_election_date(2026), date(2026, 11, 3))
        plan = build_midterm_regime_plan()
        self.assertEqual(plan.target.start, date(2026, 11, 2))
        self.assertEqual(plan.target.end, date(2026, 11, 9))
        self.assertEqual(plan.training_years, (2018, 2014, 2010, 2006))
        self.assertEqual(plan.validation_window.year, 2022)
        self.assertNotIn(2026, plan.training_years)

    def test_rejects_presidential_and_target_overlap(self):
        with self.assertRaises(ValueError):
            midterm_election_date(2024)
        with self.assertRaises(ValueError):
            build_midterm_regime_plan(historical_years=(2026, 2022))


if __name__ == "__main__":
    unittest.main()
