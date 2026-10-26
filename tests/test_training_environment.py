import json
import tempfile
import unittest
from pathlib import Path

from agent.training import HistoricalPPOEnvironment, ReplayEpisode, create_training_environment


def replay_episode(**changes):
    values = dict(
        episode_id="train-1", split="train",
        decision_at="2026-01-02T21:00:00Z", outcome_at="2026-01-03T21:00:00Z",
        messages=({"role": "user", "content": "point-in-time state"},),
        evidence_ids=("e1",), symbol="AAPL", next_open=100.0,
        outcome_close=110.0, starting_cash=1000.0,
    )
    values.update(changes)
    return ReplayEpisode(**values)


class HistoricalPPOEnvironmentTests(unittest.TestCase):
    def test_future_prices_are_private_and_valid_action_earns_net_reward(self):
        environment = HistoricalPPOEnvironment((replay_episode(),))
        observation = environment.reset(seed=7)
        self.assertNotIn("next_open", json.dumps(observation))
        action = json.dumps({"decisions": [{
            "symbol": "AAPL", "action": "buy", "qty": 1, "reason": "signal",
            "evidence_ids": ["e1"],
        }]})
        _next, reward, done, truncated, info = environment.step(action)
        self.assertGreater(reward, 0)
        self.assertTrue(done)
        self.assertFalse(truncated)
        self.assertEqual(info["invalid_proposals"], 0)

    def test_invalid_action_is_penalized_and_nontrain_episode_is_rejected(self):
        environment = HistoricalPPOEnvironment((replay_episode(),))
        environment.reset(seed=1)
        _next, invalid_reward, *_rest = environment.step("not-json")
        self.assertLess(invalid_reward, 0)
        with self.assertRaisesRegex(ValueError, "train-only"):
            replay_episode(split="validation")
        with self.assertRaisesRegex(ValueError, "future outcome"):
            replay_episode(messages=({
                "role": "user", "content": json.dumps({"next_open": 123.0}),
            },))
        with self.assertRaisesRegex(ValueError, "sensitive"):
            replay_episode(messages=({
                "role": "user", "content": {"api_key": "do-not-store"},
            },))

    def test_cli_factory_loads_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ppo.jsonl"
            row = {
                "episode_id": "train-1", "split": "train",
                "decision_at": "2026-01-02T21:00:00Z",
                "outcome_at": "2026-01-03T21:00:00Z",
                "messages": [{"role": "user", "content": "state"}],
                "evidence_ids": ["e1"], "symbol": "AAPL", "next_open": 100,
                "outcome_close": 101, "starting_cash": 1000,
            }
            path.write_text(json.dumps(row) + "\n")
            result = create_training_environment({"episodes_path": str(path)})
            self.assertEqual(result.episodes[0].episode_id, "train-1")

    def test_requires_chronology_but_allows_independent_overlapping_horizons(self):
        later = replay_episode(
            episode_id="later", decision_at="2026-01-04T21:00:00Z",
            outcome_at="2026-01-05T21:00:00Z")
        with self.assertRaisesRegex(ValueError, "chronological"):
            HistoricalPPOEnvironment((later, replay_episode()))
        overlapping = replay_episode(
            episode_id="overlap", decision_at="2026-01-03T20:00:00Z",
            outcome_at="2026-01-04T21:00:00Z")
        environment = HistoricalPPOEnvironment((replay_episode(), overlapping))
        self.assertEqual(len(environment.episodes), 2)

    def test_independent_rows_never_cross_episode_boundaries(self):
        second = replay_episode(
            episode_id="train-2", decision_at="2026-01-04T21:00:00Z",
            outcome_at="2026-01-05T21:00:00Z")
        environment = HistoricalPPOEnvironment((replay_episode(), second))
        observation = environment.reset(seed=1)
        _next, _reward, done, truncated, info = environment.step("not-json")
        self.assertTrue(done)
        self.assertFalse(truncated)
        self.assertEqual(info["episode_id"], observation["episode_id"])


if __name__ == "__main__":
    unittest.main()
