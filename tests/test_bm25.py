"""Focused tests for the time- and run-isolated BM25 memory ranker."""

from datetime import datetime, timedelta, timezone
import unittest

from agent.retrieval.bm25 import BM25Retriever, MemoryRecord, rank_bm25, tokenize


UTC = timezone.utc
AS_OF = datetime(2026, 1, 10, 15, tzinfo=UTC)


def memory(
    memory_id: str,
    situation: str,
    *,
    experiment_id: str = "experiment-a",
    run_id: str = "run-a",
    fold: str = "train",
    available_at: datetime = AS_OF,
) -> MemoryRecord:
    return MemoryRecord(memory_id, experiment_id, run_id, fold, available_at, situation)


def rank(records, query: str = "AAPL", **kwargs):
    return rank_bm25(
        records, experiment_id="experiment-a", run_id="run-a", fold="train",
        query=query, as_of=AS_OF, **kwargs,
    )


class BM25Tests(unittest.TestCase):
    def test_tokenization_is_casefolded_unicode_words(self) -> None:
        self.assertEqual(tokenize("AAPL's CAFÉ_2 + Growth"), ("aapl", "s", "café_2", "growth"))

    def test_ranking_rewards_repeated_term_and_limits_k(self) -> None:
        records = [
            memory("one", "AAPL momentum"),
            memory("two", "AAPL AAPL AAPL momentum"),
            memory("three", "momentum elsewhere"),
        ]

        results = rank(records, k=1)

        self.assertEqual([result.memory.id for result in results], ["two"])
        self.assertGreater(results[0].score, 0)

    def test_future_and_other_run_are_excluded_from_corpus_and_results(self) -> None:
        eligible = memory("eligible", "AAPL momentum")
        future = memory("future", "AAPL AAPL AAPL", available_at=AS_OF + timedelta(seconds=1))
        other_run = memory("other", "AAPL AAPL AAPL", run_id="run-b")

        isolated = rank([eligible])
        mixed = rank([eligible, future, other_run])

        self.assertEqual([item.memory.id for item in mixed], ["eligible"])
        self.assertEqual(mixed[0].score, isolated[0].score)

    def test_available_at_cutoff_is_inclusive_across_timezones(self) -> None:
        eastern = timezone(timedelta(hours=-5))
        same_instant = memory(
            "same-instant", "risk", available_at=datetime(2026, 1, 10, 10, tzinfo=eastern)
        )

        results = rank([same_instant], query="risk")

        self.assertEqual([item.memory.id for item in results], ["same-instant"])

    def test_equal_scores_use_memory_id_as_stable_tie_break(self) -> None:
        records = [memory("z-id", "sector rotation"), memory("a-id", "sector rotation")]

        results = rank(records, query="rotation")

        self.assertEqual([item.memory.id for item in results], ["a-id", "z-id"])

    def test_repeated_query_terms_do_not_change_ranking_or_scores(self) -> None:
        records = [memory("one", "AAPL momentum"), memory("two", "AAPL AAPL")]

        once = rank(records)
        repeated = rank(records, query="AAPL aapl")

        self.assertEqual(repeated, once)

    def test_zero_matches_and_empty_query_return_nothing(self) -> None:
        records = [memory("one", "AAPL momentum")]

        self.assertEqual(rank(records, query="bonds"), [])
        self.assertEqual(rank(records, query="..."), [])

    def test_callable_loader_keeps_storage_outside_ranker(self) -> None:
        calls: list[str] = []

        def load(experiment_id: str, run_id: str, fold: str) -> list[MemoryRecord]:
            calls.append(f"{experiment_id}/{run_id}/{fold}")
            return [memory("one", "profitable momentum")]

        results = BM25Retriever(load).retrieve(
            "experiment-a", "run-a", "train", "momentum", AS_OF
        )

        self.assertEqual(calls, ["experiment-a/run-a/train"])
        self.assertEqual(results[0].memory.id, "one")

    def test_invalid_k_is_rejected(self) -> None:
        for invalid in (0, -1, 1.5, True):
            with self.subTest(k=invalid), self.assertRaises(ValueError):
                rank([], query="risk", k=invalid)  # type: ignore[arg-type]

    def test_naive_timestamps_are_rejected(self) -> None:
        naive = datetime(2026, 1, 10, 15)
        with self.assertRaises(ValueError):
            MemoryRecord("id", "experiment-a", "run-a", "train", naive, "risk")
        with self.assertRaises(ValueError):
            rank_bm25([], experiment_id="experiment-a", run_id="run-a", fold="train",
                      query="risk", as_of=naive)

    def test_invalid_identity_and_record_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            rank_bm25([], experiment_id="experiment-a", run_id="", fold="train",
                      query="risk", as_of=AS_OF)
        with self.assertRaises(TypeError):
            rank([object()], query="risk")  # type: ignore[list-item]

    def test_experiment_and_fold_are_isolated(self) -> None:
        eligible = memory("eligible", "AAPL momentum")
        other_experiment = memory(
            "other-experiment", "AAPL AAPL", experiment_id="experiment-b"
        )
        other_fold = memory("other-fold", "AAPL AAPL", fold="validation")

        results = rank([eligible, other_experiment, other_fold])

        self.assertEqual([item.memory.id for item in results], ["eligible"])


if __name__ == "__main__":
    unittest.main()
