import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from agent.contracts import Decision, canonical_json
from agent.data.prices import DailyPriceBar
from agent.datasets import load_dataset_artifact
from agent.policy.evidence import alias_messages
from agent.training.core import MODEL_NAME, MODEL_REVISION
from agent.trajectories import (
    TrajectorySettings,
    collect_trajectories,
    export_good_dataset,
    load_collection,
)

DAYS = (
    "2026-01-02",
    "2026-01-05",
    "2026-01-06",
    "2026-01-07",
    "2026-01-08",
    "2026-01-09",
    "2026-01-12",
    "2026-01-13",
)
IDENTITY = {
    "model_name": MODEL_NAME,
    "model_revision": MODEL_REVISION,
    "adapter_hash": "base-model",
    "protocol_hash": "test-protocol",
}


def bars(*, flat=False):
    rows = []
    for index, day in enumerate(DAYS):
        opened = 100.0 if flat else 100.0 + index * 10
        closed = opened if flat else opened + 5
        rows.append(
            DailyPriceBar(
                "AAPL",
                day,
                opened,
                closed + 1,
                opened - 1,
                closed,
                closed,
                1000,
                0,
                0,
                f"{day}T14:30:00Z",
                f"{day}T21:00:00Z",
                "2026-02-01T00:00:00Z",
            )
        )
    return {"AAPL": tuple(rows)}


class Policy:
    policy_identity = IDENTITY

    def __init__(self, action="buy", *, bad_citation=False, aliases=False, tools=True):
        self.action = action
        self.bad_citation = bad_citation
        self.aliases = aliases
        self.tools = tools
        self.inputs = []

    def __call__(self, **kwargs):
        self.inputs.append(kwargs)
        evidence = kwargs["evidence"]
        item = max(evidence, key=lambda item: item.available_at)
        messages = [
            {"role": "system", "content": "Research, then return strict JSON."},
            {
                "role": "user",
                "content": canonical_json(
                    {
                        "as_of": kwargs["as_of"],
                        "state": kwargs["state"],
                        "evidence": [item.to_dict()],
                    }
                ),
            },
        ]
        calls = []
        if self.tools:
            request = {"tool": "get_price_history", "arguments": {"symbol": "AAPL"}}
            messages.extend(
                [
                    {"role": "assistant", "content": canonical_json(request)},
                    {
                        "role": "user",
                        "content": canonical_json(
                            {
                                "tool_result": {"status": "ok", "evidence": [item.to_dict()]},
                            }
                        ),
                    },
                ]
            )
            calls.append(request)
        original = messages
        mapping = {}
        reference = item.id
        if self.aliases:
            messages, mapping = alias_messages(messages)
            reference = next(key for key, value in mapping.items() if value == item.id)
        if self.bad_citation:
            reference = "invented"
        raw = canonical_json(
            {
                "decisions": [
                    {
                        "symbol": "AAPL",
                        "action": self.action,
                        "qty": 0 if self.action == "hold" else 1,
                        "reason": "Actual model completion, not a counterfactual replacement.",
                        "evidence_ids": [reference],
                    }
                ]
            }
        )
        full = Decision(
            "AAPL",
            self.action,
            0 if self.action == "hold" else 1,
            "Actual model completion, not a counterfactual replacement.",
            (item.id if not self.bad_citation else "invented",),
        )
        return {
            "status": "success",
            "messages": messages,
            "completion": raw,
            "decisions": (full,),
            "tool_calls": calls,
            "evidence_aliases": mapping,
            "original_messages": original,
        }


class TrajectoryTests(unittest.TestCase):
    def settings(self):
        return TrajectorySettings(
            ("AAPL",), "2026-01-07T23:59:59Z", "2026-01-12T23:59:59Z", "2026-01-14T23:59:59Z"
        )

    def collect(self, policy=None, *, rows=None, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return collect_trajectories(
                rows or bars(),
                decide=policy or Policy(),
                settings=self.settings(),
                next_session_resolver=dict(zip(DAYS, DAYS[1:], strict=False)).__getitem__,
                **kwargs,
            )

    def test_smoke_gate_requires_a_researched_trade_and_keeps_invalid_prefix(self):
        from agent.trajectory_cli import review_smoke

        with contextlib.redirect_stdout(io.StringIO()):
            self.assertGreater(review_smoke(self.collect())["executable_researched_sessions"], 0)
            for policy in (Policy("hold"), Policy(bad_citation=True), Policy(tools=False)):
                collection = self.collect(policy)
                with self.assertRaisesRegex(ValueError, "no researched executable"):
                    review_smoke(collection)
                self.assertGreater(len(collection.records), 0)

    def test_callback_failure_keeps_code_location_without_exception_values(self):
        def broken(**_kwargs):
            raise ValueError("SECRET-PROVIDER-ECHO")

        result = self.collect(broken, max_sessions=1, policy_identity=IDENTITY)
        row = result.records[0]
        self.assertEqual(row["policy_status"], "policy_setup_failure")
        self.assertEqual(row["error_location"]["function"], "broken")
        self.assertNotIn("SECRET-PROVIDER-ECHO", canonical_json(row))

    def test_actual_base_decisions_are_labeled_without_future_policy_inputs(self):
        policy = Policy()
        result = self.collect(policy)
        self.assertEqual(len(policy.inputs), 3)
        self.assertTrue(all(row["label"] == "good" for row in result.records))
        for inputs, row in zip(policy.inputs, result.records, strict=True):
            self.assertEqual(set(inputs), {"state", "as_of", "evidence", "universe"})
            self.assertTrue(all(item.eligible(inputs["as_of"]) for item in inputs["evidence"]))
            self.assertLessEqual(inputs["as_of"][:10], "2026-01-06")
            self.assertNotIn("next_open", canonical_json(inputs["state"]))
            self.assertEqual(row["completion"], Policy().__call__(**inputs)["completion"])
        self.assertEqual(result.records[0]["state"]["positions"], {})
        self.assertEqual(result.records[1]["state"]["positions"], {"AAPL": 1})

    def test_hold_invalid_and_unresearched_outputs_are_not_good(self):
        for policy, label in (
            (Policy("hold"), "unlabeled"),
            (Policy(bad_citation=True), "invalid"),
            (Policy(tools=False), "unlabeled"),
            (Policy("sell"), "invalid"),
        ):
            with self.subTest(label=label, policy=policy):
                result = self.collect(policy, max_sessions=1)
                self.assertEqual(result.records[0]["label"], label)
                self.assertNotEqual(result.records[0]["completion"], "")

    def test_costs_change_flat_price_trade_to_bad(self):
        result = self.collect(rows=bars(flat=True), max_sessions=1)
        row = result.records[0]
        self.assertEqual(row["label"], "bad")
        self.assertAlmostEqual(row["net_equity_change"], -0.05)
        self.assertAlmostEqual(row["fills"][0]["price"], 100.05)

    def test_pooling_training_folds_preserves_independent_accounts_and_rejects_overlap(self):
        from agent.trajectories import merge_train_collections

        full = self.collect()
        one = replace(full, records=full.records[:1], planned_sessions=full.planned_sessions[:1])
        two = replace(full, records=full.records[1:], planned_sessions=full.planned_sessions[1:])
        pooled = merge_train_collections((one, two))
        self.assertEqual(pooled.planned_sessions, full.planned_sessions)
        self.assertTrue(all(row["independent_fold"] for row in pooled.records))
        self.assertEqual({row["source_fold"] for row in pooled.records}, {0, 1})
        with self.assertRaisesRegex(ValueError, "overlap"):
            merge_train_collections((one, one))
        with self.assertRaisesRegex(ValueError, "complete"):
            merge_train_collections((replace(full, records=full.records[:1]),))
        with self.assertRaises(ValueError):
            merge_train_collections(())

    def test_validation_starts_separate_account_and_test_collection_is_blocked(self):
        train = self.collect()
        policy = Policy()
        validation = self.collect(policy, split="validation")
        self.assertEqual(train.split, "train")
        self.assertEqual(validation.split, "validation")
        self.assertEqual(policy.inputs[0]["state"]["positions"], {})
        self.assertEqual(policy.inputs[0]["state"]["cash"], 100000)
        with self.assertRaisesRegex(ValueError, "verified frozen release"):
            self.collect(split="test")

    def test_good_export_preserves_tool_turns_raw_completion_and_verified_splits(self):
        train, validation = (
            self.collect(Policy(aliases=True)),
            self.collect(Policy(aliases=True), split="validation"),
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "good"
            artifact = export_good_dataset(train, validation, output)
            loaded = load_dataset_artifact(output / "dataset")
            self.assertEqual(artifact, loaded)
            self.assertEqual(len(loaded.examples), 5)
            self.assertEqual(loaded.manifest.split_example_ids["test"], ())
            self.assertTrue(any(m["role"] == "assistant" for m in loaded.examples[0].messages))
            self.assertEqual(loaded.examples[0].completion, train.records[0]["completion"])
            manifest = json.loads((output / "trajectory-manifest.json").read_text())
            self.assertEqual(manifest["train_counts"]["labels"], {"good": 3})
            self.assertEqual(manifest["test_policy_calls"], 0)
            with self.assertRaisesRegex(ValueError, "already exists"):
                export_good_dataset(train, validation, output)

    def test_export_rejects_empty_good_and_sft_source_identity(self):
        train, validation = self.collect(Policy("hold")), self.collect(split="validation")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "no valid GOOD train"):
                export_good_dataset(train, validation, Path(directory) / "bad")
        with self.assertRaisesRegex(ValueError, "without adapter"):
            self.collect(policy_identity={**IDENTITY, "adapter_hash": "sft-adapter"})

    def test_future_evidence_provider_fails_before_policy_call(self):
        policy = Policy()
        future = bars()["AAPL"][-1].to_evidence()
        with self.assertRaisesRegex(ValueError, "ineligible evidence"):
            self.collect(
                policy, evidence_provider=lambda *_: (future,), evidence_source_hash="a" * 64
            )
        self.assertEqual(policy.inputs, [])

    def test_journal_resumes_prefix_without_duplicate_calls_and_rejects_changed_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = Path(directory) / "journal"
            first = Policy()
            self.collect(first, journal=journal, max_sessions=1)
            self.assertEqual(len(first.inputs), 1)
            resumed = Policy()
            result = self.collect(resumed, journal=journal, resume=True, max_sessions=3)
            self.assertEqual(len(resumed.inputs), 2)
            self.assertEqual(len(result.records), 3)
            self.assertEqual(load_collection(journal), result)
            self.assertEqual(resumed.inputs[0]["state"]["positions"], {"AAPL": 1})
            with self.assertRaisesRegex(ValueError, "resume contract changed"):
                self.collect(
                    journal=journal,
                    resume=True,
                    policy_identity={**IDENTITY, "protocol_hash": "changed"},
                )

    def test_journal_rejects_tampered_completed_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = Path(directory) / "journal"
            self.collect(journal=journal, max_sessions=1)
            path = journal / "episode-000000.json"
            envelope = json.loads(path.read_text())
            envelope["record"]["equity_after"] += 1
            path.write_text(json.dumps(envelope))
            with self.assertRaisesRegex(ValueError, "record hash"):
                load_collection(journal)

    def test_journal_recovers_single_atomic_episode_commit_interruption(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = Path(directory) / "journal"
            self.collect(journal=journal, max_sessions=1)
            marker = (journal / "collection-state.json").read_text()
            self.collect(journal=journal, resume=True, max_sessions=2)
            (journal / "collection-state.json").write_text(marker)
            restored = load_collection(journal)
            self.assertEqual(len(restored.records), 2)
            self.assertEqual(
                json.loads((journal / "collection-state.json").read_text())["record_count"], 2
            )
            policy = Policy()
            self.collect(policy, journal=journal, resume=True, max_sessions=3)
            self.assertEqual(len(policy.inputs), 1)

    def test_changed_evidence_archive_cannot_resume_same_bars_and_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = Path(directory) / "journal"

            def provider(*_):
                return ()

            with self.assertRaisesRegex(ValueError, "evidence_source_hash"):
                self.collect(evidence_provider=provider)
            self.collect(
                evidence_provider=provider,
                evidence_source_hash="a" * 64,
                journal=journal,
                max_sessions=1,
            )
            with self.assertRaisesRegex(ValueError, "resume contract changed"):
                self.collect(
                    evidence_provider=provider,
                    evidence_source_hash="b" * 64,
                    journal=journal,
                    resume=True,
                )

    def test_journal_rejects_partial_or_missing_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = Path(directory) / "journal"
            self.collect(journal=journal, max_sessions=2)
            (journal / "episode-000000.json").unlink()
            with self.assertRaisesRegex(ValueError, "complete prefix"):
                self.collect(journal=journal, resume=True)

    def test_account_prompt_mismatch_and_fabricated_tool_count_are_invalid(self):
        base = Policy()

        def wrong_state(**kwargs):
            result = base(**kwargs)
            payload = json.loads(result["messages"][1]["content"])
            payload["state"]["cash"] = 1e9
            result["messages"][1]["content"] = canonical_json(payload)
            return result

        def wrong_count(**kwargs):
            result = base(**kwargs)
            result["tool_calls"] = 4
            return result

        for callback in (wrong_state, wrong_count):
            with self.subTest(callback=callback):
                row = self.collect(callback, policy_identity=IDENTITY, max_sessions=1).records[0]
                self.assertEqual(row["label"], "invalid")

    def test_callback_cannot_mutate_actual_portfolio_state(self):
        base = Policy()

        def mutate(**kwargs):
            kwargs["state"]["positions"]["AAPL"] = 500
            return base(**kwargs)

        result = self.collect(mutate, policy_identity=IDENTITY, max_sessions=1)
        self.assertEqual(result.records[0]["state"]["positions"], {})
        self.assertEqual(result.records[0]["label"], "invalid")

    def test_selected_gap_applies_intervening_dividend_before_next_decision(self):
        data = bars()
        rows = list(data["AAPL"])
        rows[2] = replace(rows[2], dividends=2)
        data["AAPL"] = tuple(rows)
        result = self.collect(rows=data, decision_sessions=(DAYS[0], DAYS[2]))
        self.assertEqual(len(result.records), 2)
        self.assertAlmostEqual(
            result.records[1]["state"]["cash"], result.records[0]["portfolio_after"]["cash"] + 2
        )
        self.assertEqual(result.planned_sessions, (DAYS[0], DAYS[2]))


if __name__ == "__main__":
    unittest.main()
