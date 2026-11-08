"""Real-tokenizer regressions for mandatory 100-symbol research context."""

import json
import unittest
from dataclasses import replace
from unittest.mock import patch

from agent.context import ContextBudgetError, build_context
from agent.contracts import Evidence, canonical_json
from agent.policy.evidence import visible_evidence
from agent.policy.loader import POLICY_MODEL, POLICY_REVISION
from agent.research import ResearchPolicy
from agent.tools import AssignmentTools
from agent.universe import BASELINE_UNIVERSE
from tests.test_trajectories import bars

CUTOFF = "2026-01-13T21:00:00Z"


def full_source():
    rows = []
    for symbol in BASELINE_UNIVERSE:
        for bar in bars()["AAPL"]:
            rows.append(
                replace(
                    bar,
                    symbol=symbol,
                    **{
                        name: getattr(bar, name) + 0.12345678901234
                        for name in ("open", "high", "low", "close")
                    },
                ).to_evidence()
            )
        for source in ("alpaca.news", "sec.filing_text"):
            rows.append(
                Evidence(
                    source + symbol,
                    source,
                    symbol,
                    CUTOFF,
                    CUTOFF,
                    "2026-02-01T00:00:00Z",
                    "Company earnings and demand improved. " * 50,
                )
            )
    return tuple(rows)


def account():
    return {"as_of": CUTOFF, "cash": 100000, "equity": 100000, "positions": {}, "paper_only": True}


class CachedNativeBackend:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.inputs = []

    def token_count(self, messages):
        encoded = self.tokenizer.apply_chat_template(
            list(messages), tokenize=True, add_generation_prompt=True, return_dict=True
        )
        return len(encoded["input_ids"])

    def generate(self, messages, *, max_new_tokens):
        self.inputs.append(json.loads(canonical_json(messages)))
        if len(self.inputs) == 1:
            return canonical_json({"tool": "get_global_quote", "arguments": {"symbol": "MSFT"}})
        visible = visible_evidence(messages, expected_as_of=CUTOFF)
        quote = next(
            eid
            for eid, row in visible.items()
            if row["source"] == "derived.historical_quote" and row["symbol"] == "MSFT"
        )
        return canonical_json(
            {
                "decisions": [
                    {
                        "symbol": "MSFT",
                        "action": "buy",
                        "qty": 1,
                        "reason": "Visible quote.",
                        "evidence_ids": [quote],
                    }
                ]
            }
        )


class FullUniverseContextTests(unittest.TestCase):
    def test_native_context_with_precise_prices_raw_nlp_and_dynamic_outside_candidate_tool(self):
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                POLICY_MODEL, revision=POLICY_REVISION, local_files_only=True
            )
        except (ImportError, OSError):
            self.skipTest("pinned Qwen tokenizer is not cached; run cluster smoke")
        backend = CachedNativeBackend(tokenizer)
        policy = ResearchPolicy(backend)
        rows, state = full_source(), account()
        # Reproduce the failing legacy envelope with all 100 precise marks.
        initial, required, candidates, marks = policy._initial_context(
            rows, BASELINE_UNIVERSE, state, CUTOFF
        )
        with AssignmentTools(CUTOFF, BASELINE_UNIVERSE, evidence=rows) as tools:
            from agent.policy import ToolRegistry

            registry = ToolRegistry(CUTOFF)
            tools.register_tools(registry)
            with self.assertRaises(ContextBudgetError):
                build_context(
                    state={
                        **state,
                        "universe": BASELINE_UNIVERSE,
                        "research_candidates": candidates,
                        "price_marks": marks,
                    },
                    evidence=initial,
                    tools=registry.schemas,
                    token_counter=backend.token_count,
                    max_context_tokens=3320,
                    reserved_final_tokens=512,
                    config=policy.settings.context,
                    required_evidence_ids=required,
                )
        # A held symbol outside the shortlist must still retain its exact mark.
        state["positions"] = {"XLI": 1}
        with patch("transformers.AutoModelForCausalLM.from_pretrained") as load:
            result = policy(state=state, as_of=CUTOFF, evidence=rows, universe=BASELINE_UNIVERSE)
            load.assert_not_called()
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["decisions"][0].symbol, "MSFT")
        self.assertEqual(len(backend.inputs), 2)
        payload = json.loads(backend.inputs[0][1]["content"])
        self.assertEqual(payload["state"]["universe"], list(BASELINE_UNIVERSE))
        self.assertIn("XLI", payload["state"]["price_marks"])
        self.assertEqual(len(payload["state"]["price_marks"]), 8)
        self.assertNotIn("MSFT", payload["state"]["price_marks"])
        sources = {row["source"] for row in visible_evidence(backend.inputs[0]).values()}
        self.assertIn("alpaca.news", sources)
        self.assertIn("sec.filing_text", sources)
        for messages in backend.inputs:
            self.assertLessEqual(backend.token_count(messages) + 512, 4088)

    def test_budget_failure_is_explicit_and_never_calls_generation_or_exposes_data(self):
        class Tiny:
            def token_count(self, _messages):
                return 100000

            def generate(self, *_args, **_kwargs):
                raise AssertionError("generation must not run")

        result = ResearchPolicy(Tiny())(
            state=account(), as_of=CUTOFF, evidence=full_source(), universe=BASELINE_UNIVERSE
        )
        self.assertEqual(result["status"], "context_failure")
        self.assertEqual(result["failure"], "required_context_exceeds_budget")
        self.assertEqual(
            result["context_diagnostics"],
            {"minimum_input_tokens": 100000, "input_token_budget": 2808},
        )
        self.assertEqual(result["completion"], "")
        self.assertEqual(result["messages"], [])

    def test_collector_retains_numeric_context_failure_without_fabricated_output(self):
        import contextlib
        import io

        from agent.trajectories import TrajectorySettings, collect_trajectories
        from tests.test_trajectories import DAYS, IDENTITY

        class Tiny:
            def token_count(self, _messages):
                return 100000

            def generate(self, *_args, **_kwargs):
                raise AssertionError("generation must not run")

        with contextlib.redirect_stdout(io.StringIO()):
            collection = collect_trajectories(
                bars(),
                decide=ResearchPolicy(Tiny()),
                policy_identity=IDENTITY,
                settings=TrajectorySettings(
                    ("AAPL",),
                    "2026-01-07T23:59:59Z",
                    "2026-01-12T23:59:59Z",
                    "2026-01-14T23:59:59Z",
                ),
                next_session_resolver=dict(zip(DAYS, DAYS[1:], strict=False)).__getitem__,
                max_sessions=1,
            )
        row = collection.records[0]
        self.assertEqual(row["policy_status"], "context_failure")
        self.assertEqual(row["policy_failure"], "required_context_exceeds_budget")
        self.assertEqual(row["context_diagnostics"]["minimum_input_tokens"], 100000)
        self.assertEqual(row["label"], "invalid")
        self.assertEqual(row["completion"], "")
        self.assertFalse(row["fills"])
