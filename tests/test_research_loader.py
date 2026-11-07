"""Research token budgeting must stay CPU-only until inputs fit."""
import unittest
from unittest.mock import Mock

from agent.policy.loader import HuggingFaceQwenBackend


class ResearchLoaderTests(unittest.TestCase):
    def test_token_count_does_not_load_policy_weights_and_uses_native_dict(self):
        backend = HuggingFaceQwenBackend(max_context_tokens=20)
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = {'input_ids': [1, 2, 3]}
        backend._tokenizer = tokenizer
        backend.load = Mock(side_effect=AssertionError('weights must not load'))
        self.assertEqual(backend.token_count([{'role': 'user', 'content': 'state'}]), 3)
        kwargs = tokenizer.apply_chat_template.call_args.kwargs
        self.assertTrue(kwargs['return_dict'])
        self.assertTrue(kwargs['add_generation_prompt'])
        self.assertFalse(kwargs['truncation'])
        with self.assertRaisesRegex(ValueError, 'context budget'):
            backend.generate([{'role': 'user', 'content': 'state'}], max_new_tokens=18)
        backend.load.assert_not_called()

    def test_rejects_remote_code_and_invalid_budget(self):
        with self.assertRaises(ValueError):
            HuggingFaceQwenBackend(model_kwargs={'trust_remote_code': True})
        with self.assertRaises(ValueError):
            HuggingFaceQwenBackend(max_context_tokens=0)


class SupervisedCitationTests(unittest.TestCase):
    def test_supervised_trade_rejects_another_symbols_evidence_and_empty_citations(self):
        from dataclasses import replace

        from agent.context import build_context
        from agent.contracts import Evidence, canonical_json
        from agent.training.validation import validate_supervised_example
        from tests.test_training import episode

        cutoff = '2026-01-01T00:00:00Z'
        records = [Evidence(s, 'test', s, cutoff, cutoff, cutoff, 'fact')
                   for s in ('AAPL', 'MSFT')]
        context = build_context(
            state={'as_of': cutoff}, evidence=records, token_counter=lambda _: 1,
            max_context_tokens=4096, reserved_final_tokens=512)
        for citations in ([], [next(k for k, v in context.aliases.items() if v == 'MSFT')]):
            completion = canonical_json({'decisions': [{
                'symbol': 'AAPL', 'action': 'buy', 'qty': 1, 'reason': 'test',
                'evidence_ids': citations}]})
            item = replace(episode(), messages=context.packed.messages, completion=completion)
            with self.assertRaises(ValueError):
                validate_supervised_example(item)
