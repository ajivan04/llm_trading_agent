"""Selective multi-turn supervision must never learn provider text as output."""

import dataclasses
import unittest

from agent.training.core import build_trajectory_sft_example
from tests.test_training import episode


class NativeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        text = "".join(f"<{m['role']}>{m['content']}~" for m in messages)
        if kwargs.get("add_generation_prompt"):
            text += "<assistant>"
        return {"input_ids": [ord(x) for x in text], "attention_mask": [1] * len(text)}


class TrajectoryMaskTests(unittest.TestCase):
    def test_tool_requests_and_final_supervised_but_results_masked(self):
        messages = (
            {"role": "system", "content": "RULES"},
            {"role": "user", "content": "ACCOUNT"},
            {"role": "assistant", "content": '{"name":"get_rsi","arguments":{"symbol":"AAPL"}}'},
            {"role": "user", "content": "PRIVATE_PROVIDER_TEXT"},
        )
        item = dataclasses.replace(episode(), messages=messages)
        built = build_trajectory_sft_example(item, NativeTokenizer(), max_length=4096)
        learned = "".join(chr(x) for x in built.labels if x != -100)
        self.assertEqual(learned, messages[2]["content"] + "~" + item.completion + "~")
        for field in ("RULES", "ACCOUNT", "PRIVATE_PROVIDER_TEXT", "<assistant>"):
            self.assertNotIn(field, learned)
        with self.assertRaisesRegex(ValueError, "max_length"):
            build_trajectory_sft_example(item, NativeTokenizer(), max_length=20)
        with self.assertRaisesRegex(ValueError, "allowed splits"):
            build_trajectory_sft_example(
                item, NativeTokenizer(), max_length=4096, allowed_splits=("validation",)
            )

    def test_arbitrary_assistant_history_is_rejected(self):
        item = dataclasses.replace(
            episode(),
            messages=(
                {"role": "user", "content": "state"},
                {"role": "assistant", "content": "hallucinated final"},
                {"role": "user", "content": "data"},
            ),
        )
        with self.assertRaisesRegex(ValueError, "actual tool request"):
            build_trajectory_sft_example(item, NativeTokenizer(), max_length=4096)


class PinnedTokenizerMaskTests(unittest.TestCase):
    def test_cached_qwen_native_template_masks_provider_text(self):
        from agent.training.core import MODEL_NAME, MODEL_REVISION
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME, revision=MODEL_REVISION, local_files_only=True)
        except (ImportError, OSError):
            self.skipTest('pinned Qwen tokenizer not cached; cluster preflight is mandatory')
        request = '{"tool":"get_rsi","arguments":{"symbol":"AAPL"}}'
        item = dataclasses.replace(episode(), messages=(
            {'role': 'system', 'content': 'RULES'},
            {'role': 'user', 'content': 'ACCOUNT'},
            {'role': 'assistant', 'content': request},
            {'role': 'user', 'content': 'PRIVATE_PROVIDER_TEXT'},
        ))
        built = build_trajectory_sft_example(item, tokenizer, max_length=4096)
        learned = tokenizer.decode([x for x in built.labels if x != -100])
        self.assertEqual(learned, request + '<|im_end|>\n' + item.completion + '<|im_end|>\n')
        self.assertNotIn('PRIVATE_PROVIDER_TEXT', learned)


class PreallocationTests(unittest.TestCase):
    def test_oversize_fails_before_policy_weights_are_loaded(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from agent.training.execution import TrainerSettings, train_sft
        from tests.test_training_execution import FakeModel, example, stack
        with tempfile.TemporaryDirectory() as root:
            settings = TrainerSettings(Path(root), "test", ("data",), max_length=1)
            with patch.object(FakeModel, "from_pretrained") as load:
                with self.assertRaisesRegex(ValueError, "max_length"):
                    train_sft([example("train")], [example("validation")],
                              settings=settings, stack_resolver=stack)
                load.assert_not_called()
