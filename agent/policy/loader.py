"""Lazy Hugging Face backend fixed to the assignment's Qwen policy."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


POLICY_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
POLICY_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"


class HuggingFaceQwenBackend:
    def __init__(self, *, model_name: str = POLICY_MODEL, revision: str = POLICY_REVISION,
                 tokenizer_revision: str | None = None, model_kwargs: Mapping[str, Any] | None = None) -> None:
        if model_name != POLICY_MODEL:
            raise ValueError("policy model is fixed by the assignment")
        if not revision:
            raise ValueError("a pinned model revision is required")
        self.model_name = model_name
        self.revision = revision
        self.tokenizer_revision = tokenizer_revision or revision
        self.model_kwargs = dict(model_kwargs or {})
        self._tokenizer = None
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("transformers is required to load Qwen") from exc
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, revision=self.tokenizer_revision)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, revision=self.revision, **self.model_kwargs)

    def token_count(self, messages: Sequence[Mapping[str, Any]]) -> int:
        self.load()
        return len(self._tokenizer.apply_chat_template(
            list(messages), tokenize=True, add_generation_prompt=True,
        ))

    def generate(self, messages: Sequence[Mapping[str, Any]], *, max_new_tokens: int) -> str:
        self.load()
        tokens = self._tokenizer.apply_chat_template(
            list(messages), tokenize=True, add_generation_prompt=True, return_tensors="pt")
        tokens = tokens.to(self._model.device)
        self._model.eval()
        import torch
        with torch.no_grad():
            generated = self._model.generate(
                tokens, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id)
        return self._tokenizer.decode(
            generated[0, tokens.shape[-1]:], skip_special_tokens=True)

