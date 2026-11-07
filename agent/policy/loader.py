"""Lazy Hugging Face backend fixed to the assignment's Qwen policy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

POLICY_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
POLICY_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"


class HuggingFaceQwenBackend:
    def __init__(
        self,
        *,
        model_name: str = POLICY_MODEL,
        revision: str = POLICY_REVISION,
        tokenizer_revision: str | None = None,
        model_kwargs: Mapping[str, Any] | None = None,
        adapter: Path | None = None,
        max_context_tokens: int = 4096,
    ) -> None:
        if model_name != POLICY_MODEL:
            raise ValueError("policy model is fixed by the assignment")
        if not revision:
            raise ValueError("a pinned model revision is required")
        self.model_name = model_name
        self.revision = revision
        self.tokenizer_revision = tokenizer_revision or revision
        self.model_kwargs = dict(model_kwargs or {})
        if self.model_kwargs.get("trust_remote_code"):
            raise ValueError("remote model code is not enabled for this policy")
        self.adapter = Path(adapter) if adapter is not None else None
        if type(max_context_tokens) is not int or max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be positive")
        self.max_context_tokens = max_context_tokens
        self._tokenizer = None
        self._model = None

    @property
    def policy_identity(self) -> dict[str, str]:
        from agent.training.execution import _path_hash

        return {
            "model_name": self.model_name,
            "model_revision": self.revision,
            "adapter_hash": _path_hash(self.adapter) if self.adapter else "base-model",
        }

    def load_tokenizer(self) -> None:
        if self._tokenizer is not None:
            return
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("transformers is required to load Qwen") from exc
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, revision=self.tokenizer_revision, trust_remote_code=False
        )

    def load(self) -> None:
        if self._model is not None:
            return
        self.load_tokenizer()
        import torch
        from transformers import AutoModelForCausalLM

        kwargs = dict(self.model_kwargs)
        kwargs["trust_remote_code"] = False
        if torch.cuda.is_available():
            kwargs.setdefault("dtype", torch.bfloat16)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, revision=self.revision, **kwargs
        )
        if self.adapter is not None:
            from peft import PeftModel

            self._model = PeftModel.from_pretrained(
                self._model, str(self.adapter), is_trainable=False
            )
        if "device_map" not in kwargs:
            mps = getattr(getattr(torch, "backends", None), "mps", None)
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if mps is not None and mps.is_available()
                else "cpu"
            )
            self._model = self._model.to(device)

    def token_count(self, messages: Sequence[Mapping[str, Any]]) -> int:
        self.load_tokenizer()
        rendered = self._tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            truncation=False,
        )
        ids = rendered["input_ids"] if isinstance(rendered, Mapping) else rendered
        return len(ids)

    def generate(self, messages: Sequence[Mapping[str, Any]], *, max_new_tokens: int) -> str:
        self.load_tokenizer()
        if self.token_count(messages) + max_new_tokens > self.max_context_tokens:
            raise ValueError("research generation exceeds its reserved context budget")
        self.load()
        encoded = self._tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            truncation=False,
        )
        tokens = encoded["input_ids"].to(self._model.device)
        attention = encoded.get("attention_mask")
        if attention is not None:
            attention = attention.to(self._model.device)
        self._model.eval()
        import torch

        with torch.no_grad():
            generated = self._model.generate(
                input_ids=tokens,
                attention_mask=attention,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        return self._tokenizer.decode(generated[0, tokens.shape[-1] :], skip_special_tokens=True)
