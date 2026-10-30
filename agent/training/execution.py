"""Executable, dependency-injected LoRA training orchestration."""
from __future__ import annotations

import hashlib
import json
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from agent.contracts import canonical_json, content_hash

from .core import (
    MODEL_NAME,
    MODEL_REVISION,
    DPOPair,
    EpisodeExample,
    PPORollout,
    TrainingManifest,
    build_sft_example,
    validate_fresh_rollout,
)
from .integrations import resolve_training_stack


@dataclass(frozen=True)
class LoRASettings:
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")

    def __post_init__(self) -> None:
        if type(self.rank) is not int or self.rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if type(self.alpha) is not int or self.alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("LoRA dropout must be in [0, 1)")
        if not self.target_modules or any(not item for item in self.target_modules):
            raise ValueError("LoRA target_modules must be nonempty")


@dataclass(frozen=True)
class TrainerSettings:
    output_dir: Path
    prompt_version: str
    data_hashes: tuple[str, ...]
    model_revision: str = MODEL_REVISION
    tokenizer_revision: str = MODEL_REVISION
    seed: int = 42
    learning_rate: float = 2e-4
    max_steps: int = 100
    max_length: int = 4096
    train_batch_size: int = 1
    eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    save_steps: int = 25
    logging_steps: int = 5
    bf16: bool = True
    gradient_checkpointing: bool = True
    resume_from_checkpoint: str | None = None
    adapter_checkpoint: Path | None = None
    reference_adapter_checkpoint: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.adapter_checkpoint is not None:
            object.__setattr__(self, "adapter_checkpoint", Path(self.adapter_checkpoint))
        if self.reference_adapter_checkpoint is not None:
            object.__setattr__(
                self, "reference_adapter_checkpoint", Path(self.reference_adapter_checkpoint))
        for name in ("model_revision", "tokenizer_revision", "prompt_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be nonempty")
        if not self.data_hashes or any(not value for value in self.data_hashes):
            raise ValueError("data_hashes must be nonempty")
        for name in (
            "max_steps", "max_length", "train_batch_size", "eval_batch_size",
            "gradient_accumulation_steps", "save_steps", "logging_steps",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")


@dataclass(frozen=True)
class TrainingResult:
    method: str
    output_dir: str
    manifest_hash: str
    metrics: Mapping[str, Any]
    elapsed_seconds: float


def _seed_everything(seed: int, torch_module: Any) -> None:
    random.seed(seed)
    if hasattr(torch_module, "manual_seed"):
        torch_module.manual_seed(seed)
    cuda = getattr(torch_module, "cuda", None)
    if cuda is not None and hasattr(cuda, "manual_seed_all"):
        cuda.manual_seed_all(seed)


def _load_policy(stack: Mapping[str, Any], settings: TrainerSettings, lora: LoRASettings):
    modules = stack["modules"]
    transformers, peft = modules["transformers"], modules["peft"]
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL_NAME, revision=settings.tokenizer_revision, trust_remote_code=False,
    )
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    cuda = getattr(modules["torch"], "cuda", None)
    use_bf16 = (settings.bf16 and cuda is not None
                and getattr(cuda, "is_available", lambda: False)())
    model_kwargs = {"dtype": modules["torch"].bfloat16} if use_bf16 else {}
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, revision=settings.model_revision, trust_remote_code=False,
        **model_kwargs,
    )
    if settings.gradient_checkpointing and hasattr(model, "config"):
        model.config.use_cache = False
    if settings.adapter_checkpoint is not None:
        if not settings.adapter_checkpoint.exists():
            raise ValueError("adapter_checkpoint does not exist")
        return tokenizer, peft.PeftModel.from_pretrained(
            model, str(settings.adapter_checkpoint), is_trainable=True,
        )
    lora_config = peft.LoraConfig(
        r=lora.rank, lora_alpha=lora.alpha, lora_dropout=lora.dropout,
        target_modules=list(lora.target_modules), bias="none", task_type="CAUSAL_LM",
    )
    return tokenizer, peft.get_peft_model(model, lora_config)


def _argument_values(settings: TrainerSettings, *, evaluate: bool) -> dict[str, Any]:
    return {
        "output_dir": str(settings.output_dir), "max_steps": settings.max_steps,
        "learning_rate": settings.learning_rate,
        "per_device_train_batch_size": settings.train_batch_size,
        "per_device_eval_batch_size": settings.eval_batch_size,
        "gradient_accumulation_steps": settings.gradient_accumulation_steps,
        "save_steps": settings.save_steps, "logging_steps": settings.logging_steps,
        "eval_strategy": "steps" if evaluate else "no",
        "eval_steps": settings.save_steps if evaluate else None,
        "save_strategy": "steps",
        "load_best_model_at_end": evaluate,
        "metric_for_best_model": "eval_loss" if evaluate else None,
        "greater_is_better": False if evaluate else None,
        "bf16": settings.bf16, "gradient_checkpointing": settings.gradient_checkpointing,
        "report_to": [], "seed": settings.seed, "data_seed": settings.seed,
        "remove_unused_columns": False,
    }


def _manifest(method: str, settings: TrainerSettings, lora: LoRASettings,
              versions: Mapping[str, str]) -> TrainingManifest:
    adapter_config = asdict(lora)
    if settings.adapter_checkpoint is not None:
        adapter_config["initial_adapter_hash"] = _path_hash(settings.adapter_checkpoint)
    return TrainingManifest(
        model_name=MODEL_NAME, model_revision=settings.model_revision,
        tokenizer_revision=settings.tokenizer_revision,
        prompt_version=settings.prompt_version, package_versions=versions,
        data_hashes=settings.data_hashes, seed=settings.seed,
        adapter_config=adapter_config, learning_rate=settings.learning_rate,
        steps=settings.max_steps, method=method,
    )


def _path_hash(path: Path) -> str:
    """Hash checkpoint contents without recording a machine-specific path."""
    files = (sorted(item for item in path.rglob("*") if item.is_file())
             if path.is_dir() else [path])
    digest = hashlib.sha256()
    for item in files:
        name = str(item.relative_to(path)) if path.is_dir() else item.name
        digest.update(name.encode("utf-8"))
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _write_artifacts(output: Path, manifest: TrainingManifest, tokenizer: Any,
                     metrics: Mapping[str, Any], elapsed: float) -> None:
    output.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(output))
    payload = {
        "manifest_hash": manifest.manifest_hash, "manifest": asdict(manifest),
        "metrics": dict(metrics), "elapsed_seconds": elapsed,
    }
    (output / "training_manifest.json").write_text(
        canonical_json(payload) + "\n", encoding="utf-8",
    )


def _with_runtime_metrics(metrics: Mapping[str, Any], torch_module: Any) -> dict[str, Any]:
    result = dict(metrics)
    cuda = getattr(torch_module, "cuda", None)
    if cuda is not None and getattr(cuda, "is_available", lambda: False)():
        result["peak_cuda_memory_bytes"] = int(cuda.max_memory_allocated())
    return result


def _dataset(module: Any, rows: Sequence[Mapping[str, Any]]) -> Any:
    if not rows:
        raise ValueError("training dataset must be nonempty")
    return module.Dataset.from_list([dict(row) for row in rows])


def train_sft(
    train_examples: Sequence[EpisodeExample], validation_examples: Sequence[EpisodeExample],
    *, settings: TrainerSettings, lora: LoRASettings | None = None,
    stack_resolver: Callable[..., Mapping[str, Any]] = resolve_training_stack,
    expected_versions: dict[str, str] | None = None,
) -> TrainingResult:
    """Train and save a Qwen LoRA adapter with prompt/padding labels masked."""
    lora = lora or LoRASettings()
    if not train_examples or not validation_examples:
        raise ValueError("SFT requires nonempty train and validation datasets")
    if any(item.split != "train" for item in train_examples):
        raise ValueError("SFT training data must contain only train examples")
    if any(item.split != "validation" for item in validation_examples):
        raise ValueError("SFT validation data must contain only validation examples")
    stack = stack_resolver("sft", expected_versions=expected_versions)
    modules = stack["modules"]
    _seed_everything(settings.seed, modules["torch"])
    tokenizer, model = _load_policy(stack, settings, lora)

    def rows(examples: Sequence[EpisodeExample], split: str) -> list[dict[str, Any]]:
        converted = []
        for item in examples:
            if item.split != split:
                raise ValueError(f"{split} dataset contains a {item.split} example")
            built = build_sft_example(
                item, tokenizer, max_length=settings.max_length,
                allowed_splits=(split,),
            )
            converted.append({
                "input_ids": list(built.input_ids),
                "attention_mask": list(built.attention_mask), "labels": list(built.labels),
            })
        return converted

    train_data = _dataset(modules["datasets"], rows(train_examples, "train"))
    eval_data = (_dataset(modules["datasets"], rows(validation_examples, "validation"))
                 if validation_examples else None)
    args = modules["transformers"].TrainingArguments(
        **_argument_values(settings, evaluate=eval_data is not None))
    trainer = modules["transformers"].Trainer(
        model=model, args=args, train_dataset=train_data, eval_dataset=eval_data,
        data_collator=modules["transformers"].DataCollatorForSeq2Seq(
            tokenizer=tokenizer, padding=True, label_pad_token_id=-100,
            return_tensors="pt"),
    )
    started = time.monotonic()
    outcome = trainer.train(resume_from_checkpoint=settings.resume_from_checkpoint)
    elapsed = time.monotonic() - started
    trainer.save_model(str(settings.output_dir))
    trainer.save_state()
    metrics = _with_runtime_metrics(
        getattr(outcome, "metrics", {}) or {}, modules["torch"])
    manifest = _manifest("sft", settings, lora, stack["versions"])
    _write_artifacts(settings.output_dir, manifest, tokenizer, metrics, elapsed)
    return TrainingResult("sft", str(settings.output_dir), manifest.manifest_hash, metrics, elapsed)


def train_dpo(
    train_pairs: Sequence[DPOPair], validation_pairs: Sequence[DPOPair], *,
    settings: TrainerSettings, lora: LoRASettings | None = None, beta: float = 0.1,
    stack_resolver: Callable[..., Mapping[str, Any]] = resolve_training_stack,
    expected_versions: dict[str, str] | None = None,
) -> TrainingResult:
    """Run same-state DPO and save the resulting Qwen LoRA adapter."""
    lora = lora or LoRASettings()
    if beta <= 0:
        raise ValueError("DPO beta must be positive")
    if any(pair.split != "train" for pair in train_pairs):
        raise ValueError("DPO training data must contain only train-split pairs")
    if any(pair.split != "validation" for pair in validation_pairs):
        raise ValueError("DPO validation data must contain only validation-split pairs")
    if any(pair.audit_hash is None for pair in (*train_pairs, *validation_pairs)):
        raise ValueError("DPO training requires independently audited preference pairs")
    if not train_pairs or not validation_pairs:
        raise ValueError("DPO requires nonempty train and validation datasets")
    stack = stack_resolver("dpo", expected_versions=expected_versions)
    modules = stack["modules"]
    _seed_everything(settings.seed, modules["torch"])
    tokenizer, model = _load_policy(stack, settings, lora)
    tokenizer.padding_side = "left"
    def rows(pairs: Sequence[DPOPair]) -> list[dict[str, Any]]:
        converted = []
        for pair in pairs:
            prompt: Any = pair.prompt
            # Prepared artifacts store the original native chat messages as
            # canonical JSON so DPO can use the same template as SFT. Keep the
            # plain-string form accepted for small injected unit-test fixtures.
            if isinstance(prompt, str):
                try:
                    decoded = json.loads(prompt)
                except json.JSONDecodeError:
                    decoded = None
                if isinstance(decoded, list) and all(isinstance(item, Mapping) for item in decoded):
                    rendered_prompt = tokenizer.apply_chat_template(
                        decoded, tokenize=False, add_generation_prompt=True,
                    )
                    if not isinstance(rendered_prompt, str) or not rendered_prompt:
                        raise TypeError(
                            "DPO chat template must render a nonempty string prompt"
                        )
                    prompt = rendered_prompt
            converted.append({"prompt": prompt, "chosen": pair.chosen, "rejected": pair.rejected})
        return converted
    train_data = _dataset(modules["datasets"], rows(train_pairs))
    eval_data = _dataset(modules["datasets"], rows(validation_pairs)) if validation_pairs else None
    trl = modules["trl"]
    config_type = getattr(trl, "DPOConfig", None)
    if config_type is None:
        raise RuntimeError("pinned TRL must expose DPOConfig")
    args = config_type(
        **_argument_values(settings, evaluate=eval_data is not None),
        beta=beta, max_length=settings.max_length,
    )
    trainer = trl.DPOTrainer(
        model=model, ref_model=None, args=args, train_dataset=train_data,
        eval_dataset=eval_data, processing_class=tokenizer,
    )
    started = time.monotonic()
    outcome = trainer.train(resume_from_checkpoint=settings.resume_from_checkpoint)
    elapsed = time.monotonic() - started
    trainer.save_model(str(settings.output_dir))
    trainer.save_state()
    metrics = _with_runtime_metrics(
        getattr(outcome, "metrics", {}) or {}, modules["torch"])
    manifest = _manifest("dpo", settings, lora, stack["versions"])
    _write_artifacts(settings.output_dir, manifest, tokenizer, metrics, elapsed)
    return TrainingResult("dpo", str(settings.output_dir), manifest.manifest_hash, metrics, elapsed)


def smoke_test_adapter(
    adapter_checkpoint: Path, *, output_dir: Path, prompt: str,
    model_revision: str = MODEL_REVISION, tokenizer_revision: str = MODEL_REVISION,
    max_new_tokens: int = 32,
    stack_resolver: Callable[..., Mapping[str, Any]] = resolve_training_stack,
) -> Mapping[str, Any]:
    """Save/reload an adapter on CUDA and run deterministic greedy generation."""
    adapter_checkpoint, output_dir = Path(adapter_checkpoint), Path(output_dir)
    if not adapter_checkpoint.exists():
        raise ValueError("adapter checkpoint does not exist")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("smoke-test prompt must be nonempty")
    if type(max_new_tokens) is not int or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    stack = stack_resolver("sft")
    modules = stack["modules"]
    torch, transformers, peft = (
        modules["torch"], modules["transformers"], modules["peft"])
    if not torch.cuda.is_available():
        raise RuntimeError("GPU smoke test requires CUDA")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL_NAME, revision=tokenizer_revision, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def base_model():
        return transformers.AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, revision=model_revision, trust_remote_code=False,
            dtype=torch.bfloat16,
        ).to(torch.device("cuda"))

    first = peft.PeftModel.from_pretrained(
        base_model(), str(adapter_checkpoint), is_trainable=True)
    trainable = [parameter for parameter in first.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("adapter exposes no trainable parameters")
    optimizer = torch.optim.AdamW(trainable, lr=1e-6)
    first.train()
    training_inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True,
        add_generation_prompt=False, return_tensors="pt",
    ).to(first.device)
    optimizer.zero_grad()
    smoke_loss = first(input_ids=training_inputs, labels=training_inputs).loss
    smoke_loss.backward()
    optimizer.step()
    first.eval()
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True,
        add_generation_prompt=True, return_tensors="pt",
    ).to(first.device)
    with torch.no_grad():
        before_tokens = first.generate(
            input_ids=inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    before_output = tokenizer.decode(
        before_tokens[0, inputs.shape[-1]:], skip_special_tokens=True)
    reloaded_dir = output_dir / "reloaded_adapter"
    reloaded_dir.mkdir(parents=True, exist_ok=True)
    first.save_pretrained(str(reloaded_dir))
    tokenizer.save_pretrained(str(reloaded_dir))
    del first
    torch.cuda.empty_cache()
    model = peft.PeftModel.from_pretrained(
        base_model(), str(reloaded_dir), is_trainable=False)
    model.eval()
    reloaded_inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True,
        add_generation_prompt=True, return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        generated = model.generate(
            input_ids=reloaded_inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    output = tokenizer.decode(
        generated[0, reloaded_inputs.shape[-1]:], skip_special_tokens=True)
    if output != before_output:
        raise RuntimeError("adapter output changed after save/reload")
    result = {
        "model_name": MODEL_NAME, "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "source_adapter_hash": _path_hash(adapter_checkpoint),
        "reloaded_adapter_hash": _path_hash(reloaded_dir),
        "outputs_match": True,
        "optimizer_step_loss": float(smoke_loss.detach().cpu()),
        "generated_text": output,
        "max_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }
    (output_dir / "smoke_result.json").write_text(
        canonical_json(result) + "\n", encoding="utf-8")
    return result


class PPOEnvironment(Protocol):
    """Training-only environment which never exposes future observations."""
    def reset(self, *, seed: int) -> Mapping[str, Any]: ...
    def step(
        self, action: str,
    ) -> tuple[Mapping[str, Any], float, bool, bool, Mapping[str, Any]]: ...


class PPOOptimizer(Protocol):
    @property
    def policy_version(self) -> str: ...
    def collect(self, environment: PPOEnvironment, *, rollout_id: str,
                seed: int) -> PPORollout: ...
    def update(self, rollout: PPORollout) -> Mapping[str, float]: ...
    def save(self, output_dir: Path) -> None: ...
    def load(self, checkpoint: Path) -> int: ...


def train_ppo(
    environment: PPOEnvironment, optimizer: PPOOptimizer, *, settings: TrainerSettings,
    lora: LoRASettings | None = None, updates: int,
    package_versions: Mapping[str, str], clock: Callable[[], float] = time.monotonic,
) -> TrainingResult:
    """Enforce fresh on-policy collection around a Qwen-backed PPO optimizer."""
    lora = lora or LoRASettings()
    if type(updates) is not int or updates <= 0:
        raise ValueError("PPO updates must be positive")
    start_update = 0
    if settings.resume_from_checkpoint is not None:
        checkpoint = Path(settings.resume_from_checkpoint)
        if not checkpoint.exists():
            raise ValueError("PPO resume checkpoint does not exist")
        start_update = optimizer.load(checkpoint)
        if type(start_update) is not int or start_update < 0:
            raise ValueError("PPO checkpoint must report a nonnegative update index")
    started = clock()
    metrics: dict[str, Any] = {}
    update_history: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index in range(start_update, start_update + updates):
        rollout_id = content_hash({
            "policy_version": optimizer.policy_version, "update": index,
            "seed": settings.seed,
        })
        if rollout_id in seen:
            raise ValueError("PPO rollout ID was reused")
        rollout = optimizer.collect(
            environment, rollout_id=rollout_id, seed=settings.seed + index)
        validate_fresh_rollout(
            rollout, current_policy_version=optimizer.policy_version,
            expected_rollout_id=rollout_id,
        )
        metrics = dict(optimizer.update(rollout))
        update_history.append({"update": index + 1, "rollout_id": rollout_id, **metrics})
        seen.add(rollout_id)
        if (index + 1) % settings.save_steps == 0:
            optimizer.save(settings.output_dir / f"checkpoint-{index + 1}")
    elapsed = clock() - started
    optimizer.save(settings.output_dir)
    torch_module = getattr(optimizer, "torch", None)
    if torch_module is not None:
        metrics = _with_runtime_metrics(metrics, torch_module)
    metrics["completed_updates"] = len(update_history)
    manifest = _manifest("ppo", settings, lora, package_versions)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = settings.output_dir / "ppo_metrics.jsonl"
    existing_metrics = (
        metrics_path.read_text(encoding="utf-8")
        if settings.resume_from_checkpoint is not None and metrics_path.exists() else ""
    )
    metrics_path.write_text(
        existing_metrics + "".join(
            canonical_json(item) + "\n" for item in update_history), encoding="utf-8")
    (settings.output_dir / "training_manifest.json").write_text(canonical_json({
        "manifest_hash": manifest.manifest_hash, "manifest": asdict(manifest),
        "metrics": metrics, "elapsed_seconds": elapsed, "rollout_ids": sorted(seen),
    }) + "\n", encoding="utf-8")
    return TrainingResult("ppo", str(settings.output_dir), manifest.manifest_hash, metrics, elapsed)
