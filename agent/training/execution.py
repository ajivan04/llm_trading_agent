"""Executable, dependency-injected LoRA training orchestration."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import wraps
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
    build_final_trajectory_sft_example,
    build_sft_example,
    build_trajectory_sft_example,
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
    supervision_mode: str = "final"
    stop_after_steps: int | None = None
    eval_steps: int = 25

    def __post_init__(self) -> None:
        if self.supervision_mode not in {"final", "trajectory", "trajectory-final"}:
            raise ValueError("unknown SFT supervision mode")
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.adapter_checkpoint is not None:
            object.__setattr__(self, "adapter_checkpoint", Path(self.adapter_checkpoint))
        if self.reference_adapter_checkpoint is not None:
            object.__setattr__(
                self, "reference_adapter_checkpoint", Path(self.reference_adapter_checkpoint)
            )
        for name in ("model_revision", "tokenizer_revision", "prompt_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be nonempty")
        if not self.data_hashes or any(not value for value in self.data_hashes):
            raise ValueError("data_hashes must be nonempty")
        for name in (
            "max_steps",
            "max_length",
            "train_batch_size",
            "eval_batch_size",
            "gradient_accumulation_steps",
            "save_steps",
            "logging_steps",
            "eval_steps",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if self.stop_after_steps is not None and (
            type(self.stop_after_steps) is not int
            or not 0 < self.stop_after_steps <= self.max_steps
        ):
            raise ValueError("stop_after_steps must be a positive prefix of max_steps")


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


def _load_policy(
    stack: Mapping[str, Any], settings: TrainerSettings, lora: LoRASettings, tokenizer=None
):
    modules = stack["modules"]
    transformers, peft = modules["transformers"], modules["peft"]
    if tokenizer is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            MODEL_NAME,
            revision=settings.tokenizer_revision,
            trust_remote_code=False,
        )
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    cuda = getattr(modules["torch"], "cuda", None)
    use_bf16 = settings.bf16 and cuda is not None and getattr(cuda, "is_available", lambda: False)()
    model_kwargs = {"dtype": modules["torch"].bfloat16} if use_bf16 else {}
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        revision=settings.model_revision,
        trust_remote_code=False,
        **model_kwargs,
    )
    if settings.gradient_checkpointing and hasattr(model, "config"):
        model.config.use_cache = False
    if settings.adapter_checkpoint is not None:
        if not settings.adapter_checkpoint.exists():
            raise ValueError("adapter_checkpoint does not exist")
        return tokenizer, peft.PeftModel.from_pretrained(
            model,
            str(settings.adapter_checkpoint),
            is_trainable=True,
        )
    lora_config = peft.LoraConfig(
        r=lora.rank,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        target_modules=list(lora.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    return tokenizer, peft.get_peft_model(model, lora_config)


def _argument_values(settings: TrainerSettings, *, evaluate: bool) -> dict[str, Any]:
    return {
        "output_dir": str(settings.output_dir),
        "max_steps": settings.max_steps,
        "learning_rate": settings.learning_rate,
        "per_device_train_batch_size": settings.train_batch_size,
        "per_device_eval_batch_size": settings.eval_batch_size,
        "gradient_accumulation_steps": settings.gradient_accumulation_steps,
        "save_steps": settings.save_steps,
        "logging_steps": settings.logging_steps,
        "eval_strategy": "steps" if evaluate else "no",
        "eval_steps": settings.eval_steps if evaluate else None,
        "save_strategy": "steps",
        # Keep checkpoint recovery independent of validation cadence. The final
        # saved adapter is compared in the separate stateful evaluation workflow.
        "load_best_model_at_end": False,
        "bf16": settings.bf16,
        "gradient_checkpointing": settings.gradient_checkpointing,
        "report_to": [],
        "seed": settings.seed,
        "data_seed": settings.seed,
        "remove_unused_columns": False,
        "disable_tqdm": True,
        "logging_strategy": "steps",
        "logging_first_step": True,
        "prediction_loss_only": True,
        "save_safetensors": True,
        "save_only_model": False,
    }


def _manifest(
    method: str, settings: TrainerSettings, lora: LoRASettings, versions: Mapping[str, str]
) -> TrainingManifest:
    adapter_config = asdict(lora)
    if method == "sft":
        adapter_config["supervision_mode"] = settings.supervision_mode
        adapter_config["initialization"] = (
            "existing-adapter-with-prior-training-history"
            if settings.adapter_checkpoint
            else "pinned-base"
        )
    if settings.adapter_checkpoint is not None:
        adapter_config["initial_adapter_hash"] = _path_hash(settings.adapter_checkpoint)
    return TrainingManifest(
        model_name=MODEL_NAME,
        model_revision=settings.model_revision,
        tokenizer_revision=settings.tokenizer_revision,
        prompt_version=settings.prompt_version,
        package_versions=versions,
        data_hashes=settings.data_hashes,
        seed=settings.seed,
        adapter_config=adapter_config,
        learning_rate=settings.learning_rate,
        steps=settings.max_steps,
        method=method,
    )


def _path_hash(path: Path) -> str:
    """Hash checkpoint contents without recording a machine-specific path."""
    files = sorted(item for item in path.rglob("*") if item.is_file()) if path.is_dir() else [path]
    digest = hashlib.sha256()
    for item in files:
        name = str(item.relative_to(path)) if path.is_dir() else item.name
        digest.update(name.encode("utf-8"))
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _write_artifacts(
    output: Path,
    manifest: TrainingManifest,
    tokenizer: Any,
    metrics: Mapping[str, Any],
    elapsed: float,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(output))
    payload = {
        "manifest_hash": manifest.manifest_hash,
        "manifest": asdict(manifest),
        "metrics": dict(metrics),
        "elapsed_seconds": elapsed,
    }
    (output / "training_manifest.json").write_text(
        canonical_json(payload) + "\n",
        encoding="utf-8",
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


def sft_run_contract(settings, lora, versions):
    """The scheduler target is immutable; smoke stops do not change the optimizer plan."""
    values = asdict(settings)
    for key in (
        "output_dir",
        "resume_from_checkpoint",
        "stop_after_steps",
        "adapter_checkpoint",
        "reference_adapter_checkpoint",
    ):
        values.pop(key)
    return {
        "version": "trajectory-sft-v2-resume",
        "model_name": MODEL_NAME,
        "settings": values,
        "lora": asdict(lora),
        "package_versions": dict(versions),
        "initial_adapter_hash": (
            _path_hash(settings.adapter_checkpoint)
            if settings.adapter_checkpoint is not None
            else "base-model"
        ),
        "code_hashes": {
            name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in ("core.py", "execution.py", "validation.py")
        },
    }


def _sft_resume_contract(settings, lora, versions, *, write=True):
    contract = sft_run_contract(settings, lora, versions)
    path = settings.output_dir / "sft_run.json"
    if (settings.output_dir / "training_manifest.json").is_file():
        raise ValueError("SFT training is already complete; evaluate it or choose a new experiment")
    if settings.resume_from_checkpoint is not None:
        checkpoint = Path(settings.resume_from_checkpoint).resolve()
        if checkpoint.parent != settings.output_dir.resolve() or not checkpoint.name.startswith(
            "checkpoint-"
        ):
            raise ValueError(
                "SFT resume must use a checkpoint inside its original output directory"
            )
        required = (
            "trainer_state.json",
            "optimizer.pt",
            "scheduler.pt",
            "rng_state.pth",
            "adapter_config.json",
            "adapter_model.safetensors",
            "sft_checkpoint.json",
        )
        if any(not (checkpoint / name).is_file() for name in required):
            raise ValueError("SFT checkpoint is missing model/optimizer/scheduler/trainer state")
        if not path.is_file():
            raise ValueError("SFT resume requires its immutable sft_run.json contract")
        stored = json.loads(path.read_text())
        if stored.get("contract_hash") != content_hash(stored.get("contract")) or canonical_json(
            stored["contract"]
        ) != canonical_json(contract):
            raise ValueError(
                "SFT resume contract changed; use original data/model/optimizer settings"
            )
        integrity = json.loads((checkpoint / "sft_checkpoint.json").read_text())
        if integrity.get("run_contract_hash") != content_hash(contract):
            raise ValueError("SFT checkpoint belongs to a different run contract")
        hashes = {
            name: hashlib.sha256((checkpoint / name).read_bytes()).hexdigest()
            for name in required
            if name != "sft_checkpoint.json"
        }
        if integrity.get("file_hashes") != hashes:
            raise ValueError("SFT checkpoint checksum failed; use an earlier complete checkpoint")
        state = json.loads((checkpoint / "trainer_state.json").read_text())
        step = state.get("global_step")
        if type(step) is not int or not 0 <= step <= settings.max_steps:
            raise ValueError("SFT checkpoint step is outside its fixed training plan")
    else:
        if settings.output_dir.exists() and any(settings.output_dir.iterdir()):
            raise ValueError("SFT output is nonempty; resume explicitly or use a new directory")
        if write:
            settings.output_dir.mkdir(parents=True, exist_ok=True)
            _ppo_json(path, {"contract": contract, "contract_hash": content_hash(contract)})
    return contract


def _sft_callback(transformers, settings, contract):
    base = getattr(transformers, "TrainerCallback", object)

    class ResumeGuard(base):
        def on_step_end(self, args, state, control, **_kwargs):
            if (
                settings.stop_after_steps is not None
                and state.global_step >= settings.stop_after_steps
            ):
                control.should_save = True
                control.should_training_stop = True
            return control

        def on_log(self, args, state, control, logs=None, **_kwargs):
            for key, value in (logs or {}).items():
                if ("loss" in key or key == "grad_norm") and isinstance(value, (int, float)):
                    if not math.isfinite(value):
                        raise ValueError("nonfinite SFT training metric; resume only after review")
            safe = {
                key: value
                for key, value in (logs or {}).items()
                if isinstance(value, (int, float))
                and key in {"loss", "eval_loss", "grad_norm", "learning_rate", "epoch"}
            }
            print(
                canonical_json(
                    {
                        "event": "sft_training",
                        "step": state.global_step,
                        "planned_steps": settings.max_steps,
                        **safe,
                    }
                ),
                flush=True,
            )
            return control

        def on_save(self, args, state, control, **_kwargs):
            checkpoint = settings.output_dir / f"checkpoint-{state.global_step}"
            files = (
                "trainer_state.json",
                "optimizer.pt",
                "scheduler.pt",
                "rng_state.pth",
                "adapter_config.json",
                "adapter_model.safetensors",
            )
            if any(not (checkpoint / name).is_file() for name in files):
                raise ValueError("SFT checkpoint did not save complete optimizer/model/RNG state")
            hashes = {
                name: hashlib.sha256((checkpoint / name).read_bytes()).hexdigest() for name in files
            }
            _ppo_json(
                checkpoint / "sft_checkpoint.json",
                {
                    "run_contract_hash": content_hash(contract),
                    "global_step": state.global_step,
                    "file_hashes": hashes,
                },
            )
            return control

    return ResumeGuard()


def _parameter_metrics(model):
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return {"parameter_measurement_available": False}
    tensors = tuple(parameters())
    total = sum(p.numel() for p in tensors)
    trainable = sum(p.numel() for p in tensors if p.requires_grad)
    if total <= 0 or not 0 < trainable < total:
        raise ValueError("SFT must train a measured nonzero LoRA fraction, not the full base")
    return {
        "parameter_measurement_available": True,
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_parameter_fraction": trainable / total,
    }


def _sft_locked(function):
    @wraps(function)
    def wrapped(*args, settings, **kwargs):
        import fcntl

        output = settings.output_dir.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with (output.parent / (output.name + ".sft.lock")).open("a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("another SFT trainer owns this output directory") from None
            try:
                return function(*args, settings=settings, **kwargs)
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    return wrapped


@_sft_locked
def train_sft(
    train_examples: Sequence[EpisodeExample],
    validation_examples: Sequence[EpisodeExample],
    *,
    settings: TrainerSettings,
    lora: LoRASettings | None = None,
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
    contract = _sft_resume_contract(settings, lora, stack["versions"], write=False)
    _seed_everything(settings.seed, modules["torch"])
    # Tokenize and validate every selective mask before allocating policy weights.
    tokenizer = modules["transformers"].AutoTokenizer.from_pretrained(
        MODEL_NAME, revision=settings.tokenizer_revision, trust_remote_code=False
    )

    def rows(examples: Sequence[EpisodeExample], split: str) -> list[dict[str, Any]]:
        converted = []
        maximum = 0
        for index, item in enumerate(examples, 1):
            if item.split != split:
                raise ValueError(f"{split} dataset contains a {item.split} example")
            builder = {
                "trajectory": build_trajectory_sft_example,
                "trajectory-final": build_final_trajectory_sft_example,
            }.get(settings.supervision_mode, build_sft_example)
            built = builder(
                item,
                tokenizer,
                max_length=settings.max_length,
                allowed_splits=(split,),
            )
            converted.append(
                {
                    "input_ids": list(built.input_ids),
                    "attention_mask": list(built.attention_mask),
                    "labels": list(built.labels),
                }
            )
            maximum = max(maximum, len(built.input_ids))
            if index == 1 or index % 100 == 0 or index == len(examples):
                print(
                    canonical_json(
                        {
                            "event": "sft_token_preflight",
                            "split": split,
                            "checked": index,
                            "total": len(examples),
                            "max_input_tokens": maximum,
                            "supervision": settings.supervision_mode,
                        }
                    ),
                    flush=True,
                )
        return converted

    train_data = _dataset(modules["datasets"], rows(train_examples, "train"))
    eval_data = (
        _dataset(modules["datasets"], rows(validation_examples, "validation"))
        if validation_examples
        else None
    )
    _sft_resume_contract(settings, lora, stack["versions"])
    tokenizer, model = _load_policy(stack, settings, lora, tokenizer=tokenizer)
    args = modules["transformers"].TrainingArguments(
        **_argument_values(settings, evaluate=eval_data is not None)
    )
    effective_batch = settings.train_batch_size * settings.gradient_accumulation_steps
    budget_metrics = {
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "effective_batch_size": effective_batch,
        "planned_example_exposures": settings.max_steps * effective_batch,
        "planned_passes_upper_bound": settings.max_steps * effective_batch / len(train_examples),
    }
    print(canonical_json({"event": "sft_training_budget", **budget_metrics}), flush=True)
    parameter_metrics = _parameter_metrics(model)
    if settings.supervision_mode == "trajectory-final" and (
        not parameter_metrics["parameter_measurement_available"]
        or parameter_metrics["trainable_parameter_fraction"] >= 0.001
    ):
        raise ValueError(
            "assignment SFT requires a measured trainable parameter fraction below 0.1%"
        )
    callbacks = [_sft_callback(modules["transformers"], settings, contract)]
    trainer = modules["transformers"].Trainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        callbacks=callbacks,
        data_collator=modules["transformers"].DataCollatorForSeq2Seq(
            tokenizer=tokenizer, padding=True, label_pad_token_id=-100, return_tensors="pt"
        ),
    )
    started = time.monotonic()
    outcome = trainer.train(resume_from_checkpoint=settings.resume_from_checkpoint)
    elapsed = time.monotonic() - started
    trainer.save_model(str(settings.output_dir))
    trainer.save_state()
    metrics = _with_runtime_metrics(getattr(outcome, "metrics", {}) or {}, modules["torch"])
    step = getattr(getattr(trainer, "state", None), "global_step", settings.max_steps)
    if any(
        isinstance(value, (float, int)) and not math.isfinite(value) for value in metrics.values()
    ):
        raise ValueError("SFT returned nonfinite training metrics; preserve checkpoint for review")
    metrics.update(parameter_metrics)
    metrics.update(budget_metrics)
    metrics.update(
        {
            "completed_steps": step,
            "global_step": step,
            "planned_steps": settings.max_steps,
            "supervision_mode": settings.supervision_mode,
            "run_contract_hash": content_hash(contract),
            "status": "completed" if step >= settings.max_steps else "partial",
        }
    )
    from agent.policy.loader import adapter_fingerprint

    if (settings.output_dir / "adapter_config.json").is_file():
        metrics["inference_adapter_fingerprint"] = adapter_fingerprint(settings.output_dir)
    elif settings.supervision_mode == "trajectory-final":
        raise ValueError("assignment SFT did not publish inference assets")
    manifest = _manifest("sft", settings, lora, stack["versions"])
    _ppo_json(settings.output_dir / "training_status.json", metrics)
    if step >= settings.max_steps:
        _write_artifacts(settings.output_dir, manifest, tokenizer, metrics, elapsed)
    return TrainingResult("sft", str(settings.output_dir), manifest.manifest_hash, metrics, elapsed)


def train_dpo(
    train_pairs: Sequence[DPOPair],
    validation_pairs: Sequence[DPOPair],
    *,
    settings: TrainerSettings,
    lora: LoRASettings | None = None,
    beta: float = 0.1,
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
                        decoded,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    if not isinstance(rendered_prompt, str) or not rendered_prompt:
                        raise TypeError("DPO chat template must render a nonempty string prompt")
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
        beta=beta,
        max_length=settings.max_length,
    )
    trainer = trl.DPOTrainer(
        model=model,
        ref_model=None,
        args=args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=tokenizer,
    )
    started = time.monotonic()
    outcome = trainer.train(resume_from_checkpoint=settings.resume_from_checkpoint)
    elapsed = time.monotonic() - started
    trainer.save_model(str(settings.output_dir))
    trainer.save_state()
    metrics = _with_runtime_metrics(getattr(outcome, "metrics", {}) or {}, modules["torch"])
    manifest = _manifest("dpo", settings, lora, stack["versions"])
    _write_artifacts(settings.output_dir, manifest, tokenizer, metrics, elapsed)
    return TrainingResult("dpo", str(settings.output_dir), manifest.manifest_hash, metrics, elapsed)


def smoke_test_adapter(
    adapter_checkpoint: Path,
    *,
    output_dir: Path,
    prompt: str,
    model_revision: str = MODEL_REVISION,
    tokenizer_revision: str = MODEL_REVISION,
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
    torch, transformers, peft = (modules["torch"], modules["transformers"], modules["peft"])
    if not torch.cuda.is_available():
        raise RuntimeError("GPU smoke test requires CUDA")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL_NAME, revision=tokenizer_revision, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def base_model():
        return transformers.AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            revision=model_revision,
            trust_remote_code=False,
            dtype=torch.bfloat16,
        ).to(torch.device("cuda"))

    first = peft.PeftModel.from_pretrained(base_model(), str(adapter_checkpoint), is_trainable=True)
    trainable = [parameter for parameter in first.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("adapter exposes no trainable parameters")
    optimizer = torch.optim.AdamW(trainable, lr=1e-6)
    first.train()
    training_inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
        return_dict=True,
    )
    training_inputs = {key: value.to(first.device) for key, value in training_inputs.items()}
    optimizer.zero_grad()
    smoke_loss = first(**training_inputs, labels=training_inputs["input_ids"]).loss
    smoke_loss.backward()
    optimizer.step()
    first.eval()
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {key: value.to(first.device) for key, value in inputs.items()}
    with torch.no_grad():
        before_tokens = first.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    before_output = tokenizer.decode(
        before_tokens[0, inputs["input_ids"].shape[-1] :], skip_special_tokens=True
    )
    reloaded_dir = output_dir / "reloaded_adapter"
    reloaded_dir.mkdir(parents=True, exist_ok=True)
    first.save_pretrained(str(reloaded_dir))
    tokenizer.save_pretrained(str(reloaded_dir))
    del first
    torch.cuda.empty_cache()
    model = peft.PeftModel.from_pretrained(base_model(), str(reloaded_dir), is_trainable=False)
    model.eval()
    reloaded_inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    reloaded_inputs = {key: value.to(model.device) for key, value in reloaded_inputs.items()}
    with torch.no_grad():
        generated = model.generate(
            **reloaded_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    output = tokenizer.decode(
        generated[0, reloaded_inputs["input_ids"].shape[-1] :], skip_special_tokens=True
    )
    if output != before_output:
        raise RuntimeError("adapter output changed after save/reload")
    result = {
        "model_name": MODEL_NAME,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "source_adapter_hash": _path_hash(adapter_checkpoint),
        "reloaded_adapter_hash": _path_hash(reloaded_dir),
        "outputs_match": True,
        "optimizer_step_loss": float(smoke_loss.detach().cpu()),
        "generated_text": output,
        "max_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }
    (output_dir / "smoke_result.json").write_text(canonical_json(result) + "\n", encoding="utf-8")
    return result


class PPOEnvironment(Protocol):
    """Training-only environment which never exposes future observations."""

    def reset(self, *, seed: int) -> Mapping[str, Any]: ...
    def step(
        self,
        action: str,
    ) -> tuple[Mapping[str, Any], float, bool, bool, Mapping[str, Any]]: ...


class PPOOptimizer(Protocol):
    @property
    def policy_version(self) -> str: ...
    def collect(self, environment: PPOEnvironment, *, rollout_id: str, seed: int) -> PPORollout: ...
    def update(self, rollout: PPORollout) -> Mapping[str, float]: ...
    def save(self, output_dir: Path) -> None: ...
    def load(self, checkpoint: Path) -> int: ...


@dataclass(frozen=True)
class PPOTrainingLimits:
    """Wall-clock budget is checked between rollouts, not inside CUDA kernels."""

    max_seconds: float = 7200.0
    invalid_patience: int = 2

    def __post_init__(self) -> None:
        if not math.isfinite(self.max_seconds) or self.max_seconds <= 0:
            raise ValueError("PPO max_seconds must be positive and finite")
        if type(self.invalid_patience) is not int or self.invalid_patience <= 0:
            raise ValueError("PPO invalid_patience must be a positive integer")


def _ppo_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    temporary.replace(path)


def _ppo_append(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(payload) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def train_ppo(
    environment: PPOEnvironment,
    optimizer: PPOOptimizer,
    *,
    settings: TrainerSettings,
    lora: LoRASettings | None = None,
    updates: int,
    package_versions: Mapping[str, str],
    clock: Callable[[], float] = time.monotonic,
    limits: PPOTrainingLimits | None = None,
) -> TrainingResult:
    """Train toward a total update target with durable progress and safety gates.

    A stopped pilot is resumable research output, never a completed release.
    """
    lora = lora or LoRASettings()
    limits = limits or PPOTrainingLimits()
    if type(updates) is not int or updates <= 0:
        raise ValueError("PPO updates must be positive")
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = settings.output_dir / "training_status.json"
    if status_path.is_file():
        previous_status = json.loads(status_path.read_text())
        if previous_status.get("status") == "stopped" and previous_status.get("reason") in {
            "invalid_proposals",
            "reference_kl",
        }:
            raise ValueError(
                "PPO pilot failed a safety gate; review it before starting a new experiment"
            )
    if settings.resume_from_checkpoint is None and any(
        (settings.output_dir / name).exists()
        for name in (
            "ppo_state.json",
            "ppo_metrics.jsonl",
            "training_manifest.json",
            "ppo_run.json",
        )
    ):
        raise ValueError(
            "PPO output already contains a run; resume explicitly or use a new directory"
        )
    start_update = 0
    if settings.resume_from_checkpoint is not None:
        checkpoint = Path(settings.resume_from_checkpoint)
        if not checkpoint.exists():
            raise ValueError("PPO resume checkpoint does not exist")
        start_update = optimizer.load(checkpoint)
        if type(start_update) is not int or start_update < 0:
            raise ValueError("PPO checkpoint must report a nonnegative update index")
    if start_update > updates:
        raise ValueError("PPO checkpoint is beyond the requested total update target")
    completion_path = settings.output_dir / "training_manifest.json"
    history_path = settings.output_dir / "ppo_metrics.jsonl"
    recorded = []
    if history_path.exists():
        recorded = [
            json.loads(line) for line in history_path.read_text().splitlines() if line.strip()
        ]
        indices = [item["update"] for item in recorded]
        if len(set(indices)) != len(indices) or indices != sorted(indices):
            raise ValueError("PPO metrics contain duplicate or unordered updates")
        if indices and indices[-1] > start_update:
            raise ValueError(
                "PPO metrics are newer than the checkpoint; resume the latest durable state"
            )
    ppo = getattr(optimizer, "ppo", None)
    kl_limit = float(getattr(ppo, "target_kl", 0.1))
    invalid_limit = float(getattr(ppo, "max_invalid_fraction", 0.5))
    if not math.isfinite(kl_limit) or kl_limit <= 0 or not 0 <= invalid_limit <= 1:
        raise ValueError("invalid PPO safety thresholds")
    run_contract = {
        "model_revision": settings.model_revision,
        "tokenizer_revision": settings.tokenizer_revision,
        "prompt_version": settings.prompt_version,
        "data_hashes": settings.data_hashes,
        "seed": settings.seed,
        "learning_rate": settings.learning_rate,
        "max_length": settings.max_length,
        "invalid_patience": limits.invalid_patience,
        "target_kl": kl_limit,
        "max_invalid_fraction": invalid_limit,
        "optimizer_contract": getattr(optimizer, "_run_contract", {}),
    }
    run_path = settings.output_dir / "ppo_run.json"
    if run_path.exists():
        previous = json.loads(run_path.read_text())
        if canonical_json(previous["contract"]) != canonical_json(run_contract):
            raise ValueError("PPO run controls differ; resume with the original contract")
    else:
        _ppo_json(run_path, {"contract": run_contract, "package_versions": dict(package_versions)})
    if completion_path.is_file() and start_update < updates:
        # A completed pilot is not a completion certificate for its extension.
        previous_manifest = json.loads(completion_path.read_text())
        archive = settings.output_dir / (
            f"completed_manifest-{content_hash(previous_manifest)}.json"
        )
        completion_path.replace(archive)
    started = clock()
    metrics: dict[str, Any] = {}
    completed = start_update
    updates_this_invocation = 0
    sampled_decisions = 0
    invalid_streak = 0
    for item in reversed(recorded):
        if float(item.get("invalid_fraction", 0.0)) > invalid_limit:
            invalid_streak += 1
        else:
            break
    status = "completed"
    reason = None
    seen: set[str] = set()
    try:
        for index in range(start_update, updates):
            if clock() - started >= limits.max_seconds:
                status, reason = "stopped", "runtime_budget"
                break
            rollout_id = content_hash(
                {
                    "policy_version": optimizer.policy_version,
                    "update": index,
                    "seed": settings.seed,
                }
            )
            if rollout_id in seen:
                raise ValueError("PPO rollout ID was reused")
            print(
                canonical_json(
                    {"event": "ppo_collect", "update": index + 1, "target_updates": updates}
                ),
                flush=True,
            )
            rollout = optimizer.collect(
                environment, rollout_id=rollout_id, seed=settings.seed + index
            )
            validate_fresh_rollout(
                rollout,
                current_policy_version=optimizer.policy_version,
                expected_rollout_id=rollout_id,
            )
            sampled_decisions += len(rollout.transitions)
            invalid_indicators = [
                float(item.reward_components.get("invalid_proposals", 0.0))
                for item in rollout.transitions
            ]
            if any(not 0 <= value <= 1 for value in invalid_indicators):
                raise ValueError("PPO invalid-proposal indicators must be fractions in [0, 1]")
            invalid_fraction = sum(invalid_indicators) / len(rollout.transitions)
            invalid_streak = invalid_streak + 1 if invalid_fraction > invalid_limit else 0
            event = {
                "update": index + 1,
                "rollout_id": rollout_id,
                "collection_seed": settings.seed + index,
                "policy_version": rollout.policy_version,
                "sampled_decisions": len(rollout.transitions),
                "episodes": rollout.episode_count,
                "invalid_fraction": invalid_fraction,
                "action_tokens": sum(sum(item.action_mask) for item in rollout.transitions),
                "failure_counts": {
                    key.removeprefix("failure/"): sum(
                        item.reward_components.get(key, 0.0) for item in rollout.transitions
                    )
                    for key in sorted(
                        {
                            key
                            for item in rollout.transitions
                            for key in item.reward_components
                            if key.startswith("failure/")
                        }
                    )
                },
            }
            _ppo_append(settings.output_dir / "ppo_rollouts.jsonl", event)
            if invalid_streak >= limits.invalid_patience:
                status, reason = "stopped", "invalid_proposals"
                break
            if clock() - started >= limits.max_seconds:
                status, reason = "stopped", "runtime_budget"
                break
            metrics = dict(optimizer.update(rollout))
            if any(not math.isfinite(float(value)) for value in metrics.values()):
                raise ValueError("PPO optimizer returned nonfinite metrics")
            completed = index + 1
            updates_this_invocation += 1
            seen.add(rollout_id)
            record = {**event, **metrics, "elapsed_seconds": clock() - started}
            _ppo_append(history_path, record)
            print(
                canonical_json({"event": "ppo_update", "target_updates": updates, **record}),
                flush=True,
            )
            if completed % settings.save_steps == 0:
                optimizer.save(settings.output_dir / f"checkpoint-{completed}")
            measured_kl = max(
                float(metrics.get("reference_kl", 0.0)),
                float(metrics.get("reference_kl_max", 0.0)),
                float(metrics.get("post_update_reference_kl", 0.0)),
            )
            if measured_kl > kl_limit:
                status, reason = "stopped", "reference_kl"
                break
    except Exception as exc:
        _ppo_json(
            settings.output_dir / "training_status.json",
            {
                "status": "failed",
                "reason": type(exc).__name__,
                "completed_updates": completed,
                "target_updates": updates,
                "elapsed_seconds": clock() - started,
                "runtime_budget_seconds": limits.max_seconds,
                "invalid_streak": invalid_streak,
            },
        )
        raise
    elapsed = clock() - started
    optimizer.save(settings.output_dir)
    torch_module = getattr(optimizer, "torch", None)
    if torch_module is not None:
        metrics = _with_runtime_metrics(metrics, torch_module)
    metrics.update(
        {
            "completed_updates": completed,
            "updates_this_invocation": updates_this_invocation,
            "sampled_decisions_this_invocation": sampled_decisions,
            "status": status,
            "stop_reason": reason,
        }
    )
    manifest = _manifest("ppo", settings, lora, package_versions)
    _ppo_json(
        settings.output_dir / "training_status.json",
        {
            "status": status,
            "reason": reason,
            "completed_updates": completed,
            "target_updates": updates,
            "elapsed_seconds": elapsed,
            "metrics": metrics,
            "runtime_budget_seconds": limits.max_seconds,
            "invalid_streak": invalid_streak,
        },
    )
    if status == "completed":
        _ppo_json(
            settings.output_dir / "training_manifest.json",
            {
                "manifest_hash": manifest.manifest_hash,
                "manifest": asdict(manifest),
                "metrics": metrics,
                "elapsed_seconds": elapsed,
                "rollout_ids": sorted(seen),
                "optimizer_contract": getattr(optimizer, "_run_contract", {}),
            },
        )
    return TrainingResult("ppo", str(settings.output_dir), manifest.manifest_hash, metrics, elapsed)
