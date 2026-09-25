"""Typed experiment configuration, loaded from YAML.

Secrets are never read from the config file. They come from the
environment (OPENROUTER_API_KEY, HF_TOKEN) so that configs are safe to
commit.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml


@dataclass
class DefenderConfig:
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_use_double_quant: bool = True
    max_seq_length: int = 1024
    max_new_tokens: int = 512
    generation_batch_size: int = 8
    trust_remote_code: bool = True


@dataclass
class LoraConfigSpec:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


@dataclass
class RedTeamConfig:
    model_id: str = "nvidia/nemotron-3.5-lightning:free"
    temperature: float = 1.1
    max_tokens: int = 3000
    retries: int = 4
    attacks_per_behavior: int = 3
    history_window: int = 10
    request_delay_s: float = 0.5


@dataclass
class JudgeConfig:
    # Switch by model_id. OpenRouter instruct IDs use the API judge.
    # HarmBench official classifiers load locally from Hugging Face:
    #   cais/HarmBench-Mistral-7b-val-cls
    #   cais/HarmBench-Llama-2-13b-cls
    # Do NOT put llama-guard / content-safety classifiers here -- see README.
    model_id: str = "meta-llama/llama-3.3-70b-instruct"
    temperature: float = 0.0
    max_tokens: int = 16
    retries: int = 4
    # One case per request. Batching many cases into a single prompt is
    # cheaper but measurably less reliable, and a mis-aligned label silently
    # corrupts the training set. See README "Judging".
    cases_per_request: int = 1
    # Used only for local HarmBench classifiers (ignored by the API judge).
    load_in_4bit: bool = True
    # Copyright behaviours are scored by n-gram overlap against a reference
    # text, matching HarmBench's separate copyright classifier.
    copyright_ngram_n: int = 20
    copyright_reference_dir: str | None = None


@dataclass
class TrainConfig:
    epochs: int = 2
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    logging_steps: int = 5
    save_strategy: str = "no"
    fp16: bool = True
    optim: str = "paged_adamw_8bit"
    # "from_base" retrains a fresh adapter on the cumulative dataset each
    # round; "incremental" keeps training the current adapter on new data
    # only. Mixing the two double-counts early rounds. See README "Training".
    restart_policy: str = "from_base"
    # Fraction of benign retention examples mixed into the refusal data, as a
    # multiple of the refusal example count. 1.0 == equal numbers.
    retention_ratio: float = 1.0


@dataclass
class DataConfig:
    harmbench_csv: str = "data/harmbench_behaviors_text_test.csv"
    # HarmBench functional categories to include. "copyright" requires
    # reference texts (see JudgeConfig.copyright_reference_dir); without
    # them those behaviours cannot be scored and are excluded.
    functional_categories: list[str] = field(
        default_factory=lambda: ["standard"]
    )
    num_train_behaviors: int = 10
    num_probe_behaviors: int = 10
    behavior_split_seed: int = 4
    retention_dataset: str = "tatsu-lab/alpaca"
    overrefusal_dataset: str = "walledai/XSTest"


@dataclass
class ExperimentConfig:
    name: str = "jailbreak_selfplay"
    output_root: str = "runs"
    num_rounds: int = 4
    seed: int = 4
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    defender: DefenderConfig = field(default_factory=DefenderConfig)
    lora: LoraConfigSpec = field(default_factory=LoraConfigSpec)
    redteam: RedTeamConfig = field(default_factory=RedTeamConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)

    @property
    def run_dir(self) -> Path:
        return Path(self.output_root) / self.name

    @property
    def checkpoint_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def rounds_dir(self) -> Path:
        return self.run_dir / "rounds"

    @property
    def probe_dir(self) -> Path:
        return self.run_dir / "probe"


def _from_dict(cls: type, payload: dict[str, Any]) -> Any:
    """Recursively build a dataclass, rejecting unknown keys loudly.

    ``from __future__ import annotations`` makes ``Field.type`` a string, so
    resolve the real annotations before testing for nested dataclasses.
    """
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = set(payload) - known
    if unknown:
        raise ValueError(
            f"Unknown config key(s) for {cls.__name__}: {sorted(unknown)}"
        )
    kwargs: dict[str, Any] = {}
    for key, value in payload.items():
        ftype = hints.get(key)
        if is_dataclass(ftype) and isinstance(value, dict):
            kwargs[key] = _from_dict(ftype, value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path, encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or {}
    return _from_dict(ExperimentConfig, payload)


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Environment variable {name} is not set. "
            f"Export it before running (see README 'Credentials')."
        )
    return value
