"""Configuration management for WFA Profile Analyzer."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal
import os

import yaml

TorchDtypeStr = Literal["auto", "float16", "bfloat16", "float32"]


@dataclass
class ModelConfig:
    name: str = "default_model"
    enabled: bool = True
    model_id: str = "meta-llama/Llama-3-8B-Instruct"
    torch_dtype: TorchDtypeStr = "auto"
    device_map: str = "auto"
    is_instruct: bool = False
    hf_token: Optional[str] = None


@dataclass
class GenerationConfig:
    batch_size: int = 10
    max_input_tokens: int = 2000
    max_new_tokens: int = 2000
    do_sample: bool = False
    num_beams: int = 1
    # kept for backward compatibility; pipeline ignores this
    handle_long_generation: str = "hole"


@dataclass
class DataConfig:
    dataset_column: str = "about_me"
    output_column: str = "about_me_processed"
    human_label_column: str = "Human_flag"


@dataclass
class EvaluationConfig:
    save_wide_table: bool = True
    positive_class: str = "bad"


@dataclass
class AppConfig:
    models: list[ModelConfig] = field(default_factory=lambda: [ModelConfig()])
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    hf_token: Optional[str] = None

    @property
    def enabled_models(self) -> list[ModelConfig]:
        return [m for m in self.models if m.enabled]

    def validate(self) -> None:
        enabled = self.enabled_models
        if not enabled:
            raise ValueError("At least one model must be enabled")

        if len(self.models) > 3:
            raise ValueError("Maximum of 3 models allowed")

        names = [m.name for m in enabled]
        if len(names) != len(set(names)):
            raise ValueError(f"Model names must be unique. Found duplicates: {names}")

        for model in enabled:
            if not (model.model_id and model.model_id.strip()):
                raise ValueError(f"Model '{model.name}' must have a valid model_id")

    @classmethod
    def from_yaml(cls, config_path: Path) -> "AppConfig":
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f) or {}

        models_data = config_dict.get("models", [])
        models = [ModelConfig(**m) for m in models_data]
        # Prefer explicit token in YAML; fall back to HF_TOKEN env var if not provided
        global_hf_token = config_dict.get("hf_token") or os.getenv("HF_TOKEN")

        for model in models:
            if model.hf_token is None:
                model.hf_token = global_hf_token

        return cls(
            models=models,
            hf_token=global_hf_token,
            generation=GenerationConfig(**config_dict.get("generation", {})),
            data=DataConfig(**config_dict.get("data", {})),
            evaluation=EvaluationConfig(**config_dict.get("evaluation", {})),
        )


def load_config(config_path: Optional[Path] = None) -> AppConfig:
    config = AppConfig.from_yaml(config_path) if (config_path and config_path.exists()) else AppConfig()
    config.validate()
    return config
