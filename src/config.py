"""Configuration management for WFA Profile Analyzer."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    """Configuration for a single LLM model."""
    
    name: str = "default_model"
    enabled: bool = True
    model_id: str = "meta-llama/Llama-3-8B-Instruct"
    torch_dtype: str = "auto"
    device_map: str = "auto"
    hf_token: Optional[str] = None


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    
    batch_size: int = 10
    max_input_tokens: int = 2000
    max_new_tokens: int = 2000
    do_sample: bool = False
    num_beams: int = 1
    handle_long_generation: str = "hole"


@dataclass
class DataConfig:
    """Configuration for data processing."""
    
    dataset_column: str = "about_me"
    output_column: str = "about_me_processed"
    human_label_column: str = "Human_flag"


@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics."""
    
    save_wide_table: bool = True
    positive_class: str = "bad"


@dataclass
class AppConfig:
    """Main application configuration."""
    
    models: list[ModelConfig] = field(default_factory=lambda: [ModelConfig()])
    hf_token: Optional[str] = None
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    def get_enabled_models(self) -> list[ModelConfig]:
        """Get list of enabled models only.
        
        Returns:
            List of ModelConfig instances where enabled=True.
        """
        enabled = [m for m in self.models if m.enabled]
        
        # Apply hf_token to all enabled models if set
        if self.hf_token:
            for model in enabled:
                if model.hf_token is None:
                    model.hf_token = self.hf_token
        
        return enabled
    
    def validate(self) -> None:
        """Validate configuration.
        
        Raises:
            ValueError: If configuration is invalid.
        """
        enabled = self.get_enabled_models()
        
        if len(enabled) == 0:
            raise ValueError("At least one model must be enabled")
        
        if len(self.models) > 3:
            raise ValueError("Maximum of 3 models allowed")
        
        # Check for duplicate names
        names = [m.name for m in enabled]
        if len(names) != len(set(names)):
            raise ValueError(f"Model names must be unique. Found duplicates: {names}")
        
        # Validate each model has required fields
        for model in enabled:
            if not model.model_id or not model.model_id.strip():
                raise ValueError(f"Model '{model.name}' must have a valid model_id")
    
    @classmethod
    def from_yaml(cls, config_path: Path) -> "AppConfig":
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file.
            
        Returns:
            AppConfig instance with loaded settings.
        """
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        
        # Parse models list
        models_data = config_dict.get("models", [])
        models = [ModelConfig(**model_dict) for model_dict in models_data]
        
        return cls(
            models=models,
            hf_token=config_dict.get("hf_token"),
            generation=GenerationConfig(**config_dict.get("generation", {})),
            data=DataConfig(**config_dict.get("data", {})),
            evaluation=EvaluationConfig(**config_dict.get("evaluation", {})),
        )
    
    @classmethod
    def from_env(cls, hf_token_env: str = "HF_TOKEN") -> "AppConfig":
        """Create config with HuggingFace token from environment.
        
        Args:
            hf_token_env: Environment variable name for HF token.
            
        Returns:
            AppConfig with token loaded from environment.
        """
        config = cls()
        config.hf_token = os.getenv(hf_token_env)
        return config


def load_config(config_path: Optional[Path] = None) -> AppConfig:
    """Load application configuration.
    
    Args:
        config_path: Optional path to config file. If None, uses defaults.
        
    Returns:
        Loaded AppConfig instance.
    
    Raises:
        ValueError: If configuration is invalid.
    """
    if config_path and config_path.exists():
        config = AppConfig.from_yaml(config_path)
    else:
        config = AppConfig()
    
    # Override with environment variable for token
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        config.hf_token = hf_token
    
    # Validate configuration
    config.validate()
    
    # Set HuggingFace environment variables for performance
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    os.environ["HF_HUB_DISABLE_EXPERIMENTAL"] = "1"
    
    return config