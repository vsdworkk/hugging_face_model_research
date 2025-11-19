"""
Utility Functions for Profile Analyzer
This module contains helper functions used by the main analyzer module.
It handles:
- Configuration loading
- Model output parsing and cleaning
- GPU and memory management
- Hugging Face cache management
- Pipeline argument construction
"""

import json
import os
import shutil
from typing import Any, Dict, Optional

import torch
import yaml
from transformers.pipelines import Pipeline


def load_config(path: str) -> Dict[str, Any]:
    """
    Load the YAML configuration that defines model settings and runtime options.

    Args:
        path: Absolute or relative path to the config YAML file.

    Returns:
        Dictionary containing the parsed configuration structure.
    """
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def parse_json_output(text: str) -> Optional[Dict[str, Any]]:
    """
    Convert a model response into a JSON dictionary, tolerating stray text.

    Args:
        text: Raw string produced by a model invocation.

    Returns:
        Parsed JSON object if decoding succeeds, otherwise None.
    """
    if not isinstance(text, str):
        return None
    
    try:
        obj = json.loads(text.strip())
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    
    # Fallback: find the first '{' and last '}'
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            pass
    return None


def clean_harmony_output(text: str) -> str:
    """
    Clean Harmony model output by extracting content after 'assistantfinal'.
    
    Args:
        text: Raw model output string
        
    Returns:
        Cleaned string containing only the final response
    """
    if not isinstance(text, str):
        return ""
    
    # Look for 'assistantfinal' marker
    marker = "assistantfinal"
    idx = text.find(marker)
    
    if idx != -1:
        return text[idx + len(marker):].strip()
    return text.strip()


def build_pipeline_args(model_config: Dict[str, Any], hf_token: Optional[str]) -> Dict[str, Any]:
    """
    Translate a model configuration entry into kwargs for `transformers.pipeline`.

    Args:
        model_config: Single model entry from the config file.
        hf_token: Optional Hugging Face token supplied by the user.

    Returns:
        Keyword arguments that can be expanded into `pipeline("text-generation", **kwargs)`.
    """
    return {
        "model": model_config["model_id"],
        "device_map": model_config.get("device_map", "auto"),
        "torch_dtype": model_config.get("torch_dtype", "auto"),
        "token": hf_token,
    }


def report_model_memory(pipe: Pipeline) -> None:
    """
    Print how much GPU memory the loaded model occupies.

    Args:
        pipe: Transformers pipeline whose underlying model was just loaded.
    """
    try:
        footprint_gb = pipe.model.get_memory_footprint() / (1024 ** 3)
        print(f"Model memory footprint: {footprint_gb:.2f} GB")
    except Exception:
        # Fallback or silence if model doesn't support get_memory_footprint
        pass


def report_gpu_memory(label: str) -> None:
    """
    Log current CUDA memory usage, skipping when CUDA is unavailable.

    Args:
        label: Prefix label to include in the printed message.
    """
    if not torch.cuda.is_available():
        return
    gpu_allocated = torch.cuda.memory_allocated() / (1024**3)
    gpu_reserved = torch.cuda.memory_reserved() / (1024**3)
    print(f"{label}: allocated {gpu_allocated:.2f} GB, reserved {gpu_reserved:.2f} GB")


def cleanup_gpu(label: str = "GPU memory after cleanup") -> None:
    """
    Release cached CUDA memory and report utilization for visibility.

    Args:
        label: Message prefix for the post-cleanup memory report.
    """
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    report_gpu_memory(label)


def free_disk_gb() -> float:
    """
    Report how much free disk space is available in gigabytes.

    Returns:
        Floating point number representing free disk size in GB.
    """
    return shutil.disk_usage(".").free / (1024**3)


def cleanup_hf_cache(cache_dir: Optional[str] = None) -> None:
    """
    Remove the Hugging Face cache directory to reclaim disk space.

    Args:
        cache_dir: Optional override path; defaults to the standard HF cache.
    """
    cache_path = cache_dir or os.path.expanduser("~/.cache/huggingface/hub")
    free_before = free_disk_gb()
    if os.path.isdir(cache_path):
        shutil.rmtree(cache_path, ignore_errors=True)
    free_after = free_disk_gb()
    freed = free_after - free_before
    print(
        f"Hugging Face cache cleared: freed {freed:.2f} GB "
        f"(free now {free_after:.2f} GB)"
    )


def prepare_tokenizer(pipe: Pipeline) -> None:
    """
    Ensure the pipeline tokenizer has padding configured for batch generation.

    Args:
        pipe: Transformers pipeline whose tokenizer may need pad token setup.
    """
    if pipe.tokenizer.pad_token_id is None:
        pipe.tokenizer.pad_token = (
            getattr(pipe.tokenizer, 'unk_token', None) or 
            pipe.tokenizer.eos_token
        )
    pipe.tokenizer.padding_side = 'left'

