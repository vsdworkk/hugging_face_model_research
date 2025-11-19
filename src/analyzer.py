"""
Profile Quality Analyzer
This module provides the main functionality for analyzing job seeker profile quality using
multiple language models. It serves as the central orchestrator that:
- Loads and configures language models (LLMs) based on config.yaml settings
- Processes profile text through multiple models using efficient batching
- Generates structured quality assessments (good/bad ratings with reasoning)
- Handles tokenization and memory management (quantization temporarily disabled)
- Outputs results as structured data for evaluation
"""

import json
import re
from typing import List, Dict, Optional, Any
import os
import shutil # for clearing cache
import torch
from transformers import pipeline
from transformers.pipelines import Pipeline
from transformers.pipelines.pt_utils import KeyDataset
import datasets
import pandas as pd
from tqdm import tqdm
import yaml
from .prompt import SYSTEM_PROMPT, generate_prompt


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


# Helpers to support model loading and inference.

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
    footprint_gb = pipe.model.get_memory_footprint() / (1024 ** 3)
    print(f"Model memory footprint: {footprint_gb:.2f} GB")


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


def generate_prompts(texts: List[str], model_config: Dict[str, Any], tokenizer: Any) -> List[Any]:
    """
    Build per-profile prompts formatted according to the model family requirements.

    Args:
        texts: Collection of raw profile strings.
        model_config: Model configuration dict describing type and features.
        tokenizer: Tokenizer associated with the current pipeline model.

    Returns:
        List containing either raw strings (standard/instruct models) or token
        ID lists (Harmony models).
    """
    return [generate_prompt(text, model_config, tokenizer) for text in texts]


def process_in_batches(
    pipe: Pipeline,
    prompts: List[Any],
    batch_size: int,
    max_new_tokens: int
) -> List[str]:
    """
    Process prompts in batches through the model pipeline using Hugging Face datasets.

    Args:
        pipe: Configured transformers pipeline.
        prompts: Prompt payloads (strings).
        batch_size: Batch size.
        max_new_tokens: Maximum tokens to generate per prompt.

    Returns:
        List of generated raw strings for every prompt.
    """
    outputs: List[str] = []
    # Create a dataset for the prompts to leverage pipeline's batching
    dataset = datasets.Dataset.from_dict({'text': prompts})
    
    for output in pipe(
        KeyDataset(dataset, 'text'),
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_full_text=False
    ):
        # Each output corresponds to one prompt's result
        if isinstance(output, list) and output and isinstance(output[0], dict):
            outputs.append(output[0].get('generated_text', ''))
        elif isinstance(output, dict):
            outputs.append(output.get('generated_text', ''))
        else:
            outputs.append('')
            
    return outputs


def add_model_results_to_dataframe(
    df: pd.DataFrame,
    model_name: str,
    raw_outputs: List[str],
    parsed_outputs: List[Optional[Dict[str, Any]]]
) -> None:
    """
    Persist raw and structured model outputs onto the working DataFrame.

    Args:
        df: DataFrame being augmented with model columns.
        model_name: Human-readable model identifier used as column prefix.
        raw_outputs: Unparsed text strings returned by the model.
        parsed_outputs: JSON dictionaries derived from each raw output.
    """
    df[f'{model_name}_raw_output'] = raw_outputs
    df[f'{model_name}_quality'] = [p.get('quality') if p else None for p in parsed_outputs]
    df[f'{model_name}_reasoning'] = [p.get('reasoning') if p else None for p in parsed_outputs]
    df[f'{model_name}_tags'] = [p.get('tags', []) if p else [] for p in parsed_outputs]
    df[f'{model_name}_improvement_points'] = [p.get('improvement_points', []) if p else [] for p in parsed_outputs]


def analyze_single_model(
    df: pd.DataFrame,
    model_config: Dict[str, Any],
    hf_token: Optional[str],
    input_col: str,
    batch_size: int,
    max_new_tokens: int
) -> None:
    """
    Run one configured model over the dataset and append its outputs.

    Args:
        df: Working DataFrame containing profile text and accumulating results.
        model_config: Configuration entry describing the active model.
        hf_token: Optional Hugging Face token for gated checkpoints.
        input_col: Column name containing profile text.
        batch_size: Number of prompts to process per batch (non-Harmony only).
        max_new_tokens: Truncation limit for generations.
    """
    model_name = model_config['name']
    print(f"\nProcessing with {model_name}: {model_config['model_id']}")

    args = build_pipeline_args(model_config, hf_token)
    
    # IMPORTANT: Pass the task positionally as in your working snippet
    pipe = pipeline(
        "text-generation",
        **args
    )
    prepare_tokenizer(pipe)

    report_model_memory(pipe)
    report_gpu_memory("GPU memory before generation")

    texts = df[input_col].fillna('').astype(str).tolist()
    prompts = generate_prompts(texts, model_config, pipe.tokenizer)

    raw_outputs = process_in_batches(pipe, prompts, batch_size, max_new_tokens)

    if model_config.get('is_harmony', False):
        raw_outputs = [clean_harmony_output(out) for out in raw_outputs]

    parsed_outputs = [parse_json_output(out) for out in raw_outputs]

    add_model_results_to_dataframe(df, model_name, raw_outputs, parsed_outputs)

    del pipe
    cleanup_gpu()
    cleanup_hf_cache()


def analyze_profiles(
    df: pd.DataFrame,
    config: Dict[str, Any],
    input_col: Optional[str] = None,
    batch_size: Optional[int] = None,
    max_new_tokens: Optional[int] = None
) -> pd.DataFrame:
    """
    Orchestrate multi-model evaluation over a dataset of profile snippets.

    Args:
        df: Source dataframe with at least the input column populated.
        config: Parsed configuration dict listing model definitions.
        input_col: Column storing the profile "about me" text.
        batch_size: Default batch size for non-Harmony models.
        max_new_tokens: Generation limit shared across models.

    Returns:
        Copy of the input dataframe with additional model result columns.
    """
    results = df.copy()
    
    # Resolve defaults from config if not provided
    input_col = input_col or config.get('input_column', 'about_me')
    batch_size = batch_size or config.get('batch_size', 10)
    max_new_tokens = max_new_tokens or config.get('max_new_tokens', 2000)

    hf_token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )

    if hf_token:
        print(f"HF token detected: True ({hf_token[:4]}...{hf_token[-4:]})")
    else:
        print(f"HF token detected: False")
        
    for model_config in config['models']:
        if not model_config.get('enabled', True):
            continue
        analyze_single_model(
            results,
            model_config,
            hf_token,
            input_col,
            batch_size,
            max_new_tokens
        )
    
    return results