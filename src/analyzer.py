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
import pandas as pd
from tqdm import tqdm
import yaml
from .prompt import SYSTEM_PROMPT, generate_prompt
from .harmony_utils import (
    is_harmony_model,
    parse_harmony_response,
    render_harmony_prompt,
    build_harmony_conversation,
    get_harmony_stop_tokens
)


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


def _process_harmony_prompts(
    pipe: Pipeline,
    prompts: List[List[int]],
    max_new_tokens: int
) -> List[str]:
    """
    Run Harmony-formatted prompts sequentially to avoid batching issues.

    Args:
        pipe: Transformers pipeline hosting the Harmony-capable model.
        prompts: Pre-rendered Harmony token sequences.
        max_new_tokens: Generation cap for each prompt.

    Returns:
        List of decoded JSON strings (empty string when parsing fails).
    """
    pad_token_id = pipe.tokenizer.pad_token_id or pipe.tokenizer.eos_token_id
    stop_token_ids = get_harmony_stop_tokens()
    device = pipe.model.device
    outputs: List[str] = []

    with torch.inference_mode():
        for prompt_ids in tqdm(prompts, desc="Processing Harmony prompts"):
            try:
                input_ids = torch.tensor([prompt_ids], device=device)
                attention_mask = torch.ones_like(input_ids, device=device)

                result = pipe.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=stop_token_ids,
                    do_sample=False,
                    pad_token_id=pad_token_id,
                )[0].tolist()

                completion_tokens = result[len(prompt_ids):]
                parsed_output = parse_harmony_response(completion_tokens)
                outputs.append(parsed_output or "")
            except Exception as exc:
                print(f"Harmony generation failed: {exc}")
                outputs.append("")
    return outputs


def _batched(iterable: List[Any], batch_size: int):
    """
    Yield successive slices of an iterable to support manual batching.

    Args:
        iterable: Sequence to iterate over.
        batch_size: Number of elements to include in each batch.

    Yields:
        Lists containing up to `batch_size` items from the iterable.
    """
    for idx in range(0, len(iterable), batch_size):
        yield iterable[idx : idx + batch_size]


def process_in_batches(
    pipe: Pipeline,
    prompts: List[Any],
    batch_size: int,
    max_new_tokens: int,
    model_config: Dict[str, Any]
) -> List[str]:
    """
    Execute prompts against a pipeline, respecting Harmony vs standard batching.

    Args:
        pipe: Configured transformers pipeline.
        prompts: Prompt payloads (strings or Harmony token lists).
        batch_size: Batch size for non-Harmony models.
        max_new_tokens: Maximum tokens to generate per prompt.
        model_config: Model definition used to choose Harmony logic.

    Returns:
        List of generated raw strings for every prompt.
    """
    if is_harmony_model(model_config):
        return _process_harmony_prompts(pipe, prompts, max_new_tokens)

    outputs: List[str] = []
    for batch in _batched(prompts, batch_size):
        try:
            batch_outputs = pipe(
                batch,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_full_text=False
            )
            for output in batch_outputs:
                if isinstance(output, list) and output and isinstance(output[0], dict):
                    outputs.append(output[0].get('generated_text', ''))
                elif isinstance(output, dict):
                    outputs.append(output.get('generated_text', ''))
                else:
                    outputs.append('')
        except Exception as exc:
            print(f"Batch generation failed: {exc}")
            outputs.extend([''] * len(batch))

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

    raw_outputs = process_in_batches(pipe, prompts, batch_size, max_new_tokens, model_config)
    parsed_outputs = [parse_json_output(out) for out in raw_outputs]

    add_model_results_to_dataframe(df, model_name, raw_outputs, parsed_outputs)

    del pipe
    cleanup_gpu()
    cleanup_hf_cache()


def analyze_profiles(
    df: pd.DataFrame,
    config: Dict[str, Any],
    input_col: str = 'about_me',
    batch_size: int = 10,
    max_new_tokens: int = 2000
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