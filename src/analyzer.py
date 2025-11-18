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
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def parse_json_output(text: str) -> Optional[Dict[str, Any]]:
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

#
# Helpers to build & debug the final pipeline call (updated)
#

def build_pipeline_args(model_config: Dict[str, Any], hf_token: Optional[str]) -> Dict[str, Any]:
    """
    Build the exact keyword args that will be sent to transformers.pipeline
    *excluding* the task (we pass "text-generation" positionally).
    """
    return {
        "model": model_config["model_id"],
        "device_map": model_config.get("device_map", "auto"),
        "torch_dtype": model_config.get("torch_dtype", "auto"),
        "token": hf_token,
    }


def load_model_pipeline(model_config: Dict[str, Any], hf_token: Optional[str] = None) -> Pipeline:
    """
    Load and configure a model pipeline.
    Quantization disabled; token passed directly (matches your working snippet).
    """
    args = build_pipeline_args(model_config, hf_token)
    
    # IMPORTANT: Pass the task positionally as in your working snippet
    return pipeline(
        "text-generation",
        **args
    )

def prepare_tokenizer(pipe: Pipeline) -> None:
    if pipe.tokenizer.pad_token_id is None:
        # Try to use a different token for padding to avoid the warning
        if hasattr(pipe.tokenizer, 'unk_token') and pipe.tokenizer.unk_token is not None:
            pipe.tokenizer.pad_token = pipe.tokenizer.unk_token
        else:
            # Fallback to eos_token if no unk_token available
            pipe.tokenizer.pad_token = pipe.tokenizer.eos_token
    pipe.tokenizer.padding_side = 'left'


def generate_prompts(texts: List[str], model_config: Dict[str, Any], tokenizer: Any) -> List[Any]:
    """Generate prompts - returns strings for standard models, token IDs for Harmony models."""
    return [generate_prompt(text, model_config, tokenizer) for text in texts]


def _process_harmony_prompts(
    pipe: Pipeline,
    prompts: List[List[int]],
    max_new_tokens: int
) -> List[str]:
    """Generate outputs for Harmony models with inference safeguards."""
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
    for idx in range(0, len(iterable), batch_size):
        yield iterable[idx : idx + batch_size]


def process_in_batches(
    pipe: Pipeline,
    prompts: List[Any],
    batch_size: int,
    max_new_tokens: int,
    model_config: Dict[str, Any]
) -> List[str]:
    """Process prompts (Harmony sequentially, standard models via pipeline batching)."""
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
    model_name = model_config['name']
    print(f"\nProcessing with {model_name}: {model_config['model_id']}")

    pipe = load_model_pipeline(model_config, hf_token)
    prepare_tokenizer(pipe)

    footprint_gb = pipe.model.get_memory_footprint() / (1024 ** 3)
    print(f"Model memory footprint: {footprint_gb:.2f} GB")

    if torch.cuda.is_available():
        gpu_allocated = torch.cuda.memory_allocated() / (1024**3)
        gpu_reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"GPU memory allocated: {gpu_allocated:.2f} GB, reserved: {gpu_reserved:.2f} GB")

    texts = df[input_col].fillna('').astype(str).tolist()
    prompts = generate_prompts(texts, model_config, pipe.tokenizer)

    raw_outputs = process_in_batches(pipe, prompts, batch_size, max_new_tokens, model_config)
    parsed_outputs = [parse_json_output(out) for out in raw_outputs]

    add_model_results_to_dataframe(df, model_name, raw_outputs, parsed_outputs)

    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gpu_allocated_after = torch.cuda.memory_allocated() / (1024**3)
        gpu_reserved_after = torch.cuda.memory_reserved() / (1024**3)
        print(f"GPU memory after cleanup: allocated: {gpu_allocated_after:.2f} GB, reserved: {gpu_reserved_after:.2f} GB")

    # free_before = shutil.disk_usage(".").free / (1024**3)
    # shutil.rmtree(os.path.expanduser("~/.cache/huggingface/hub"))
    # free_after = shutil.disk_usage(".").free / (1024**3)
    # print(f"Free disk GB after (approx): {free_after:.2f} (before: {free_before:.2f})")


def analyze_profiles(
    df: pd.DataFrame,
    config: Dict[str, Any],
    input_col: str = 'about_me',
    batch_size: int = 10,
    max_new_tokens: int = 2000
) -> pd.DataFrame:
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