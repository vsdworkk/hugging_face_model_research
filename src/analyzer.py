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
from transformers.pipelines.pt_utils import KeyDataset # for efficient dataset streaming
import datasets # Hugging Face datasets library
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
    
    match = re.search(r'\{[^{}]*?(\{[^{}]*?\}[^{}]*?)*?\}', text)
    if match:
        try:
            obj = json.loads(match.group())
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


def debug_print_pipeline_args(args: Dict[str, Any]) -> None:
    """Print the final pipeline call exactly as it will be executed (token masked)."""
    tok_mask = "***" if args.get("token") else None
    td_str = args.get("torch_dtype")
    td_str = "auto" if td_str == "auto" else str(td_str)
    print(
        f"  pipeline(\"text-generation\",\n"
        f"    model=\"{args['model']}\",\n"
        f"    device_map=\"{args.get('device_map', 'auto')}\",\n"
        f"    torch_dtype=\"{td_str}\",\n"
        f"    token={tok_mask}\n"
        f"  )"
    )


def load_model_pipeline(model_config: Dict[str, Any], hf_token: Optional[str] = None) -> Pipeline:
    """
    Load and configure a model pipeline using a debuggable builder.
    Quantization disabled; token passed directly (matches your working snippet).
    """
    args = build_pipeline_args(model_config, hf_token)
    debug_print_pipeline_args(args)
    
    # IMPORTANT: Pass the task positionally as in your working snippet
    return pipeline(
        "text-generation",
        model=args["model"],
        device_map=args.get("device_map", "auto"),
        torch_dtype=args.get("torch_dtype", "auto"),
        token=args.get("token"),
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


def process_in_batches(
    pipe: Pipeline,
    prompts: List[Any],
    batch_size: int,
    max_new_tokens: int,
    model_config: Dict[str, Any]
) -> List[str]:
    """Process prompts in batches, handling both standard and Harmony models."""
    outputs = []
    
    if is_harmony_model(model_config):
        # Process Harmony models with manual batching for token IDs
        pad_token_id = pipe.tokenizer.pad_token_id
        stop_token_ids = get_harmony_stop_tokens()
        debug_enabled = os.getenv("DEBUG_HARMONY", "0") == "1"

        for i in tqdm(range(0, len(prompts), batch_size), desc="Processing Harmony batches"):
            batch_prompts_ids = prompts[i:i + batch_size]
            
            # Pad batch to the length of the longest sequence
            max_len = max(len(p) for p in batch_prompts_ids)
            
            padded_prompts = []
            attention_masks = []
            for prompt_ids in batch_prompts_ids:
                padding_len = max_len - len(prompt_ids)
                padded_prompts.append([pad_token_id] * padding_len + prompt_ids)
                attention_masks.append([0] * padding_len + [1] * len(prompt_ids))

            input_ids = torch.tensor(padded_prompts).to(pipe.model.device)
            attention_mask = torch.tensor(attention_masks).to(pipe.model.device)

            # Generate responses for the batch
            result = pipe.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                eos_token_id=stop_token_ids,
                do_sample=False,
                pad_token_id=pad_token_id,
            )
            
            # Helper: find prompt boundary by searching for the last occurrence of the prompt
            def search_boundary_by_suffix(row_tokens: List[int], prompt_tokens: List[int]) -> int:
                if not prompt_tokens or not row_tokens or len(row_tokens) < len(prompt_tokens):
                    return 0
                last_possible = len(row_tokens) - len(prompt_tokens)
                for start in range(last_possible, -1, -1):
                    if row_tokens[start:start + len(prompt_tokens)] == prompt_tokens:
                        return start + len(prompt_tokens)
                return 0
            
            # Process each response in the batch, slicing out the prompt part
            for j, res_tensor in enumerate(result):
                prompt_ids = batch_prompts_ids[j]
                prompt_len = len(prompt_ids)
                pad_len = max_len - prompt_len
                row = res_tensor.tolist()

                # First try to locate the prompt directly in the generated sequence
                boundary = search_boundary_by_suffix(row, prompt_ids)

                # Detect if returned sequence includes the left padding we added
                has_left_pad_prefix = (
                    pad_len > 0 and len(row) >= pad_len and all(tok == pad_token_id for tok in row[:pad_len])
                )
                
                if boundary == 0:
                    # Fall back to heuristic boundary using padding info
                    boundary = max_len if has_left_pad_prefix else prompt_len

                if boundary > len(row):
                    boundary = min(len(row), max_len)
                
                completion_tokens = row[boundary:]
                parsed_output = parse_harmony_response(completion_tokens)

                if debug_enabled and (not parsed_output):
                    # Attempt a safer boundary search
                    alt_boundary = search_boundary_by_suffix(row, prompt_ids)
                    alt_completion_tokens = row[alt_boundary:]
                    alt_parsed = parse_harmony_response(alt_completion_tokens)

                    # Prepare debug preview strings
                    preview = pipe.tokenizer.decode(completion_tokens, skip_special_tokens=False)
                    alt_preview = pipe.tokenizer.decode(alt_completion_tokens, skip_special_tokens=False)
                    preview_short = preview[:240].replace("\n", "\\n")
                    alt_preview_short = alt_preview[:240].replace("\n", "\\n")

                    # Check if EOS/stop tokens appear at the end of either completion
                    row_last = row[-8:]
                    stop_hits = [tok for tok in row_last if tok in stop_token_ids]

                    print(
                        f"[Harmony DEBUG] batch={i//batch_size} item={j} len(row)={len(row)} max_len={max_len} "
                        f"prompt_len={prompt_len} pad_len={pad_len} left_pad={has_left_pad_prefix} "
                        f"boundary={boundary} alt_boundary={alt_boundary} gen_len={len(completion_tokens)} "
                        f"alt_gen_len={len(alt_completion_tokens)} stop_tail_hits={len(stop_hits)}"
                    )
                    print(f"[Harmony DEBUG] preview: {preview_short}")
                    print(f"[Harmony DEBUG] alt_preview: {alt_preview_short}")

                    # Prefer alt_parsed if it yields a result
                    if alt_parsed:
                        parsed_output = alt_parsed

                outputs.append(parsed_output or "")
    else:
        # Standard processing for non-Harmony models
        dataset = datasets.Dataset.from_dict({'text': prompts})
        for output in pipe(
            KeyDataset(dataset, 'text'),
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_full_text=False
        ):
            if isinstance(output, list) and output and isinstance(output[0], dict):
                outputs.append(output[0].get('generated_text', ''))
            elif isinstance(output, dict):
                outputs.append(output.get('generated_text', ''))
    
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