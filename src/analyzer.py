"""
Profile Quality Analyzer
This module provides the main functionality for analyzing job seeker profile quality using
multiple language models. It serves as the central orchestrator that:
- Orchestrates the multi-model evaluation process
- Processes profile text through multiple models using efficient batching
- Generates structured quality assessments (good/bad ratings with reasoning)
- Outputs results as structured data for evaluation

Helper functions for configuration, memory management, and output parsing have been moved 
to src/utils.py to keep this module focused on the core analysis logic.
"""

from typing import List, Dict, Optional, Any
import os

from transformers import pipeline
from transformers.pipelines import Pipeline
from transformers.pipelines.pt_utils import KeyDataset
import datasets
import pandas as pd
from tqdm.auto import tqdm

from .prompt import generate_prompt
from .utils import (
    load_config,
    parse_json_output,
    clean_harmony_output,
    build_pipeline_args,
    report_model_memory,
    report_gpu_memory,
    cleanup_gpu,
    cleanup_hf_cache,
    prepare_tokenizer
)


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
    
    for output in tqdm(pipe(
        KeyDataset(dataset, 'text'),
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_full_text=False
    ), total=len(dataset), desc=f"Generating ({len(dataset)} profiles)"):
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
