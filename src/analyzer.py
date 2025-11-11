"""
Profile Quality Analyzer

This module provides the main functionality for analyzing job seeker profile quality using 
multiple language models. It serves as the central orchestrator that:

- Loads and configures language models (LLMs) based on config.yaml settings
- Processes profile text through multiple models using efficient batching
- Generates structured quality assessments (good/bad ratings with reasoning)
- Handles model quantization, tokenization, and memory management
- Outputs results as structured data for evaluation

Key Components:
- Model pipeline management (loading, quantization, cleanup)
- Batch processing for efficient inference
- JSON output parsing and validation
- Integration with prompt templates from prompt.py
- Results formatting for downstream evaluation in evaluate.py

Dependencies:
- config.yaml: Model configurations and processing parameters
- prompt.py: System prompts and message formatting
- evaluate.py: Metrics calculation and model comparison
"""
import json
import re
from typing import List, Dict, Optional, Any, Tuple
import os

import torch
from transformers import pipeline, Pipeline, BitsAndBytesConfig
import pandas as pd
from tqdm import tqdm
import yaml

from .prompt import SYSTEM_PROMPT, generate_prompt


def load_config(path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        path: Path to the YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def parse_json_output(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON object from model output text.
    
    Tries multiple strategies:
    1. Direct JSON parse
    2. Regex extraction of JSON-like content
    
    Args:
        text: Raw model output text
        
    Returns:
        Parsed JSON dictionary or None if parsing fails
    """
    if not isinstance(text, str):
        return None
    
    # Try direct parse first
    try:
        obj = json.loads(text.strip())
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON object using regex
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text)
    if match:
        try:
            obj = json.loads(match.group())
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            pass
    
    return None


def get_quantization_config(quantization: str) -> Optional[BitsAndBytesConfig]:
    """
    Get quantization configuration based on string identifier.
    
    Args:
        quantization: Quantization type ('none', '8bit', '4bit')
        
    Returns:
        BitsAndBytesConfig object or None
        
    Raises:
        ValueError: If quantization type is unknown
    """
    if quantization == "none":
        return None
    elif quantization == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    elif quantization == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    else:
        raise ValueError(f"Unknown quantization type: {quantization}")


def load_model_pipeline(model_config: Dict[str, Any], hf_token: Optional[str] = None) -> Pipeline:
    """
    Load and configure a model pipeline.
    
    Args:
        model_config: Model configuration dictionary
        hf_token: Hugging Face token for authentication
        
    Returns:
        Configured text generation pipeline
    """
    # Setup hub kwargs for authentication
    hub_kwargs = {"token": hf_token} if hf_token else {}
    
    # Setup quantization and model_kwargs
    model_kwargs = {}
    quantization = model_config.get('quantization', 'none')
    torch_dtype = model_config.get('torch_dtype', 'auto')
    
    if quantization != 'none':
        model_kwargs['quantization_config'] = get_quantization_config(quantization)
        # Force auto dtype when using quantization
        torch_dtype = 'auto'
    
    # Create pipeline - use dtype parameter for transformers 4.55+
    return pipeline(
        "text-generation",
        model=model_config['model_id'],
        dtype=torch_dtype,  # Changed from torch_dtype to dtype
        device_map=model_config.get('device_map', 'auto'),
        model_kwargs=model_kwargs,
        **hub_kwargs
    )


def prepare_tokenizer(pipe: Pipeline) -> None:
    """
    Configure tokenizer settings for consistent behavior.
    
    Args:
        pipe: The pipeline to configure
    """
    if pipe.tokenizer.pad_token_id is None:
        pipe.tokenizer.pad_token = pipe.tokenizer.eos_token
    pipe.tokenizer.padding_side = 'left'


def generate_prompts(texts: List[str], model_config: Dict[str, Any], tokenizer: Any) -> List[str]:
    """
    Generate prompts for all texts using the model configuration.
    
    Args:
        texts: List of profile texts
        model_config: Model configuration
        tokenizer: Model tokenizer
        
    Returns:
        List of formatted prompts
    """
    return [generate_prompt(text, model_config, tokenizer) for text in texts]


def process_in_batches(
    pipe: Pipeline,
    prompts: List[str],
    batch_size: int,
    max_new_tokens: int
) -> List[str]:
    """
    Process prompts in batches through the model pipeline.
    """
    outputs = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
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
    return outputs



def add_model_results_to_dataframe(
    df: pd.DataFrame,
    model_name: str,
    raw_outputs: List[str],
    parsed_outputs: List[Optional[Dict[str, Any]]]
) -> None:
    """
    Add model results as new columns to the dataframe.
    
    Args:
        df: DataFrame to modify (in-place)
        model_name: Name prefix for columns
        raw_outputs: Raw model outputs
        parsed_outputs: Parsed JSON outputs
    """
    # Add raw output column
    df[f'{model_name}_raw'] = raw_outputs
    
    # Add parsed fields
    df[f'{model_name}_quality'] = [
        p.get('quality', '') if p else '' for p in parsed_outputs
    ]
    df[f'{model_name}_reasoning'] = [
        p.get('reasoning', '') if p else '' for p in parsed_outputs
    ]
    df[f'{model_name}_tags'] = [
        p.get('tags', []) if p else [] for p in parsed_outputs
    ]
    df[f'{model_name}_improvement_points'] = [
        p.get('improvement_points', []) if p else [] for p in parsed_outputs
    ]


def analyze_single_model(
    df: pd.DataFrame,
    model_config: Dict[str, Any],
    hf_token: Optional[str],
    input_col: str,
    batch_size: int,
    max_new_tokens: int
) -> None:
    """
    Analyze profiles using a single model.
    
    Args:
        df: DataFrame with profiles (modified in-place)
        model_config: Model configuration
        hf_token: Hugging Face token
        input_col: Column name containing profile text
        batch_size: Batch size for processing
        max_new_tokens: Maximum tokens to generate
    """
    model_name = model_config['name']
    print(f"\nProcessing with {model_name}: {model_config['model_id']}")
    
    # Load model pipeline
    pipe = load_model_pipeline(model_config, hf_token)
    prepare_tokenizer(pipe)
    
    # Print memory footprint
    footprint_gb = pipe.model.get_memory_footprint() / (1024 ** 3)
    print(f"Model memory footprint: {footprint_gb:.2f} GB")
    
    # Prepare texts and prompts
    texts = df[input_col].fillna('').astype(str).tolist()
    prompts = generate_prompts(texts, model_config, pipe.tokenizer)
    
    # Generate outputs
    raw_outputs = process_in_batches(pipe, prompts, batch_size, max_new_tokens)
    
    # Parse outputs
    parsed_outputs = [parse_json_output(out) for out in raw_outputs]
    
    # Add results to dataframe
    add_model_results_to_dataframe(df, model_name, raw_outputs, parsed_outputs)
    
    # Cleanup
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def analyze_profiles(
    df: pd.DataFrame, 
    config: Dict[str, Any],
    input_col: str = 'about_me',
    batch_size: int = 10,
    max_new_tokens: int = 2000
) -> pd.DataFrame:
    """
    Analyze profiles using multiple models specified in configuration.
    
    This is the main entry point for profile analysis. It:
    1. Loads each enabled model from the configuration
    2. Generates quality assessments for all profiles
    3. Adds results as new columns to the dataframe
    
    Args:
        df: DataFrame containing profiles to analyze
        config: Configuration dictionary with model definitions
        input_col: Column name containing profile text
        batch_size: Number of profiles to process at once
        max_new_tokens: Maximum tokens to generate per response
        
    Returns:
        DataFrame with added columns for each model's results
    """
    results = df.copy()
    
    # Get HF token from environment
    hf_token = os.getenv('HF_TOKEN')
    
    # Process each enabled model
    for model_config in config['models']:
        if not model_config.get('enabled', True):
            continue
        
        # Analyze with this model
        analyze_single_model(
            results,
            model_config,
            hf_token,
            input_col,
            batch_size,
            max_new_tokens
        )
    
    return results