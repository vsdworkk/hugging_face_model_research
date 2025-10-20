"""Simplified profile analyzer for data science workflows."""
import json
import re
from typing import List, Dict, Optional
import torch
from transformers import pipeline
import pandas as pd
from tqdm import tqdm
import yaml

# System prompt for profile analysis
SYSTEM_PROMPT = """You are an expert AI Recruitment and Profile Quality Analyst.

Your task is to review the "about me" sections that candidates include in their online job seeker profiles and rate them as either "good" or "bad" quality.

---

### Criteria to score as "bad":

A response should be scored as "bad" if *any* of the following conditions are met:

1. **Contains Personal Information**
   - Includes names, addresses, phone numbers, email addresses, or any other demographic data that could bias the employer.

2. **Includes Inappropriate Content**
   - Contains offensive, discriminatory, violent, or sexually explicit language or references.

3. **Poor Grammar or Language Quality**
   - Contains multiple grammatical errors, spelling mistakes, or awkward phrasing that affects clarity or professionalism.

---

### Output Format

After review, output a **single JSON object** with the following structure. Do not include any explanation, commentary, or formatting outside the JSON.

{
  "quality": "good" | "bad",
  "reasoning": "One sentence summary of why the quality is bad. Leave empty if quality is good.",
  "tags": ["personal_info", "inappropriate_content", "grammar"],
  "recommendation_email": "Hi,\\n\\nWe've reviewed your 'About Me' section and noticed a few areas that could be improved to help you stand out more confidently to potential employers. [Concise recommendation...]\\n\\nWarm regards,\\nRecruitment Team"
}"""


def load_config(path: str) -> dict:
    """Load configuration from YAML file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def parse_json_output(text: str) -> Optional[dict]:
    """Extract JSON from model output."""
    if not isinstance(text, str):
        return None
    
    # Try direct parse
    try:
        obj = json.loads(text.strip())
        return obj if isinstance(obj, dict) else None
    except:
        pass
    
    # Try to find JSON object in text
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text)
    if match:
        try:
            obj = json.loads(match.group())
            return obj if isinstance(obj, dict) else None
        except:
            pass
    
    return None


def get_quantization_config(quantization: str):
    """Get quantization config if needed."""
    if quantization == "none":
        return None
    
    from transformers import BitsAndBytesConfig
    
    if quantization == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    elif quantization == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    else:
        raise ValueError(f"Unknown quantization: {quantization}")


def analyze_profiles(df: pd.DataFrame, 
                    model_configs: List[Dict], 
                    input_col: str = 'about_me',
                    batch_size: int = 10,
                    max_new_tokens: int = 2000) -> pd.DataFrame:
    """Run profile analysis with multiple models."""
    import os
    results = df.copy()
    
    for model_cfg in model_configs:
        if not model_cfg.get('enabled', True):
            continue
            
        model_name = model_cfg['name']
        print(f"\nProcessing with {model_name}: {model_cfg['model_id']}")
        
        # Get HF token from model config or environment
        hf_token = model_cfg.get('hf_token') or os.getenv('HF_TOKEN')
        
        # Setup hub_kwargs for token
        hub_kwargs = {}
        if hf_token:
            hub_kwargs["token"] = hf_token
        
        # Setup quantization
        model_kwargs = {}
        quantization = model_cfg.get('quantization', 'none')
        torch_dtype = model_cfg.get('torch_dtype', 'auto')
        
        if quantization != 'none':
            model_kwargs['quantization_config'] = get_quantization_config(quantization)
            torch_dtype = 'auto'
        
        # Create pipeline
        pipe = pipeline(
            "text-generation",
            model=model_cfg['model_id'],
            device_map=model_cfg.get('device_map', 'auto'),
            torch_dtype=torch_dtype,
            model_kwargs=model_kwargs,
            **hub_kwargs
        )
        
        # Rest of the function remains the same...
        
        # Set padding
        if pipe.tokenizer.pad_token_id is None:
            pipe.tokenizer.pad_token = pipe.tokenizer.eos_token
        pipe.tokenizer.padding_side = 'left'
        
        # Print memory footprint
        footprint_gb = pipe.model.get_memory_footprint() / (1024 ** 3)
        print(f"Model memory footprint: {footprint_gb:.2f} GB")
        
        # Process in batches
        texts = df[input_col].fillna('').astype(str).tolist()
        outputs = [''] * len(texts)
        
        # Build prompts
        prompts = []
        for text in texts:
            if model_cfg.get('is_instruct', False):
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"About me:\n{text}"}
                ]
                prompt = pipe.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
            else:
                prompt = f"{SYSTEM_PROMPT}\n\nAbout me:\n{text}\n\nAnalysis:"
            prompts.append(prompt)
        
        # Generate in batches
        for i in tqdm(range(0, len(prompts), batch_size), desc=f"{model_name}"):
            batch = prompts[i:i+batch_size]
            batch_outputs = pipe(
                batch,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_full_text=False
            )
            
            for j, output in enumerate(batch_outputs):
                outputs[i+j] = output[0]['generated_text']
        
        # Parse outputs
        parsed_outputs = [parse_json_output(out) for out in outputs]
        
        # Add columns
        results[f'{model_name}_raw'] = outputs
        results[f'{model_name}_quality'] = [
            p.get('quality', '') if p else '' for p in parsed_outputs
        ]
        results[f'{model_name}_reasoning'] = [
            p.get('reasoning', '') if p else '' for p in parsed_outputs
        ]
        results[f'{model_name}_tags'] = [
            p.get('tags', []) if p else [] for p in parsed_outputs
        ]
        results[f'{model_name}_recommendation'] = [
            p.get('recommendation_email', '') if p else '' for p in parsed_outputs
        ]
        
        # Cleanup
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results