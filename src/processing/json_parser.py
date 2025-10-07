"""Robust JSON parsing utilities for LLM outputs."""

import json
import logging
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def extract_first_json_object(text: str) -> Optional[dict]:
    """Extract the first valid JSON object from potentially malformed text.
    
    Handles:
    - Text before/after JSON
    - Nested braces
    - Escaped quotes
    - Multiple JSON objects (returns first)
    
    Args:
        text: Raw text potentially containing JSON.
        
    Returns:
        Parsed dict if found, None otherwise.
    """
    if not isinstance(text, str):
        return None
    
    text = text.strip()
    
    # Fast path: try parsing entire string
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    
    # Robust extraction: find balanced braces
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        
        for i in range(start, len(text)):
            char = text[i]
            
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            else:
                if char == '"':
                    in_string = True
                elif char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        # Found balanced JSON, try to parse
                        try:
                            return json.loads(text[start : i + 1])
                        except Exception:
                            break  # Try next '{'
        
        # Move to next potential start
        start = text.find("{", start + 1)
    
    return None


def coerce_to_tag_list(value: Any) -> list[str]:
    """Coerce various inputs to a list of tag strings.
    
    Args:
        value: Value from parsed JSON (could be list, string, None, etc.)
        
    Returns:
        List of tag strings.
    """
    if isinstance(value, list):
        return [str(item) for item in value]
    
    if isinstance(value, str):
        trimmed = value.strip()
        # Try parsing as JSON array
        if trimmed.startswith("[") and trimmed.endswith("]"):
            try:
                parsed = json.loads(trimmed)
                return [str(item) for item in parsed] if isinstance(parsed, list) else []
            except Exception:
                return []
    
    return [] if pd.isna(value) else []


def parse_model_outputs_to_dataframe(
    raw_outputs: pd.Series,
    prefix: str = "ai_",
) -> pd.DataFrame:
    """Parse model JSON outputs into a structured DataFrame.
    
    Args:
        raw_outputs: Series containing raw JSON strings from model.
        prefix: Prefix for output column names.
        
    Returns:
        DataFrame with normalized columns.
    """
    logger.info(f"Parsing {len(raw_outputs)} model outputs")
    
    # Extract JSON objects
    parsed = raw_outputs.apply(extract_first_json_object)
    
    # Normalize to flat structure
    df = pd.json_normalize(parsed).add_prefix(prefix)
    
    # Ensure expected columns exist
    expected_cols = [
        f"{prefix}quality",
        f"{prefix}reasoning",
        f"{prefix}tags",
        f"{prefix}recommendation_email",
    ]
    
    for col in expected_cols:
        if col not in df.columns:
            df[col] = None
    
    # Type coercion and validation
    quality_col = f"{prefix}quality"
    df[quality_col] = (
        df[quality_col]
        .astype(str)
        .str.strip()
        .str.lower()
        .where(lambda s: s.isin(["good", "bad"]))
    )
    
    # Clean up tags
    tags_col = f"{prefix}tags"
    df[tags_col] = df[tags_col].apply(coerce_to_tag_list)
    
    # Fill text fields
    df[f"{prefix}recommendation_email"] = (
        df[f"{prefix}recommendation_email"].fillna("").astype(str)
    )
    df[f"{prefix}reasoning"] = df[f"{prefix}reasoning"].fillna("").astype(str)
    
    logger.info(f"Successfully parsed {df[quality_col].notna().sum()} valid outputs")
    
    return df