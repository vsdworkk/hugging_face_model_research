"""Robust JSON parsing utilities for LLM outputs."""

import json
import re
import logging
from typing import Any, Optional

import pandas as pd
from ..constants import OutputColumn

logger = logging.getLogger(__name__)


def extract_first_json_object(text: str) -> Optional[dict]:
    """Extract the first JSON object using regex with a fast-path parse.

    Tries to parse the whole string first; if that fails, searches for the
    first brace-delimited object and attempts to parse that substring.
    """
    if not isinstance(text, str):
        return None

    s = text.strip()

    # Fast path: try parsing entire string
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # Regex search for a JSON-like object (handles simple nesting patterns)
    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", s)
    if match:
        try:
            obj = json.loads(match.group(0))
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

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
        f"{prefix}{OutputColumn.QUALITY}",
        f"{prefix}{OutputColumn.REASONING}",
        f"{prefix}{OutputColumn.TAGS}",
        f"{prefix}{OutputColumn.RECOMMENDATION}",
    ]
    
    for col in expected_cols:
        if col not in df.columns:
            df[col] = None
    
    # Type coercion and validation
    quality_col = f"{prefix}{OutputColumn.QUALITY}"
    df[quality_col] = (
        df[quality_col]
        .astype(str)
        .str.strip()
        .str.lower()
        .where(lambda s: s.isin(["good", "bad"]))
    )
    
    # Clean up tags
    tags_col = f"{prefix}{OutputColumn.TAGS}"
    df[tags_col] = df[tags_col].apply(coerce_to_tag_list)
    
    # Fill text fields
    df[f"{prefix}{OutputColumn.RECOMMENDATION}"] = (
        df[f"{prefix}{OutputColumn.RECOMMENDATION}"].fillna("").astype(str)
    )
    df[f"{prefix}{OutputColumn.REASONING}"] = df[f"{prefix}{OutputColumn.REASONING}"].fillna("").astype(str)
    
    logger.info(f"Successfully parsed {df[quality_col].notna().sum()} valid outputs")
    
    return df