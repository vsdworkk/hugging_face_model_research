"""Robust JSON parsing utilities for LLM outputs."""
import json
import re
from typing import Any, Optional

import pandas as pd
from ..constants import OutputColumn

_JSON_OBJECT_RE = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}")


def extract_first_json_object(text: str) -> Optional[dict]:
    """Extract first JSON object from text. Returns dict or None."""
    if not isinstance(text, str):
        return None

    s = text.strip()
    # Fast path
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # Fallback: first {...}
    m = _JSON_OBJECT_RE.search(s)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _coerce_to_tag_list(value: Any) -> list[str]:
    """Always return a list[str] for tags."""
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        t = value.strip()
        if t.startswith("[") and t.endswith("]"):
            try:
                parsed = json.loads(t)
                return [str(x) for x in parsed] if isinstance(parsed, list) else []
            except Exception:
                return []
    return []


def parse_model_outputs_to_dataframe(raw_outputs: pd.Series, prefix: str = "ai_") -> pd.DataFrame:
    """
    Parse model JSON outputs into a normalized DataFrame with guaranteed columns:
      <prefix>quality, <prefix>reasoning, <prefix>tags, <prefix>recommendation_email
    """
    parsed = raw_outputs.apply(extract_first_json_object)
    df = pd.json_normalize(parsed).add_prefix(prefix)

    # Ensure columns
    want = [
        f"{prefix}{OutputColumn.QUALITY}",
        f"{prefix}{OutputColumn.REASONING}",
        f"{prefix}{OutputColumn.TAGS}",
        f"{prefix}{OutputColumn.RECOMMENDATION}",
    ]
    for col in want:
        if col not in df.columns:
            df[col] = None

    # Clean types
    qcol = f"{prefix}{OutputColumn.QUALITY}"
    df[qcol] = (
        df[qcol]
        .astype(str)
        .str.strip()
        .str.lower()
        .where(lambda s: s.isin(["good", "bad"]))
    )

    tcol = f"{prefix}{OutputColumn.TAGS}"
    df[tcol] = df[tcol].apply(_coerce_to_tag_list)

    rcol = f"{prefix}{OutputColumn.RECOMMENDATION}"
    df[rcol] = df[rcol].fillna("").astype(str)

    rsn = f"{prefix}{OutputColumn.REASONING}"
    df[rsn] = df[rsn].fillna("").astype(str)

    return df
