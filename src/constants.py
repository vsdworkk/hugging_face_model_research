"""Shared constants and column builders used across the project."""

from dataclasses import dataclass


@dataclass(frozen=True)
class OutputColumn:
    QUALITY: str = "quality"
    REASONING: str = "reasoning"
    TAGS: str = "tags"
    RECOMMENDATION: str = "recommendation_email"


def make_model_columns(model_name: str, prefix: str = "ai", base_output: str = "about_me_processed") -> dict[str, str]:
    """
    Standardize column names used for a given model.

    Returns keys:
      - output          -> where the raw model string output is stored
      - quality         -> parsed ai_<model>_quality
      - reasoning       -> parsed ai_<model>_reasoning
      - tags            -> parsed ai_<model>_tags
      - recommendation  -> parsed ai_<model>_recommendation_email
    """
    model = model_name.strip()
    return {
        "output":        f"{base_output}_{model}",
        "quality":       f"{prefix}_{model}_{OutputColumn.QUALITY}",
        "reasoning":     f"{prefix}_{model}_{OutputColumn.REASONING}",
        "tags":          f"{prefix}_{model}_{OutputColumn.TAGS}",
        "recommendation":f"{prefix}_{model}_{OutputColumn.RECOMMENDATION}",
    }
