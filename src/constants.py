"""Shared constants for column naming and conventions."""

from enum import Enum


class OutputColumn(str, Enum):
    QUALITY = "quality"
    REASONING = "reasoning"
    TAGS = "tags"
    RECOMMENDATION = "recommendation_email"


def make_model_columns(model_name: str, prefix: str = "ai") -> dict[str, str]:
    """Generate consistent column names for a model.

    Args:
        model_name: Short name of the model (e.g., "llama_3b").
        prefix: Prefix for parsed columns (default: "ai").

    Returns:
        Mapping of logical names to concrete column names.
    """
    return {
        "output": f"about_me_processed_{model_name}",
        "quality": f"{prefix}_{model_name}_{OutputColumn.QUALITY}",
        "reasoning": f"{prefix}_{model_name}_{OutputColumn.REASONING}",
        "tags": f"{prefix}_{model_name}_{OutputColumn.TAGS}",
        "recommendation": f"{prefix}_{model_name}_{OutputColumn.RECOMMENDATION}",
    }


