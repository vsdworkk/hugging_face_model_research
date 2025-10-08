"""Evaluation utilities for multi-model comparison."""

import pandas as pd

from src.evaluation.metrics import ModelEvaluator
from src.constants import make_model_columns


def evaluate_multi_model_dataframe(df: pd.DataFrame, config, enabled_models: list):
    """Print side-by-side model comparison metrics."""
    human_col = config.data.human_label_column
    
    if human_col not in df.columns:
        print(f"Human labels column '{human_col}' not found")
        return
    
    print("COMPARISON")
    print("Model      Acc   Prec  Rec   F1")
    print("-" * 30)
    
    for model in enabled_models:
        quality_col = make_model_columns(model.name)["quality"]
        if quality_col in df.columns:
            evaluator = ModelEvaluator(
                ai_quality_col=quality_col,
                human_label_col=human_col,
            )
            
            try:
                _, metrics = evaluator.evaluate_dataframe(df)
                print(f"{model.name[:10]:<10} {metrics.accuracy:.3f} {metrics.precision:.3f} {metrics.recall:.3f} {metrics.f1_score:.3f}")
            except Exception:
                print(f"{model.name[:10]:<10} N/A   N/A   N/A   N/A")