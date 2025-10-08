"""Compare multiple models side-by-side on a DataFrame."""
import pandas as pd
from .metrics import ModelEvaluator
from ..constants import make_model_columns


def evaluate_multi_model_dataframe(df: pd.DataFrame, config, enabled_models: list) -> None:
    human_col = config.data.human_label_column
    if human_col not in df.columns:
        print(f"⚠️ Human labels column '{human_col}' not found; skipping evaluation.")
        return

    print("COMPARISON")
    print("Model        Acc    Prec   Rec    F1")
    print("-" * 40)

    for model in enabled_models:
        quality_col = make_model_columns(model.name)["quality"]
        if quality_col in df.columns:
            evaluator = ModelEvaluator(ai_quality_col=quality_col, human_label_col=human_col)
            try:
                _, m = evaluator.evaluate_dataframe(df)
                print(f"{model.name[:12]:<12} {m.accuracy:0.3f}  {m.precision:0.3f}  {m.recall:0.3f}  {m.f1_score:0.3f}")
            except Exception:
                print(f"{model.name[:12]:<12}  N/A    N/A    N/A    N/A")
        else:
            print(f"{model.name[:12]:<12}  N/A    N/A    N/A    N/A")
