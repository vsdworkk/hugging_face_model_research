"""Evaluation utilities using sklearn."""
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


def label_to_binary(value) -> int:
    """Convert labels to binary (1=bad, 0=good)."""
    if pd.isna(value):
        return -1
    
    value_str = str(value).strip().lower()
    if value_str in ('bad', '1', 'true', 'yes', 'y'):
        return 1
    elif value_str in ('good', '0', 'false', 'no', 'n'):
        return 0
    else:
        return -1


def evaluate_model(df: pd.DataFrame, 
                  pred_col: str, 
                  true_col: str = 'Human_flag',
                  model_name: str = '') -> dict:
    """Evaluate a single model's predictions."""
    # Convert to binary
    y_true = df[true_col].apply(label_to_binary)
    y_pred = df[pred_col].apply(label_to_binary)
    
    # Remove invalid labels
    mask = (y_true != -1) & (y_pred != -1)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        print(f"No valid predictions for {model_name}")
        return {}
    
    # Get metrics
    print(f"\n{'='*50}")
    print(f"Evaluation for: {model_name or pred_col}")
    print(f"{'='*50}")
    
    # Classification report
    report = classification_report(y_true, y_pred, 
                                 labels=[0, 1],
                                 target_names=['good', 'bad'])
    print("\nClassification Report:")
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print("\nConfusion Matrix:")
    print("                Predicted")
    print("                good  bad")
    print(f"Actual good     {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"       bad      {cm[1,0]:4d}  {cm[1,1]:4d}")
    
    # Return dict of metrics
    return classification_report(y_true, y_pred, 
                               labels=[0, 1],
                               target_names=['good', 'bad'],
                               output_dict=True)


def evaluate_all_models(df: pd.DataFrame, 
                       model_names: list,
                       true_col: str = 'Human_flag') -> pd.DataFrame:
    """Evaluate all models and return comparison DataFrame."""
    results = []
    
    for model_name in model_names:
        pred_col = f'{model_name}_quality'
        if pred_col in df.columns:
            metrics = evaluate_model(df, pred_col, true_col, model_name)
            if metrics:
                # Extract key metrics
                results.append({
                    'model': model_name,
                    'accuracy': metrics['accuracy'],
                    'precision_bad': metrics['bad']['precision'],
                    'recall_bad': metrics['bad']['recall'],
                    'f1_bad': metrics['bad']['f1-score'],
                    'support': metrics['bad']['support'] + metrics['good']['support']
                })
    
    comparison_df = pd.DataFrame(results)
    
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    print(comparison_df.to_string(index=False))
    
    return comparison_df