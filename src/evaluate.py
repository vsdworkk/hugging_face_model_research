"""Model evaluation utilities for profile quality predictions."""
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime


def label_to_binary(value) -> int:
    """
    Convert various label formats to binary representation.
    
    Binary mapping:
    - 1: bad quality profile
    - 0: good quality profile
    - -1: invalid/missing value
    
    Args:
        value: Label value in various formats
        
    Returns:
        Binary representation (0, 1, or -1)
    """
    if pd.isna(value):
        return -1
    
    value_str = str(value).strip().lower()
    
    # Bad quality indicators
    if value_str == 'bad':
        return 1
    # Good quality indicators
    elif value_str == 'good':
        return 0
    # Invalid value
    else:
        return -1


def prepare_labels(
    y_true: pd.Series, 
    y_pred: pd.Series
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert and filter labels for evaluation.
    
    Args:
        y_true: True labels series
        y_pred: Predicted labels series
        
    Returns:
        Tuple of (filtered_true_labels, filtered_pred_labels) as numpy arrays
    """
    # Convert to binary
    y_true_binary = y_true.apply(label_to_binary)
    y_pred_binary = y_pred.apply(label_to_binary)
    
    # Filter out invalid values
    mask = (y_true_binary != -1) & (y_pred_binary != -1)
    
    return y_true_binary[mask].values, y_pred_binary[mask].values


def calculate_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray
) -> Dict[str, any]:
    """
    Calculate classification metrics.
    
    Args:
        y_true: True labels (binary)
        y_pred: Predicted labels (binary)
        
    Returns:
        Dictionary containing metrics and confusion matrix
    """
    # Get classification report
    report = classification_report(
        y_true, 
        y_pred, 
        target_names=['good', 'bad'], 
        output_dict=True,
        zero_division=0
    )
    
    # Get confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    return {
        'accuracy': report['accuracy'],
        'good': report['good'],
        'bad': report['bad'],
        'confusion_matrix': cm,
        'support': len(y_true)
    }


def evaluate_all_models(
    df: pd.DataFrame, 
    model_names: Union[str, List[str]],
    true_col: str = 'Human_flag'
) -> pd.DataFrame:
    """
    Evaluate one or more models and return comparison DataFrame.
    
    Args:
        df: DataFrame with predictions
        model_names: Single model name or list of model names to evaluate
        true_col: Column name for true labels
        
    Returns:
        DataFrame comparing all models' performance (single row if one model)
    """
    # Ensure model_names is a list
    if isinstance(model_names, str):
        model_names = [model_names]
    
    results = []
    
    for model_name in model_names:
        pred_col = f'{model_name}_quality'
        
        if pred_col not in df.columns:
            print(f"Warning: Column {pred_col} not found, skipping {model_name}")
            continue
        
        # Prepare labels
        y_true, y_pred = prepare_labels(df[true_col], df[pred_col])
        
        if len(y_true) == 0:
            print(f"No valid predictions found for {model_name}")
            continue
        
        # Calculate metrics
        metrics = calculate_metrics(y_true, y_pred)
        
        # Collect results
        results.append({
            'model': model_name,
            'accuracy': metrics['accuracy'],
            'precision_bad': metrics['bad']['precision'],
            'recall_bad': metrics['bad']['recall'],
            'f1_bad': metrics['bad']['f1-score'],
            'support': metrics['support']
        })
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame(results)
    
    # Sort by F1 score for bad profiles (descending) if multiple models
    if len(comparison_df) > 1:
        comparison_df = comparison_df.sort_values('f1_bad', ascending=False)
    
    # Print comparison only if multiple models
    if len(comparison_df) > 1:
        print("\n" + "="*70)
        print("MODEL COMPARISON")
        print("="*70)
        print(comparison_df.to_string(index=False))
    
    return comparison_df