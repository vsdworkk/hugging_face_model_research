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
    
    label_map = {'bad': 1, 'good': 0}
    return label_map.get(str(value).strip().lower(), -1)


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


def calculate_binary_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, Union[float, int]]:
    """
    Calculates detailed binary classification metrics from a confusion matrix.

    Args:
        y_true: True labels (binary).
        y_pred: Predicted labels (binary).

    Returns:
        Dictionary containing Precision, Recall, F1-Score, Support,
        TP, FP, FN, and Accuracy.
    """
    # Ensure labels are [0, 1] to handle cases with no true positives or negatives
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    support = tp + fn

    return {
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Support": int(support),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "Accuracy": accuracy,
    }


def evaluate_all_models(
    df: pd.DataFrame,
    model_names: Union[str, List[str]],
    true_col: str = "Human_flag",
    tags: Optional[List[str]] = None,
) -> None:
    """
    Generates and prints a detailed performance report for one or more models,
    including overall and per-tag metrics.

    Args:
        df: DataFrame with predictions and true labels.
        model_names: Single model name or list of model names to evaluate.
        true_col: Column name for overall true quality labels.
        tags: A list of tag names to evaluate. If None, defaults to a standard list.
    """
    if isinstance(model_names, str):
        model_names = [model_names]

    if tags is None:
        tags = [
            "personal_information",
            "sensitive_information",
            "inappropriate_information",
            "poor_grammar",
        ]

    for model_name in model_names:
        print("\n" + "=" * 80)
        print(f"PER-TAG PERFORMANCE METRICS FOR MODEL: {model_name}")
        print("=" * 80)

        pred_quality_col = f"{model_name}_quality"
        pred_tags_col = f"{model_name}_tags"
        if pred_quality_col not in df.columns or pred_tags_col not in df.columns:
            print(f"Warning: Prediction columns for {model_name} not found. Skipping.")
            continue

        results = []

        # 1. Per-tag metrics calculation
        for tag in tags:
            if tag not in df.columns:
                print(f"Warning: Ground truth column '{tag}' not found. Skipping.")
                continue

            y_true = df[tag].fillna(0).astype(int)
            y_pred = df[pred_tags_col].apply(
                lambda t: 1 if isinstance(t, list) and tag in t else 0
            )

            metrics = calculate_binary_classification_metrics(y_true, y_pred)
            metrics["Tag"] = tag
            results.append(metrics)

        # 2. Overall metrics calculation
        y_true_overall, y_pred_overall = prepare_labels(
            df[true_col], df[pred_quality_col]
        )

        if len(y_true_overall) > 0:
            overall_metrics = calculate_binary_classification_metrics(
                y_true_overall, y_pred_overall
            )
            overall_metrics["Tag"] = "overall"
            # Support for overall is the total number of evaluated samples
            overall_metrics["Support"] = len(y_true_overall)
            results.append(overall_metrics)

        # 3. Create, format, and print the report DataFrame
        if not results:
            print("No results to display.")
            continue
            
        report_df = pd.DataFrame(results)
        
        cols = [
            "Tag",
            "Precision",
            "Recall",
            "F1-Score",
            "Support",
            "TP",
            "FP",
            "FN",
            "Accuracy",
        ]
        # Ensure all columns exist before trying to order them
        report_df = report_df.reindex(columns=cols)

        # Format float columns to 2 decimal places for cleaner output
        for col in ["Precision", "Recall", "F1-Score", "Accuracy"]:
            report_df[col] = report_df[col].map("{:.2f}".format)

        print(report_df.to_string(index=False))
        print("─" * 80)