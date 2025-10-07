"""Evaluation metrics and confusion matrix calculation."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ClassificationMetrics:
    """Container for binary classification metrics."""
    
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    total_samples: int
    
    @property
    def accuracy(self) -> float:
        """Overall accuracy."""
        return self._divide(
            self.true_positives + self.true_negatives,
            self.total_samples
        )
    
    @property
    def precision(self) -> float:
        """Precision for positive class."""
        return self._divide(
            self.true_positives,
            self.true_positives + self.false_positives
        )
    
    @property
    def recall(self) -> float:
        """Recall (sensitivity) for positive class."""
        return self._divide(
            self.true_positives,
            self.true_positives + self.false_negatives
        )
    
    @property
    def f1_score(self) -> float:
        """F1 score (harmonic mean of precision and recall)."""
        p, r = self.precision, self.recall
        return self._divide(2 * p * r, p + r)
    
    @property
    def specificity(self) -> float:
        """Specificity (true negative rate)."""
        return self._divide(
            self.true_negatives,
            self.true_negatives + self.false_positives
        )
    
    @staticmethod
    def _divide(numerator: float, denominator: float) -> float:
        """Safe division returning 0.0 if denominator is zero."""
        return numerator / denominator if denominator > 0 else 0.0
    
    def confusion_matrix_df(self) -> pd.DataFrame:
        """Generate confusion matrix as DataFrame.
        
        Returns:
            DataFrame with true labels as rows, predicted as columns.
        """
        return pd.DataFrame(
            [
                [self.true_negatives, self.false_positives],
                [self.false_negatives, self.true_positives],
            ],
            index=pd.Index(["true_good(0)", "true_bad(1)"], name="Actual"),
            columns=pd.Index(["pred_good(0)", "pred_bad(1)"], name="Predicted"),
        )
    
    def summary_dict(self) -> dict[str, float]:
        """Return metrics as dictionary."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "specificity": self.specificity,
        }


def label_to_binary(value) -> Optional[int]:
    """Convert various label formats to binary (1=bad/positive, 0=good/negative).
    
    Args:
        value: Label value (can be str, int, bool, etc.)
        
    Returns:
        1 for positive class, 0 for negative, None for invalid/missing.
    """
    if pd.isna(value):
        return None
    
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("bad", "1", "true", "yes", "y"):
            return 1
        if normalized in ("good", "0", "false", "no", "n"):
            return 0
    
    if isinstance(value, (int, bool)):
        return 1 if value else 0
    
    return None


def calculate_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> ClassificationMetrics:
    """Calculate classification metrics from true and predicted labels.
    
    Args:
        y_true: Series of ground truth labels.
        y_pred: Series of predicted labels.
        
    Returns:
        ClassificationMetrics object with calculated values.
    """
    # Convert to binary
    y_true_bin = y_true.apply(label_to_binary)
    y_pred_bin = y_pred.apply(label_to_binary)
    
    # Create evaluation DataFrame (only complete pairs)
    eval_df = pd.DataFrame({
        "y_true": y_true_bin,
        "y_pred": y_pred_bin,
    }).dropna()
    
    if len(eval_df) == 0:
        logger.warning("No valid label pairs found for evaluation")
        return ClassificationMetrics(0, 0, 0, 0, 0)
    
    logger.info(f"Evaluating {len(eval_df)} samples with complete labels")
    
    # Calculate confusion matrix components
    tp = int(((eval_df["y_true"] == 1) & (eval_df["y_pred"] == 1)).sum())
    tn = int(((eval_df["y_true"] == 0) & (eval_df["y_pred"] == 0)).sum())
    fp = int(((eval_df["y_true"] == 0) & (eval_df["y_pred"] == 1)).sum())
    fn = int(((eval_df["y_true"] == 1) & (eval_df["y_pred"] == 0)).sum())
    
    return ClassificationMetrics(
        true_positives=tp,
        true_negatives=tn,
        false_positives=fp,
        false_negatives=fn,
        total_samples=len(eval_df),
    )


class ModelEvaluator:
    """Evaluates model predictions against ground truth labels."""
    
    def __init__(
        self,
        ai_quality_col: str = "ai_quality",
        human_label_col: str = "Human_flag",
    ):
        """Initialize evaluator.
        
        Args:
            ai_quality_col: Column name for AI predictions.
            human_label_col: Column name for human ground truth.
        """
        self.ai_quality_col = ai_quality_col
        self.human_label_col = human_label_col
    
    def evaluate_dataframe(
        self,
        df: pd.DataFrame,
        save_path: Optional[Union[str, Path]] = None,
    ) -> tuple[pd.DataFrame, ClassificationMetrics]:
        """Evaluate predictions in a DataFrame.
        
        Args:
            df: DataFrame with AI predictions and human labels.
            save_path: Optional path to save wide evaluation table.
            
        Returns:
            Tuple of (wide_df with binary labels, metrics object).
        """
        logger.info("Starting evaluation")
        
        # Add binary label columns
        wide_df = df.copy()
        wide_df["y_pred"] = df[self.ai_quality_col].apply(label_to_binary)
        wide_df["y_true"] = df[self.human_label_col].apply(label_to_binary)
        
        # Calculate metrics
        metrics = calculate_metrics(
            y_true=wide_df["y_true"],
            y_pred=wide_df["y_pred"],
        )
        
        # Log results
        logger.info(f"Evaluation complete: {metrics.total_samples} samples")
        logger.info(f"Accuracy: {metrics.accuracy:.3f}")
        logger.info(f"Precision: {metrics.precision:.3f}")
        logger.info(f"Recall: {metrics.recall:.3f}")
        logger.info(f"F1 Score: {metrics.f1_score:.3f}")
        
        # Save if requested
        if save_path:
            wide_df.to_csv(save_path, index=False)
            logger.info(f"Saved evaluation table to {save_path}")
        
        return wide_df, metrics