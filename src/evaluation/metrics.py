"""Evaluation metrics and confusion matrix calculation."""
import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ClassificationMetrics:
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    total_samples: int

    @staticmethod
    def _div(n: float, d: float) -> float:
        return n / d if d > 0 else 0.0

    @property
    def accuracy(self) -> float:
        return self._div(self.true_positives + self.true_negatives, self.total_samples)

    @property
    def precision(self) -> float:
        return self._div(self.true_positives, self.true_positives + self.false_positives)

    @property
    def recall(self) -> float:
        return self._div(self.true_positives, self.true_positives + self.false_negatives)

    @property
    def f1_score(self) -> float:
        p, r = self.precision, self.recall
        return self._div(2 * p * r, p + r)

    @property
    def specificity(self) -> float:
        return self._div(self.true_negatives, self.true_negatives + self.false_positives)

    def confusion_matrix_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                [self.true_negatives, self.false_positives],
                [self.false_negatives, self.true_positives],
            ],
            index=pd.Index(["true_good(0)", "true_bad(1)"], name="Actual"),
            columns=pd.Index(["pred_good(0)", "pred_bad(1)"], name="Predicted"),
        )

    def summary_dict(self) -> dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "specificity": self.specificity,
        }


def label_to_binary(value) -> Optional[int]:
    if pd.isna(value):
        return None
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("bad", "1", "true", "yes", "y"):
            return 1
        if s in ("good", "0", "false", "no", "n"):
            return 0
    if isinstance(value, (int, bool)):
        return 1 if value else 0
    return None


def calculate_metrics(y_true: pd.Series, y_pred: pd.Series) -> ClassificationMetrics:
    df = pd.DataFrame({"y_true": y_true.apply(label_to_binary), "y_pred": y_pred.apply(label_to_binary)}).dropna()
    if df.empty:
        logger.warning("No valid label pairs found for evaluation")
        return ClassificationMetrics(0, 0, 0, 0, 0)

    tp = int(((df["y_true"] == 1) & (df["y_pred"] == 1)).sum())
    tn = int(((df["y_true"] == 0) & (df["y_pred"] == 0)).sum())
    fp = int(((df["y_true"] == 0) & (df["y_pred"] == 1)).sum())
    fn = int(((df["y_true"] == 1) & (df["y_pred"] == 0)).sum())

    return ClassificationMetrics(tp, tn, fp, fn, len(df))


class ModelEvaluator:
    def __init__(self, ai_quality_col: str = "ai_quality", human_label_col: str = "Human_flag"):
        self.ai_quality_col = ai_quality_col
        self.human_label_col = human_label_col

    def evaluate_dataframe(self, df: pd.DataFrame) -> tuple[pd.DataFrame, ClassificationMetrics]:
        wide = df.copy()
        wide["y_pred"] = wide[self.ai_quality_col].apply(label_to_binary)
        wide["y_true"] = wide[self.human_label_col].apply(label_to_binary)

        metrics = calculate_metrics(wide["y_true"], wide["y_pred"])
        return wide, metrics
