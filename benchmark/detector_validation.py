"""Metrics for a detector evaluated on fixed, labeled responses."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class DetectorValidator:
    """Evaluate risk scores where 1 means hallucinated and 0 means factual."""

    @staticmethod
    def evaluate(
        labels: Iterable[int],
        scores: Iterable[float],
        threshold: float,
    ) -> dict:
        labels = list(labels)
        scores = list(scores)
        if not labels or len(labels) != len(scores):
            raise ValueError("labels and scores must be non-empty and equal length")
        predictions = [int(score >= threshold) for score in scores]
        has_both_classes = len(set(labels)) == 2
        return {
            "n_cases": len(labels),
            "threshold": threshold,
            "roc_auc": roc_auc_score(labels, scores) if has_both_classes else np.nan,
            "average_precision": (
                average_precision_score(labels, scores)
                if has_both_classes else np.nan
            ),
            "accuracy": accuracy_score(labels, predictions),
            "precision": precision_score(labels, predictions, zero_division=0),
            "recall": recall_score(labels, predictions, zero_division=0),
            "f1": f1_score(labels, predictions, zero_division=0),
        }

    def evaluate_frame(
        self,
        frame: pd.DataFrame,
        score_columns: Iterable[str],
        thresholds: dict[str, float],
    ) -> pd.DataFrame:
        rows = []
        for column in score_columns:
            usable = frame[["label", column]].dropna()
            row = self.evaluate(
                usable["label"], usable[column], thresholds[column]
            )
            row["detector"] = column.removesuffix("_score")
            rows.append(row)
        return pd.DataFrame(rows)
