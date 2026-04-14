"""
Evaluator
=========
Computes per-method, per-model, per-dataset metrics:
  - Accuracy, Precision, Recall, F1
  - AUC-ROC
  - Confusion matrix stats
  - Latency statistics
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger


# Detection method columns present in the results DataFrame
METHODS = {
    "token_similarity":   ("token_score",    "token_pred"),
    "semantic_similarity": ("semantic_score", "semantic_pred"),
    "llm_based":          ("llm_score",      "llm_pred"),
    "bert_stochastic":    ("bert_score",     "bert_pred"),
    "ensemble":           ("ensemble_score", "ensemble_pred"),
}


class Evaluator:
    """Computes hallucination detection metrics from a results DataFrame."""

    def evaluate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute metrics for every (model, dataset, method) combination.
        Returns a summary DataFrame.
        """
        rows = []

        for model_name, model_df in df.groupby("model"):
            for dataset_name, ds_df in model_df.groupby("dataset"):
                y_true = ds_df["gt_hallucinated"].astype(bool).values

                for method, (score_col, pred_col) in METHODS.items():
                    if score_col not in ds_df.columns:
                        continue
                    valid = ds_df[[score_col, pred_col]].dropna()
                    if len(valid) == 0:
                        continue

                    idx     = valid.index
                    y_score = valid[score_col].values.astype(float)
                    y_pred  = valid[pred_col].astype(bool).values
                    y_gt    = y_true[ds_df.index.get_indexer(idx)]

                    metrics = self._compute_metrics(y_gt, y_pred, y_score)
                    metrics.update({
                        "model":   model_name,
                        "dataset": dataset_name,
                        "method":  method,
                        "n_samples": len(valid),
                    })

                    # Latency (only meaningful for LLM-based / BERT / Ensemble)
                    if "latency_s" in ds_df.columns:
                        metrics["mean_latency_s"] = round(
                            float(ds_df["latency_s"].mean()), 3
                        )

                    rows.append(metrics)

        summary = pd.DataFrame(rows)
        # Reorder columns
        front = ["model", "dataset", "method", "n_samples",
                 "accuracy", "precision", "recall", "f1", "auc_roc"]
        rest  = [c for c in summary.columns if c not in front]
        summary = summary[[c for c in front if c in summary.columns] + rest]
        return summary

    # ── per-model aggregate ───────────────────────────────────

    def model_summary(self, summary_df: pd.DataFrame) -> pd.DataFrame:
        """Average metrics across datasets for each (model, method)."""
        numeric = ["accuracy", "precision", "recall", "f1", "auc_roc"]
        cols    = [c for c in numeric if c in summary_df.columns]
        return (
            summary_df
            .groupby(["model", "method"])[cols]
            .mean()
            .round(4)
            .reset_index()
            .sort_values(["method", "f1"], ascending=[True, False])
        )

    # ── metrics ───────────────────────────────────────────────

    @staticmethod
    def _compute_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_score: np.ndarray,
    ) -> dict:
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score, confusion_matrix,
        )

        # Guard: need both classes for AUC
        acc  = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec  = float(recall_score(y_true, y_pred, zero_division=0))
        f1   = float(f1_score(y_true, y_pred, zero_division=0))

        try:
            auc = float(roc_auc_score(y_true, y_score))
        except ValueError:
            auc = float("nan")

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[False, True]).ravel()

        return {
            "accuracy":  round(acc,  4),
            "precision": round(prec, 4),
            "recall":    round(rec,  4),
            "f1":        round(f1,   4),
            "auc_roc":   round(auc,  4),
            "tp": int(tp), "fp": int(fp),
            "tn": int(tn), "fn": int(fn),
        }

    # ── method comparison table (mirrors the AWS blog table) ─

    @staticmethod
    def method_comparison_table(summary_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build a concise table that mirrors Table 1 in the AWS blog post:
        | Method | Accuracy | Precision | Recall | F1 | AUC |
        Averaged across all models and datasets.
        """
        numeric = ["accuracy", "precision", "recall", "f1", "auc_roc"]
        cols    = [c for c in numeric if c in summary_df.columns]
        return (
            summary_df
            .groupby("method")[cols]
            .mean()
            .round(4)
            .reset_index()
            .sort_values("accuracy", ascending=False)
        )
