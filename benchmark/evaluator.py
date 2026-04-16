"""
Evaluator — Reduction Experiment
=================================
Computes reducer-performance metrics from the results DataFrame.

For each (model × reducer) combination, computes:
  - Mean score per detector (lower = better)
  - Score reduction vs baseline (how much did this reducer help?)
  - Win rate: % of samples where reducer scored lower than baseline
  - Accuracy: % of samples classified as factual (score < threshold)

Since we don't have ground-truth hallucination labels (we removed them
in our workflow), we compare each reducer's scores against the baseline
scores to measure how effective each reduction method is.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from loguru import logger


# Score columns the detectors produce
SCORE_COLS = ["token_score", "semantic_score", "bert_score", "llm_score"]
PRED_COLS  = ["token_pred",  "semantic_pred",  "bert_pred",  "llm_pred"]


class Evaluator:
    """Computes reduction-experiment metrics from a results DataFrame."""

    def evaluate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute metrics for every (model × dataset × reducer) combination.

        Returns a summary DataFrame with one row per combination, containing
        mean scores across all samples for each detector.
        """
        # Only use score columns that actually exist in the results
        score_cols = [c for c in SCORE_COLS if c in df.columns]
        pred_cols  = [c for c in PRED_COLS  if c in df.columns]

        if not score_cols:
            logger.warning("No detector score columns found in results.")
            return pd.DataFrame()

        # Average scores per (model, dataset, reducer)
        summary = (
            df.groupby(["model", "dataset", "reducer"])[score_cols]
              .mean()
              .round(4)
              .reset_index()
        )

        # Also compute "accuracy" per detector per combination:
        # % of samples where the detector labeled the answer as NOT hallucinated
        # (i.e., pred == False means "factual")
        for pred_col in pred_cols:
            score_col = pred_col.replace("_pred", "_score")
            if pred_col not in df.columns:
                continue
            acc = (
                df.groupby(["model", "dataset", "reducer"])[pred_col]
                  .apply(lambda x: (~x.astype(bool)).mean())
                  .round(4)
                  .reset_index(name=score_col.replace("_score", "_accuracy"))
            )
            summary = summary.merge(
                acc, on=["model", "dataset", "reducer"], how="left"
            )

        # Sample count per combination
        n_samples = (
            df.groupby(["model", "dataset", "reducer"])
              .size()
              .reset_index(name="n_samples")
        )
        summary = summary.merge(n_samples, on=["model", "dataset", "reducer"])

        # Mean latency
        if "latency_s" in df.columns:
            latency = (
                df.groupby(["model", "dataset", "reducer"])["latency_s"]
                  .mean()
                  .round(3)
                  .reset_index(name="mean_latency_s")
            )
            summary = summary.merge(
                latency, on=["model", "dataset", "reducer"], how="left"
            )

        return summary

    # ── reducer comparison ────────────────────────────────────

    def reducer_comparison(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute score reduction per reducer vs baseline.

        For each (model × sample), we compute:
            reduction = baseline_score - reducer_score
        A positive reduction means the reducer lowered the hallucination
        score (good). A negative reduction means it got worse.

        Returns a DataFrame with average reductions per (model × reducer × detector).
        """
        score_cols = [c for c in SCORE_COLS if c in df.columns]
        if not score_cols:
            return pd.DataFrame()

        # Separate baseline from reducers
        baseline_df = df[df["reducer"] == "baseline"].copy()
        reducer_df  = df[df["reducer"] != "baseline"].copy()

        if len(baseline_df) == 0 or len(reducer_df) == 0:
            logger.warning("Cannot compute reducer comparison: missing baseline or reducer rows.")
            return pd.DataFrame()

        # Index baseline rows by (model, sample_id) for easy lookup
        baseline_lookup = baseline_df.set_index(["model", "sample_id"])

        # For each reducer row, subtract the corresponding baseline score
        rows = []
        for _, r in reducer_df.iterrows():
            key = (r["model"], r["sample_id"])
            if key not in baseline_lookup.index:
                continue
            b = baseline_lookup.loc[key]
            row = {
                "model":     r["model"],
                "sample_id": r["sample_id"],
                "reducer":   r["reducer"],
            }
            for col in score_cols:
                if col in b and col in r:
                    b_val = b[col] if not isinstance(b[col], pd.Series) else b[col].iloc[0]
                    row[f"{col}_reduction"] = (
                        float(b_val) - float(r[col])
                        if pd.notna(b_val) and pd.notna(r[col])
                        else None
                    )
            rows.append(row)

        reduction_df = pd.DataFrame(rows)

        # Average reductions per (model × reducer)
        reduction_cols = [c for c in reduction_df.columns if c.endswith("_reduction")]
        summary = (
            reduction_df.groupby(["model", "reducer"])[reduction_cols]
                        .mean()
                        .round(4)
                        .reset_index()
        )

        # Win rate: % of samples where reducer beat baseline (reduction > 0)
        for col in reduction_cols:
            wins = (
                reduction_df.groupby(["model", "reducer"])[col]
                            .apply(lambda x: (x > 0).mean())
                            .round(4)
                            .reset_index(name=col.replace("_reduction", "_win_rate"))
            )
            summary = summary.merge(wins, on=["model", "reducer"], how="left")

        return summary

    # ── simple per-reducer table (averaged across models) ────

    def reducer_summary(self, summary_df: pd.DataFrame) -> pd.DataFrame:
        """
        Average scores per reducer across all models and datasets.
        Useful for the final overall comparison table.
        """
        score_cols = [c for c in SCORE_COLS if c in summary_df.columns]
        if not score_cols:
            return pd.DataFrame()

        return (
            summary_df
            .groupby("reducer")[score_cols]
            .mean()
            .round(4)
            .reset_index()
            .sort_values(score_cols[0])  # Sort by first score column (lower = better)
        )