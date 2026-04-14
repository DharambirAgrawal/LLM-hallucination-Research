"""
Benchmark Runner
================
Orchestrates:
  1. Load datasets
  2. For each model:
       a. Generate answers for each sample (RAG-style)
       b. Run all 4 detectors on the generated answers
       c. Save intermediate results
  3. Aggregate and return results
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger
from tqdm import tqdm

from data.datasets import BenchmarkSample
from models.base_model import BaseModel
from detectors.llm_detector import LLMDetector
from detectors.semantic_detector import SemanticSimilarityDetector
from detectors.bert_detector import BERTStochasticDetector
from detectors.token_detector import TokenSimilarityDetector
from detectors.ensemble import EnsembleDetector


class BenchmarkRunner:
    """
    Main benchmark orchestrator.

    For each model × dataset combination:
      - Optionally generate answers via the model (or use pre-generated)
      - Run each enabled detection method
      - Record predictions vs ground-truth labels
    """

    def __init__(self, config: dict):
        self.config    = config
        self.output_dir = Path(config["benchmark"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = config["benchmark"].get("seed", 42)

        # Shared detectors (model-agnostic)
        det_cfg = config.get("detectors", {})

        self._semantic_det = None
        self._token_det    = None

        if det_cfg.get("semantic_similarity", {}).get("enabled", True):
            sem_cfg = det_cfg["semantic_similarity"]
            self._semantic_det = SemanticSimilarityDetector(
                embedding_model=sem_cfg.get("embedding_model",
                    "sentence-transformers/all-mpnet-base-v2"),
                threshold=sem_cfg.get("threshold", 0.35),
            )

        if det_cfg.get("token_similarity", {}).get("enabled", True):
            tok_cfg = det_cfg["token_similarity"]
            self._token_det = TokenSimilarityDetector(
                bleu_threshold=tok_cfg.get("bleu_threshold", 0.25),
                intersection_threshold=tok_cfg.get("intersection_threshold", 0.35),
                rouge_threshold=tok_cfg.get("rouge_threshold", 0.30),
            )

    # ── main entry point ─────────────────────────────────────

    def run(
        self,
        models: List[BaseModel],
        datasets: Dict[str, List[BenchmarkSample]],
    ) -> pd.DataFrame:
        """Run the full benchmark. Returns a DataFrame of results."""

        all_rows = []

        for model in models:
            logger.info(f"\n{'='*60}")
            logger.info(f"  Benchmarking model: {model.name}")
            logger.info(f"{'='*60}")

            det_cfg = self.config.get("detectors", {})

            # Build model-specific detectors
            llm_det = None
            if det_cfg.get("llm_based", {}).get("enabled", True):
                llm_det = LLMDetector(
                    judge_model=model,
                    threshold=det_cfg["llm_based"].get("threshold", 0.5),
                )

            bert_det = None
            if det_cfg.get("bert_stochastic", {}).get("enabled", True):
                b_cfg = det_cfg["bert_stochastic"]
                bert_det = BERTStochasticDetector(
                    model=model,
                    n_samples=b_cfg.get("n_samples", 5),
                    temperature=b_cfg.get("temperature", 1.0),
                    threshold=b_cfg.get("threshold", 0.75),
                    use_fast_bert=True,   # use bert-base for speed
                )

            ensemble = EnsembleDetector(
                token_detector=self._token_det,
                semantic_detector=self._semantic_det,
                llm_detector=llm_det,
                bert_detector=bert_det,
            )

            for dataset_name, samples in datasets.items():
                logger.info(f"\n  Dataset: {dataset_name} ({len(samples)} samples)")
                rows = self._run_model_on_dataset(
                    model, samples, dataset_name,
                    llm_det, bert_det, ensemble,
                )
                all_rows.extend(rows)

                # Save intermediate checkpoint
                if self.config["benchmark"].get("save_intermediate", True):
                    self._save_checkpoint(model.name, dataset_name, rows)

            # Free GPU memory before next model
            if hasattr(model, "unload"):
                model.unload()

        df = pd.DataFrame(all_rows)
        out_path = self.output_dir / "all_results.csv"
        df.to_csv(out_path, index=False)
        logger.info(f"\n✓ Results saved to {out_path}")
        return df

    # ── per-model/dataset ────────────────────────────────────

    def _run_model_on_dataset(
        self,
        model:        BaseModel,
        samples:      List[BenchmarkSample],
        dataset_name: str,
        llm_det:      Optional[LLMDetector],
        bert_det:     Optional[BERTStochasticDetector],
        ensemble:     EnsembleDetector,
    ) -> List[dict]:
        rows = []

        for sample in tqdm(samples, desc=f"{model.name}/{dataset_name}", ncols=90):
            row = {
                "model":          model.name,
                "dataset":        dataset_name,
                "sample_id":      sample.sample_id,
                "question":       sample.question,
                "context":        sample.context[:300],   # truncate for storage
                "answer":         sample.answer,
                "gt_hallucinated": sample.is_hallucinated,
            }

            t0 = time.perf_counter()

            # ── Token similarity (no LLM) ──────────────────
            if self._token_det:
                try:
                    r = self._token_det.detect(sample.context, sample.answer)
                    row.update({
                        "token_score":         r.hallucination_score,
                        "token_pred":          r.is_hallucinated,
                        "token_bleu":          r.bleu_score,
                        "token_rouge_l":       r.rouge_l_score,
                        "token_intersection":  r.intersection_score,
                    })
                except Exception as e:
                    logger.debug(f"TokenDet error: {e}")
                    row.update({"token_score": None, "token_pred": None})

            # ── Semantic similarity ────────────────────────
            if self._semantic_det:
                try:
                    r = self._semantic_det.detect(sample.context, sample.answer)
                    row.update({
                        "semantic_score": r.score,
                        "semantic_pred":  r.is_hallucinated,
                        "semantic_cosine": r.cosine_similarity,
                    })
                except Exception as e:
                    logger.debug(f"SemanticDet error: {e}")
                    row.update({"semantic_score": None, "semantic_pred": None})

            # ── LLM-based ──────────────────────────────────
            if llm_det:
                try:
                    r = llm_det.detect(sample.context, sample.answer)
                    row.update({
                        "llm_score": r.score,
                        "llm_pred":  r.is_hallucinated,
                    })
                except Exception as e:
                    logger.debug(f"LLMDet error: {e}")
                    row.update({"llm_score": None, "llm_pred": None})

            # ── BERT Stochastic ────────────────────────────
            if bert_det:
                try:
                    r = bert_det.detect(sample.question, sample.context, sample.answer)
                    row.update({
                        "bert_score":    r.score,
                        "bert_pred":     r.is_hallucinated,
                        "bert_mean_f1":  r.mean_bert_f1,
                    })
                except Exception as e:
                    logger.debug(f"BERTDet error: {e}")
                    row.update({"bert_score": None, "bert_pred": None})

            # ── Ensemble ───────────────────────────────────
            try:
                r = ensemble.detect(sample.question, sample.context, sample.answer)
                row.update({
                    "ensemble_score": r.final_score,
                    "ensemble_pred":  r.is_hallucinated,
                })
            except Exception as e:
                logger.debug(f"Ensemble error: {e}")
                row.update({"ensemble_score": None, "ensemble_pred": None})

            row["latency_s"] = round(time.perf_counter() - t0, 3)
            rows.append(row)

        return rows

    # ── utils ─────────────────────────────────────────────────

    def _save_checkpoint(self, model_name: str, dataset_name: str, rows: List[dict]):
        path = self.output_dir / f"{model_name}_{dataset_name}.json"
        with open(path, "w") as f:
            json.dump(rows, f, indent=2, default=str)
