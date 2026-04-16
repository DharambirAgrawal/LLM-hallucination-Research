"""
Benchmark Runner — Reduction Experiment
========================================
Orchestrates the hallucination-reduction experiment:

For each model × sample:
  1. Generate BASELINE answer (just ask the question, no reducer)
  2. Run all detectors: compare baseline answer vs right_answer
  3. For each REDUCER:
       a. Generate answer using that reducer's strategy
       b. Run all detectors: compare reduced answer vs right_answer
  4. Save one row per (model × sample × reducer) to results table

Output structure (one row per combination):
  model, sample_id, reducer, generated_answer, right_answer,
  token_score, semantic_score, bert_score, llm_score
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
from reducers import (
    BaseReducer,
    RAGReducer,
    ConstrainedDecodingReducer,
    SelfVerificationReducer,
)


class BenchmarkRunner:
    """
    Main benchmark orchestrator for the reduction experiment.

    Generates a baseline answer first, then one answer per reducer,
    scoring each answer against the dataset's right_answer using
    all enabled detectors.
    """

    def __init__(self, config: dict):
        self.config     = config
        self.output_dir = Path(config["benchmark"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed       = config["benchmark"].get("seed", 42)

        # Build model-agnostic detectors (same instance reused across models)
        det_cfg = config.get("detectors", {})
        self._semantic_det = None
        self._token_det    = None

        if det_cfg.get("semantic_similarity", {}).get("enabled", True):
            sem_cfg = det_cfg["semantic_similarity"]
            self._semantic_det = SemanticSimilarityDetector(
                embedding_model=sem_cfg.get("embedding_model",
                    "sentence-transformers/all-MiniLM-L6-v2"),
                threshold=sem_cfg.get("threshold", 0.35),
            )

        if det_cfg.get("token_similarity", {}).get("enabled", True):
            tok_cfg = det_cfg["token_similarity"]
            self._token_det = TokenSimilarityDetector(
                bleu_threshold=tok_cfg.get("bleu_threshold", 0.25),
                intersection_threshold=tok_cfg.get("intersection_threshold", 0.35),
                rouge_threshold=tok_cfg.get("rouge_threshold", 0.30),
            )

        # Build reducers from config
        self._reducers = self._build_reducers(config.get("reducers", {}))
        logger.info(f"Enabled reducers: {[r.name for r in self._reducers]}")

    # ── reducer setup ─────────────────────────────────────────

    def _build_reducers(self, reducer_cfg: dict) -> List[BaseReducer]:
        """Build the list of enabled reducers from config.yaml."""
        reducers: List[BaseReducer] = []

        if reducer_cfg.get("rag", {}).get("enabled", True):
            reducers.append(RAGReducer(config=reducer_cfg.get("rag")))

        if reducer_cfg.get("constrained_decoding", {}).get("enabled", True):
            reducers.append(ConstrainedDecodingReducer(
                config=reducer_cfg.get("constrained_decoding")
            ))

        if reducer_cfg.get("self_verification", {}).get("enabled", True):
            reducers.append(SelfVerificationReducer(
                config=reducer_cfg.get("self_verification")
            ))

        return reducers

    # ── main entry point ─────────────────────────────────────

    def run(
        self,
        models: List[BaseModel],
        datasets: Dict[str, List[BenchmarkSample]],
    ) -> pd.DataFrame:
        """Run the reduction experiment. Returns a DataFrame of results."""
        all_rows: List[dict] = []

        for model in models:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"  Benchmarking model: {model.name}")
            logger.info(f"{'=' * 60}")

            # Build model-specific detectors (judge + BERT use the model)
            llm_det, bert_det = self._build_model_detectors(model)

            for dataset_name, samples in datasets.items():
                logger.info(f"\n  Dataset: {dataset_name} ({len(samples)} samples)")
                rows = self._run_model_on_dataset(
                    model, samples, dataset_name,
                    llm_det, bert_det,
                )
                all_rows.extend(rows)

                if self.config["benchmark"].get("save_intermediate", True):
                    self._save_checkpoint(model.name, dataset_name, rows)

            # Free memory before next model
            if hasattr(model, "unload"):
                model.unload()

        df = pd.DataFrame(all_rows)
        out_path = self.output_dir / "all_results.csv"
        df.to_csv(out_path, index=False)
        logger.info(f"\n✓ Results saved to {out_path}")
        return df

    # ── per-model setup ──────────────────────────────────────

    def _build_model_detectors(
        self, model: BaseModel
    ) -> tuple[Optional[LLMDetector], Optional[BERTStochasticDetector]]:
        """Detectors that depend on a specific model (judge / stochastic sampler)."""
        det_cfg = self.config.get("detectors", {})
        llm_det = None
        bert_det = None

        if det_cfg.get("llm_based", {}).get("enabled", True):
            llm_det = LLMDetector(
                judge_model=model,
                threshold=det_cfg["llm_based"].get("threshold", 0.5),
            )

        if det_cfg.get("bert_stochastic", {}).get("enabled", True):
            b_cfg = det_cfg["bert_stochastic"]
            bert_det = BERTStochasticDetector(
                model=model,
                n_samples=b_cfg.get("n_samples", 5),
                temperature=b_cfg.get("temperature", 1.0),
                threshold=b_cfg.get("threshold", 0.75),
                use_fast_bert=True,
            )

        return llm_det, bert_det

    # ── per-model/dataset loop ───────────────────────────────

    def _run_model_on_dataset(
        self,
        model:        BaseModel,
        samples:      List[BenchmarkSample],
        dataset_name: str,
        llm_det:      Optional[LLMDetector],
        bert_det:     Optional[BERTStochasticDetector],
    ) -> List[dict]:
        """
        For each sample, run baseline + every reducer, score each answer,
        and return one result row per (sample × reducer-or-baseline).
        """
        rows: List[dict] = []

        for sample in tqdm(samples, desc=f"{model.name}/{dataset_name}", ncols=90):
            # === 1. BASELINE: no reducer, just ask the question ===
            baseline_prompt = (
                "Answer the following question concisely and factually.\n\n"
                f"Question: {sample.question}\n\n"
                "Answer:"
            )
            t0 = time.perf_counter()
            baseline_answer = model.generate(baseline_prompt)
            baseline_latency = time.perf_counter() - t0

            baseline_row = self._score_answer(
                model=model,
                sample=sample,
                reducer_name="baseline",
                generated_answer=baseline_answer,
                llm_det=llm_det,
                bert_det=bert_det,
                latency=baseline_latency,
                dataset_name=dataset_name,
            )
            rows.append(baseline_row)

            # === 2. Each REDUCER ===
            for reducer in self._reducers:
                t0 = time.perf_counter()
                try:
                    reduced_answer = reducer.generate(sample, model)
                except Exception as exc:
                    logger.warning(f"Reducer '{reducer.name}' failed: {exc}")
                    reduced_answer = ""
                latency = time.perf_counter() - t0

                reducer_row = self._score_answer(
                    model=model,
                    sample=sample,
                    reducer_name=reducer.name,
                    generated_answer=reduced_answer,
                    llm_det=llm_det,
                    bert_det=bert_det,
                    latency=latency,
                    dataset_name=dataset_name,
                )
                rows.append(reducer_row)

        return rows

    # ── scoring a single answer with all detectors ───────────

    def _score_answer(
        self,
        model:            BaseModel,
        sample:           BenchmarkSample,
        reducer_name:     str,
        generated_answer: str,
        llm_det:          Optional[LLMDetector],
        bert_det:         Optional[BERTStochasticDetector],
        latency:          float,
        dataset_name:     str,
    ) -> dict:
        """
        Run all enabled detectors on the generated answer,
        comparing it against sample.right_answer.
        """
        row = {
            "model":            model.name,
            "dataset":          dataset_name,
            "sample_id":        sample.sample_id,
            "reducer":          reducer_name,
            "question":         sample.question,
            "right_answer":     sample.right_answer,
            "generated_answer": generated_answer,
            "latency_s":        round(latency, 3),
        }

        # --- Token similarity: generated vs right_answer ---
        if self._token_det:
            try:
                r = self._token_det.detect(sample.right_answer, generated_answer)
                row.update({
                    "token_score":        r.hallucination_score,
                    "token_pred":         r.is_hallucinated,
                })
            except Exception as e:
                logger.debug(f"TokenDet error: {e}")
                row.update({"token_score": None, "token_pred": None})

        # --- Semantic similarity: generated vs right_answer ---
        if self._semantic_det:
            try:
                r = self._semantic_det.detect(sample.right_answer, generated_answer)
                row.update({
                    "semantic_score": r.score,
                    "semantic_pred":  r.is_hallucinated,
                })
            except Exception as e:
                logger.debug(f"SemanticDet error: {e}")
                row.update({"semantic_score": None, "semantic_pred": None})

        # --- LLM judge: is generated_answer factually correct vs right_answer? ---
        if llm_det:
            try:
                r = llm_det.detect(sample.right_answer, generated_answer)
                row.update({
                    "llm_score": r.score,
                    "llm_pred":  r.is_hallucinated,
                })
            except Exception as e:
                logger.debug(f"LLMDet error: {e}")
                row.update({"llm_score": None, "llm_pred": None})

        # --- BERT stochastic: consistency-based (uses question + context) ---
        # This detector works differently - it checks consistency across
        # multiple samples from the model, not direct comparison to right_answer.
        if bert_det:
            try:
                r = bert_det.detect(sample.question, sample.context, generated_answer)
                row.update({
                    "bert_score": r.score,
                    "bert_pred":  r.is_hallucinated,
                })
            except Exception as e:
                logger.debug(f"BERTDet error: {e}")
                row.update({"bert_score": None, "bert_pred": None})

        return row

    # ── utils ────────────────────────────────────────────────

    def _save_checkpoint(self, model_name: str, dataset_name: str, rows: List[dict]):
        """Save intermediate results after each (model × dataset)."""
        path = self.output_dir / f"{model_name}_{dataset_name}.json"
        with open(path, "w") as f:
            json.dump(rows, f, indent=2, default=str)