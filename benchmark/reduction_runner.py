"""Stage B: paired baseline vs. reduced-answer comparison.

For each sample, a generator answers once (baseline), the configured
reduction method revises that answer, and the SAME already-configured
detector scores both the baseline and the revised answer. This is an
engineering harness for the Stage B protocol in docs/REPRODUCIBILITY.md, not
a substitute for its full paired statistical analysis (bootstrap confidence
intervals, blinded human review, latency/cost accounting across a held-out
split). Treat its output as a smoke comparison, not a reportable result.

Only SelfCheckGPT is supported as the scoring detector today: it is the only
detector this harness has verified end-to-end, and it is also the only one
that needs a generator model, which the reduction loop already requires.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
from loguru import logger
from tqdm import tqdm

from data.datasets import BenchmarkSample
from detectors.selfcheckgpt_detector import SelfCheckGPTDetector
from models.base_model import BaseModel

REDUCERS = {}


def _load_reducers() -> dict:
    """Registry keyed by each reducer's own `METHOD_ID`, not a bare paper name.

    Requiring the config to spell "self_refine_adapted" (not "self_refine") is
    deliberate: it stops a config or results table from ever citing this as
    plain, unqualified "Self-Refine".
    """
    if not REDUCERS:
        from reducers.self_refine import SelfRefineReducer
        REDUCERS[SelfRefineReducer.METHOD_ID] = SelfRefineReducer
    return REDUCERS


class ReductionRunner:
    """Runs one reduction method for each generator and scores before/after."""

    def __init__(self, config: dict, detector: SelfCheckGPTDetector):
        if not isinstance(detector, SelfCheckGPTDetector):
            raise ValueError(
                "Reduction stage currently supports only the 'selfcheckgpt' "
                "detector, since it is the only detector verified end-to-end "
                "and the only one that already needs a generator model."
            )
        cfg = config.get("reduction", {})
        reducers = _load_reducers()
        method = cfg.get("method", "self_refine_adapted")
        if method not in reducers:
            raise ValueError(f"Unsupported reduction method '{method}'")
        self.reducer_cls = reducers[method]
        self.max_iterations = cfg.get("max_iterations", 3)
        self.detector = detector
        self.output_dir = Path(config.get("benchmark", {}).get("output_dir", "results/current"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        datasets: Dict[str, List[BenchmarkSample]],
        generators: List[BaseModel],
    ) -> pd.DataFrame:
        samples = [sample for group in datasets.values() for sample in group]
        rows = []

        for model in generators:
            reducer = self.reducer_cls(model=model, max_iterations=self.max_iterations)
            for sample in tqdm(samples, desc=f"reduction[{model.name}]", ncols=90):
                row = {
                    "sample_id": sample.sample_id,
                    "dataset": sample.dataset,
                    "model": model.name,
                    # Stamped from the reducer class itself (not hand-typed
                    # here) so this label can't drift from reducers/self_refine.py.
                    "method": reducer.METHOD_ID,
                    "reproduction_status": reducer.REPRODUCTION_STATUS,
                }
                start = time.monotonic()
                try:
                    result = reducer.reduce(sample.question, sample.context)
                    baseline = self.detector.detect(
                        sample.question, sample.context, result.initial_answer, model=model
                    )
                    refined = self.detector.detect(
                        sample.question, sample.context, result.final_answer, model=model
                    )
                    row.update({
                        "baseline_answer": result.initial_answer,
                        "baseline_score": baseline.score,
                        "refined_answer": result.final_answer,
                        "refined_score": refined.score,
                        "score_delta": refined.score - baseline.score,
                        "iterations": result.iterations,
                        "stopped_reason": result.stopped_reason,
                        "latency_seconds": time.monotonic() - start,
                        "error": None,
                    })
                except Exception as exc:
                    detail = str(exc).strip() or repr(exc)
                    row["error"] = f"{type(exc).__name__}: {detail}"
                    logger.exception(
                        "Reduction failed for {} ({}): {}", sample.sample_id, model.name, detail
                    )
                rows.append(row)

        frame = pd.DataFrame(rows)
        output = self.output_dir / "reduction_comparison.csv"
        frame.to_csv(output, index=False)
        logger.info("Saved reduction comparison to {}", output)
        return frame
