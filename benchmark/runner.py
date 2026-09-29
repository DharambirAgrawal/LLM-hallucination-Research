"""Run pinned upstream detectors on fixed, labeled responses."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from data.datasets import BenchmarkSample, DatasetLoader
from detectors import (
    AlignScoreDetector,
    MiniCheckDetector,
    SelfCheckGPTDetector,
    SummaCDetector,
)
from models.base_model import BaseModel
from utils import console

# Stop a model/detector after this many failures in a row: the cause is
# almost always environmental (server down, model missing, package broken),
# and continuing would spend hours recording the same error.
MAX_CONSECUTIVE_FAILURES = 10


class BenchmarkRunner:
    """Detector-validation harness; it contains no detection algorithm."""

    def __init__(self, config: dict, generator: Optional[BaseModel] = None):
        self.config = config
        self.generator = generator
        self.output_dir = Path(config.get("benchmark", {}).get("output_dir", "results/current"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.detectors = self._build_detectors(config.get("detectors", {}))

    def _build_detectors(self, detector_config: dict) -> dict:
        detectors = {}

        cfg = detector_config.get("selfcheckgpt", {})
        if cfg.get("enabled", False):
            # A generator is not required here: `validate(generators=...)`
            # supplies one (or more) per call and raises if none is usable.
            detectors["selfcheckgpt"] = SelfCheckGPTDetector(
                model=self.generator,
                method=cfg.get("method", "ngram"),
                n_samples=cfg.get("n_samples", 5),
                temperature=cfg.get("temperature", 1.0),
                threshold=cfg.get("threshold", 0.5),
                device=cfg.get("device", "cpu"),
            )

        cfg = detector_config.get("summac", {})
        if cfg.get("enabled", False):
            detectors["summac"] = SummaCDetector(
                threshold=cfg.get("threshold", 0.5),
                device=cfg.get("device", "cpu"),
                model_name=cfg.get("model_name", "vitc"),
            )

        cfg = detector_config.get("minicheck", {})
        if cfg.get("enabled", False):
            detectors["minicheck"] = MiniCheckDetector(
                model_name=cfg.get("model_name", "flan-t5-large"),
                cache_dir=cfg.get("cache_dir", "external_models/minicheck"),
                threshold=cfg.get("threshold", 0.5),
            )

        cfg = detector_config.get("alignscore", {})
        if cfg.get("enabled", False):
            detectors["alignscore"] = AlignScoreDetector(
                checkpoint_path=cfg["checkpoint_path"],
                threshold=cfg.get("threshold", 0.5),
                device=cfg.get("device", "cpu"),
                model_name=cfg.get("model_name", "roberta-base"),
            )

        if not detectors:
            logger.warning("No detector is enabled in the configuration")
        return detectors

    def validate(
        self,
        datasets: Dict[str, List[BenchmarkSample]],
        generators: Optional[List[BaseModel]] = None,
    ) -> pd.DataFrame:
        """Score fixed factual/hallucinated pairs and preserve every failure.

        Model-independent detectors (SummaC/MiniCheck/AlignScore) score each
        case once. SelfCheckGPT then runs one model at a time over every
        case, so Ollama keeps a single model loaded instead of swapping
        models on every case; a `model` column tags each of its rows.
        """
        cases = []
        for samples in datasets.values():
            cases.extend(DatasetLoader.detection_cases(samples))

        selfcheck = self.detectors.get("selfcheckgpt")
        other_detectors = {
            name: detector for name, detector in self.detectors.items()
            if name != "selfcheckgpt"
        }
        active_generators = (
            generators if generators is not None
            else ([self.generator] if self.generator else [])
        )
        if selfcheck is not None and not any(active_generators):
            raise ValueError("SelfCheckGPT is enabled but no generator model was provided")

        base_rows = [dict(case) for case in cases]
        for name, detector in other_detectors.items():
            self._score_all(
                base_rows, name, label=name,
                score=lambda case, d=detector: d.detect(case["context"], case["answer"]),
            )

        if selfcheck is None:
            rows = base_rows
        else:
            rows = []
            total = len(active_generators)
            for index, generator in enumerate(active_generators, 1):
                model_name = getattr(generator, "name", None)
                model_rows = [{**row, "model": model_name} for row in base_rows]
                self._score_all(
                    model_rows, "selfcheckgpt",
                    label=f"[{index}/{total}] {model_name}",
                    score=lambda case, g=generator: selfcheck.detect(
                        case["question"], case["context"], case["answer"], model=g,
                    ),
                )
                rows.extend(model_rows)

        frame = pd.DataFrame(rows)
        output = self.output_dir / "detector_validation_raw.csv"
        frame.to_csv(output, index=False)
        logger.debug("Saved raw detector validation to {}", output)
        return frame

    @staticmethod
    def _score_all(rows: List[dict], name: str, label: str, score) -> None:
        """Fill `<name>_score` / `<name>_error` on every row, with one
        progress bar and a one-line outcome. Tracebacks go to run.log."""
        started = time.monotonic()
        failures = 0
        consecutive = 0
        first_error = None
        bar = console.progress(rows, desc=f"{label:<26}", total=len(rows))
        for row in bar:
            if consecutive >= MAX_CONSECUTIVE_FAILURES:
                row[f"{name}_score"] = None
                row[f"{name}_error"] = (
                    f"skipped after {MAX_CONSECUTIVE_FAILURES} consecutive failures"
                )
                failures += 1
                continue
            try:
                row[f"{name}_score"] = score(row).score
                row[f"{name}_error"] = None
                consecutive = 0
            except Exception as exc:
                detail = str(exc).strip() or repr(exc)
                row[f"{name}_score"] = None
                row[f"{name}_error"] = f"{type(exc).__name__}: {detail}"
                logger.opt(exception=exc).debug(
                    "{} failed on {}: {}", label, row["case_id"], detail
                )
                failures += 1
                consecutive += 1
                first_error = first_error or row[f"{name}_error"]
                bar.set_postfix_str(f"failed={failures}")
        bar.close()

        elapsed = console.duration(time.monotonic() - started)
        if failures == 0:
            console.line(f"✓ {label.strip()} · {len(rows)} cases in {elapsed}")
        else:
            aborted = consecutive >= MAX_CONSECUTIVE_FAILURES
            logger.warning(
                f"{label.strip()}: {failures}/{len(rows)} cases failed in {elapsed}"
                + (" (stopped early)" if aborted else "")
                + f" · first error: {first_error[:160]}"
            )

    def thresholds(self) -> dict[str, float]:
        """Return configured decision thresholds for enabled detectors."""
        config = self.config.get("detectors", {})
        return {
            f"{name}_score": float(config.get(name, {}).get("threshold", 0.5))
            for name in self.detectors
        }
