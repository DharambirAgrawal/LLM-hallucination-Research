"""Run pinned upstream detectors on fixed, labeled responses."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger
from tqdm import tqdm

from data.datasets import BenchmarkSample, DatasetLoader
from detectors import (
    AlignScoreDetector,
    MiniCheckDetector,
    SelfCheckGPTDetector,
    SummaCDetector,
)
from models.base_model import BaseModel


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
        else:
            logger.info("Enabled official detectors: {}", ", ".join(detectors))
        return detectors

    def validate(
        self,
        datasets: Dict[str, List[BenchmarkSample]],
        generators: Optional[List[BaseModel]] = None,
    ) -> pd.DataFrame:
        """Score fixed factual/hallucinated pairs and preserve every failure.

        `generators` scores SelfCheckGPT once per model (a `model` column
        tags each row) instead of only the first configured model.
        Model-independent detectors (SummaC/MiniCheck/AlignScore) are scored
        once per case and merged into every model's row, not recomputed per
        model.
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

        rows = []
        for case in tqdm(cases, desc="official detector validation", ncols=90):
            base_row = dict(case)
            for name, detector in other_detectors.items():
                try:
                    result = detector.detect(case["context"], case["answer"])
                    base_row[f"{name}_score"] = result.score
                    base_row[f"{name}_error"] = None
                except Exception as exc:
                    base_row[f"{name}_score"] = None
                    detail = str(exc).strip() or repr(exc)
                    base_row[f"{name}_error"] = f"{type(exc).__name__}: {detail}"
                    logger.exception(
                        "{} failed on {}: {}", name, case["case_id"], detail
                    )

            if selfcheck is None:
                rows.append(base_row)
                continue

            for generator in active_generators:
                row = dict(base_row)
                row["model"] = getattr(generator, "name", None)
                try:
                    result = selfcheck.detect(
                        case["question"], case["context"], case["answer"],
                        model=generator,
                    )
                    row["selfcheckgpt_score"] = result.score
                    row["selfcheckgpt_error"] = None
                except Exception as exc:
                    row["selfcheckgpt_score"] = None
                    detail = str(exc).strip() or repr(exc)
                    row["selfcheckgpt_error"] = f"{type(exc).__name__}: {detail}"
                    logger.exception(
                        "selfcheckgpt failed on {} ({}): {}",
                        case["case_id"], row["model"], detail,
                    )
                rows.append(row)

        frame = pd.DataFrame(rows)
        output = self.output_dir / "detector_validation_raw.csv"
        frame.to_csv(output, index=False)
        logger.info("Saved raw detector validation to {}", output)
        return frame

    def thresholds(self) -> dict[str, float]:
        """Return configured decision thresholds for enabled detectors."""
        config = self.config.get("detectors", {})
        return {
            f"{name}_score": float(config.get(name, {}).get("threshold", 0.5))
            for name in self.detectors
        }
