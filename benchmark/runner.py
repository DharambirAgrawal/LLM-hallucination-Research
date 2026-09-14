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
            if self.generator is None:
                raise ValueError("SelfCheckGPT requires a configured generator")
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
        self, datasets: Dict[str, List[BenchmarkSample]]
    ) -> pd.DataFrame:
        """Score fixed factual/hallucinated pairs and preserve every failure."""
        cases = []
        for samples in datasets.values():
            cases.extend(DatasetLoader.detection_cases(samples))

        rows = []
        for case in tqdm(cases, desc="official detector validation", ncols=90):
            row = dict(case)
            for name, detector in self.detectors.items():
                try:
                    if name == "selfcheckgpt":
                        result = detector.detect(
                            case["question"], case["context"], case["answer"]
                        )
                    else:
                        result = detector.detect(case["context"], case["answer"])
                    row[f"{name}_score"] = result.score
                    row[f"{name}_error"] = None
                except Exception as exc:
                    row[f"{name}_score"] = None
                    row[f"{name}_error"] = f"{type(exc).__name__}: {exc}"
                    logger.warning("{} failed on {}: {}", name, case["case_id"], exc)
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
