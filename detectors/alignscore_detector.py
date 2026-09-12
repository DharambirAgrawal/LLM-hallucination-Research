"""Adapter for the published AlignScore factual-consistency detector.

Upstream implementation: https://github.com/yuh-zha/AlignScore
Paper: https://arxiv.org/abs/2305.07035

AlignScore requires an explicit downloaded checkpoint. The adapter stays
optional so the benchmark can still run when that checkpoint is unavailable.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class AlignScoreDetectionResult:
    score: float
    factuality_score: float
    is_hallucinated: bool


class AlignScoreDetector:
    """Use an upstream AlignScore checkpoint to score evidence entailment."""

    def __init__(
        self,
        checkpoint_path: str,
        threshold: float = 0.5,
        device: str = "cpu",
        model_name: str = "roberta-base",
    ):
        self.checkpoint_path = checkpoint_path
        self.threshold = threshold
        self.device = device
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        if not Path(self.checkpoint_path).exists():
            raise FileNotFoundError(
                f"AlignScore checkpoint not found: {self.checkpoint_path}"
            )
        try:
            from alignscore import AlignScore
        except ImportError as exc:
            raise RuntimeError(
                "AlignScore is not installed. See METHOD_SOURCES.md for the "
                "upstream installation instructions."
            ) from exc
        self._model = AlignScore(
            model=self.model_name,
            batch_size=8,
            device=self.device,
            ckpt_path=self.checkpoint_path,
            evaluation_mode="nli_sp",
        )

    def detect(self, context: str, answer: str) -> AlignScoreDetectionResult:
        self._load()
        if not context or not answer:
            factuality = 0.0
        else:
            factuality = float(
                self._model.score(contexts=[context], claims=[answer])[0]
            )
        factuality = max(0.0, min(1.0, factuality))
        score = 1.0 - factuality
        return AlignScoreDetectionResult(
            score=score,
            factuality_score=factuality,
            is_hallucinated=score >= self.threshold,
        )
