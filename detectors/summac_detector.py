"""Adapter for the published SummaC factual-consistency detector.

The model implementation is provided by the external ``summac`` package:
https://github.com/tingofurro/summac

This module only adapts its public scoring API to this benchmark's detector
interface. It does not reimplement the SummaC model.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SummaCDetectionResult:
    score: float
    factuality_score: float
    is_hallucinated: bool


class SummaCDetector:
    """Use a published SummaC model to score context entailment."""

    def __init__(
        self,
        threshold: float = 0.5,
        device: str = "cpu",
        model_name: str = "vitc",
    ):
        self.threshold = threshold
        self.device = device
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from summac.model_summac import SummaCConv, SummaCZS
        except ImportError as exc:
            raise RuntimeError(
                "SummaC is not installed. Install it with `pip install summac`."
            ) from exc
        if self.model_name == "zs":
            self._model = SummaCZS(
                granularity="sentence",
                model_name="vitc",
                device=self.device,
            )
        else:
            self._model = SummaCConv(
                models=[self.model_name],
                bins="percentile",
                granularity="sentence",
                nli_labels="e",
                device=self.device,
                start_file="default",
                agg="mean",
            )

    def detect(self, context: str, answer: str) -> SummaCDetectionResult:
        self._load()
        if not context or not answer:
            factuality = 0.0
        else:
            result = self._model.score([context], [answer])
            factuality = float(result["scores"][0])
        factuality = max(0.0, min(1.0, factuality))
        score = 1.0 - factuality
        return SummaCDetectionResult(
            score=score,
            factuality_score=factuality,
            is_hallucinated=score >= self.threshold,
        )
