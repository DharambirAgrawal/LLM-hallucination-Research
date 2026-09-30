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
        conv_weights: str = "external_models/summac/summac_conv_vitc_sent_perc_e.bin",
    ):
        self.threshold = threshold
        self.device = device
        self.model_name = model_name
        # Upstream's start_file="default" wgets these weights from the master
        # branch into the working directory. The same file at the pinned
        # commit is downloaded and checksummed by utils/resources.py instead
        # (identical bytes to master at the time of review).
        self.conv_weights = conv_weights
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from summac.model_summac import SummaCConv, SummaCZS
        except ImportError as exc:
            raise RuntimeError(
                "SummaC is not installed. Install the pinned revision with "
                "`pip install -r requirements/summac.txt`."
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
                start_file=self.conv_weights,
                agg="mean",
            )

    def detect(self, context: str, answer: str) -> SummaCDetectionResult:
        self._load()
        if not context.strip() or not answer.strip():
            raise ValueError("empty context or answer cannot be scored")
        raw = float(self._model.score([context], [answer])["scores"][0])
        # SummaC-Conv outputs a probability in [0, 1]; SummaC-ZS outputs
        # entailment − contradiction in [−1, 1], mapped linearly (not clipped,
        # which would collapse every negative score to the same value).
        factuality = (raw + 1.0) / 2.0 if self.model_name == "zs" else raw
        score = 1.0 - factuality
        return SummaCDetectionResult(
            score=score,
            factuality_score=factuality,
            is_hallucinated=score >= self.threshold,
        )
