"""Thin adapter around the official MiniCheck package."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class MiniCheckResult:
    score: float
    is_hallucinated: bool
    sentence_support_scores: List[float] = field(default_factory=list)


class MiniCheckDetector:
    """Score whether each response sentence is supported by its document."""

    def __init__(
        self,
        model_name: str = "flan-t5-large",
        cache_dir: str = "external_models/minicheck",
        threshold: float = 0.5,
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.threshold = threshold
        self._scorer = None

    def _load(self):
        if self._scorer is not None:
            return
        try:
            from minicheck.minicheck import MiniCheck
        except ImportError as exc:
            raise RuntimeError(
                "MiniCheck is not installed. Use requirements-colab.txt or "
                "install its pinned revision from provenance/sources.yaml."
            ) from exc
        self._scorer = MiniCheck(
            model_name=self.model_name,
            cache_dir=self.cache_dir,
        )

    @staticmethod
    def _sentences(answer: str) -> List[str]:
        parts = re.split(r"(?<=[.!?])\s+", answer.strip())
        return [part.strip() for part in parts if part.strip()]

    def detect(self, context: str, answer: str) -> MiniCheckResult:
        self._load()
        sentences = self._sentences(answer)
        if not context.strip() or not sentences:
            return MiniCheckResult(score=1.0, is_hallucinated=True)
        _, raw_prob, _, _ = self._scorer.score(
            docs=[context] * len(sentences),
            claims=sentences,
        )
        support = [float(value) for value in np.asarray(raw_prob).reshape(-1)]
        # Conservative aggregation: the least-supported sentence controls risk.
        factuality = min(support) if support else 0.0
        risk = max(0.0, min(1.0, 1.0 - factuality))
        return MiniCheckResult(
            score=risk,
            is_hallucinated=risk >= self.threshold,
            sentence_support_scores=support,
        )
