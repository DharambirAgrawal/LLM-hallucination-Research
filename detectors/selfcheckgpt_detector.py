"""Thin adapter around the official SelfCheckGPT package.

Upstream source and immutable revision are recorded in
``provenance/sources.yaml``. Generation remains provider-independent: sampled
passages can come from a remote Ollama server or another ``BaseModel`` backend.
The upstream scoring implementation is not copied or reimplemented here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from models.base_model import BaseModel


@dataclass
class SelfCheckGPTResult:
    score: float
    is_hallucinated: bool
    sentence_scores: List[float] = field(default_factory=list)


class SelfCheckGPTDetector:
    """Run the official SelfCheckGPT NLI or BERTScore scorer."""

    def __init__(
        self,
        model: "BaseModel",
        method: str = "nli",
        n_samples: int = 5,
        temperature: float = 1.0,
        threshold: float = 0.5,
        device: str = "cpu",
    ):
        if method not in {"nli", "bertscore", "ngram"}:
            raise ValueError(
                "SelfCheckGPT method must be 'nli', 'bertscore', or 'ngram'"
            )
        self.model = model
        self.method = method
        self.n_samples = n_samples
        self.temperature = temperature
        self.threshold = threshold
        self.device = device
        self._scorer = None

    def _load(self):
        if self._scorer is not None:
            return
        try:
            from selfcheckgpt.modeling_selfcheck import (
                SelfCheckBERTScore,
                SelfCheckNLI,
                SelfCheckNgram,
            )
        except ImportError as exc:
            raise RuntimeError(
                "SelfCheckGPT is not installed. Install the optional upstream "
                "environment described in requirements-upstream.txt."
            ) from exc

        if self.method == "nli":
            self._scorer = SelfCheckNLI(device=self.device)
        elif self.method == "bertscore":
            self._scorer = SelfCheckBERTScore(rescale_with_baseline=True)
        else:
            self._scorer = SelfCheckNgram(n=1)

    @staticmethod
    def _sentences(answer: str) -> List[str]:
        # Sentence segmentation is adapter plumbing, not a replacement for the
        # upstream scorer. Preserve non-empty short answers as one claim.
        parts = re.split(r"(?<=[.!?])\s+", answer.strip())
        return [part.strip() for part in parts if part.strip()]

    def detect(
        self, question: str, context: str, answer: str
    ) -> SelfCheckGPTResult:
        self._load()
        if not answer.strip():
            return SelfCheckGPTResult(score=1.0, is_hallucinated=True)

        prompt = (
            "Answer the question based on the supplied context. Do not add "
            "unsupported information.\n\n"
            f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        )
        if hasattr(self.model, "sample_n"):
            samples = self.model.sample_n(
                prompt, n=self.n_samples, temperature=self.temperature
            )
        else:
            samples = self.model.generate_batch(
                [prompt] * self.n_samples, temperature=self.temperature
            )
        samples = [sample for sample in samples if sample.strip()]
        if not samples:
            raise RuntimeError("Generator returned no SelfCheckGPT samples")

        score_kwargs = {
            "sentences": self._sentences(answer),
            "sampled_passages": samples,
        }
        if self.method == "ngram":
            score_kwargs["passage"] = answer
        sentence_scores = self._scorer.predict(**score_kwargs)
        if self.method == "ngram":
            # The official n-gram variant returns a nested dictionary and its
            # negative-log-probability score is not bounded to [0, 1].
            sentence_scores = sentence_scores["sent_level"]["avg_neg_logprob"]
        values = [float(value) for value in np.asarray(sentence_scores).reshape(-1)]
        score = float(np.mean(values)) if values else 1.0
        if self.method != "ngram":
            score = max(0.0, min(1.0, score))
        return SelfCheckGPTResult(
            score=score,
            is_hallucinated=score >= self.threshold,
            sentence_scores=values,
        )
