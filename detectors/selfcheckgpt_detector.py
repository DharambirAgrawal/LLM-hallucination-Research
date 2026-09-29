"""Thin adapter around the official SelfCheckGPT package.

Upstream source and immutable revision are recorded in
``provenance/sources.yaml``. Generation remains provider-independent: sampled
passages can come from a remote Ollama server or another ``BaseModel`` backend.
The upstream scoring implementation is not copied or reimplemented here.

Sampled passages depend only on the generator and the prompt (question +
context), never on the answer being checked. They are therefore drawn once
per (generator, prompt) and reused: the factual and hallucinated answers of
one sample, and the baseline/refined answers of the reduction stage, are all
scored against the same stochastic samples. That is SelfCheckGPT's own
setup (N samples per prompt, any number of responses checked against them),
it makes paired comparisons lower-variance, and it avoids regenerating
identical requests. Every sample set is kept for export to JSONL.
"""
from __future__ import annotations

import contextlib
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

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
        model: Optional["BaseModel"] = None,
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
        self._nlp = None
        self._samples: Dict[Tuple[str, str], List[str]] = {}

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

        with contextlib.redirect_stdout(io.StringIO()):
            if self.method == "nli":
                self._scorer = SelfCheckNLI(device=self.device)
            elif self.method == "bertscore":
                self._scorer = SelfCheckBERTScore(rescale_with_baseline=True)
            else:
                self._scorer = SelfCheckNgram(n=1)

    def _sentences(self, answer: str) -> List[str]:
        """Split the response the way the upstream README does:
        ``[sent.text.strip() for sent in nlp(passage).sents]`` with spaCy
        ``en_core_web_sm``. The n-gram scorer builds its vocabulary with that
        same spaCy segmentation, so a different splitter can produce tokens
        it never counted, which score -log(0) = inf. Falls back to a regex
        only when spaCy or the model is missing (offline tests)."""
        if self._nlp is None:
            try:
                import spacy
                self._nlp = spacy.load("en_core_web_sm")
            except Exception:
                self._nlp = False
        if self._nlp:
            parts = [sent.text.strip() for sent in self._nlp(answer.strip()).sents]
        else:
            parts = re.split(r"(?<=[.!?])\s+", answer.strip())
        return [part.strip() for part in parts if part.strip()]

    def detect(
        self,
        question: str,
        context: str,
        answer: str,
        model: Optional["BaseModel"] = None,
    ) -> SelfCheckGPTResult:
        """Score one fixed answer. `model` overrides the constructor's model
        for this call only, so one detector/scorer instance can be reused
        across several generators instead of reloading a checkpoint per model."""
        self._load()
        if not answer.strip():
            # No neutral score exists: n-gram scores are unbounded, so a
            # placeholder such as 1.0 would read as "very factual". Record
            # the case as a failure instead.
            raise ValueError("empty answer cannot be scored")

        generator = model or self.model
        if generator is None:
            raise ValueError(
                "SelfCheckGPT needs a generator model, either bound at "
                "construction or passed to detect(model=...)"
            )

        prompt = (
            "Answer the question based on the supplied context. Do not add "
            "unsupported information.\n\n"
            f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        )
        samples = self._sample(generator, prompt)

        score_kwargs = {
            "sentences": self._sentences(answer),
            "sampled_passages": samples,
        }
        if self.method == "ngram":
            score_kwargs["passage"] = answer
        # Upstream prints status lines ("SelfCheck-1gram initialized") on
        # every call; keep them out of the terminal.
        with contextlib.redirect_stdout(io.StringIO()):
            sentence_scores = self._scorer.predict(**score_kwargs)
        if self.method == "ngram":
            # The official n-gram variant returns a nested dictionary and its
            # negative-log-probability score is not bounded to [0, 1].
            sentence_scores = sentence_scores["sent_level"]["avg_neg_logprob"]
        values = [float(value) for value in np.asarray(sentence_scores).reshape(-1)]
        if not values or not np.isfinite(values).all():
            raise ValueError(f"upstream scorer returned a non-finite score: {values}")
        score = float(np.mean(values))
        if self.method != "ngram":
            score = max(0.0, min(1.0, score))
        return SelfCheckGPTResult(
            score=score,
            is_hallucinated=score >= self.threshold,
            sentence_scores=values,
        )

    def _sample(self, generator: "BaseModel", prompt: str) -> List[str]:
        key = (getattr(generator, "name", repr(generator)), prompt)
        if key in self._samples:
            return self._samples[key]
        if hasattr(generator, "sample_n"):
            samples = generator.sample_n(
                prompt, n=self.n_samples, temperature=self.temperature
            )
        else:
            samples = generator.generate_batch(
                [prompt] * self.n_samples, temperature=self.temperature
            )
        samples = [sample for sample in samples if sample.strip()]
        if not samples:
            raise RuntimeError("Generator returned no SelfCheckGPT samples")
        self._samples[key] = samples
        return samples

    def export_samples(self, path: Path) -> int:
        """Write every sampled passage set to JSONL (one line per model and
        prompt) so the exact generations behind each score are archived."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for (model_name, prompt), samples in self._samples.items():
                handle.write(json.dumps({
                    "model": model_name,
                    "temperature": self.temperature,
                    "n_samples": len(samples),
                    "prompt": prompt,
                    "samples": samples,
                }, ensure_ascii=False) + "\n")
        return len(self._samples)
