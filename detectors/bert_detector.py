"""
Method 3: BERT Stochastic Checker
===================================
Generate N stochastic samples from the model, then compute BERTScore
between the original answer and each sample.

Hypothesis:
  - Factual sentences → consistent across multiple generations → HIGH BERTScore
  - Hallucinated sentences → vary across generations → LOW BERTScore

Hallucination score = 1 - mean(BERTScore F1 over N samples)

From AWS blog: highest recall (0.90) — best for catching subtle hallucinations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from models.base_model import BaseModel


@dataclass
class BERTStochasticResult:
    score: float                    # 0=factual, 1=hallucinated
    mean_bert_f1: float             # average BERTScore F1
    all_f1_scores: List[float] = field(default_factory=list)
    is_hallucinated: bool = False


class BERTStochasticDetector:
    """
    BERT stochastic hallucination checker.

    For each input, generate N diverse samples, compute BERTScore
    between original answer and each sample, average the F1 scores.
    Low average F1 → inconsistent → likely hallucinated.
    """

    DEFAULT_THRESHOLD  = 0.75   # low BERT F1 = hallucination
    DEFAULT_N_SAMPLES  = 5
    DEFAULT_BERT_MODEL = "microsoft/deberta-xlarge-mnli"  # best BERTScore model
    FAST_BERT_MODEL    = "bert-base-uncased"

    def __init__(
        self,
        model: "BaseModel",
        n_samples: int = DEFAULT_N_SAMPLES,
        temperature: float = 1.0,
        threshold: float = DEFAULT_THRESHOLD,
        bert_model: str = None,
        use_fast_bert: bool = False,
    ):
        """
        Parameters
        ----------
        model          : The LLM used to generate stochastic samples
        n_samples      : Number of random generations to compare against
        temperature    : Sampling temperature for diversity
        threshold      : BERTScore F1 below this → hallucinated
        bert_model     : BERTScore scoring model (None = auto-select)
        use_fast_bert  : Use bert-base-uncased instead of DeBERTa (faster)
        """
        self.model       = model
        self.n_samples   = n_samples
        self.temperature = temperature
        self.threshold   = threshold

        if bert_model is None:
            self._bert_model = self.FAST_BERT_MODEL if use_fast_bert else self.DEFAULT_BERT_MODEL
        else:
            self._bert_model = bert_model

    # ── public API ────────────────────────────────────────────

    def detect(
        self,
        question: str,
        context: str,
        answer: str,
    ) -> BERTStochasticResult:
        """
        Generate N samples and compare to original answer via BERTScore.
        """
        prompt = self._build_prompt(question, context)

        # Generate N stochastic samples
        try:
            if hasattr(self.model, "sample_n"):
                samples = self.model.sample_n(
                    prompt, n=self.n_samples, temperature=self.temperature
                )
            else:
                samples = self.model.generate_batch(
                    [prompt] * self.n_samples,
                    temperature=self.temperature,
                )
        except Exception as exc:
            logger.warning(f"BERTStochasticDetector: generation failed: {exc}")
            return BERTStochasticResult(score=0.5, mean_bert_f1=0.5, is_hallucinated=False)

        # Filter empty samples
        samples = [s for s in samples if s.strip()]
        if not samples:
            return BERTStochasticResult(score=0.5, mean_bert_f1=0.5)

        # Compute BERTScore between original answer and each sample
        f1_scores = self._compute_bert_f1(answer, samples)

        mean_f1 = float(np.mean(f1_scores))
        # Invert: high F1 = consistent = factual → low hallucination score
        hallucination_score = 1.0 - mean_f1

        return BERTStochasticResult(
            score=hallucination_score,
            mean_bert_f1=mean_f1,
            all_f1_scores=f1_scores,
            is_hallucinated=mean_f1 < self.threshold,
        )

    def detect_batch(
        self,
        questions: List[str],
        contexts: List[str],
        answers: List[str],
    ) -> List[BERTStochasticResult]:
        results = []
        for q, ctx, ans in zip(questions, contexts, answers):
            try:
                results.append(self.detect(q, ctx, ans))
            except Exception as exc:
                logger.warning(f"BERTStochasticDetector batch item error: {exc}")
                results.append(BERTStochasticResult(score=0.5, mean_bert_f1=0.5))
        return results

    # ── internal ─────────────────────────────────────────────

    def _compute_bert_f1(self, reference: str, candidates: List[str]) -> List[float]:
        """Compute BERTScore F1 between reference and each candidate."""
        try:
            from bert_score import score as bert_score_fn
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            refs = [reference] * len(candidates)

            P, R, F1 = bert_score_fn(
                cands=candidates,
                refs=refs,
                model_type=self._bert_model,
                lang="en",
                device=device,
                verbose=False,
                rescale_with_baseline=True,
            )
            return F1.tolist()

        except Exception as exc:
            logger.warning(f"BERTScore computation failed, falling back to cosine: {exc}")
            return self._fallback_cosine_f1(reference, candidates)

    def _fallback_cosine_f1(self, reference: str, candidates: List[str]) -> List[float]:
        """Fallback: simple token-overlap F1 if BERTScore unavailable."""
        ref_tokens = set(reference.lower().split())
        scores = []
        for cand in candidates:
            cand_tokens = set(cand.lower().split())
            if not ref_tokens or not cand_tokens:
                scores.append(0.5)
                continue
            intersection = ref_tokens & cand_tokens
            p = len(intersection) / len(cand_tokens) if cand_tokens else 0
            r = len(intersection) / len(ref_tokens)  if ref_tokens  else 0
            f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0
            scores.append(f1)
        return scores

    def _build_prompt(self, question: str, context: str) -> str:
        return (
            "Answer the following question based on the context provided. "
            "Be concise and accurate.\n\n"
            f"Context: {context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
