"""
Ensemble Detector
=================
Combines all four methods with configurable weights/voting.

Recommended combinations from AWS blog analysis:
  - For HIGH PRECISION: Token + LLM-based
  - For HIGH RECALL:    BERT Stochastic alone
  - BALANCED:          LLM-based + BERT Stochastic (weighted average)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .llm_detector import LLMDetector, LLMDetectionResult
    from .semantic_detector import SemanticSimilarityDetector, SemanticDetectionResult
    from .bert_detector import BERTStochasticDetector, BERTStochasticResult
    from .token_detector import TokenSimilarityDetector, TokenDetectionResult


@dataclass
class EnsembleResult:
    final_score:        float
    is_hallucinated:    bool
    method_scores:      Dict[str, float]  = field(default_factory=dict)
    method_predictions: Dict[str, bool]   = field(default_factory=dict)
    details:            dict              = field(default_factory=dict)


class EnsembleDetector:
    """
    Weighted-average ensemble of all four hallucination detectors.

    Weights (default, can be tuned per dataset):
      token_similarity:    0.10   (high precision, low recall — use as pre-filter)
      semantic_similarity: 0.15   (useful for obvious hallucinations)
      llm_based:           0.40   (best accuracy/cost trade-off)
      bert_stochastic:     0.35   (highest recall)
    """

    DEFAULT_WEIGHTS = {
        "token_similarity":   0.10,
        "semantic_similarity": 0.15,
        "llm_based":           0.40,
        "bert_stochastic":    0.35,
    }

    def __init__(
        self,
        token_detector:    Optional["TokenSimilarityDetector"]    = None,
        semantic_detector: Optional["SemanticSimilarityDetector"] = None,
        llm_detector:      Optional["LLMDetector"]                = None,
        bert_detector:     Optional["BERTStochasticDetector"]     = None,
        weights:           Optional[Dict[str, float]]             = None,
        threshold:         float = 0.5,
    ):
        self.token_detector    = token_detector
        self.semantic_detector = semantic_detector
        self.llm_detector      = llm_detector
        self.bert_detector     = bert_detector
        self.weights           = weights or self.DEFAULT_WEIGHTS
        self.threshold         = threshold

    def detect(
        self,
        question: str,
        context: str,
        answer: str,
    ) -> EnsembleResult:
        scores      = {}
        predictions = {}
        details     = {}

        # ── Token similarity ──────────────────────────────────
        if self.token_detector:
            r = self.token_detector.detect(context, answer)
            scores["token_similarity"]      = r.hallucination_score
            predictions["token_similarity"] = r.is_hallucinated
            details["token_similarity"]     = r.details

        # ── Semantic similarity ───────────────────────────────
        if self.semantic_detector:
            r = self.semantic_detector.detect(context, answer)
            scores["semantic_similarity"]      = r.score
            predictions["semantic_similarity"] = r.is_hallucinated
            details["semantic_similarity"]     = {
                "cosine_similarity": r.cosine_similarity
            }

        # ── LLM-based ─────────────────────────────────────────
        if self.llm_detector:
            r = self.llm_detector.detect(context, answer)
            scores["llm_based"]      = r.score
            predictions["llm_based"] = r.is_hallucinated
            details["llm_based"]     = {"raw_response": r.raw_response}

        # ── BERT Stochastic ───────────────────────────────────
        if self.bert_detector:
            r = self.bert_detector.detect(question, context, answer)
            scores["bert_stochastic"]      = r.score
            predictions["bert_stochastic"] = r.is_hallucinated
            details["bert_stochastic"]     = {
                "mean_bert_f1": r.mean_bert_f1,
                "f1_scores":    r.all_f1_scores,
            }

        # ── Weighted average ──────────────────────────────────
        total_w, weighted_sum = 0.0, 0.0
        for method, score in scores.items():
            w = self.weights.get(method, 0.25)
            weighted_sum += w * score
            total_w      += w

        final_score = weighted_sum / total_w if total_w > 0 else 0.5

        return EnsembleResult(
            final_score=final_score,
            is_hallucinated=final_score >= self.threshold,
            method_scores=scores,
            method_predictions=predictions,
            details=details,
        )
