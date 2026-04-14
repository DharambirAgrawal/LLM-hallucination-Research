"""
Method 4: Token Similarity Hallucination Detector
==================================================
Computes token-level overlap between context and answer using:
  - Token intersection (proportion of answer tokens found in context)
  - BLEU score (n-gram precision)
  - ROUGE-L score (longest common subsequence recall)

Hallucination score = 1 - overlap_score

From AWS blog: highest precision (0.96) but very low recall (0.03).
Best for filtering obvious hallucinations cheaply (no LLM calls).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from loguru import logger


@dataclass
class TokenDetectionResult:
    intersection_score: float   # proportion of answer tokens in context
    bleu_score:         float   # BLEU n-gram precision
    rouge_l_score:      float   # ROUGE-L F1
    hallucination_score: float  # average of the three (inverted)
    is_hallucinated:    bool
    details: dict = None


class TokenSimilarityDetector:
    """
    Token-level overlap hallucination detector.
    Zero LLM calls needed — fast and cheap.
    """

    STOPWORDS = {
        "a", "an", "the", "is", "it", "in", "on", "at", "to", "of", "and",
        "or", "but", "for", "with", "this", "that", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "shall", "can",
        "not", "no", "nor", "so", "yet", "both", "either", "neither",
        "i", "you", "he", "she", "we", "they", "it", "its",
    }

    def __init__(
        self,
        bleu_threshold:         float = 0.25,
        intersection_threshold: float = 0.35,
        rouge_threshold:        float = 0.30,
        length_cutoff:          int   = 3,
        remove_stopwords:       bool  = True,
    ):
        self.bleu_threshold         = bleu_threshold
        self.intersection_threshold = intersection_threshold
        self.rouge_threshold        = rouge_threshold
        self.length_cutoff          = length_cutoff
        self.remove_stopwords       = remove_stopwords

    # ── public API ────────────────────────────────────────────

    def detect(self, context: str, answer: str) -> TokenDetectionResult:
        ctx_clean = self._clean(context)
        ans_clean = self._clean(answer)

        ans_tokens = self._tokenize(ans_clean)
        ctx_tokens = self._tokenize(ctx_clean)

        if len(ans_tokens) < self.length_cutoff:
            return TokenDetectionResult(
                intersection_score=1.0, bleu_score=1.0, rouge_l_score=1.0,
                hallucination_score=0.0, is_hallucinated=False,
                details={"reason": "answer too short"},
            )

        # Token intersection
        ctx_set  = set(ctx_tokens)
        ans_set  = set(ans_tokens)
        if ans_set:
            intersection = sum(t in ctx_set for t in ans_set) / len(ans_set)
        else:
            intersection = 0.0

        # BLEU score
        bleu = self._compute_bleu(ans_tokens, ctx_tokens)

        # ROUGE-L
        rouge_l = self._compute_rouge_l(ans_tokens, ctx_tokens)

        # Combined hallucination score (average of inverted overlaps)
        inv_intersection = 1.0 - intersection
        inv_bleu         = 1.0 - bleu
        inv_rouge        = 1.0 - rouge_l
        combined_score   = (inv_intersection + inv_bleu + inv_rouge) / 3.0

        # Majority vote
        votes = [
            intersection < (1 - self.intersection_threshold),
            bleu          < self.bleu_threshold,
            rouge_l       < self.rouge_threshold,
        ]
        is_hallucinated = sum(votes) >= 2

        return TokenDetectionResult(
            intersection_score=intersection,
            bleu_score=bleu,
            rouge_l_score=rouge_l,
            hallucination_score=combined_score,
            is_hallucinated=is_hallucinated,
            details={
                "intersection": intersection,
                "bleu":         bleu,
                "rouge_l":      rouge_l,
                "votes":        votes,
            },
        )

    def detect_batch(
        self, contexts: List[str], answers: List[str]
    ) -> List[TokenDetectionResult]:
        return [self.detect(ctx, ans) for ctx, ans in zip(contexts, answers)]

    # ── internal helpers ─────────────────────────────────────

    def _clean(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _tokenize(self, text: str) -> List[str]:
        tokens = text.split()
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.STOPWORDS]
        return tokens

    def _compute_bleu(
        self, hypothesis: List[str], reference: List[str], max_n: int = 4
    ) -> float:
        """Compute BLEU-4 score (corpus level)."""
        if not hypothesis or not reference:
            return 0.0

        import math

        ref_ngrams   = {}
        hyp_ngrams   = {}
        matches      = 0
        total_hyp    = 0

        scores = []
        for n in range(1, min(max_n, len(hypothesis)) + 1):
            ref_ng  = self._ngram_counts(reference,   n)
            hyp_ng  = self._ngram_counts(hypothesis,  n)

            match = sum(min(c, ref_ng.get(ng, 0)) for ng, c in hyp_ng.items())
            total = sum(hyp_ng.values())
            scores.append(match / total if total > 0 else 0.0)

        if not scores or all(s == 0 for s in scores):
            return 0.0

        # Geometric mean with brevity penalty
        log_avg = sum(math.log(s + 1e-10) for s in scores) / len(scores)
        bp      = min(1.0, len(hypothesis) / len(reference)) if reference else 1.0
        bleu    = bp * math.exp(log_avg)
        return min(max(bleu, 0.0), 1.0)

    @staticmethod
    def _ngram_counts(tokens: List[str], n: int) -> dict:
        counts = {}
        for i in range(len(tokens) - n + 1):
            ng = tuple(tokens[i : i + n])
            counts[ng] = counts.get(ng, 0) + 1
        return counts

    def _compute_rouge_l(
        self, hypothesis: List[str], reference: List[str]
    ) -> float:
        """Compute ROUGE-L F1 (longest common subsequence)."""
        if not hypothesis or not reference:
            return 0.0

        lcs_len = self._lcs_length(hypothesis, reference)
        if lcs_len == 0:
            return 0.0

        precision = lcs_len / len(hypothesis)
        recall    = lcs_len / len(reference)
        if precision + recall == 0:
            return 0.0
        f1 = 2 * precision * recall / (precision + recall)
        return f1

    @staticmethod
    def _lcs_length(a: List[str], b: List[str]) -> int:
        """Dynamic programming LCS length."""
        m, n = len(a), len(b)
        # Space-optimised DP
        prev = [0] * (n + 1)
        for i in range(1, m + 1):
            curr = [0] * (n + 1)
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = max(prev[j], curr[j - 1])
            prev = curr
        return prev[n]
