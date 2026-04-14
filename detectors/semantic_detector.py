"""
Method 2: Semantic Similarity Hallucination Detector
=====================================================
Embeds context and answer using a sentence-transformer model, then
computes cosine similarity.  Low similarity → likely hallucination.

Hallucination score = 1 - cosine_similarity(context_emb, answer_emb)

From AWS blog: precision ~90% but recall only ~2% → catches obvious ones.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class SemanticDetectionResult:
    score: float            # 0=factual, 1=hallucinated (1 - cosine_sim)
    cosine_similarity: float
    is_hallucinated: bool


class SemanticSimilarityDetector:
    """
    Cosine-similarity based hallucination detector using
    dense sentence embeddings (sentence-transformers).
    """

    DEFAULT_MODEL    = "sentence-transformers/all-mpnet-base-v2"
    DEFAULT_THRESHOLD = 0.35   # hallucination if score > threshold

    def __init__(
        self,
        embedding_model: str = DEFAULT_MODEL,
        threshold: float = DEFAULT_THRESHOLD,
        device: str = "cuda",
    ):
        self.threshold = threshold
        self.device    = device
        self._embed_model = None
        self._model_name  = embedding_model

    # ── lazy-load embedding model ─────────────────────────────

    def _load(self):
        if self._embed_model is not None:
            return
        from sentence_transformers import SentenceTransformer
        import torch
        dev = self.device if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading embedding model: {self._model_name} on {dev}")
        self._embed_model = SentenceTransformer(self._model_name, device=dev)

    # ── public API ────────────────────────────────────────────

    def detect(self, context: str, answer: str) -> SemanticDetectionResult:
        """Compute semantic similarity score for one (context, answer) pair."""
        self._load()
        if not context or not answer:
            return SemanticDetectionResult(score=0.0, cosine_similarity=1.0, is_hallucinated=False)

        ctx_emb = self._embed_model.encode([context], convert_to_numpy=True, show_progress_bar=False)
        ans_emb = self._embed_model.encode([answer],  convert_to_numpy=True, show_progress_bar=False)

        sim = float(cosine_similarity(ctx_emb, ans_emb)[0][0])
        sim = max(0.0, min(1.0, sim))   # clamp
        score = 1.0 - sim

        return SemanticDetectionResult(
            score=score,
            cosine_similarity=sim,
            is_hallucinated=score >= self.threshold,
        )

    def detect_batch(
        self, contexts: List[str], answers: List[str]
    ) -> List[SemanticDetectionResult]:
        """Batch version — more efficient due to batched encoding."""
        self._load()
        results = []
        ctx_embs = self._embed_model.encode(
            contexts, convert_to_numpy=True, show_progress_bar=False, batch_size=32
        )
        ans_embs = self._embed_model.encode(
            answers,  convert_to_numpy=True, show_progress_bar=False, batch_size=32
        )

        for i, (ce, ae) in enumerate(zip(ctx_embs, ans_embs)):
            if not contexts[i] or not answers[i]:
                results.append(SemanticDetectionResult(0.0, 1.0, False))
                continue
            sim = float(cosine_similarity(ce.reshape(1, -1), ae.reshape(1, -1))[0][0])
            sim = max(0.0, min(1.0, sim))
            score = 1.0 - sim
            results.append(SemanticDetectionResult(
                score=score,
                cosine_similarity=sim,
                is_hallucinated=score >= self.threshold,
            ))
        return results
