"""Abstract base class for all models."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional


class BaseModel(ABC):
    """Common interface for every model backend."""

    def __init__(self, name: str, config: dict):
        self.name   = name
        self.config = config

    # ── must implement ────────────────────────────────────────

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a single response given a prompt."""

    @abstractmethod
    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses for a list of prompts."""

    # ── helpers ──────────────────────────────────────────────

    def answer_question(self, question: str, context: str) -> str:
        """Convenience: answer a question given context (RAG-style)."""
        prompt = self._rag_prompt(question, context)
        return self.generate(prompt)

    def score_hallucination(self, question: str, context: str, answer: str) -> float:
        """
        Use THIS model as a judge to score hallucination (0 = factual, 1 = hallucinated).
        Returns float in [0, 1].
        """
        prompt = self._judge_prompt(context, answer)
        raw = self.generate(prompt)
        return self._parse_score(raw)

    # ── prompt templates ─────────────────────────────────────

    def _rag_prompt(self, question: str, context: str) -> str:
        return (
            "You are a helpful assistant. Answer the question using ONLY the "
            "provided context. Be concise.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

    def _judge_prompt(self, context: str, statement: str) -> str:
        return (
            "You are an expert assistant helping to check if statements are based on the context.\n"
            "Read the context and the statement, then provide a hallucination score.\n\n"
            "Rules:\n"
            "- Return ONLY a float number between 0 and 1.\n"
            "- 0.0 means the statement is directly supported by the context (factual).\n"
            "- 1.0 means the statement contradicts or is not supported by the context (hallucinated).\n"
            "- Do not include any explanation, just the number.\n\n"
            "<example>\n"
            "Context: AWS is a cloud computing subsidiary of Amazon.\n"
            "Statement: AWS provides cloud computing services.\n"
            "Score: 0.05\n"
            "</example>\n\n"
            "<example>\n"
            "Context: AWS was founded in 2006.\n"
            "Statement: AWS was founded by Bill Gates in 1990.\n"
            "Score: 0.97\n"
            "</example>\n\n"
            f"Context: {context}\n"
            f"Statement: {statement}\n"
            "Score:"
        )

    def _parse_score(self, raw: str) -> float:
        import re
        raw = raw.strip()
        # find first float or int in the response
        match = re.search(r"\d+\.?\d*", raw)
        if match:
            val = float(match.group())
            return min(max(val, 0.0), 1.0)
        return 0.5  # fallback: uncertain

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
