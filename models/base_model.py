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

    # ── prompt templates ─────────────────────────────────────

    def _rag_prompt(self, question: str, context: str) -> str:
        return (
            "You are a helpful assistant. Answer the question using ONLY the "
            "provided context. Be concise.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
