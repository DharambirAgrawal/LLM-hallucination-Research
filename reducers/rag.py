"""
RAG (Retrieval-Augmented Generation) Reducer
=============================================
Provides a reference passage alongside the question.

The model uses the passage as its primary source to ground
its answer in factual information, reducing the need to rely
on potentially incorrect memorized knowledge.

In a real-world RAG system, the passage would be retrieved
from a vector database. In our benchmark, each HaluEval sample
already provides a relevant `context` (knowledge passage),
so we use that directly.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from reducers.base_reducer import BaseReducer

if TYPE_CHECKING:
    from models.base_model import BaseModel
    from data.datasets import BenchmarkSample


class RAGReducer(BaseReducer):
    """Provides the reference passage as context before asking the question."""

    def __init__(self, config: dict | None = None):
        super().__init__(name="rag", config=config)

    def generate(self, sample: "BenchmarkSample", model: "BaseModel") -> str:
        """Give the model the question PLUS the reference passage."""
        prompt = (
            "You are given a reference passage and a question. "
            "Use the information in the passage as your primary source "
            "to answer the question. Give a concise, factual answer.\n\n"
            f"Reference Passage: {sample.context}\n\n"
            f"Question: {sample.question}\n\n"
            "Answer:"
        )
        return model.generate(prompt)