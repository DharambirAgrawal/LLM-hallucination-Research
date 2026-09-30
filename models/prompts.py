"""The prompts every stage shares, in one place.

SelfCheckGPT compares an answer with other answers sampled from the SAME
prompt (Manakul et al., 2023). The stochastic samples, the baseline answer of
the reduction stage, and every reduction method's first draft therefore all
use `grounded_prompt`. Wording is local benchmark policy, not taken from any
dataset or paper.
"""
from __future__ import annotations


def grounded_prompt(question: str, context: str) -> str:
    """Answer from the supplied context (the retrieval-augmented setting)."""
    return (
        "Answer the question using ONLY the supplied context. Be concise and "
        "do not add unsupported information.\n\n"
        f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    )


def closed_book_prompt(question: str) -> str:
    """The same request without any context (the no-retrieval condition)."""
    return (
        "Answer the question. Be concise and do not add unsupported "
        f"information.\n\nQuestion: {question}\n\nAnswer:"
    )
