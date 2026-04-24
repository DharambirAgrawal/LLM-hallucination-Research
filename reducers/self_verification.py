"""
Self-Verification Reducer
==========================
A two-step process:
  Step 1: Generate an answer to the question (like baseline)
  Step 2: Ask the model to verify its own answer

If the model catches its own mistake (says "INCORRECT"), we use
the verification to try generating a better answer. Otherwise we
keep the original answer.

The intuition: models sometimes "know" they're wrong when asked
directly. Forcing the model to critically evaluate its answer
can reveal hallucinations that weren't caught the first time.

Trade-off: This method is ~2x slower because it generates twice
per question (answer + verification).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from reducers.base_reducer import BaseReducer

if TYPE_CHECKING:
    from models.base_model import BaseModel
    from data.datasets import BenchmarkSample


class SelfVerificationReducer(BaseReducer):
    """Generates an answer, then asks the model to verify it."""

    def __init__(self, config: dict | None = None):
        super().__init__(name="self_verification", config=config)

    def generate(self, sample: "BenchmarkSample", model: "BaseModel") -> str:
        """
        Two-pass generation:
          1. Generate initial answer
          2. Ask the model to verify its own answer
          3. If the model thinks it's wrong, keep the verification response
             (which often contains a corrected answer); otherwise keep the
             original answer.
        """
        # Step 1: Generate initial answer
        initial_prompt = (
            "Answer the following question concisely and factually.\n\n"
            f"Question: {sample.question}\n\n"
            "Answer:"
        )
        initial_answer = model.generate(initial_prompt)

        # Step 2: Ask model to verify its own answer
        verification_prompt = (
            "You are a fact-checker. Evaluate whether the following answer "
            "is factually correct for the given question. If it is incorrect, "
            "provide the corrected answer. Respond with 'CORRECT' or "
            "'INCORRECT' followed by a brief explanation or correction.\n\n"
            f"Question: {sample.question}\n"
            f"Answer: {initial_answer}\n\n"
            "Verification:"
        )
        verification = model.generate(verification_prompt)

        # If the model thinks its answer is correct, return the original answer.
        # If the model flags it as incorrect, return the verification response
        # (which typically contains the corrected answer).
        verification_upper = verification.upper()
        if "INCORRECT" in verification_upper:
            return verification
        return initial_answer