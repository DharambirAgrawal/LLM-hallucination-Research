"""Local baseline inspired by the Self-Refine workflow.

The baseline uses the high-level Self-Refine workflow:
1. generate an initial answer;
2. ask the same model for actionable feedback;
3. revise the answer using that feedback;
4. optionally repeat for a configured number of iterations.

Paper: https://arxiv.org/abs/2303.17651
Reference repository: https://github.com/madaan/self-refine

The upstream repository provides task-specific prompts and experiments rather
than a generic factual-QA reducer. These prompts were written locally, so this
condition must not be reported as an upstream reproduction.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from reducers.base_reducer import BaseReducer

if TYPE_CHECKING:
    from data.datasets import BenchmarkSample
    from models.base_model import BaseModel


class SelfRefineReducer(BaseReducer):
    """Run a local generate-feedback-revise baseline."""

    def __init__(self, config: dict | None = None):
        super().__init__(name="self_refine_inspired", config=config)
        cfg = config or {}
        self.iterations = max(1, int(cfg.get("iterations", 1)))

    def generate(self, sample: "BenchmarkSample", model: "BaseModel") -> str:
        answer = model.generate(
            self._initial_prompt(sample),
            temperature=self.config.get("temperature", 0.7),
        )

        for _ in range(self.iterations):
            feedback = model.generate(
                self._feedback_prompt(sample, answer),
                temperature=self.config.get("feedback_temperature", 0.0),
            )
            answer = model.generate(
                self._revision_prompt(sample, answer, feedback),
                temperature=self.config.get("revision_temperature", 0.2),
            )

        return answer

    @staticmethod
    def _initial_prompt(sample: "BenchmarkSample") -> str:
        return (
            "Answer the question concisely and factually. Do not invent facts.\n\n"
            f"Question: {sample.question}\n\nAnswer:"
        )

    @staticmethod
    def _feedback_prompt(sample: "BenchmarkSample", answer: str) -> str:
        return (
            "Provide actionable feedback on the answer below. Check factuality, "
            "unsupported claims, and whether it answers the question. Identify "
            "specific corrections. Do not rewrite the complete answer.\n\n"
            f"Question: {sample.question}\n"
            f"Answer: {answer}\n\nFeedback:"
        )

    @staticmethod
    def _revision_prompt(
        sample: "BenchmarkSample", answer: str, feedback: str
    ) -> str:
        return (
            "Rewrite the answer using the feedback. Return only the final concise "
            "answer. Keep claims supported by the question and remove unsupported "
            "details.\n\n"
            f"Question: {sample.question}\n"
            f"Previous answer: {answer}\n"
            f"Feedback: {feedback}\n\nFinal answer:"
        )
