"""Generic-QA adaptation of Self-Refine's feedback/refine control loop.

This is a LOCAL INSPIRED BASELINE, not an upstream reproduction — report it
as such, per docs/REPRODUCIBILITY.md ("Stage C": upstream reproductions,
local inspired baselines, provider services, and the proposed method must be
reported as separate conditions).

Why this can't be an "official" integration: the upstream repository ships
task-specific prompts and harnesses (GSM8K math, code optimization, dialogue
response generation, acronym generation, ...); it has no grounded-QA /
hallucination-reduction task. Calling this code "Self-Refine" without
qualification would misrepresent what was actually run. What is reused from
the paper is only its control loop — generate an answer, ask the same model
for feedback, revise, repeat until the model reports no remaining issues or
a fixed iteration budget is spent. The prompts here are local and written for
the grounded-QA setting used by this harness; they are not copied from the
upstream repository.

Paper:       Madaan et al., 2023 - https://arxiv.org/abs/2303.17651
Reference:   https://github.com/madaan/self-refine (task-specific prompts only;
             not vendored here)
Provenance:  see `self_refine` in provenance/sources.yaml.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from models.base_model import BaseModel


@dataclass
class SelfRefineResult:
    initial_answer: str
    final_answer: str
    iterations: int
    feedback_history: List[str] = field(default_factory=list)
    stopped_reason: str = "max_iterations"


class SelfRefineReducer:
    """Local, generic-QA port of the Self-Refine generate/feedback/refine loop."""

    # Carried into every output row (see benchmark/reduction_runner.py) so the
    # "not upstream" fact survives in the raw CSV even if nobody reads the
    # surrounding docs. Do not soften this string or drop it from outputs.
    METHOD_ID = "self_refine_adapted"
    REPRODUCTION_STATUS = "local_inspired_baseline_NOT_an_upstream_reproduction"
    SOURCE_PAPER = "https://arxiv.org/abs/2303.17651"

    STOP_TOKEN = "NO_ISSUES"

    def __init__(
        self,
        model: "BaseModel",
        max_iterations: int = 3,
        temperature: float = 0.7,
    ):
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        self.model = model
        self.max_iterations = max_iterations
        self.temperature = temperature

    # ── prompts (local; not copied from the upstream repository) ──────────

    def _initial_prompt(self, question: str, context: str) -> str:
        return (
            "Answer the question using ONLY the supplied context. Be concise "
            "and do not add unsupported information.\n\n"
            f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        )

    def _feedback_prompt(self, question: str, context: str, answer: str) -> str:
        return (
            "Review the candidate answer for claims the supplied context does "
            "not support. List each unsupported claim briefly. If every claim "
            f"in the answer is supported by the context, reply with exactly "
            f"'{self.STOP_TOKEN}' and nothing else.\n\n"
            f"Context: {context}\n\nQuestion: {question}\n\n"
            f"Candidate answer: {answer}\n\nFeedback:"
        )

    def _refine_prompt(
        self, question: str, context: str, answer: str, feedback: str
    ) -> str:
        return (
            "Rewrite the candidate answer to fix every issue raised in the "
            "feedback, using ONLY the supplied context. Be concise.\n\n"
            f"Context: {context}\n\nQuestion: {question}\n\n"
            f"Candidate answer: {answer}\n\nFeedback: {feedback}\n\n"
            "Revised answer:"
        )

    # ── loop ────────────────────────────────────────────────────────────

    def reduce(
        self,
        question: str,
        context: str,
        initial_answer: Optional[str] = None,
    ) -> SelfRefineResult:
        """Run generate -> feedback -> refine until the model reports no
        remaining issues or `max_iterations` refine steps have run."""
        answer = initial_answer or self.model.generate(
            self._initial_prompt(question, context), temperature=self.temperature
        )
        initial = answer
        history: List[str] = []

        for step in range(1, self.max_iterations + 1):
            feedback = self.model.generate(
                self._feedback_prompt(question, context, answer),
                temperature=self.temperature,
            ).strip()
            history.append(feedback)
            if not feedback or self.STOP_TOKEN in feedback.upper():
                return SelfRefineResult(initial, answer, step, history, "no_issues_reported")
            answer = self.model.generate(
                self._refine_prompt(question, context, answer, feedback),
                temperature=self.temperature,
            ).strip()

        return SelfRefineResult(initial, answer, self.max_iterations, history, "max_iterations")
