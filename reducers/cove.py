"""Chain-of-Verification (CoVe), factored variant, for grounded QA.

Paper: Dhuliawala et al., "Chain-of-Verification Reduces Hallucination in
Large Language Models", Findings of ACL 2024 (arXiv:2309.11495).

No official code was released. This follows the paper's four steps and is
reported as a LOCAL IMPLEMENTATION of the published method:

  1. baseline response        (the grounded baseline answer, reused)
  2. plan verifications       the model writes verification questions about
                              the facts in its answer
  3. execute verifications    each question is answered SEPARATELY, without
                              the baseline answer in the prompt ("factored",
                              the paper's variant that avoids copying the
                              original mistakes), using the supplied context
  4. final verified response  the model revises its answer given the
                              question/answer pairs

Differences from the paper, all deliberate and reported: prompts are local
and zero-shot (the paper's are few-shot and task-specific: list questions,
closed-book QA, biographies); steps 3 and 4 include the supplied context,
because this benchmark is grounded QA while the paper's tasks are
closed-book; generation uses the model's default temperature (0.7).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

MAX_QUESTIONS = 5


@dataclass
class CoVeResult:
    initial_answer: str
    final_answer: str
    verification_questions: List[str] = field(default_factory=list)
    verification_answers: List[str] = field(default_factory=list)
    n_calls: int = 0


class ChainOfVerificationReducer:
    METHOD_ID = "cove_adapted"
    REPRODUCTION_STATUS = "local_implementation_of_published_method_no_official_code"
    SOURCE_PAPER = "https://arxiv.org/abs/2309.11495"

    def __init__(self, model, temperature: float = 0.7):
        self.model = model
        self.temperature = temperature

    @staticmethod
    def _plan_prompt(question: str, answer: str) -> str:
        return (
            "Below is a question and a draft answer. Write short verification "
            "questions that check each factual claim in the draft answer "
            f"(at most {MAX_QUESTIONS}). Write one question per line and nothing else.\n\n"
            f"Question: {question}\n\nDraft answer: {answer}\n\nVerification questions:"
        )

    @staticmethod
    def _verify_prompt(context: str, verification_question: str) -> str:
        return (
            "Answer the question using ONLY the supplied context. Be brief. If the "
            "context does not contain the answer, say so.\n\n"
            f"Context: {context}\n\nQuestion: {verification_question}\n\nAnswer:"
        )

    @staticmethod
    def _final_prompt(question: str, context: str, answer: str, qa: List[tuple]) -> str:
        checks = "\n".join(f"Q: {q}\nA: {a}" for q, a in qa)
        return (
            "Revise the draft answer so it is consistent with the verification "
            "results and the supplied context. Remove or correct anything the "
            "verification contradicts or cannot support. Reply with the revised "
            "answer only.\n\n"
            f"Context: {context}\n\nQuestion: {question}\n\nDraft answer: {answer}\n\n"
            f"Verification:\n{checks}\n\nRevised answer:"
        )

    @staticmethod
    def parse_questions(text: str) -> List[str]:
        """One question per line; a preamble such as "Here are the
        questions:" is skipped by preferring lines that end with '?'."""
        lines = []
        for line in text.splitlines():
            line = re.sub(r"^\s*(?:[-*•]|\d+[.)]|Q\d*[:.])\s*", "", line).strip()
            if line and len(line) > 3:
                lines.append(line)
        questions = [line for line in lines if line.endswith("?")] or lines
        return questions[:MAX_QUESTIONS]

    def reduce(self, question: str, context: str, initial_answer: str) -> CoVeResult:
        calls = 0
        plan = self.model.generate(self._plan_prompt(question, initial_answer), temperature=self.temperature)
        calls += 1
        questions = self.parse_questions(plan)
        if not questions:
            raise ValueError(f"no verification questions could be parsed from: {plan[:120]!r}")
        answers = []
        for vq in questions:  # factored: each in its own call, without the draft
            answers.append(self.model.generate(self._verify_prompt(context, vq), temperature=self.temperature).strip())
            calls += 1
        final = self.model.generate(
            self._final_prompt(question, context, initial_answer, list(zip(questions, answers))),
            temperature=self.temperature,
        ).strip()
        calls += 1
        return CoVeResult(initial_answer, final, questions, answers, calls)
