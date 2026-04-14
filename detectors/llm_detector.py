"""
Method 1: LLM Prompt-Based Hallucination Detector
==================================================
Uses an LLM as a judge.  The model reads (context, answer) and returns
a float in [0,1]:  0 = factual,  1 = hallucinated.

From the AWS blog: "The LLM prompt-based detector outperforms... accuracy ~75%"
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from models.base_model import BaseModel


@dataclass
class LLMDetectionResult:
    score: float            # 0=factual, 1=hallucinated
    is_hallucinated: bool
    raw_response: str


class LLMDetector:
    """
    LLM-as-a-judge hallucination detector.

    Can use any BaseModel as the judge — typically the same model
    that generated the answer, or a separate more powerful judge model.
    """

    # Threshold above which we call something hallucinated
    DEFAULT_THRESHOLD = 0.5

    # Few-shot prompt (mirroring the AWS blog's prompt design)
    JUDGE_PROMPT = """\
You are an expert assistant helping to check if a statement is based on the provided context.

Your task: Read the Context and the Statement. Return ONLY a single float between 0 and 1.
- 0.0 = the statement is fully supported by the context (factual)
- 1.0 = the statement contradicts or is not supported by the context (hallucinated)
- Values between 0 and 1 reflect your level of uncertainty.

DO NOT explain your reasoning. Output ONLY the number.

--- EXAMPLES ---

Context: Amazon Web Services (AWS) is a subsidiary of Amazon providing cloud computing services.
Statement: AWS is Amazon's cloud computing division.
Score: 0.05

Context: Amazon Web Services (AWS) is a subsidiary of Amazon providing cloud computing services.
Statement: AWS was founded in 1985 by Bill Gates.
Score: 0.98

Context: The Eiffel Tower is located in Paris, France. It was built between 1887 and 1889.
Statement: The Eiffel Tower is in London.
Score: 0.97

Context: The Eiffel Tower is located in Paris, France. It was built between 1887 and 1889.
Statement: The Eiffel Tower was constructed in the late 1880s.
Score: 0.04

--- YOUR TASK ---

Context: {context}
Statement: {statement}
Score:"""

    def __init__(
        self,
        judge_model: "BaseModel",
        threshold: float = DEFAULT_THRESHOLD,
    ):
        self.judge_model = judge_model
        self.threshold   = threshold

    # ── public API ────────────────────────────────────────────

    def detect(self, context: str, answer: str) -> LLMDetectionResult:
        """Score a single (context, answer) pair."""
        prompt = self.JUDGE_PROMPT.format(
            context=context.strip(),
            statement=answer.strip(),
        )
        try:
            raw = self.judge_model.generate(prompt, temperature=0.0, max_new_tokens=16)
            score = self._parse(raw)
        except Exception as exc:
            logger.warning(f"LLMDetector error: {exc}")
            raw, score = "", 0.5

        return LLMDetectionResult(
            score=score,
            is_hallucinated=score >= self.threshold,
            raw_response=raw,
        )

    def detect_batch(
        self, contexts: list[str], answers: list[str]
    ) -> list[LLMDetectionResult]:
        prompts = [
            self.JUDGE_PROMPT.format(
                context=ctx.strip(), statement=ans.strip()
            )
            for ctx, ans in zip(contexts, answers)
        ]
        try:
            raws = self.judge_model.generate_batch(
                prompts, temperature=0.0, max_new_tokens=16
            )
        except Exception as exc:
            logger.warning(f"LLMDetector batch error: {exc}")
            raws = [""] * len(prompts)

        return [
            LLMDetectionResult(
                score=self._parse(r),
                is_hallucinated=self._parse(r) >= self.threshold,
                raw_response=r,
            )
            for r in raws
        ]

    # ── helpers ──────────────────────────────────────────────

    @staticmethod
    def _parse(raw: str) -> float:
        import re
        m = re.search(r"\d+\.?\d*", raw.strip())
        if m:
            return min(max(float(m.group()), 0.0), 1.0)
        return 0.5
