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
You are an expert fact-checker. Your task is to determine whether a Generated Answer
conveys the same core factual information as the Correct Answer.
 
Return ONLY a single float between 0 and 1:
- 0.0 = Generated Answer matches the Correct Answer (factual)
- 1.0 = Generated Answer contradicts or is completely different from the Correct Answer (hallucinated)
- Values in between reflect partial matches or uncertainty.
 
Synonyms, rephrasings, and added details are acceptable as long as the core fact is right.
For example, "ethanol" and "alcohol" should score low (they mean the same thing).
 
DO NOT explain your reasoning. Output ONLY the number.
 
--- EXAMPLES ---
 
Correct Answer: Arthur's Magazine
Generated Answer: Arthur's Magazine was started first in 1844.
Score: 0.05
 
Correct Answer: Arthur's Magazine
Generated Answer: First for Women was started first.
Score: 0.95
 
Correct Answer: Delhi
Generated Answer: The head office is in New Delhi, India.
Score: 0.08
 
Correct Answer: alcohol
Generated Answer: Cadmium chloride is slightly soluble in ethanol.
Score: 0.10
 
Correct Answer: American
Generated Answer: James Henry Miller's wife was Canadian.
Score: 0.92
 
--- YOUR TASK ---
 
Correct Answer: {correct_answer}
Generated Answer: {generated_answer}
Score:"""

    def __init__(
        self,
        judge_model: "BaseModel",
        threshold: float = DEFAULT_THRESHOLD,
    ):
        self.judge_model = judge_model
        self.threshold   = threshold

    # ── public API ────────────────────────────────────────────

    def detect(self, correct_answer: str, generated_answer: str) -> LLMDetectionResult:
        """
        Score how well the generated answer matches the correct answer.
 
        Parameters
        ----------
        correct_answer : str
            The ground-truth correct answer from the dataset.
        generated_answer : str
            The answer produced by the model (after applying any reducer).
 
        Returns
        -------
        LLMDetectionResult with:
          - score: 0.0 (match) to 1.0 (hallucinated)
          - is_hallucinated: True if score >= threshold
          - raw_response: the raw output from the judge model
        """
        prompt = self.JUDGE_PROMPT.format(
            correct_answer=correct_answer.strip(),
            generated_answer=generated_answer.strip(),
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
        self, correct_answers: list[str], generated_answers: list[str]
    ) -> list[LLMDetectionResult]:
        """Batch version of detect() - compares each generated answer to its correct answer."""
        prompts = [
            self.JUDGE_PROMPT.format(
                correct_answer=correct.strip(),
                generated_answer=gen.strip(),
            )
            for correct, gen in zip(correct_answers, generated_answers)
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
