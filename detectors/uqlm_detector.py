"""Adapters around UQLM (CVS Health), Apache-2.0.

Upstream: https://github.com/cvs-health/uqlm (version pinned in
requirements.txt and provenance/sources.yaml). Paper: Bouchard et al.,
"UQLM: A Python Package for Uncertainty Quantification in Large Language
Models", arXiv:2507.06196.

UQLMConsistencyDetector: the official black-box (consistency) scorers,
    BlackBoxUQ(...).score(responses, sampled_responses), given the fixed
    answer and the shared SelfCheckGPT samples (detectors/sampling.py):
      semantic_negentropy  1 - normalized semantic entropy (Farquhar et al., 2024)
      noncontradiction     NLI non-contradiction probability
      entailment           NLI entailment probability
      cosine_sim           sentence-embedding cosine similarity
      exact_match          share of samples identical to the answer (short
                           answers only; not used by default, see config.yaml)
      bert_score           BERTScore F1
    `use_best=False` is required: with UQLM's default (True) the answer is
    replaced by the "best" sample BEFORE the consistency and similarity
    scorers run, so they would score a different answer than the one
    labeled. (Verified against uqlm 0.6.6 BlackBoxUQ.score.)

UQLMJudgeDetector: the official LLM-as-a-judge, LLMJudge(...).judge_responses,
    with its default "true_false_uncertain" template (Chen & Mueller, 2023),
    a local Ollama model as the judge, and the grounded prompt (context +
    question) as the judged question.

UQLM returns confidence (1 = trustworthy). Every value is reported here as
risk = 1 - confidence so that, like every other detector, higher means more
likely hallucinated. No UQLM scoring logic is reimplemented.
"""
from __future__ import annotations

import asyncio
import contextlib
import io
from typing import Dict, List, Optional, Sequence

from detectors.sampling import SampleBank
from models.prompts import grounded_prompt

CONSISTENCY_SCORERS = ("semantic_negentropy", "noncontradiction", "entailment",
                       "cosine_sim", "exact_match", "bert_score")
DEFAULT_SCORERS = ("semantic_negentropy", "noncontradiction", "entailment", "cosine_sim", "bert_score")


class UQLMConsistencyDetector:
    def __init__(self, bank: SampleBank, scorers: Sequence[str] = CONSISTENCY_SCORERS,
                 device: Optional[str] = None):
        unknown = set(scorers) - set(CONSISTENCY_SCORERS)
        if unknown or not scorers:
            raise ValueError(f"UQLM scorers must be among {CONSISTENCY_SCORERS}, got {list(scorers)}")
        self.bank = bank
        self.scorers = list(scorers)
        self.device = device
        self._uq = None

    def _load(self):
        if self._uq is not None:
            return
        try:
            from uqlm import BlackBoxUQ
        except ImportError as exc:
            raise RuntimeError("UQLM is not installed: pip install -r requirements.txt") from exc
        with contextlib.redirect_stdout(io.StringIO()):
            self._uq = BlackBoxUQ(scorers=self.scorers, device=self.device, use_best=False)

    def shared_nli(self):
        """The NLI model UQLM loaded for these scorers, so the best-response
        reducer can reuse it instead of loading a second copy (~1.6 GB)."""
        self._load()
        entropy = getattr(self._uq, "scorer_objects", {}).get("semantic_negentropy")
        return getattr(entropy, "nli", None)

    def detect(self, question: str, context: str, answer: str, model) -> Dict[str, float]:
        self._load()
        if not answer.strip():
            raise ValueError("empty answer cannot be scored")
        samples = self.bank.get(model, question, context)
        with contextlib.redirect_stdout(io.StringIO()):
            result = self._uq.score(responses=[answer], sampled_responses=[samples],
                                    show_progress_bars=False)
        return {name: 1.0 - float(result.data[name][0]) for name in self.scorers}


class UQLMJudgeDetector:
    def __init__(self, judge_model: str, host: str = "http://localhost:11434",
                 scoring_template: str = "true_false_uncertain"):
        self.judge_model = judge_model
        self.host = host
        self.scoring_template = scoring_template
        self._judge = None

    def _load(self):
        if self._judge is not None:
            return
        try:
            from langchain_ollama import ChatOllama
            from uqlm.judges import LLMJudge
        except ImportError as exc:
            raise RuntimeError("UQLM / langchain-ollama not installed: pip install -r requirements.txt") from exc
        llm = ChatOllama(model=self.judge_model, base_url=self.host, temperature=0)
        self._judge = LLMJudge(llm=llm, scoring_template=self.scoring_template)
        self._loop = asyncio.new_event_loop()

    def score_many(self, questions: List[str], contexts: List[str], answers: List[str]) -> List[float]:
        self._load()
        prompts = [grounded_prompt(q, c) for q, c in zip(questions, contexts)]
        with contextlib.redirect_stdout(io.StringIO()):
            # One event loop for the judge's lifetime: asyncio.run() would close
            # its loop after each call, while ChatOllama's HTTP client stays
            # bound to it, so every call after the first failed with
            # "Event loop is closed" (found in the first real smoke run).
            result = self._loop.run_until_complete(
                self._judge.judge_responses(prompts=prompts, responses=answers))
        scores = []
        for value in result["scores"]:
            if value is None or value != value:  # NaN: judge output could not be parsed
                raise ValueError("judge output could not be parsed after UQLM's retries")
            scores.append(1.0 - float(value))
        return scores

    def detect(self, question: str, context: str, answer: str) -> float:
        if not answer.strip():
            raise ValueError("empty answer cannot be scored")
        return self.score_many([question], [context], [answer])[0]
