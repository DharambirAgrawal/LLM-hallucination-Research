"""Synthetic offline backends. Never use these scores for research conclusions.

Only scripts/offline_replay.py activates these in-process stand-ins for
external neural packages. Production detector and reducer code is unchanged.
"""
from __future__ import annotations

import hashlib
import re
import types
import sys
from contextlib import contextmanager

import numpy as np

from models.replay_model import ReplayModel

QUESTIONS = [
    {"question": "When was Python first released?", "context": "Python was created by Guido van Rossum and first released in 1991.",
     "right_answer": "Python was first released in 1991.", "hallucinated_answer": "Python was first released in 1985 by James Gosling."},
    {"question": "What is the main pigment involved in photosynthesis?", "context": "Chlorophyll is the primary pigment that absorbs light during photosynthesis in plants.",
     "right_answer": "Chlorophyll is the primary pigment in photosynthesis.", "hallucinated_answer": "Melanin is the main pigment in photosynthesis and absorbs ultraviolet light."},
]
GOOD = [
    ["Python first became available in 1991.", "Guido van Rossum released Python in 1991.", "The initial Python release was in 1991."],
    ["Chlorophyll absorbs light for photosynthesis.", "The primary photosynthetic pigment is chlorophyll.", "Plants use chlorophyll as their main photosynthetic pigment."],
]
BAD = [
    ["Python was first released in 1985.", "James Gosling released Python in 1991.", "Python was released in 1985 by James Gosling."],
    ["Melanin is the primary pigment in photosynthesis.", "Chlorophyll primarily absorbs ultraviolet light in photosynthesis.", "Melanin absorbs ultraviolet light for photosynthesis."],
]


def fixture_risk(answer: str, evidence: str = "", variant: str = "") -> float:
    """A toy score determined by planted errors and stable textual variation."""
    errors = sum(word in answer.lower() for word in ("1985", "james gosling", "melanin", "ultraviolet"))
    jitter = int(hashlib.sha256((answer + evidence + variant).encode()).hexdigest()[:6], 16) % 65 / 1000
    return min(.96, .08 + .35 * errors + jitter)


class ScriptedReplayModel(ReplayModel):
    def __init__(self, name: str, model_index: int, run_index: int):
        super().__init__(["prepared replay response"], name=name)
        self.model_index, self.run_index = model_index, run_index
        self.calls = []

    def _question(self, prompt):
        return 0 if "Python" in prompt else 1

    def prepared(self, question, condition):
        shift = (self.model_index + self.run_index + question) % 3
        # Deliberately include improved, unchanged, and worsened methods.
        bad = {"baseline": (self.model_index + question) % 2 == 0,
               "closed_book": True, "greedy": self.model_index == 1,
               "self_refine_adapted": self.model_index == 3,
               "cove_adapted": self.model_index == 4}[condition]
        return (BAD if bad else GOOD)[question][shift]

    def generate(self, prompt, **kwargs):
        q = self._question(prompt)
        if prompt.endswith("Verification questions:"):
            kind, answer = "verification_plan", "1. When was Python first released?\n2. Who created Python?" if q == 0 else "1. What is the primary photosynthetic pigment?"
        elif prompt.endswith("Feedback:"):
            kind, answer = "feedback", "Correct unsupported details using the supplied context."
        elif "Verification:\n" in prompt:
            kind, answer = "cove_adapted", self.prepared(q, "cove_adapted")
        elif prompt.endswith("Revised answer:"):
            kind, answer = "self_refine_adapted", self.prepared(q, "self_refine_adapted")
        elif prompt.startswith("Answer the question using ONLY the supplied context. Be brief."):
            kind, answer = "verification_answer", GOOD[q][0]
        elif "Context:" not in prompt:
            kind, answer = "closed_book", self.prepared(q, "closed_book")
        elif kwargs.get("temperature") == 0:
            kind, answer = "greedy", self.prepared(q, "greedy")
        else:
            kind, answer = "baseline", self.prepared(q, "baseline")
        self.calls.append({"run": self.run_index, "model": self.name, "question": q,
                           "kind": kind, "answer": answer})
        return answer

    def sample_n(self, prompt, n=2, temperature=1.0):
        q = self._question(prompt)
        shift = (self.model_index + self.run_index) % 3
        samples = [GOOD[q][shift], "1991." if q == 0 else "Chlorophyll."]
        result = [samples[i % 2] for i in range(n)]
        self.calls.append({"run": self.run_index, "model": self.name, "question": q,
                           "kind": "sample_set", "answer": result})
        return result


class Span:
    def __init__(self, text):
        self.text = text
    def __len__(self):
        return len(re.findall(r"\w+|[^\w\s]", self.text))


class Tensor:
    def __init__(self, values):
        self.array = np.asarray(values)
    def reshape(self, *shape):
        return Tensor(self.array.reshape(*shape))
    def max(self, axis):
        return types.SimpleNamespace(values=types.SimpleNamespace(numpy=lambda: self.array.max(axis=axis)))


def fixture_modules():
    """Mock only external inference APIs, preserving real local adapters."""
    def module(name, **attrs):
        value = types.ModuleType(name)
        value.__dict__.update(attrs)
        return value
    def nlp(text):
        return types.SimpleNamespace(sents=[Span(s) for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()])

    class FakeSelfCheck:
        def __init__(self, **kwargs):
            pass
        def predict(self, sentences, sampled_passages, passage=None, **kwargs):
            values = [fixture_risk(s, "|".join(sampled_passages), "selfcheck") for s in sentences]
            return {"sent_level": {"avg_neg_logprob": [1 + 4 * v for v in values]}} if passage is not None else np.array(values)

    class FakeBERTScore(FakeSelfCheck):
        def __init__(self, **kwargs):
            self.nlp, self.default_model, self.rescale_with_baseline = nlp, "en", True
        def predict(self, sentences, sampled_passages, **kwargs):
            if any(not any(len(span) > 3 for span in nlp(sample).sents) for sample in sampled_passages):
                raise IndexError("synthetic reproduction of upstream empty sentence filter")
            return np.mean([[fixture_risk(sentence, sample, "bert") for sentence in sentences]
                            for sample in sampled_passages], axis=0)

    class FakeBERTScorer:
        def __init__(self, **kwargs):
            pass
        def score(self, cands, refs, **kwargs):
            f1 = Tensor([1 - fixture_risk(ref, cand, "bert") for cand, ref in zip(cands, refs)])
            return f1, f1, f1

    class FakeUQ:
        def __init__(self, scorers, **kwargs):
            self.scorers = scorers
        def score(self, responses, sampled_responses, **kwargs):
            return types.SimpleNamespace(data={s: [1 - fixture_risk(responses[0], "|".join(sampled_responses[0]), s)] for s in self.scorers})

    class FakeSelection:
        def __init__(self, **kwargs):
            pass
        def score(self, responses, sampled_responses, **kwargs):
            best = min([responses[0], *sampled_responses[0]], key=lambda s: fixture_risk(s))
            return types.SimpleNamespace(data={"responses": [best]})

    class FakeJudge:
        def __init__(self, **kwargs):
            pass
        async def judge_responses(self, prompts, responses):
            return {"scores": [1 - fixture_risk(answer, prompt, "judge") for prompt, answer in zip(prompts, responses)]}

    class FakeContextScorer:
        def __init__(self, **kwargs):
            pass
        def score(self, docs=None, claims=None, contexts=None, **kwargs):
            if isinstance(docs, list) and claims is not None and contexts is None:
                probs = [1 - fixture_risk(c, d, "minicheck") for d, c in zip(docs, claims)]
                return [int(p >= .5) for p in probs], probs, None, None
            return [1 - fixture_risk(c, d, "alignscore") for d, c in zip(contexts, claims)]

    class FakeSummaC:
        def __init__(self, **kwargs):
            pass
        def score(self, documents, claims):
            return {"scores": [1 - fixture_risk(c, d, "summac") for d, c in zip(documents, claims)]}

    return {"spacy": module("spacy", load=lambda name: nlp),
        "bert_score": module("bert_score", BERTScorer=FakeBERTScorer),
        "selfcheckgpt": module("selfcheckgpt"),
        "selfcheckgpt.modeling_selfcheck": module("selfcheckgpt.modeling_selfcheck", SelfCheckNgram=FakeSelfCheck, SelfCheckNLI=FakeSelfCheck, SelfCheckBERTScore=FakeBERTScore),
        "selfcheckgpt.modeling_selfcheck_apiprompt": module("selfcheckgpt.modeling_selfcheck_apiprompt", SelfCheckAPIPrompt=FakeSelfCheck),
        "uqlm": module("uqlm", BlackBoxUQ=FakeUQ, SemanticEntropy=FakeSelection),
        "uqlm.judges": module("uqlm.judges", LLMJudge=FakeJudge),
        "langchain_ollama": module("langchain_ollama", ChatOllama=lambda **kwargs: kwargs),
        "minicheck": module("minicheck"), "minicheck.minicheck": module("minicheck.minicheck", MiniCheck=FakeContextScorer),
        "summac": module("summac"), "summac.model_summac": module("summac.model_summac", SummaCConv=FakeSummaC, SummaCZS=FakeSummaC),
        "alignscore": module("alignscore", AlignScore=FakeContextScorer)}


@contextmanager
def synthetic_backends():
    """Restore only replaced packages, preserving newly imported real modules.

    patch.dict(sys.modules) restores its entire snapshot; that can unload
    multiprocessing's resource tracker while real metric libraries still own
    semaphores, producing spurious cleanup errors at interpreter shutdown.
    """
    missing = object()
    overrides = fixture_modules()
    previous = {name: sys.modules.get(name, missing) for name in overrides}
    sys.modules.update(overrides)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
